# bare bones training script using sparse kernels and sharding/data parallel.
# the main purpose of this code is to provide a reference implementation to compare
# against when implementing our training methodology into other codebases, and to
# demonstrate how sharding/DP can be implemented for autoencoders. some limitations:
# - many basic features (e.g checkpointing, data loading, validation) are not implemented,
# - the codebase is not designed to be extensible or easily hackable.
# - this code is not guaranteed to run efficiently out of the box / in
#   combination with other changes, so you should profile it and make changes as needed.
#
# example launch command:
#    torchrun --nproc-per-node 8 train.py


import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
import glob
from tqdm import tqdm
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from sparse_autoencoder.kernels import *
from torch.distributed import ReduceOp
from torch.utils.data import TensorDataset, DataLoader

RANK = int(os.environ.get("RANK", "0")) # 定义了一个全局变量 RANK，用于分布式训练中的进程排名


## parallelism

# 定义通信类 Comm , 封装了分布式通信操作
@dataclass
class Comm:
    group: torch.distributed.ProcessGroup

    def all_reduce(self, x, op=ReduceOp.SUM, async_op=False):
        return dist.all_reduce(x, op=op, group=self.group, async_op=async_op)

    def all_gather(self, x_list, x, async_op=False):
        return dist.all_gather(list(x_list), x, group=self.group, async_op=async_op)

    def broadcast(self, x, src, async_op=False):
        return dist.broadcast(x, src, group=self.group, async_op=async_op)

    def barrier(self):
        return dist.barrier(group=self.group)

    def size(self):
        return self.group.size()

# 定义分片通信类 ShardingComms, 封装了分片通信操作，用于分布式训练中的数据并行和模型并行
@dataclass
class ShardingComms:
    n_replicas: int
    n_op_shards: int
    dp_rank: int
    sh_rank: int
    dp_comm: Comm | None
    sh_comm: Comm | None
    _rank: int

    def sh_allreduce_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sh_comm is None:
            return x

        class AllreduceForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                assert self.sh_comm is not None
                self.sh_comm.all_reduce(input, async_op=True)
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return AllreduceForward.apply(x)  # type: ignore

    def sh_allreduce_backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sh_comm is None:
            return x

        class AllreduceBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = grad_output.clone()
                assert self.sh_comm is not None
                self.sh_comm.all_reduce(grad_output, async_op=True)
                return grad_output

        return AllreduceBackward.apply(x)  # type: ignore

    def init_broadcast_(self, autoencoder):
        if self.dp_comm is not None:
            for p in autoencoder.parameters():
                self.dp_comm.broadcast(
                    maybe_transpose(p.data),
                    replica_shard_to_rank(
                        replica_idx=0,
                        shard_idx=self.sh_rank,
                        n_op_shards=self.n_op_shards,
                    ),
                )
        
        if self.sh_comm is not None:
            # pre_bias is the same across all shards
            self.sh_comm.broadcast(
                autoencoder.pre_bias.data,
                replica_shard_to_rank(
                    replica_idx=self.dp_rank,
                    shard_idx=0,
                    n_op_shards=self.n_op_shards,
                ),
            )

    def dp_allreduce_(self, autoencoder) -> None:
        if self.dp_comm is None:
            return

        for param in autoencoder.parameters():
            if param.grad is not None:
                self.dp_comm.all_reduce(maybe_transpose(param.grad), op=ReduceOp.AVG, async_op=True)

        # make sure statistics for dead neurons are correct
        self.dp_comm.all_reduce(  # type: ignore
            autoencoder.stats_last_nonzero, op=ReduceOp.MIN, async_op=True
        )

    def sh_allreduce_scale(self, scaler):
        if self.sh_comm is None:
            return

        if hasattr(scaler, "_scale") and scaler._scale is not None:
            self.sh_comm.all_reduce(scaler._scale, op=ReduceOp.MIN, async_op=True)
            self.sh_comm.all_reduce(scaler._growth_tracker, op=ReduceOp.MIN, async_op=True)

    def _sh_comm_op(self, x, op):
        if isinstance(x, (float, int)):
            x = torch.tensor(x, device="cuda")

        if not x.is_cuda:
            x = x.cuda()

        if self.sh_comm is None:
            return x

        out = x.clone()
        self.sh_comm.all_reduce(x, op=op, async_op=True)
        return out

    def sh_sum(self, x: torch.Tensor) -> torch.Tensor:
        return self._sh_comm_op(x, ReduceOp.SUM)

    def all_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        if self.dp_comm is not None:
            self.dp_comm.broadcast(
                x,
                replica_shard_to_rank(
                    replica_idx=0,
                    shard_idx=self.sh_rank,
                    n_op_shards=self.n_op_shards,
                ),
            )

        if self.sh_comm is not None:
            self.sh_comm.broadcast(
                x,
                replica_shard_to_rank(
                    replica_idx=self.dp_rank,
                    shard_idx=0,
                    n_op_shards=self.n_op_shards,
                ),
            )

        return x


# def make_torch_comms(n_op_shards=4, n_replicas=2):
#     if "RANK" not in os.environ:
#         assert n_op_shards == 1
#         assert n_replicas == 1
#         return TRIVIAL_COMMS

#     rank = int(os.environ.get("RANK"))
#     world_size = int(os.environ.get("WORLD_SIZE", 1))
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)

#     print(f"{rank=}, {world_size=}")
#     dist.init_process_group("nccl")

#     my_op_shard_idx = rank % n_op_shards
#     my_replica_idx = rank // n_op_shards

#     shard_rank_lists = [list(range(i, i + n_op_shards)) for i in range(0, world_size, n_op_shards)]

#     shard_groups = [dist.new_group(shard_rank_list) for shard_rank_list in shard_rank_lists]

#     my_shard_group = shard_groups[my_replica_idx]

#     replica_rank_lists = [
#         list(range(i, n_op_shards * n_replicas, n_op_shards)) for i in range(n_op_shards)
#     ]

#     replica_groups = [dist.new_group(replica_rank_list) for replica_rank_list in replica_rank_lists]

#     my_replica_group = replica_groups[my_op_shard_idx]

#     torch.distributed.all_reduce(torch.ones(1).cuda())
#     torch.cuda.synchronize()

#     dp_comm = Comm(group=my_replica_group)
#     sh_comm = Comm(group=my_shard_group)

#     return ShardingComms(
#         n_replicas=n_replicas,
#         n_op_shards=n_op_shards,
#         dp_comm=dp_comm,
#         sh_comm=sh_comm,
#         dp_rank=my_replica_idx,
#         sh_rank=my_op_shard_idx,
#         _rank=rank,
#     )


# 这是构建分布式通信组的关键代码, 尤其用于多 GPU, 多卡训练时做 data parallel (DP) 和 operation sharding (OP shard)
def make_torch_comms(n_op_shards=4, n_replicas=2):

    # 如果是单机单卡训练，直接返回 Dummy 通信对象 TRIVIAL_COMMS
    if "RANK" not in os.environ:
        assert n_op_shards == 1
        assert n_replicas == 1
        return TRIVIAL_COMMS

    # 获取当前进程在分布式中的 rank 和总 rank 数 (通常由 torchrun 自动注入)
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)

    print(f"{rank=}, {world_size=}")
    dist.init_process_group("nccl")

    my_op_shard_idx = rank % n_op_shards
    my_replica_idx = rank // n_op_shards

    shard_rank_lists = [list(range(i, i + n_op_shards)) for i in range(0, world_size, n_op_shards)]
    shard_groups = [dist.new_group(shard_rank_list) for shard_rank_list in shard_rank_lists]
    my_shard_group = shard_groups[my_replica_idx]

    replica_rank_lists = [
        list(range(i, n_op_shards * n_replicas, n_op_shards)) for i in range(n_op_shards)
    ]
    replica_groups = [dist.new_group(replica_rank_list) for replica_rank_list in replica_rank_lists]
    my_replica_group = replica_groups[my_op_shard_idx]

    torch.distributed.all_reduce(torch.ones(1).cuda())
    torch.cuda.synchronize()

    dp_comm = Comm(group=my_replica_group)
    sh_comm = Comm(group=my_shard_group)

    return ShardingComms(
        n_replicas=n_replicas,
        n_op_shards=n_op_shards,
        dp_comm=dp_comm,
        sh_comm=sh_comm,
        dp_rank=my_replica_idx,
        sh_rank=my_op_shard_idx,
        _rank=rank,
    )


def replica_shard_to_rank(replica_idx, shard_idx, n_op_shards):
    return replica_idx * n_op_shards + shard_idx


class DummyComm:
    def broadcast(self, x, src=0, group=None, async_op=False):
        return x

    def allreduce(self, x, op=None, group=None, async_op=False):
        return x

    def gather(self, x, dst=0):
        return [x]

    def all_gather(self, output_tensor_list, input_tensor, async_op=False):
        # 假设只有一个 replica，复制 input_tensor 到每个 output
        for i in range(len(output_tensor_list)):
            output_tensor_list[i].copy_(input_tensor)
        return output_tensor_list

    def size(self):
        return 1


class DummyComms:
    def __init__(self):
        self.n_replicas = 1
        self.n_op_shards = 1
        self.dp_rank = 0
        self.sh_rank = 0
        self._rank = 0
        self.dp_comm = DummyComm()
        self.sh_comm = DummyComm()

    def dp_allreduce_(self, model): pass
    def sh_allreduce_scale(self, scaler): pass
    def sh_allreduce_forward(self, x): return x
    def sh_allreduce_backward(self, x): return x
    def all_broadcast(self, x): return x 
    def all_reduce(self, x): return x
    def broadcast(self, x): return x
    def gather(self, x): return [x]
    def init_broadcast_(self, model): pass


TRIVIAL_COMMS = DummyComms()


# TRIVIAL_COMMS = ShardingComms(
#     n_replicas=1,
#     n_op_shards=1,
#     dp_rank=0,
#     sh_rank=0,
#     dp_comm=None,
#     sh_comm=None,
#     _rank=0,
# )

# 定义了一个名为 sharded_topk 的分片 TopK 函数, 目的是在分布式训练环境中高效地执行 Top-K 操作
# 在分布式环境中, 这个函数需要跨多个分片 (shards) 聚合信息以确保每个样本的 Top-K 值是全局最大的 K 个值
def sharded_topk(x, k, sh_comm, capacity_factor=None):

    '''
    x: 输入的张量，形状为 [batch, n_dirs_total], 其中 batch 是批次大小, n_dirs_total 是潜在空间的总维度
    k: 每个样本要选出的最大值的数量
    sh_comm: 分片通信对象, 用于在不同的分片之间进行通信。
    capacity_factor: 可选参数, 用于调整每个分片处理的元素数量。
    '''

    batch = x.shape[0] # [batch_size, n_dirs_local]

    # 根据给定的 capacity_factor 调整每个分片 (shard) 在分布式环境中处理的 Top-K 操作的数量
    if capacity_factor is not None:
        k_in = min(int(k * capacity_factor // sh_comm.size()), k)
    else:
        k_in = k

    topk = torch.topk(x, k=k_in, dim=-1) # 在最后一个维度 (特征维度) 上执行 Top-K 操作, 选出每个样本的 k_in 个最大值及其索引
    inds = topk.indices # 每个样本的 k_in 个最大值对应的索引
    vals = topk.values # 每个样本的 k_in 个最大值

    if sh_comm is None: # 如果没有提供分片通信对象 (即不在分布式环境中), 直接返回当前分片计算得到的索引和值
        return inds, vals

    # 创建一个形状为 [sh_comm.size(), batch, k_in] 的空张量 all_vals, 用于存储所有分片的 Top-K 值
    all_vals = torch.empty(sh_comm.size(), batch, k_in, dtype=vals.dtype, device=vals.device)

    # 使用分片通信对象 sh_comm 的 all_gather 方法, 将 vals 收集到 all_vals 中分片保存
    # all_vals 的形状: [sh_comm.size(), batch, k_in], vals 的形状: [batch, k_in]
    # async_op=True 表示这个操作是异步的, 即函数会立即返回, 而不会等待操作完成
    sh_comm.all_gather(all_vals, vals, async_op=True)

    all_vals = all_vals.permute(1, 0, 2) # put shard dim next to k ([sh_comm.size(), batch, k_in]-->[batch, sh_comm.size(), k_in])
    all_vals = all_vals.reshape(batch, -1) # flatten shard into k ([batch, sh_comm.size(), k_in]-->[batch, sh_comm.size()*k_in])

    all_topk = torch.topk(all_vals, k=k, dim=-1) 
    global_topk = all_topk.values

    # 创建与 vals 和 inds 形状相同的零张量 dummy_vals 和 dummy_inds, 用于后续的掩码操作
    dummy_vals = torch.zeros_like(vals)
    dummy_inds = torch.zeros_like(inds)

    my_inds = torch.where(vals >= global_topk[:, [-1]], inds, dummy_inds)
    my_vals = torch.where(vals >= global_topk[:, [-1]], vals, dummy_vals)

    return my_inds, my_vals


## autoencoder
class FastAutoencoder(nn.Module):
    """
    Top-K Autoencoder with sparse kernels. Implements:

        latents = relu(topk(encoder(x - pre_bias) + latent_bias))
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None,
        dead_steps_threshold: int,
        comms: ShardingComms | None = None,
    ):
        
        '''
        n_dirs_local: 每个 GPU 或分片上的潜在空间维度数
        d_model: 输入数据的维度
        k: Top-K 激活函数中保留的最大值的数量
        auxk: 辅助 Top-K 激活函数中保留的最大值的数量, 可能为 None
        dead_steps_threshold: 用于跟踪死亡神经元 (长时间未激活的神经元) 的阈值
        comms: 分片通信对象, 如果为 None, 则使用默认的 TRIVIAL_COMMS
        '''

        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.comms = comms if comms is not None else TRIVIAL_COMMS
        self.dead_steps_threshold = dead_steps_threshold

        # 初始化编码器 (self.encoder) 和解码器 (self.decoder) 为线性层, 且不包含偏置项 (bias=False)
        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        # 初始化 pre_bias 和 latent_bias 为可学习的参数, 分别用于输入数据和潜在表示的偏置
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        # 注册一个名为 stats_last_nonzero 的缓冲区, 用于跟踪每个潜在维度最后一次非零激活的时间步数
        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))

        # 定义一个辅助函数 auxk_mask_fn, 用于根据 stats_last_nonzero 和 dead_steps_threshold 生成掩码, 该掩码用于稀疏化潜在表示
        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_steps_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

        ## initialization

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone() # 仅仅是在初始化的时候共享参数, 后续训练更新参数时不共享

        # store decoder in column major layout for kernel
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T

        unit_norm_decoder_(self)

    # 定义了一个 Python 属性, 用于获取模型的总维度, 考虑了数据并行的情况
    @property
    def n_dirs(self):
        return self.n_dirs_local * self.comms.n_op_shards

    def forward(self, x): # x: [batch_size, d_model]

        '''
        为何要在 forward 里定义 EncWrapper?

        1) 捕获实例状态
        将 EncWrapper 嵌在 forward 里, 可以直接闭包引用各种实例属性 (self.k, self.auck 等), 而不必把它们都当作参数显式传给 apply

        2) 自定义高效梯度 (主要原因)
        论文中需要对 Top-K 进行稀疏散射 (scatter) 和梯度汇总, 并更新 dead-latent 统计, 这些都超出 PyTorch 原生算子的能力
        用自定义 autograd.Function 可以在前向/反向中手写高性能 kernel (如 Triton 内核), 并且一次性处理好所有逻辑

        3) 隔离命名空间
        将专门针对这个模块的前/反向逻辑封装成局部类, 不会污染全局命名, 也方便以后维护和阅读

        4) 灵活切换行为
        在 forward 中根据 self.auxk is not None 动态决定是否启用 auxiliary Top-K, 且能在同一个 Function 内完成主、辅两套逻辑
        '''

        class EncWrapper(torch.autograd.Function): # 使用 EncWrapper, 一个自定义的 torch.autograd.Function, 来处理稀疏 Top-K 激活和编码过程
            
            @staticmethod # 静态方法装饰器, 不依赖于类的实例 (不依赖于 self 参数)
            def forward(ctx, x, pre_bias, weight, latent_bias):

                '''
                ctx: 上下文对象, 用于在前向传播和反向传播之间传递信息
                x: 输入张量, 形状为 [batch_size, d_model]
                pre_bias: 偏置项, 用于调整输入数据
                weight: 编码器的权重
                latent_bias: 潜在空间的偏置项
                '''

                x = x - pre_bias
                latents_pre_act = F.linear(x, weight, latent_bias) # [batch_size, n_dirs_local]

                # 调用 sharded_topk 函数, 选择每个样本的前 k 个最大值及其索引
                # inds 是索引张量, 形状为 [batch_size, k], vals 是值张量, 形状为 [batch_size, k]
                inds, vals = sharded_topk(
                    latents_pre_act,
                    k=self.k,
                    sh_comm=self.comms.sh_comm,
                    capacity_factor=4,
                )

                ## set num nonzero stat ##
                # 创建一个与 self.stats_last_nonzero 形状相同的零张量 tmp, 该张量用于跟踪每个潜在维度自上次激活以来的步数
                tmp = torch.zeros_like(self.stats_last_nonzero) # tmp: [n_dirs_local]

                # scatter_add_ 是一个原地操作, 先将 vals 和 inds>1e-3 (bool张量) 展平为一维张量
                # 然后将 (vals > 1e-3).to(tmp.dtype).reshape(-1) 的值累加到 tmp 中由 inds.reshape(-1) 指定的位置
                tmp.scatter_add_(
                    0,
                    inds.reshape(-1), # [batch_size*k]
                    (vals > 1e-3).to(tmp.dtype).reshape(-1), # [batch_size*k]
                ) # 至此, tmp 张量包含了每个潜在维度自上次更新以来的激活次数

                # 如果某个潜在维度在当前批次中被激活, stats_last_nonzero 对应位置归零; 否则 stats_last_nonzero 对应位置取值不变
                self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
                self.stats_last_nonzero += 1 # 将 stats_last_nonzero 所有分量加一, 表示每个潜在维度自上次更新以来又经过了一个时间步
                ## end stats ##

                ## auxk (辅助 Top-K 操作)
                # sparse autoencoder 在训练过程中会出现  dead latents 现象, 导致网络结构和计算资源的浪费
                # 为了解决该问题, 我们利用辅助 Top-K 函数定向激活那些 dead latents
                if self.auxk is not None:  # for auxk
                    # IMPORTANT: has to go after stats update!
                    # WARN: auxk_mask_fn can mutate latents_pre_act!
                    auxk_inds, auxk_vals = sharded_topk(
                        self.auxk_mask_fn(latents_pre_act), # 应用一个掩码函数 auxk_mask_fn 到潜在表示 latents_pre_act 上
                        k=self.auxk,
                        sh_comm=self.comms.sh_comm,
                        capacity_factor=2,
                    )

                    # ctx (context) 是 torch.autograd.Function 类的一个属性, 用于在自定义的 autograd 函数中保存用于反向传播所需的中间变量
                    # ctx.save_for_backward 是 ctx 上的一个方法, 它的作用是保存前向传播中的中间结果, 以便在反向传播时可以访问这些结果
                    # 这对于自定义 autograd 函数来说非常重要, 因为它允许你手动实现反向传播的逻辑
                    ctx.save_for_backward(x, weight, inds, auxk_inds)
                else:
                    ctx.save_for_backward(x, weight, inds)
                    auxk_inds = None
                    auxk_vals = None

                ## end auxk

                return (
                    inds,
                    vals,
                    auxk_inds,
                    auxk_vals,
                )

            @staticmethod
            def backward(ctx, _, grad_vals, __, grad_auxk_vals):

                '''
                ctx: ctx 是一个上下文对象, 用于在前向传播和反向传播之间传递信息。它保存了前向传播中需要的张量, 以便在反向传播中使用
                _: 在反向传播中, 这个参数代表输入 x 的梯度, 但在自定义的 backward 方法中通常不需要使用, 所以用 _ 忽略
                grad_vals: 损失函数相对于主 Top-K 激活值 vals 的梯度, 形状为 [batch_size, k]
                **__: 在 backward 方法中, 这个参数代表 inds 的梯度, 但在自定义的 backward 方法中通常不需要使用, 所以用 __ 忽略
                grad_auxk_vals: 损失函数相对于辅助 Top-K 激活值 auxk_vals 的梯度, 形状为 [batch_size, auxk]
                '''

                # encoder backwards
                if self.auxk is not None: # 如果启用了辅助 Top-K

                    # 从 ctx 中提取保存的张量: 输入 x, 权重 weight, 主 Top-K 的索引 inds 和辅助 Top-K 的索引 auxk_inds
                    x, weight, inds, auxk_inds = ctx.saved_tensors # 这里就体现出 forward 函数中 ctx.save_for_backward 的作用
                    
                    # 将 Top-K 和辅助 Top-K 得到的索引拼接, all_inds: [batch_size, k+auxk]
                    all_inds = torch.cat((inds, auxk_inds), dim=-1)

                    # 将 Top-K 和辅助 Top-K 对应激活值的梯度拼接, all_grad_vals: [batch_size, k+auxk]
                    all_grad_vals = torch.cat((grad_vals, grad_auxk_vals), dim=-1)

                else: # 如果没有启用辅助 Top-K

                    # 从 ctx 中提取保存的张量: 输入 x, 权重 weight 和主 Top-K 的索引 inds
                    x, weight, inds = ctx.saved_tensors

                    all_inds = inds
                    all_grad_vals = grad_vals

                # 创建一个形状为 [self.n_dirs_local] 的零张量 grad_sum, 用于存储每个潜在维度的梯度和
                grad_sum = torch.zeros(self.n_dirs_local, dtype=torch.float32, device=grad_vals.device)
                grad_sum.scatter_add_(
                    -1, all_inds.flatten(), all_grad_vals.flatten().to(torch.float32)
                ) # 使用 scatter_add_ 将 all_grad_vals 中的值根据 all_inds 中的索引累加到 grad_sum 中

                '''
                回顾前向传播 forward:
                x: [B, d_model]
                pre_bias: [d_model]
                weight: [n_dirs_local, d_model]
                latent_bias: [n_dirs_local]

                1) 先做 bias 校正 x' = x - pre_bias  (x': [B, d_model])
                2) 然后进行线性映射 Z = x' @ weight.T + latent_bias  (Z: [B, n_dirs_local])
                3) 最后利用 TopK 选出 k 个最大值 inds, vals = topk(Z, k)  (inds: [B, k], vals: [B, k], vals[b,i] = Z[b, inds[b,i]])
                '''

                return (
                    None, # 输入 x 的梯度 (由于我们并不需要所以赋为 None)
                    # pre_bias grad optimization - can reduce before mat-vec multiply
                    -(grad_sum @ weight), # pre_bias 的梯度
                    triton_sparse_transpose_dense_matmul(all_inds, all_grad_vals, x, N=self.n_dirs_local), # 权重 weight 的梯度
                    grad_sum, # latent_bias 的梯度
                )

        '''
        sh_allreduce_backward 是跨 shard (切分) 的后向 all-reduce 操作封装, 它的行为是:
        前向：直接返回输入的 pre_bias, 不做修改
        反向：在反向传播时，对梯度做一次 all-reduce, 保证每个 shard 上的 pre_bias.grad 都是所有 shard 梯度的和
        作用：确保多卡/多 shard 下 pre_bias 的梯度正确汇总, 而在前向不用额外通信, 节省开销
        '''
        pre_bias = self.comms.sh_allreduce_backward(self.pre_bias)

        # encoder
        '''
        回顾: EncWrapper 是一个嵌在 forward 里的自定义 torch.autograd.Function, 
        
        在前向传播过程中:
        1) 先做 bias 校正 x' = x - pre_bias  (x': [B, d_model])
        2) 接着进行线性映射 Z = x' @ weight.T + latent_bias  (Z: [B, n_dirs_local])
        3) 然后利用 TopK 选出 k 个最大值 inds, vals = topk(Z, k)  (inds: [B, k], vals: [B, k], vals[b,i] = Z[b, inds[b,i]])
        4) 最后返回 inds, vals (主 Top-K 的索引和值) 以及 auxk_inds, auxk_vals (辅助 Top-K (如果启用) 的索引和值)

        在反向传播过程中:
        EncWrapper.backward 会根据这些 inds 把梯度 scatter 回对应的 latent 维度，并计算出 pre_bias, weight, latent_bias 的梯度
        '''
        inds, vals, auxk_inds, auxk_vals = EncWrapper.apply(
            x, pre_bias, self.encoder.weight, self.latent_bias
        )

        '''
        在 PyTorch 里, 任何继承自 torch.autograd.Function 的自定义自动求导函数都不直接实例化来调用, 而是通过它们的类方法 .apply() 来执行

        当你写了一个类似于上面代码中 EncWrapper 的类, 调用时你并不会写 EncWrapper(), 而是写 EncWrapper.apply(), 这时:
        PyTorch 在内部创建了一个临时的 Function 对象来管理本次前向/后向计算的上下文 (ctx)
        调用 EncWrapper.forward(ctx, ...) 来执行前向逻辑, 将返回的张量包裹成支持自动求导的节点, 并且把 ctx 中保存的信息链接到计算图上
        当后续执行 .backward() 时, PyTorch 会自动调用 EncWrapper.backward(ctx, grad_outputs…), 并把梯度正确地分发到输入参数上

        为什么用 .apply()?
        1) 统一接口: 所有自定义 Function 都用 .apply(...) 来执行, 用户无需关心内部实例化细节
        2) 自动构建计算图: .apply() 会帮你把前向的输入输出和保存的上下文挂到计算图里, 以便在反向时正确触发 backward
        3) 参数透明：.apply 接受的参数就是 forward 里定义的那些张量, 返回值就是 forward 的输出, 使用上非常直观
        '''

        vals = torch.relu(vals)
        if auxk_vals is not None:
            auxk_vals = torch.relu(auxk_vals)

        recons = self.decode_sparse(inds, vals) # recons 的计算没用到

        return recons, {
            "auxk_inds": auxk_inds,
            "auxk_vals": auxk_vals,
        } # forward 函数的返回值包含两部分, recons 和辅助 Top-K 函数的返回值以及索引构成的字典

    def decode_sparse(self, inds, vals):
        recons = TritonDecoderAutograd.apply(inds, vals, self.decoder.weight)
        recons = self.comms.sh_allreduce_forward(recons)

        return recons + self.pre_bias


def unit_norm_decoder_(autoencoder: FastAutoencoder) -> None:
    """
    Unit normalize the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)


def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:
    """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""

    assert autoencoder.decoder.weight.grad is not None

    triton_add_mul_(
        autoencoder.decoder.weight.grad,
        torch.einsum("bn,bn->n", autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad),
        autoencoder.decoder.weight.data,
        c=-1,
    )


def maybe_transpose(x):
    return x.T if not x.is_contiguous() and x.T.is_contiguous() else x


def sharded_grad_norm(autoencoder, comms, exclude=None):
    if exclude is None:
        exclude = []
    total_sq_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    exclude = set(exclude)

    total_num_params = 0
    for param in autoencoder.parameters():
        if param in exclude:
            continue
        if param.grad is not None:
            sq_norm = ((param.grad).float() ** 2).sum()
            if param is autoencoder.pre_bias:
                total_sq_norm += sq_norm  # pre_bias is the same across all shards
            else:
                total_sq_norm += comms.sh_sum(sq_norm)

            param_shards = comms.n_op_shards if param is autoencoder.pre_bias else 1
            total_num_params += param.numel() * param_shards

    return total_sq_norm.sqrt()


def batch_tensors(
    it: Iterable[torch.Tensor],
    batch_size: int,
    drop_last=True,
    stream=None,
) -> Iterator[torch.Tensor]:
    """
    input is iterable of tensors of shape [batch_old, ...]
    output is iterable of tensors of shape [batch_size, ...]
    batch_old does not need to be divisible by batch_size
    """

    tensors = []
    batch_so_far = 0

    for t in it:
        tensors.append(t)
        batch_so_far += t.shape[0]

        if sum(t.shape[0] for t in tensors) < batch_size:
            continue

        while batch_so_far >= batch_size:
            if len(tensors) == 1:
                (concat,) = tensors
            else:
                with torch.cuda.stream(stream):
                    concat = torch.cat(tensors, dim=0)

            offset = 0
            while offset + batch_size <= concat.shape[0]:
                yield concat[offset : offset + batch_size]
                batch_so_far -= batch_size
                offset += batch_size

            tensors = [concat[offset:]] if offset < concat.shape[0] else []

    if len(tensors) > 0 and not drop_last:
        yield torch.cat(tensors, dim=0)


def print0(*a, **k):
    if RANK == 0:
        print(*a, **k)


import wandb


class Logger:
    def __init__(self, **kws):
        self.vals = {}
        self.enabled = (RANK == 0) and not kws.pop("dummy", False)
        if self.enabled:
            wandb.init(
                **kws
            )

    def logkv(self, k, v):
        if self.enabled:
            self.vals[k] = v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        if self.enabled:
            wandb.log(self.vals)
            self.vals = {}


def training_loop_(
    ae, train_acts_iter, loss_fn, lr, comms, eps=6.25e-10, clip_grad=None, ema_multiplier=0.999, logger=None
):
    if logger is None:
        logger = Logger(dummy=True)

    scaler = torch.cuda.amp.GradScaler()
    autocast_ctx_manager = torch.cuda.amp.autocast()

    opt = torch.optim.Adam(ae.parameters(), lr=lr, eps=eps, fused=True)
    if ema_multiplier is not None:
        ema = EmaModel(ae, ema_multiplier=ema_multiplier)

    num_epochs = 200
    pbar = tqdm(range(num_epochs), desc="Epoch", dynamic_ncols=True)

    for epoch in pbar:
        for i, flat_acts_train_batch in enumerate(train_acts_iter):

            # print(print(f"flat_acts_train_batch shape: {flat_acts_train_batch.shape}"))
            # exit()

            flat_acts_train_batch = flat_acts_train_batch[0].cuda()

            with autocast_ctx_manager:
                recons, info = ae(flat_acts_train_batch)
                loss = loss_fn(ae, flat_acts_train_batch, recons, info, logger)

            # print0(i, loss)
            logger.logkv("loss_scale", scaler.get_scale())

            if RANK == 0:
                if wandb.run is not None:
                    wandb.log({"train_loss": loss.item()})

            # 显示实时训练信息在 tqdm 尾部
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            loss = scaler.scale(loss)
            loss.backward()

            unit_norm_decoder_(ae)
            unit_norm_decoder_grad_adjustment_(ae)

            # allreduce gradients
            comms.dp_allreduce_(ae)

            # keep fp16 loss scale synchronized across shards
            comms.sh_allreduce_scale(scaler)

            # if you want to do anything with the gradients that depends on the absolute scale (e.g clipping, do it after the unscale_)
            scaler.unscale_(opt)

            # gradient clipping
            if clip_grad is not None:
                grad_norm = sharded_grad_norm(ae, comms)
                logger.logkv("grad_norm", grad_norm)
                grads = [x.grad for x in ae.parameters() if x.grad is not None]
                torch._foreach_mul_(grads, clip_grad / torch.clamp(grad_norm, min=clip_grad))

            if ema_multiplier is not None:
                ema.step()

            # take step with optimizer
            scaler.step(opt)
            scaler.update()
            logger.dumpkvs()
        
        # Checkpoint 每 20 个 epoch 保存一次
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(epoch)
            save_path = os.path.join("/data2/angli/SAE_checkpoint", f"epoch_{epoch + 1}.pt")
            torch.save({
                "model_state_dict": ae.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch + 1,
            }, save_path)
            tqdm.write(f"Saved checkpoint at {save_path}")


# 这是 SAE 模型训练前的初始化步骤 (从数据中提取统计量进行初始化)
def init_from_data_(ae, stats_acts_sample, comms):
    from geom_median.torch import compute_geometric_median

    ae.pre_bias.data = (
        compute_geometric_median(stats_acts_sample[:32768].float().cpu()).median.cuda().float()
        # 从前 32768 条样本中提取几何中位数, 作为 pre_bias (减去这个偏置后能让每个样本都大致居中分布)
    )
    comms.all_broadcast(ae.pre_bias.data) # 如果使用多卡训练，要让每个 GPU 的 pre_bias 一致

    # encoder initialization (note: in our ablations we couldn't find clear evidence that this is beneficial, this is just to ensure exact match with internal codebase)
    d_model = ae.d_model

    # 初始化 encoder 的 scale
    with torch.no_grad():
        x = torch.randn(256, d_model).cuda().to(stats_acts_sample.dtype) # 构造一组均值为 0 的输入
        x /= x.norm(dim=-1, keepdim=True) # 归一化
        x += ae.pre_bias.data
        comms.all_broadcast(x)
        recons, _ = ae(x) # 加上 pre_bias 后输入模型, 得到重构输出 recons
        recons_norm = (recons - ae.pre_bias.data).norm(dim=-1).mean()

        # 计算输出的均值 norm 后, 用它来缩放 encoder 的权重, 目的是让输出的重构初始大小 ≈ 输入大小 (避免一开始梯度太爆或太小)
        ae.encoder.weight.data /= recons_norm.item()
        print0("x norm", x.norm(dim=-1).mean().item())
        print0("out norm", (ae(x)[0] - ae.pre_bias.data).norm(dim=-1).mean().item())


from contextlib import contextmanager


@contextmanager
def temporary_weight_swap(model: torch.nn.Module, new_weights: list[torch.Tensor]):
    for _p, new_p in zip(model.parameters(), new_weights, strict=True):
        assert _p.shape == new_p.shape
        _p.data, new_p.data = new_p.data, _p.data

    yield

    for _p, new_p in zip(model.parameters(), new_weights, strict=True):
        assert _p.shape == new_p.shape
        _p.data, new_p.data = new_p.data, _p.data


class EmaModel:
    def __init__(self, model, ema_multiplier):
        self.model = model
        self.ema_multiplier = ema_multiplier
        self.ema_weights = [torch.zeros_like(x, requires_grad=False) for x in model.parameters()]
        self.ema_steps = 0

    def step(self):
        torch._foreach_lerp_(
            self.ema_weights,
            list(self.model.parameters()),
            1 - self.ema_multiplier,
        )
        self.ema_steps += 1

    # context manager for setting the autoencoder weights to the EMA weights
    @contextmanager
    def use_ema_weights(self):
        assert self.ema_steps > 0

        # apply bias correction
        bias_correction = 1 - self.ema_multiplier**self.ema_steps
        ema_weights_bias_corrected = torch._foreach_div(self.ema_weights, bias_correction)

        with torch.no_grad():
            with temporary_weight_swap(self.model, ema_weights_bias_corrected):
                yield


@dataclass
class Config:
    n_op_shards: int = 1 # 操作并行
    n_replicas: int = 1 # 数据并行

    n_dirs: int = 32768
    bs: int = 131072
    d_model: int = 768
    k: int = 32
    auxk: int = 256

    lr: float = 1e-4
    eps: float = 6.25e-10
    clip_grad: float | None = None
    auxk_coef: float = 1 / 32
    dead_toks_threshold: int = 10_000_000
    ema_multiplier: float | None = None
    
    # 训练完成后, 可以使用 wandb 的 API 来查看训练过程中的指标和参数
    wandb_project: str | None = None
    wandb_name: str | None = None


def load_all_activations(data_dir):
    paths = sorted([
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir) if fname.endswith(".pt")
    ])
    return torch.cat([torch.load(p) for p in paths], dim=0) # 将 25 个 chunks 的数据拼接成一个完整的 chunk


def build_epoch_dataloader(all_acts_tensor, batch_size):
    dataset = TensorDataset(all_acts_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,     # 每个 epoch 打乱顺序
        drop_last=True,   # 确保每个 batch 大小一致
    )
    return loader


def main():
    cfg = Config()
    comms = make_torch_comms(n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)

    ## dataloading is left as an exercise for the reader
    chunk_paths = sorted(glob.glob("/data4/angli/SAE_training_data/chunk_*.pt")) # 25 个排好序的路径: chunk_1-25

    # acts_iter = ...
    # print(f"First chunk shape: {next(iter(acts_iter)).shape}")

    # 加载整个训练所用的数据集 (25_000_000 tokens 的激活值)
    all_acts = load_all_activations("/data4/angli/SAE_training_data") # [25_000_000, d_model]
    print(f"all_acts shape: {all_acts.shape}")
    # exit()

    sample_list = []
    total = 0
    target_sample_count = 100_000

    for path in chunk_paths:
        data = torch.load(path, map_location="cpu")
        needed = target_sample_count - total
        if needed <= 0:
            break
        sample_list.append(data[:needed])
        total += min(needed, len(data))

    # 从全体训练数据 all_acts 中获取前 100_000 tokens 的激活值, 用于计算样本均值和方差
    stats_acts_sample = torch.cat(sample_list, dim=0)  # shape [100_000, d_model]

    print(f"stats_acts_sample shape: {stats_acts_sample.shape}")
    # exit()

    n_dirs_local = cfg.n_dirs // cfg.n_op_shards
    bs_local = cfg.bs // cfg.n_replicas

    # 用加载好的整个数据集构建可迭代的 dataloader
    train_acts_loader = build_epoch_dataloader(all_acts, batch_size=bs_local)

    ae = FastAutoencoder(
        n_dirs_local=n_dirs_local,
        d_model=cfg.d_model,
        k=cfg.k,
        auxk=cfg.auxk,
        dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
        comms=comms,
    )
    ae.cuda()
    init_from_data_(ae, stats_acts_sample, comms)
    # IMPORTANT: make sure all DP ranks have the same params
    comms.init_broadcast_(ae)

    mse_scale = (
        1 / ((stats_acts_sample.float().mean(dim=0) - stats_acts_sample.float()) ** 2).mean()
    )
    mse_scale = mse_scale.to(torch.device("cuda"))
    comms.all_broadcast(mse_scale)
    mse_scale = mse_scale.item()

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=cfg.wandb_project is None,
    )

    # training_loop_(
    #     ae,
    #     batch_tensors(
    #         acts_iter,
    #         bs_local,
    #         drop_last=True,
    #     ),
    #     lambda ae, flat_acts_train_batch, recons, info, logger: (
    #         # MSE
    #         logger.logkv("train_recons", mse_scale * mse(recons, flat_acts_train_batch))
    #         # AuxK
    #         + logger.logkv(
    #             "train_maxk_recons",
    #             cfg.auxk_coef
    #             * normalized_mse(
    #                 ae.decode_sparse(
    #                     info["auxk_inds"],
    #                     info["auxk_vals"],
    #                 ),
    #                 flat_acts_train_batch - recons.detach() + ae.pre_bias.detach(),
    #             ).nan_to_num(0),
    #         )
    #     ),
    #     lr=cfg.lr,
    #     eps=cfg.eps,
    #     clip_grad=cfg.clip_grad,
    #     ema_multiplier=cfg.ema_multiplier,
    #     logger=logger,
    #     comms=comms,
    # )

    training_loop_(
        ae,
        train_acts_loader,
        lambda ae, flat_acts_train_batch, recons, info, logger: (
            # MSE
            logger.logkv("train_recons", mse_scale * mse(recons, flat_acts_train_batch))
            # AuxK
            + logger.logkv(
                "train_maxk_recons",
                cfg.auxk_coef
                * normalized_mse(
                    ae.decode_sparse(
                        info["auxk_inds"],
                        info["auxk_vals"],
                    ),
                    flat_acts_train_batch - recons.detach() + ae.pre_bias.detach(),
                ).nan_to_num(0),
            )
        ),
        lr=cfg.lr,
        eps=cfg.eps,
        clip_grad=cfg.clip_grad,
        ema_multiplier=cfg.ema_multiplier,
        logger=logger,
        comms=comms,
    )

    # 保存训练后的模型
    save_path = "/data2/angli/SAE_checkpoint/sae_final.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ae.state_dict(), save_path)
    print(f"SAE checkpoint saved to: {save_path}")

if __name__ == "__main__":
    main()
