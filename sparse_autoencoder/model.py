from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps) # eps: 一个小的常数, 用于防止除以零
    return x, mu, std


class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False,
        normalize: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        # latent_slice: slice = slice(None): 一个切片对象，用于选择部分潜在表示。默认值为 slice(None)，表示选择所有潜在表示。

        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """

        # 计算线性变换的输出，即编码器的输出(在应用激活函数之前)，计算公式为：
        # latents_pre_act=(x-pre_bias)*encoder.weight[latent_slice]+latent_bias[latent_slice]
        # x: [batch, n_inputs], pre_bias: [n_inputs], encoder.weight: [n_input, n_latent], latent_bias: [n_latent]
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        # 这里使用 F.linear 是因为其更加灵活，适用于需要动态选择权重和偏置的场景。
        # 而 nn.Linear 是更高级别的模块，封装了线性变换的逻辑，适用于构建标准的神经网络层。
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """

        # 数据进入 encoder 后先进行 layer normalization, 再通过线性变换实现 pre_activation, 最后再完成 relu activation 得到 latents
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] | None = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1
        # 这两行代码的作用是更新 self.stats_last_nonzero, 这是一个用于跟踪潜在表示 (latents) 中每个维度最后一次非零激活的时间步的统计变量
        # latents == 0 逐元素比较 latents 是否等于 0, 生成一个布尔张量, 形状与 latents 相同, 值为 True 或 False
        # .all(dim=0) 沿着第 0 维对布尔张量进行 all 操作, 生成形状为 [n_latents] 的布尔张量, 表示每个潜在维度在所有批量样本中是否都为 0
        # .long() 将布尔张量转换为长整型张量 (torch.long), 生成一个布尔张量，形状为 [n_latents], 表示每个潜在维度在所有批量样本中是否都为 0
        # self.stats_last_nonzero *= whether_all_zero_long, 将二者逐元素相乘,
        # 如果某个潜在维度在所有批量样本中都为 0, 则 self.stats_last_nonzero 中对应的值保持不变; 否则, 该值被置为 0
        # self.stats_last_nonzero += 1, 将 self.stats_last_nonzero 每个分量的值加 1, 记录每个潜在维度自上次非零激活以来的时间步数。

        return latents_pre_act, latents, recons

    # 下面这段代码定义了一个类方法 from_state_dict, 用于从一个状态字典 (state_dict) 中加载并初始化一个 Autoencoder 实例
    # 在 Python 中，类中定义的函数通常被称为实例方法 (Instance Method), 实例方法的第一个参数通常是 self, 用来表示类的实例
    # 在这里利用装饰器定义的是类方法, 其第一个参数是类本身 (cls) 而不是类的实例 (self), 这意味着类方法可以访问类本身, 但不能访问类的实例
    # 在 from_state_dict 方法中, 需要调用类的构造函数 cls(n_latents, d_model, activation=activation, normalize=normalize) 
    # 来创建一个新的 Autoencoder 实例, 需要将类本身 (cls) 作为参数传入, 因此这里使用的是类方法, 而非一般的实例方法
    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU") # 从状态字典中提取激活函数的名称。如果键 activation 不存在, 则默认为 ReLU
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU) # 获取激活函数的类。如果键 activation_class_name 不存在, 则默认为 nn.ReLU
        normalize = activation_class_name == "TopK"  # 根据激活函数的名称判断是否启用归一化。如果激活函数是 TopK, 则启用归一化
        activation_state_dict = state_dict.pop("activation_state_dict", {}) # 从状态字典中提取激活函数的状态字典。如果键 activation_state_dict 不存在，则默认为空字典。
        
        # 检查激活函数类是否有一个 from_state_dict 方法
        if hasattr(activation_class, "from_state_dict"): # 如果有, 直接调用 from_state_dict 方法来创建激活函数实例
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else: # 如果没有, 先创建激活函数实例, 然后再调用 load_state_dict 方法加载状态字典
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, d_model, activation=activation, normalize=normalize)
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    # 下面这段代码定义了一个名为 state_dict 的方法, 用于保存 Autoencoder 类的当前状态
    # 这个方法扩展了 PyTorch 中 torch.nn.Module 的默认 state_dict 方法, 以包含自定义的激活函数信息
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars) # 调用 torch.nn.Module 的默认 state_dict 方法, 保存模型的参数和缓冲区
        sd[prefix + "activation"] = self.activation.__class__.__name__ # 将激活函数的类名保存到状态字典中
        if hasattr(self.activation, "state_dict"): # 检查激活函数是否有一个 state_dict 方法。如果有, 则调用该方法并保存激活函数的状态字典
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd

# 下面这段代码实现了一个特殊的线性变换, 其中解码器的权重与编码器的权重是绑定的 (即权重共享)
# 这种设计在某些神经网络架构中非常有用, 尤其是在自编码器中, 可以减少模型的参数数量并提高训练效率
class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    # 定义了两个属性方法, weight 和 bias, 使得 TiedTranspose 模块的行为类似于一个标准的 nn.Linear 模块。
    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

# 这段代码实现了一个特殊的激活函数, 该激活函数只保留输入张量中每个样本的前 k 个最大值, 并将其他值设置为 0
class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn # 后置激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1) # 在最后一个维度上选择每个样本的前 k 个最大值及其索引

        # topk.values: 前 k 个最大值; topk.indices: 前 k 个最大值的索引
        values = self.postact_fn(topk.values) # 对前 k 个最大值应用后置激活函数
        # make all other values 0
        result = torch.zeros_like(x) # 创建一个与输入张量 x 形状相同但所有值为 0 的张量
        result.scatter_(-1, topk.indices, values) # 将前 k 个最大值 (经过激活函数处理后) 散列到结果张量中, 其他位置保持为 0
        # torch.Tensor.scatter_ 是 PyTorch 中一个非常强大的方法, 用于将一个张量的值散列到另一个张量中
        # dim：指定在哪个维度上进行散列操作 index：一个张量, 指定 src 中的值应该散列到 result 的哪些位置 src：一个张量, 包含要散列的值
        # scatter() 是一个普通方法, 它返回一个新的张量, 其中指定位置的值被更新, 而原始张量保持不变
        # scatter_() 是一个原地方法, 它直接修改调用它的张量, 而不返回新的张量

        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}
