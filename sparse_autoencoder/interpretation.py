import argparse
import os
import random
import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from sparse_autoencoder.train import *

# 解析命令行参数
parser = argparse.ArgumentParser(description="Top-activation analyzer")
parser.add_argument("--split_id", type=int, required=True, help="which 1/50 split to analyze (0-49)")
args = parser.parse_args()

# 参数配置
DATA_PATH = "/data2/datasets/openwebtext"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_LEN = 64
LAYER_IDX = 7
NUM_LATENTS = 32768
SELECTED = 384

# 加载模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2 = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(DEVICE)
gpt2.eval()

cfg = Config()
comms = make_torch_comms(n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)
sae = FastAutoencoder(
    n_dirs_local=cfg.n_dirs,
    d_model=cfg.d_model,
    k=cfg.k,
    auxk=cfg.auxk,
    dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
    comms=comms,
).to(DEVICE)

checkpoint = torch.load('/data2/angli/SAE_checkpoint/epoch_120.pt')
sae.load_state_dict(checkpoint["model_state_dict"], strict=True)
sae.eval()

# 加载数据
dataset = load_from_disk(DATA_PATH)["train"]
split_len = len(dataset) // 50
start_idx = args.split_id * split_len
end_idx = (args.split_id + 1) * split_len
dataset = dataset.select(range(start_idx, end_idx))
print(f"分析 split {args.split_id+1}/50，样本范围：[{start_idx}, {end_idx})")

# 选定 latent 维度
random.seed(42)
selected_latent_dims = sorted(random.sample(range(NUM_LATENTS), SELECTED))
max_vals = torch.full((SELECTED,), -float("inf"), device=DEVICE)
max_indices = torch.full((SELECTED,), -1, dtype=torch.long)
text_buffer = [""] * SELECTED

# 构建 DataLoader
def collate_fn(batch):
    texts = [x["text"] for x in batch]
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    return tokens, texts

loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 主分析循环
sample_idx = start_idx * MAX_LEN  # 全局 token 索引

for batch, texts in tqdm(loader, desc=f"Analyzing split {args.split_id+1}/50"):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = gpt2(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.hidden_states[LAYER_IDX]
        residual = gpt2.h[LAYER_IDX].ln_1(h)
        residual = residual.view(-1, residual.shape[-1])  # [B*T, 768]

        latent_input = residual - sae.pre_bias
        latents = sae.encoder(latent_input) + sae.latent_bias
        topk_vals, topk_inds = torch.topk(latents, cfg.k, dim=-1)

        for i in range(topk_vals.size(0)):
            for j in range(cfg.k):
                latent_idx = topk_inds[i, j].item()
                val = topk_vals[i, j].item()

                if latent_idx in selected_latent_dims:
                    pos = selected_latent_dims.index(latent_idx)
                    if val > max_vals[pos]:
                        max_vals[pos] = val
                        max_indices[pos] = sample_idx + i
                        text_buffer[pos] = texts[i // MAX_LEN]

    sample_idx += residual.size(0)

# 保存结果
os.makedirs("latents_analysis", exist_ok=True)
save_path = f"latents_analysis/latent_top384_split_{args.split_id+1}.pt"

torch.save({
    "split_id": args.split_id,
    "selected_latent_dims": selected_latent_dims,
    "max_vals": max_vals.cpu(),
    "max_indices": max_indices.cpu(),
    "text_samples": text_buffer,
}, save_path)

print(f"Split {args.split_id+1}/50 分析完成，结果保存在：{save_path}")
