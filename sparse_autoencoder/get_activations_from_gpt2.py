# 提取 GPT-2 第8层 residual stream 激活，用于训练 sparse autoencoder

import os
import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

# 配置参数
SAVE_DIR = "/data4/angli/SAE_training_data"
LAYER_IDX = 7  # GPT-2 第8层（0-based）
MAX_LEN = 64   # 每条文本最大token数
CHUNK_SIZE = 1000000  # 每个文件最多保存多少个激活样本
BATCH_SIZE = 512      # 每批处理64条文本
TARGET_TOKEN_COUNT = 25000000
current_token_count = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(DEVICE)
model.eval()

# 加载 openwebtext 数据集
dataset = load_from_disk("/data2/datasets/openwebtext")["train"]

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据处理函数
def collate_fn(batch):
    texts = [x["text"] for x in batch]
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    return tokens

# 构建 DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 缓冲区
buffer = []
sample_count = 0
chunk_id = 0

# 主提取循环
with torch.no_grad():
    for batch in tqdm(loader, desc="Extracting GPT-2 activations"):
        input_ids = batch["input_ids"].to(DEVICE)       # [B, T]
        attention_mask = batch["attention_mask"].to(DEVICE)

        # 获取 GPT-2 的中间表示
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple of 13 tensors

        # 取第8层 LayerNorm 后的 residual stream 激活
        h = hidden_states[LAYER_IDX]               # [B, T, d_model]
        ln_out = model.h[LAYER_IDX].ln_1(h)        # [B, T, d_model]
        ln_out = ln_out.reshape(-1, ln_out.shape[-1])  # [B*T, d_model]

        # 累加 token 数量
        current_token_count += ln_out.shape[0]

        buffer.append(ln_out.cpu())
        sample_count += ln_out.shape[0]

        # 存盘
        if sample_count >= CHUNK_SIZE:
            all_acts = torch.cat(buffer, dim=0)
            save_path = os.path.join(SAVE_DIR, f"chunk_{chunk_id}.pt")
            torch.save(all_acts, save_path)
            print(f"Saved {save_path}, shape={all_acts.shape}")

            # 清空缓冲区
            buffer = []
            sample_count = 0
            chunk_id += 1

        # 如果已经达到目标，提前结束循环
        if current_token_count >= TARGET_TOKEN_COUNT:
            break

# 保存最后剩余部分
if buffer:
    all_acts = torch.cat(buffer, dim=0)
    save_path = os.path.join(SAVE_DIR, f"chunk_{chunk_id}.pt")
    torch.save(all_acts, save_path)
    print(f"Saved final {save_path}, shape={all_acts.shape}")
