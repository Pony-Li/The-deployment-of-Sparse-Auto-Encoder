import os
import json
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm
from collections import defaultdict

# 参数配置
JSON_DIR = "latents_analysis/latents_analysis_readable"
SELECTED_LATENT_COUNT = 364
OUTPUT_PATH = "latents_analysis/latent_similarity_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 77

# 加载 CLIP 模型
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_model.eval()

# 第一步：收集所有 latent_xxxx 对应的文本
latent_to_texts = defaultdict(list)

json_files = sorted([
    f for f in os.listdir(JSON_DIR) if f.endswith(".json")
])

for fname in json_files:
    path = os.path.join(JSON_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        if k.startswith("latent_"):
            latent_to_texts[k].append(v)

# 第二步：计算每个 latent 的平均余弦相似度
results = {}

for latent_key, texts in tqdm(latent_to_texts.items(), desc="Computing similarities"):
    if len(texts) < 2:
        continue  # 至少需要两个样本才能算相似度

    with torch.no_grad():
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        embeddings = clip_model.get_text_features(**tokens)  # [N, 512]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 相似度矩阵 & 平均余弦相似度
        sim_matrix = embeddings @ embeddings.T  # [N, N]
        upper = sim_matrix.triu(diagonal=1)
        mean_sim = upper[upper != 0].mean().item()

        results[latent_key] = round(mean_sim, 4)

mean_similarity = sum(results.values()) / SELECTED_LATENT_COUNT
print(f"所有 384 个 latent 维度的平均余弦相似度为: {mean_similarity:.4f}")

# 第三步：保存结果
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"共处理 {len(results)} 个 latent, 结果已保存至 {OUTPUT_PATH}")
