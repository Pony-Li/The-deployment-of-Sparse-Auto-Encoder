import matplotlib.pyplot as plt
import json
import os

# 参数
INPUT_JSON = "latents_analysis/latent_similarity_results.json"
GRAPH_DIR = "graph"
os.makedirs(GRAPH_DIR, exist_ok=True)

# 加载数据
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为 list 并排序
latent_ids = []
scores = []
for k in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
    latent_ids.append(int(k.split("_")[1]))
    scores.append(data[k])

mean_val = 0.6375  # 平均余弦相似度

# 柱状图
plt.figure(figsize=(20, 6))
plt.bar(range(len(scores)), scores, color="black")
plt.axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.4f}")
plt.xlabel("Latent Dimension Index")
plt.ylabel("Cosine Similarity")
plt.title("CLIP Cosine Similarity per Latent Dimension")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/latent_similarity_barplot.pdf")
plt.close()

# 分布直方图
plt.figure(figsize=(8, 6))
plt.hist(scores, bins=20, color="gray", edgecolor="black")
plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.4f}")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Latents")
plt.title("Distribution of Cosine Similarity across Latent Dimensions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/latent_similarity_distribution.pdf")
plt.close()

