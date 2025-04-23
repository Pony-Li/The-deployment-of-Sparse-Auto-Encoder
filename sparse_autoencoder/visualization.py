import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm

# 设置路径和参数
json_dir = "latents_analysis/latents_analysis_readable"
similarity_path = "latents_analysis/latent_similarity_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_len = 77
max_latents = 50
samples_per_latent = 20

# 加载 similarity 结果
with open(similarity_path, "r", encoding="utf-8") as f:
    similarity_scores = json.load(f)
selected_latents = [
    k for k, v in similarity_scores.items() if 0.72 <= v <= 0.95
][:max_latents]

# 加载文本
latent_to_texts = {k: [] for k in selected_latents}
for fname in sorted(os.listdir(json_dir)):
    if fname.endswith(".json"):
        with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in selected_latents:
            if k in data and len(latent_to_texts[k]) < samples_per_latent:
                latent_to_texts[k].append(data[k])

# 展平文本和标签
all_texts = []
all_labels = []
for i, (k, texts) in enumerate(latent_to_texts.items()):
    for t in texts:
        all_texts.append(t)
        all_labels.append(i)

# 编码为向量
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

embeddings = []
for i in tqdm(range(0, len(all_texts), 64)):
    batch = all_texts[i:i+64]
    with torch.no_grad():
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        embs = model.get_text_features(**tokens)
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        embeddings.append(embs.cpu())
embeddings = torch.cat(embeddings).numpy()

# 降维到 3D
points_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings)

# 创建 graph 文件夹
os.makedirs("graph", exist_ok=True)

# 静态图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=all_labels, cmap='tab10')
plt.title("3D t-SNE Visualization of SAE Latent-triggered Texts")
plt.savefig("graph/latent_tsne_3d_static.pdf")

# 动图
def rotate(angle):
    ax.view_init(azim=angle)

ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)
ani.save("graph/latent_tsne_3d_rotation.gif", writer="pillow")
