import os
import json
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_id",
        type=int,
        required=True,
        help="要分析的 SAE latent 维度编号"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="latents_analysis/latents_analysis_readable",
        help="存放各 split JSON 文件的目录"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="最多读取多少个 JSON 文件"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=77,
        help="CLIP tokenizer 的最大 token 长度"
    )
    args = parser.parse_args()

    LATENT_KEY = f"latent_{args.latent_id}"
    print(f"分析 SAE latent 维度: {LATENT_KEY}")

    # 1. 加载 CLIP tokenizer + model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()

    # 2. 从 JSON 文件里收集对应的文本
    texts = []
    json_files = sorted([
        fn for fn in os.listdir(args.json_dir)
        if fn.endswith(".json")
    ])[: args.top_n]

    for fname in json_files:
        path = os.path.join(args.json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if LATENT_KEY in data:
            texts.append(data[LATENT_KEY])

    print(f"共找到 {len(texts)} 条文本用于分析")

    if len(texts) < 2:
        print("文本数不足，无法计算相似度")
        return

    # 3. Tokenize 并做调试打印，确认输入确实不同
    with torch.no_grad():
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt"
        ).to(device)


        '''
        texts 是一个 Python 列表, 包含 20 条文本: len(texts) = 20
        max_length = 77, 也就是每条文本最多保留 77 个 token
        设置了 padding=True 和 truncation=True, 表示:
        不足 77 个 token 的文本右侧补 <pad>, 超过的部分会被截断
        return_tensors="pt"：返回 PyTorch Tensor 格式
        最终的 tokenized 是一个字典对象 {'input_ids': tensor1, 'attention_mask': tensor2}
        '''

    # 4. 计算真正的 CLIP 文本嵌入
    with torch.no_grad():
        # get_text_features 会执行 Transformer + text_projection + layer norm
        embeddings = model.get_text_features(**tokenized)  # [N, 512]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 归一化

    # 5. 计算余弦相似度矩阵，并取上三角平均
    similarity_matrix = embeddings @ embeddings.T  # [N, N]
    # 只取上三角（不含对角线）
    n = similarity_matrix.size(0)
    upper_tri = similarity_matrix.triu(diagonal=1)
    mean_sim = upper_tri[upper_tri != 0].mean().item()

    print(f"相似度矩阵大小: {similarity_matrix.shape}")
    print(f"平均余弦相似度: {mean_sim:.4f}")

    # 6. 绘图并保存
    os.makedirs("graph", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix.cpu().numpy(),
        xticklabels=False,
        yticklabels=False,
        cmap="gray_r",
        cbar=True,
        ax=ax
    )
    ax.set_title(f"Cosine Similarity Heatmap for Latent {args.latent_id}\nMean: {mean_sim:.4f}")
    fig.tight_layout()
    fig.savefig(f"graph/latent_{args.latent_id}.pdf")
    print(f"热力图已保存为 graph/latent_{args.latent_id}.pdf")

if __name__ == "__main__":
    main()
