from transformers import CLIPTokenizer, CLIPTextModel
import torch
from torch.nn.functional import normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

# 正确初始化
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

texts = [
    "Preview | Recap | Notebook\nBulls-Bobcats Preview",
    "Grizzlies-Rockets Preview",
    "Some totally different topic",
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=77,
    return_tensors="pt"
)
print(tokenizer.batch_decode(inputs["input_ids"]))
exit()

# 编码
inputs = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 向量
    embeddings = normalize(embeddings, p=2, dim=1)

# 打印嵌入
print("Embeddings:", embeddings)
print("Cosine similarity matrix:")
cos_sim = embeddings @ embeddings.T
print(cos_sim)
