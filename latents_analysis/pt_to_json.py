import os
import json
import torch
import argparse

parser = argparse.ArgumentParser(description="Convert .pt latent analysis file to readable JSON")
parser.add_argument("--pt_path", type=str, required=True, help="Path to the .pt file to convert")
parser.add_argument("--output_dir", type=str, default="latents_analysis_readable", help="Where to save the .json")
args = parser.parse_args()

# 加载 .pt 文件
data = torch.load(args.pt_path)

selected_dims = data["selected_latent_dims"]
text_samples = data["text_samples"]

assert len(selected_dims) == len(text_samples)

# 构造 JSON-friendly 字典
result_dict = {
    f"latent_{dim}": text_samples[i]
    for i, dim in enumerate(selected_dims)
}

# 输出路径处理
os.makedirs(args.output_dir, exist_ok=True)
filename = os.path.splitext(os.path.basename(args.pt_path))[0] + ".json"
output_path = os.path.join(args.output_dir, filename)

# 写入 JSON 文件（UTF-8 编码支持中文、特殊字符）
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False)

print(f"转换完成, JSON 已保存到：{output_path}")
