import os
import json

# 源图片目录（将在该目录下递归查找）
image_dir = "/data/250010183/Datasets/Celeb-DF-v3-process/Celeb-real"
# 目标 json 文件路径
output_json = "/data/250010183/workspace/Data_json/data_json/Celeb-DF-v3/Celeb-real.json"

# 支持的图片扩展名（统一用小写比较）
image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in image_exts

def main():
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"目录不存在：{image_dir}")

    image_paths = []
    # 递归遍历
    for root, dirs, files in os.walk(image_dir, followlinks=False):
        # 可选：跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    # 生成相对路径（相对于当前工作目录）
                    rel_path = os.path.relpath(fpath, os.getcwd())
                    image_paths.append(rel_path)

    # 排序保证稳定输出
    image_paths = sorted(image_paths)

    # 构造 json 数据
    json_data = {
        "Description": "",
        "images": [{"image_path": p} for p in image_paths]
    }

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # 写入 json 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"已生成 {output_json}，共包含 {len(image_paths)} 张图片。")

if __name__ == "__main__":
    main()
