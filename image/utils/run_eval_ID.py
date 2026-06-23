import os
import sys
from glob import glob
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# --- 配置路径 ---
SRC_DIR = "/home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/demo"
EDIT_DIR = "/home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/examples/glass_eval/Face_ID"

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- 设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Face ID (facenet-pytorch) ---
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("ERROR: facenet-pytorch 未安装，请先执行: pip install facenet-pytorch==2.5.3")
    sys.exit(1)


# Facenet 预处理：InceptionResnetV1 通常输入 160x160，范围 [0,1]
facenet_transform = T.Compose([
    T.Resize((160, 160), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),   # [0,1]
])

def to_facenet_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = facenet_transform(img).unsqueeze(0).to(device)  # [1,3,160,160]
    return x

@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum(dim=-1).mean().item())

def find_source_image(img_id: str) -> str:
    """根据图片 ID 在原图文件夹中寻找对应的源文件"""
    for ext in SUPPORTED_EXTS:
        path = os.path.join(SRC_DIR, f"{img_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def main():
    # 1. 加载模型
    print("正在加载 Face ID 模型 (InceptionResnetV1)...")
    id_model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

    if not os.path.exists(EDIT_DIR):
        print(f"ERROR: 编辑图总文件夹不存在: {EDIT_DIR}")
        return

    # 2. 获取所有的图片 ID (即下一级的文件夹名称)
    img_ids = [d for d in os.listdir(EDIT_DIR) if os.path.isdir(os.path.join(EDIT_DIR, d))]
    img_ids = sorted(img_ids)
    print(f"共找到 {len(img_ids)} 个图片 ID 文件夹。")

    all_scores = []       # 保存所有编辑图片的单个分数 (用于计算总体微平均)
    folder_averages = []  # 保存每个文件夹的平均分 (用于计算总体宏平均)

    # 3. 遍历图片 ID
    for img_id in img_ids:
        # 获取源图路径
        src_path = find_source_image(img_id)
        if not src_path:
            print(f"[跳过] 找不到 ID={img_id} 对应的原图 (搜索目录: {SRC_DIR})")
            continue
        
        # 计算源图特征
        try:
            src_img = Image.open(src_path)
            x0_id = to_facenet_tensor(src_img)
            with torch.no_grad():
                emb_src = id_model(x0_id)
        except Exception as e:
            print(f"[跳过] 无法读取或处理原图 {src_path}，原因: {e}")
            continue

        # 获取该 ID 文件夹下所有的编辑图片
        folder_path = os.path.join(EDIT_DIR, img_id)
        edit_files = []
        for ext in SUPPORTED_EXTS:
            edit_files.extend(glob(os.path.join(folder_path, f"*{ext}")))
        
        if not edit_files:
            print(f"[提示] ID={img_id} 的文件夹中没有找到支持的图片文件。")
            continue

        folder_scores = []
        # 遍历每张编辑图，和原图比对
        for edit_path in edit_files:
            try:
                edit_img = Image.open(edit_path)
                x_edit_id = to_facenet_tensor(edit_img)
                with torch.no_grad():
                    emb_edit = id_model(x_edit_id)
                
                score = cosine_sim(emb_src, emb_edit)
                folder_scores.append(score)
            except Exception as e:
                print(f"[警告] 无法处理编辑后的图片 {edit_path}，原因: {e}")
                continue

        # 统计当前 ID 的平均值
        if folder_scores:
            folder_avg = np.mean(folder_scores)
            folder_averages.append(folder_avg)
            all_scores.extend(folder_scores)
            
            print(f"ID={img_id:10s} | 编辑图片数: {len(folder_scores):2d} | 平均 ID 相似度: {folder_avg:.4f}")

    # 4. 汇总求所有图片的平均值
    if folder_averages:
        # Micro-average: 把所有比较过的图片得分全部加起来除以图片总数
        global_avg_micro = np.mean(all_scores)
        # Macro-average: 把每个 ID 的平均分加起来除以 ID 总数
        global_avg_macro = np.mean(folder_averages)

        print("\n" + "="*50)
        print("最终统计结果")
        print("="*50)
        print(f"共处理 {len(folder_averages)} 个图片 ID，总计测试了 {len(all_scores)} 张编辑图。")
        print(f"总体平均 ID 相似度 (以全部图片为基准): {global_avg_micro:.6f}")
        print(f"总体平均 ID 相似度 (以 ID 文件夹为基准): {global_avg_macro:.6f}")
        print("="*50)
    else:
        print("\n未成功计算任何图片的相似度，请检查文件路径或格式。")

if __name__ == "__main__":
    main()