# coding=utf-8
"""
评估脚本：对生成结果按外部参数指定的 类型 / prompt组合 / 图片id 进行打分。

用法示例:
  # 评估 gcar (训练后) 结果, 自动跑该目录下 sad_angry 的全部图片
  python utils/run_eval.py --dir /home/users/meiyi/CAR-guidance-image/image/examples/gcar_20260624_020209  --prefix sad_angry --type gcar

  # 评估 单prompt 结果 (singleprompt1 / singleprompt2)
  python utils/run_eval.py --dir /home/users/meiyi/CAR-guidance-image/image/examples/gcar_20260624_020209 --prefix sad_angry --type gcov-A_multi

  # 评估 线性叠加(multiprompt) 结果, 只跑指定的图片id
  python run_eval.py --dir <目录> --prefix sad_smile --type gcov-A_multi --ids 000442 celeba

目录结构约定:  {dir}/{prefix}_{id}/{id}{suffix}.jpg
  --type gcar          -> 后缀 _gcar
  --type gcov-A_multi   -> 后缀 _gcov-A_multiprompt
  --type gcov-A_single -> 后缀 _gcov-A_singleprompt1 / _gcov-A_singleprompt2 ... (每张图按其对应的单个 prompt 评估)
"""
import os
import sys
import argparse

import torch
import imageio
import numpy as np
import lpips

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.dirname(CURRENT_DIR)
if IMAGE_DIR not in sys.path:
    sys.path.insert(0, IMAGE_DIR)

from eval_metrics import (
    _compute_clipiqa_score,
    _compute_dsd_score,
    _compute_blip_itm_scores,
    _compute_vqa_scores,
)
from gcar_utils import clip_semantic_loss

# ===========================================================================
# 全局配置
# ===========================================================================
PROMPT_MAP = {
    "angry": "A photo of a angry face.",
    "smile": "A photo of a smiling face.",
    "sad":   "A photo of a sad face.",
    "curly": "A photo of a curly hair face.",
}

# 每种评估类型 -> 文件名后缀 + 是否按"单 prompt"逐个评估
EVAL_TYPES = {
    "gcar":          {"suffix": "_gcar",                "single": False},
    "gcov-A_multi":   {"suffix": "_gcov-A_multiprompt",  "single": False},
    "gcov-A_single": {"suffix": "_gcov-A_singleprompt", "single": True},
}

DEFAULT_ORIG_DIR = "/home/users/meiyi/CAR-guidance-image/image/demo"
IMAGE_SIZE = 256
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def build_prompts_from_prefix(prefix):
    """prefix='sad_angry' -> [sad_prompt, angry_prompt] (顺序与生成时一致)。"""
    prompts = []
    for p in prefix.split("_"):
        if p not in PROMPT_MAP:
            raise ValueError(f"Unknown prompt keyword: '{p}' (prefix={prefix})")
        prompts.append(PROMPT_MAP[p])
    return prompts


def find_image(dir_path, stem):
    """在 dir_path 下查找文件名(不含扩展名)为 stem 的图片，返回完整路径或 None。"""
    for ext in VALID_EXTS:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def discover_ids(orig_dir):
    """以原图目录 (demo) 下的图片文件名作为完整、规范的 id 列表。

    demo 里的 id 不重不漏, 与生成时使用的图片集一致;
    若某个 id 在生成目录下不存在, 由主循环负责跳过。
    """
    if not os.path.isdir(orig_dir):
        return []
    ids = []
    for f in os.listdir(orig_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in VALID_EXTS:
            ids.append(stem)
    return sorted(ids)


def load_and_preprocess_image(image_path, device, size=256):
    img = imageio.imread(image_path)
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    img_tensor = torch.tensor(img).float().to(device)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=size)
    img_tensor = img_tensor.clamp(0.0, 1.0)
    return img_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="对生成结果进行多指标评估")
    parser.add_argument("--dir", required=True,
                        help="结果根目录, 例如 .../examples/gcar_20260624_013503")
    parser.add_argument("--prefix", required=True,
                        help="prompt 组合, 例如 sad_angry / sad_smile / sad_curly")
    parser.add_argument("--type", required=True, choices=list(EVAL_TYPES.keys()),
                        help="评估对象: gcar / gcov-A_single / gcov-A_multi")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="指定图片 id (可多个); 不指定则自动评估目录下全部图片")
    parser.add_argument("--orig_dir", default=DEFAULT_ORIG_DIR,
                        help="原图目录 (用于 LPIPS / DSD / CLIP 参照)")
    return parser.parse_args()


# ===========================================================================
# 主验证入口
# ===========================================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = build_prompts_from_prefix(args.prefix)
    num_prompts = len(prompts)
    type_info = EVAL_TYPES[args.type]

    # 确定要评估的图片 id
    id_list = args.ids if args.ids else discover_ids(args.orig_dir)
    if not id_list:
        print(f"[Error] 原图目录 {args.orig_dir} 下没有图片，无法确定要评估的 id；请检查 --orig_dir 或用 --ids 指定。")
        return

    print(f"Initializing metrics on {device}...")
    lpips_f = lpips.LPIPS(net='alex').to(device)
    identity_scaler = lambda x: x

    # 统计字典: clip/blip/vqa 按 prompt 分桶; lpips/clipiqa/dsd 全局
    global_stats = {
        "clip_loss": [[] for _ in range(num_prompts)],
        "lpips": [],
        "clipiqa": [],
        "dsd": [],
        "blip_itm": [[] for _ in range(num_prompts)],
        "vqa": [[] for _ in range(num_prompts)],
    }

    print("=" * 60)
    print(f"Type   : {args.type}   (suffix='{type_info['suffix']}')")
    print(f"Prefix : {args.prefix}  ->  prompts={prompts}")
    print(f"IDs    : {id_list}")
    print("=" * 60)

    for img_id in id_list:
        print(f"\n>>> Processing ID: {img_id}")

        orig_img_path = os.path.join(args.orig_dir, f"{img_id}.jpg")
        gen_dir = os.path.join(args.dir, f"{args.prefix}_{img_id}")

        if not os.path.isdir(gen_dir):
            print(f"  [Warning] 生成目录不存在: {gen_dir}, 跳过。")
            continue

        # 加载原图 (用于 LPIPS / DSD / CLIP 参照)
        img_orig = None
        img_orig_norm = None
        if os.path.exists(orig_img_path):
            img_orig = load_and_preprocess_image(orig_img_path, device, IMAGE_SIZE)
            img_orig_norm = img_orig * 2.0 - 1.0
        else:
            print(f"  [Warning] 原图未找到: {orig_img_path}, 将跳过 DSD / LPIPS / CLIP。")

        # 针对每个 prompt 预构建 CLIP Loss 对象
        clip_loss_evals = []
        if img_orig is not None:
            for prompt in prompts:
                clip_loss_evals.append(clip_semantic_loss(
                    prompt, img_orig, device, alpha=1.0, inverse_scaler=identity_scaler
                ))

        # 构造待评估文件列表: [(文件路径, [对应的 prompt 下标])]
        eval_files = []
        if type_info["single"]:
            # 单 prompt: singleprompt{k} 只对第 k 个 prompt 评估
            for k in range(1, num_prompts + 1):
                stem = f"{img_id}{type_info['suffix']}{k}"
                fp = find_image(gen_dir, stem)
                if fp:
                    eval_files.append((fp, [k - 1]))
                else:
                    print(f"  [Warning] 缺少文件: {stem}.* in {gen_dir}")
        else:
            # gcar / multiprompt: 一张图同时对所有 prompt 评估
            stem = f"{img_id}{type_info['suffix']}"
            fp = find_image(gen_dir, stem)
            if fp:
                eval_files.append((fp, list(range(num_prompts))))
            else:
                print(f"  [Warning] 缺少文件: {stem}.* in {gen_dir}")

        for img_path, p_indices in eval_files:
            img_name = os.path.basename(img_path)
            img_edit = load_and_preprocess_image(img_path, device, IMAGE_SIZE)
            img_edit_norm = img_edit * 2.0 - 1.0
            prompts_subset = [prompts[i] for i in p_indices]

            print(f"\n  Evaluating File: {img_name}  (prompts={[i + 1 for i in p_indices]})")

            # 1. CLIP Loss (仅对该文件负责的 prompt)
            clip_losses = {}
            if img_orig is not None:
                with torch.no_grad():
                    for i in p_indices:
                        clip_losses[i] = clip_loss_evals[i].L_N(img_edit).item()

            # 2. LPIPS
            lpips_score = None
            if img_orig_norm is not None:
                with torch.no_grad():
                    lpips_score = lpips_f(img_edit_norm, img_orig_norm).item()

            # 3. 其他指标
            clipiqa_score = _compute_clipiqa_score(img_edit, device)
            dsd_score = _compute_dsd_score(img_orig, img_edit, device) if img_orig is not None else None
            blip = _compute_blip_itm_scores(img_edit, prompts_subset, device)
            vqa = _compute_vqa_scores(img_edit, prompts_subset, device)

            # --- 收集全局(与 prompt 无关)指标 ---
            if lpips_score is not None:
                global_stats["lpips"].append(lpips_score)
            if clipiqa_score is not None:
                global_stats["clipiqa"].append(clipiqa_score)
            if dsd_score is not None:
                global_stats["dsd"].append(dsd_score)

            # --- 收集分 prompt 指标 ---
            for j, i in enumerate(p_indices):
                if i in clip_losses and clip_losses[i] is not None:
                    global_stats["clip_loss"][i].append(clip_losses[i])
                if blip and j < len(blip) and blip[j] is not None:
                    global_stats["blip_itm"][i].append(blip[j])
                if vqa and j < len(vqa) and vqa[j] is not None:
                    global_stats["vqa"][i].append(vqa[j])

            # --- 打印单图结果 ---
            for j, i in enumerate(p_indices):
                if i in clip_losses:
                    print(f"    [Prompt {i+1}] CLIP loss : {clip_losses[i]:.4f}")
            print(f"    LPIPS score   : {lpips_score if lpips_score is not None else 'N/A'}")
            print(f"    CLIPIQA score : {clipiqa_score if clipiqa_score is not None else 'N/A'}")
            print(f"    DSD score     : {dsd_score if dsd_score is not None else 'N/A'}")
            for j, i in enumerate(p_indices):
                s = blip[j] if (blip and j < len(blip)) else None
                print(f"    [Prompt {i+1}] BLIP-ITM : {s:.4f}" if s is not None else f"    [Prompt {i+1}] BLIP-ITM : N/A")
            for j, i in enumerate(p_indices):
                s = vqa[j] if (vqa and j < len(vqa)) else None
                print(f"    [Prompt {i+1}] VQAScore : {s:.4f}" if s is not None else f"    [Prompt {i+1}] VQAScore : N/A")

    # ===========================================================================
    # FINAL STATISTICS
    # ===========================================================================
    print("\n" + "=" * 60)
    print("FINAL GLOBAL STATISTICS")
    print("=" * 60)

    single_metrics = {
        "lpips": "LPIPS Score (Lower is better)",
        "clipiqa": "CLIPIQA Score (Higher is better)",
        "dsd": "DSD Score (Lower is better)",
    }
    multi_prompt_metrics = {
        "clip_loss": "CLIP Loss (Lower is better)",
        "blip_itm": "BLIP-ITM Score (Higher is better)",
        "vqa": "VQAScore (Higher is better)",
    }

    for key, name in single_metrics.items():
        vals = global_stats[key]
        if vals:
            print(f"  {name:<46} : {np.mean(vals):.4f}  (N={len(vals)})")
        else:
            print(f"  {name:<46} : N/A")

    print("  " + "-" * 56)

    for key, name in multi_prompt_metrics.items():
        for i in range(num_prompts):
            vals = global_stats[key][i]
            display_name = f"{name} [Prompt {i+1}]"
            if vals:
                print(f"  {display_name:<46} : {np.mean(vals):.4f}  (N={len(vals)})")
            else:
                print(f"  {display_name:<46} : N/A")

    print("=" * 60)


if __name__ == "__main__":
    main()
