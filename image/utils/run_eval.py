# coding=utf-8
import os
import sys
import torch
import imageio
import numpy as np
import lpips  # 【同步代码B】导入 LPIPS 库

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.dirname(CURRENT_DIR)

if IMAGE_DIR not in sys.path:
    sys.path.insert(0, IMAGE_DIR)

# 直接从师姐的 utils 中导入独立的打分函数
from eval_metrics import (
    _compute_clipiqa_score,
    _compute_dsd_score,
    _compute_blip_itm_scores,
    _compute_vqa_scores,
    _compute_fer_scores
)
from flowgrad_utils import clip_semantic_loss

# ===========================================================================
# 全局配置
# ===========================================================================
# 在 CONFIG 定义之前，添加：
PREFIX = "sad_curly"  # 改这里即可切换，如 "sad_smile", "sad_curly", "angry_sad"

PROMPT_MAP = {
    "angry": "A photo of a angry face.",
    "smile": "A photo of a smiling face.",
    "sad":   "A photo of a sad face.",
    "curly": "A photo of a curly hair face.",
}

def build_prompts_from_prefix(prefix):
    parts = prefix.split("_")  # e.g. ["sad", "angry"]
    prompts = []
    for p in reversed(parts):  # reversed 使得第二个词对应 prompt1
        if p in PROMPT_MAP:
            prompts.append(PROMPT_MAP[p])
        else:
            raise ValueError(f"Unknown prompt keyword: '{p}'")
    return prompts

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "id_list": ["000442", "011472", "013624", "176801", "celeba"],
    "base_orig_dir": "/home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/demo/",

    # 自动根据 PREFIX 生成
    "base_gen_dir": "/home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/examples/glass_eval",
    "prefix": PREFIX,
    "prompts": build_prompts_from_prefix(PREFIX),

    "image_size": 256
}
# ===========================================================================
# 前处理对齐函数
# ===========================================================================
def load_and_preprocess_image(image_path, device, size=256):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
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

def get_identity_scaler():
    return lambda x: x

def safe_mean(values):
    valid_vals = [v for v in values if v is not None]
    return np.mean(valid_vals) if valid_vals else None

# ===========================================================================
# 主验证入口
# ===========================================================================
def main():
    device = CONFIG["device"]
    prompts = CONFIG["prompts"]
    id_list = CONFIG["id_list"]
    
    print(f"Initializing metrics on {device}...")
    identity_scaler = get_identity_scaler()

    # 【同步代码B】初始化 LPIPS 评估器
    lpips_f = lpips.LPIPS(net='alex').to(device)

    # 全局统计字典（增加 lpips 统计项）
    global_stats = {
        "clip_loss": [[] for _ in prompts], # 为每个 prompt 初始化一个独立的列表
        "lpips": [],  
        "clipiqa": [],
        "dsd": [],
        "blip_itm": [[] for _ in prompts],  # 同上
        "vqa": [[] for _ in prompts]        # 同上
    }

    print("=" * 60)
    print(f"Starting evaluation for {len(id_list)} IDs...")
    print("=" * 60)

    for img_id in id_list:
        print(f"\n>>> Processing ID: {img_id}")
        
        orig_img_path = os.path.join(CONFIG["base_orig_dir"], f"{img_id}.jpg")
        # 拼接子目录：如 sad_angry_000442
        gen_img_dir = os.path.join(
            CONFIG["base_gen_dir"],
            f"{CONFIG['prefix']}_{img_id}"
        )
        
        # 加载 Original Image
        img_orig = None
        img_orig_norm = None  # 【同步代码B】LPIPS 专用
        if os.path.exists(orig_img_path):
            img_orig = load_and_preprocess_image(orig_img_path, device, CONFIG["image_size"])
            img_orig_norm = img_orig * 2.0 - 1.0  # 【同步代码B】映射到 [-1, 1]
        else:
            print(f"  [Warning] Original image not found at {orig_img_path}. DSD & LPIPS will be skipped.")

        # 预先构建 CLIP Loss 对象（代码A逻辑）
        clip_loss_evals = []
        if img_orig is not None:
            for prompt in prompts:
                clip_eval = clip_semantic_loss(
                    prompt, img_orig, device,
                    alpha=1.0, inverse_scaler=identity_scaler
                )
                clip_loss_evals.append(clip_eval)

        if not os.path.exists(gen_img_dir):
            print(f"  [Warning] Generated dir '{gen_img_dir}' does not exist. Skipping.")
            continue

# 在子目录下递归查找后缀为 _gcar_gcovA_multiprompt_cw0.3_prompt_combined 的图片
        target_suffix = "_fk_edited"
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        img_files = []
        for root, dirs, files in os.walk(gen_img_dir):
            for f in files:
                name_no_ext, ext = os.path.splitext(f)
                if ext.lower() in valid_extensions and name_no_ext.endswith(target_suffix):
                    img_files.append(os.path.join(root, f))

        if not img_files:
            print(f"  [Warning] No '*{target_suffix}' images found in '{gen_img_dir}'. Skipping.")
            continue

        for img_path in sorted(img_files):
            img_name = os.path.basename(img_path)
            img_edit = load_and_preprocess_image(img_path, device, CONFIG["image_size"])
            
            # 【同步代码B】准备 LPIPS 输入
            img_edit_norm = img_edit * 2.0 - 1.0
            
            print(f"\n  Evaluating File: {img_name}")
            
            # --- 1. CLIP Loss (保持代码A原始方式) ---
            clip_losses = []
            if img_orig is not None:
                with torch.no_grad():
                    for clip_eval in clip_loss_evals:
                        clip_losses.append(clip_eval.L_N(img_edit).item())
            
            # --- 2. LPIPS (【同步代码B】新增) ---
            lpips_score = None
            if img_orig_norm is not None:
                with torch.no_grad():
                    lpips_score = lpips_f(img_edit_norm, img_orig_norm).item()

            # --- 3. 其他指标 ---
            clipiqa_score = _compute_clipiqa_score(img_edit, device)
            dsd_score = None
            if img_orig is not None:
                dsd_score = _compute_dsd_score(img_orig, img_edit, device)
                
            blip_itm = _compute_blip_itm_scores(img_edit, prompts, device)
            vqa = _compute_vqa_scores(img_edit, prompts, device)
            fer = _compute_fer_scores(img_edit)

            # # --- 统计汇总 ---
            # avg_clip_loss = safe_mean(clip_losses)
            # avg_blip = safe_mean(blip_itm)
            # avg_vqa = safe_mean(vqa)

            # 收集数据到 global_stats
            if lpips_score is not None: global_stats["lpips"].append(lpips_score)
            if clipiqa_score is not None: global_stats["clipiqa"].append(clipiqa_score)
            if dsd_score is not None: global_stats["dsd"].append(dsd_score)

            # --- 收集多 prompt 指标（修改这里） ---
            for i in range(len(prompts)):
                if i < len(clip_losses) and clip_losses[i] is not None:
                    global_stats["clip_loss"][i].append(clip_losses[i])
                
                if blip_itm and blip_itm[i] is not None:
                    global_stats["blip_itm"][i].append(blip_itm[i])
                    
                if vqa and vqa[i] is not None:
                    global_stats["vqa"][i].append(vqa[i])

            # --- 打印单图结果 ---
            if clip_losses:
                for i, (prompt, c_loss) in enumerate(zip(prompts, clip_losses)):
                    print(f"    [Prompt {i+1}] CLIP loss : {c_loss:.4f}")
                # print(f"    -> [Avg 1] Prompt Mean CLIP Loss : {avg_clip_loss:.4f}")
            
            # 【同步代码B】增加 LPIPS 打印
            print(f"    LPIPS score   : {lpips_score if lpips_score is not None else 'N/A'}")
            print(f"    CLIPIQA score : {clipiqa_score if clipiqa_score is not None else 'N/A'}")
            print(f"    DSD score     : {dsd_score if dsd_score is not None else 'N/A'}")
            
            for i, (prompt, s) in enumerate(zip(prompts, blip_itm)):
                print(f"    [Prompt {i+1}] BLIP-ITM : {s:.4f}" if s is not None else "    BLIP-ITM: N/A")
            
            for i, (prompt, s) in enumerate(zip(prompts, vqa)):
                print(f"    [Prompt {i+1}] VQAScore : {s:.4f}" if s is not None else "    VQAScore: N/A")

    # ===========================================================================
    # FINAL STATISTICS
    # ===========================================================================
    print("\n" + "=" * 60)
    print("FINAL GLOBAL STATISTICS")
    print("=" * 60)
    
    metrics_display_names = {
        "clip_loss": "CLIP Loss (Lower is better)",
        "lpips": "LPIPS Score (Lower is better)",  # 【同步代码B】
        "clipiqa": "CLIPIQA Score",
        "dsd": "DSD Score",
        "blip_itm": "BLIP-ITM Score",
        "vqa": "VQAScore"
    }

# 分开定义单指标和多 prompt 指标的名称
    single_metrics = {
        "lpips": "LPIPS Score (Lower is better)",
        "clipiqa": "CLIPIQA Score",
        "dsd": "DSD Score"
    }
    
    multi_prompt_metrics = {
        "clip_loss": "CLIP Loss (Lower is better)",
        "blip_itm": "BLIP-ITM Score",
        "vqa": "VQAScore"
    }

    # 1. 打印不需要分 prompt 的全局指标
    for key, name in single_metrics.items():
        vals = global_stats[key]
        if vals:
            print(f"  {name:<30} : {np.mean(vals):.4f}  (N={len(vals)})")
        else:
            print(f"  {name:<30} : N/A")

    print("  " + "-" * 56)

    # 2. 打印分 prompt 的全局指标
    for key, name in multi_prompt_metrics.items():
        for i in range(len(prompts)):
            vals = global_stats[key][i]
            display_name = f"{name} [Prompt {i+1}]"
            if vals:
                print(f"  {display_name:<30} : {np.mean(vals):.4f}  (N={len(vals)})")
            else:
                print(f"  {display_name:<30} : N/A")
            
    print("=" * 60)

if __name__ == "__main__":
    main()