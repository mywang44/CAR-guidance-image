# coding=utf-8

import os
from datetime import datetime
import numpy as np
import imageio
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

# Custom library imports
from utils import run_lib_flowgrad_oc

FLAGS = flags.FLAGS

# Configuration
config_flags.DEFINE_config_file("config", 'RectifiedFlow/configs/celeba_hq_pytorch_rf_gaussian.py', "Rectified Flow Model configuration.", lock_config=True)

# Method selection
flags.DEFINE_string('method', 'gcar_gcovA_multiprompt', '[gcar_gcovA_multiprompt]')

# Optimization Hyperparameters
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("index", 0, "Position of samples")

# Conflict & Hybrid Parameters
flags.DEFINE_float('conflict_weight', 0.3, 'Weight for conflict score minimization')
flags.DEFINE_float('conflict_lr', 2.5, 'Learning rate for conflict methods')
flags.DEFINE_bool('use_true_landscape', False, 'Use true sampling for Loss Landscape')
flags.DEFINE_bool('use_L_best', True, 'Return controls from best metric (L_best); else return last step')

# Global Constants
ALPHA = 0.7
LR_DEFAULT = 5.0 # 2.5
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(_SCRIPT_DIR, 'demo')
_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
# 定义所有需要处理的图片路径：demo 目录下所有图片
IMAGE_PATHS = sorted([
    os.path.join(DEMO_DIR, f) for f in os.listdir(DEMO_DIR)
    if os.path.isfile(os.path.join(DEMO_DIR, f)) and f.lower().endswith(_IMAGE_EXTENSIONS)
])
MODEL_PATH = os.path.join(_SCRIPT_DIR, '/home/users/meiyi/RectifiedFlow/ImageGeneration/logs/celebahq/checkpoints/checkpoint_10.pth')

def get_img(path=None):
    """Helper to load and preprocess single image."""
    img = imageio.imread(path)
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    print('Read image from:', path, 'Range:', img.min(), img.max())
    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)
    return img

def main(argv):
    # --- Prompt Definitions ---
    # Index: 0=old, 1=sad, 2=smiling, 3=angry, 4=curly hair
    text_prompts = [
        'A photo of an old face.',
        'A photo of a sad face.',
        'A photo of a smiling face.',
        'A photo of an angry face.',
        'A photo of a face with curly hair.'
    ]

    print(f"=== Starting Optimization: {FLAGS.method} ===")

    if FLAGS.method != 'gcar_gcovA_multiprompt':
        raise ValueError(f"Unknown method: {FLAGS.method}")

    # 整轮运行共用一个输出文件夹: gcar_{时间戳}
    run_folder = f'gcar_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # --- Execution Logic: gcar_gcovA_multiprompt (Hybrid gcovA + Residual OC) ---
    # 定义任务列表：包含 (Prompt列表, 任务名称前缀)
    tasks = [
        # Task 1: Sad + Angry
        (
            [text_prompts[1], text_prompts[3]],  # Sad, Angry
            'sad_angry'
        ),
        # Task 2: Sad + Smile
        (
            [text_prompts[1], text_prompts[2]],  # Sad, Smile
            'sad_smile'
        ),
        # Task 3: Sad + Curly Hair
        (
            [text_prompts[1], text_prompts[4]],  # Sad, Curly
            'sad_curly'
        )
    ]

    # 外层循环：遍历任务 (Prompt组合)
    for current_prompts, task_prefix in tasks:

        # 内层循环：遍历所有图片
        for img_path in IMAGE_PATHS:
            # 获取图片文件名 (不带扩展名), 例如 'celeba'
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # 拼接完整的后缀: task_imgname (e.g., angry_sad_celeba)
            full_suffix = f"{task_prefix}_{img_name}"

            # 最终输出路径: gcar_{时间戳}/full_suffix
            output_dir = f"{run_folder}/{full_suffix}"

            print(f"\n--- Processing Image: {img_name} | Task: {task_prefix} ---")
            print(f"Prompts: {current_prompts}")
            print(f"Output Directory: {output_dir}")

            current_batch_paths = [img_path]

            print("Using Hybrid (gcovA + Residual OC)")
            run_lib_flowgrad_oc.flowgrad_edit_batch_hybrid_multiprompt(
                FLAGS.config, MODEL_PATH, current_batch_paths, current_prompts, output_dir,
                method=FLAGS.method, alpha=ALPHA, lr_gcov=LR_DEFAULT, lr_res=FLAGS.conflict_lr,
                conflict_weight=FLAGS.conflict_weight,
                use_true_landscape=FLAGS.use_true_landscape,
                use_L_best=FLAGS.use_L_best
            )

    print("\nAll Tasks & Images Processed!")

if __name__ == "__main__":
    app.run(main)


# CUDA_VISIBLE_DEVICES=7 nohup python -u /home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/main_data.py > generate_sad_smile_sad_curly.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u /home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/main_data.py > train4400-sm.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u /home/users/meiyi/Guided-Flow-Matching-with-Optimal-Control/image/main_data.py > PCGrad_baseline.log 2>&1 &