import gc
import io
import os
import time

import numpy as np
import logging
import json
from tqdm import tqdm
# Keep the import below for registering all model definitions
from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from RectifiedFlow.utils import save_checkpoint, restore_checkpoint
import RectifiedFlow.datasets as datasets

from RectifiedFlow.models.utils import get_model_fn
from RectifiedFlow.models import utils as mutils

from .flowgrad_utils import get_img, embed_to_latent, clip_semantic_loss, save_img, generate_traj
from .conflict_utils import (
    visualize_spatial_conflict,
    visualize_weighted_conflict,
    overlay_heatmap_on_image,
    visualize_pca_trajectory,
    visualize_pca_trajectory_with_landscape,
    visualize_gradient_angle,
    plot_gradient_map,
)
# from id_loss.loss_fn import IDLoss

import torch.nn.functional as F
import torch.backends.cuda
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, PCA visualization will be disabled")

import warnings
warnings.filterwarnings("ignore")
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'



FLAGS = flags.FLAGS

def flowgrad_edit_batch_hybrid_multiprompt(config, model_path, image_paths, text_prompts, output_dir, 
                                           method='hybrid_multiprompt', alpha=0.7, 
                                           lr_gcov=1.0, lr_res=2.5, conflict_weight=None,
                                           use_true_landscape=False,
                                           use_L_best=True,
                                           save_single_prompt=True,
                                           save_combined=True):
    # ... (初始化模型、Scaler等代码与之前相同) ...
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    model_fn = mutils.get_model_fn(score_model, train=False)
    
    N = 100
    num_prompts = len(text_prompts)

    print(f"\n{'='*60}")
    print(f"Hybrid Multi-prompt (gcovA + Residual OC):")
    for idx, prompt in enumerate(text_prompts):
        print(f"  Prompt {idx+1}: {prompt}")
    print(f"{'='*60}\n")

    for img_path in tqdm(image_paths):
        # 输出目录: examples/{output_dir} (output_dir 形如 gcar_{时间戳}/{task}_{id})
        target_dir = f'examples/{output_dir}'
        os.makedirs(target_dir, exist_ok=True)
        # 图片 id, 例如 000442
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        # 训练后的 gcar 结果图: {id}_gcar.jpg
        opt_img_path = os.path.join(target_dir, f'{img_id}_gcar.jpg')

        image = get_img(img_path)  
        original_img = image.to(config.device)
        
        clip_loss_list = []
        for prompt in text_prompts:
            clip_loss = clip_semantic_loss(prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)
            clip_loss_list.append(clip_loss)
        
        import math # 记得在文件开头 import math
        t_s = time.time()
        # 1. 获得原图的倒推确定性潜变量 y(0) (shape: [1, C, H, W])
        y_0 = embed_to_latent(model_fn, scaler(original_img)) 
        # 也可以跑一条基准轨迹供对比参考
        traj = generate_traj(model_fn, y_0, N=N)

        print(f'\nHybrid optimization starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        L_N_list = [clip_loss.L_N for clip_loss in clip_loss_list]
        

        vis_dir = os.path.join(target_dir, f'conflict_maps_cw{conflict_weight}') if conflict_weight is not None else None
        # =========================================================
        # 🔴 新增：构造用于训练的鲁棒 Batch Latent
        # =========================================================
        train_batch_size = getattr(config, "guidance_batch_size", 4) # 比如用4条轨迹同时训练
        # 注意: 函数签名里已经有个 alpha 是给 CLIP Loss 用的，这里给公式里的 alpha 取名 init_alpha
        init_alpha = getattr(config, "init_alpha", 0.9) 

        # 将单张 y_0 复制扩展成 Batch (shape: [B, C, H, W])
        y_0_batch = y_0.repeat(train_batch_size, 1, 1, 1)
        
        # 采样标准高斯噪声 z ~ p_0(x_0)
        z = torch.randn_like(y_0_batch)

        # 严格执行你提供的初始化公式：x_0 = sqrt(α)*y(0) + sqrt(1-α)*z
        latent_batch = math.sqrt(init_alpha) * y_0_batch + math.sqrt(1.0 - init_alpha) * z


        # =====================================================================
        # 🔴 Step 4 接口: 为每种分类器(Prompt)组合构建引导向量场 🌟
        # 参考源代码: guided_fields['c0c0'] = make_guidance([classifier_1, classifier_2], ...)
        # 迁移说明: 
        # 1. 以前的 classifiers 现在对应这里的 L_N_list (即 CLIP Loss 的梯度提供者)
        # 2. 我们初始化一个 ComposedGuidanceImage 对象，内部包含可学习的 Residual Net
        # =====================================================================
        is_learnable = True # 开启 Residual Net 的训练模式
        
        from utils.composed_guidance import ImageGCovAGMOnlineGuidance
        lr_gcov = 4400
        guided_field = ImageGCovAGMOnlineGuidance(
            base_model=model_fn,                 # 对应源流匹配的 `vf` (基础速度场)
            loss_fns=L_N_list,                   # 对应源流匹配的 `classifiers`
            scales=[lr_gcov] * num_prompts,       # 对应源流匹配的 `scales`
            config=config,
            learnable=is_learnable,
            conflict_weight=conflict_weight,
            vis_dir=vis_dir
        )

        # 触发 Residual Net 的在线训练 (调解冲突)
        # 对应源流匹配中隐式包含在 ComposedGuidance 或 Solver 内部的训练循环
        print("\n--- Training Residual Net to resolve prompt conflicts ---")
        # 2. 训练时：喂入带有扰动的 Batch (提升泛化性和收敛速度)
        guided_field.train_model(latent_batch, num_steps=N, steps=15)

        # =====================================================================
        # 🔴 Step 7 接口: 核心采样与效果拆解 (The Execution Engine)
        # 参考源代码: solvers = [ODESolver(velocity_model=ModelWrapper(field))]
        #            visualizer.visualize_sampling_process(...)
        # 迁移说明:
        # 1. 最优控制时代，我们需要把离散的字典 u_total 传给 generate_traj。
        # 2. 现在，`guided_field` 本身就是一个动态的 Callable 函数，它会自动计算
        #    v_total(x, t) = v_base(x, t) + v_gcov(x, t) + v_res_net(x, t)。
        # 3. 因此，我们不再需要传递 `u`，直接把 `guided_field` 当作 `dynamic` 传给求解器。
        # =====================================================================
        print("\n--- Generating final trajectory with Train-based Guidance ---")
        # 3. 最终推理时：回归纯净！
        # 此时 Residual Net 已经具备了处理冲突的能力，我们对原汁原味的 y_0 进行生成，
        # 以确保最大限度保留原图的 Structure Prior (结构先验)，不被随机噪声破坏。
        traj_oc = generate_traj(guided_field, z0=y_0, u=None, N=N)

        # # 这里的效果拆解（对应源流匹配的 likelihood/residual 分解）
        # # 可以在 guided_field 内部或者此处调用类似 visualize_residual_decomposition 的方法
        # if vis_dir:
        #     guided_field.visualize_residual_decomposition(z0=y_0, N=N)
        # =====================================================================

        # 保存图像
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)
            
        # =========================================================
        # 🔴 Debug1: 仿照 OC 范式，单独保存每个 Prompt 的独立引导结果
        # 仅当外部传入 save_single_prompt=True 时才运行
        # =========================================================
        if save_single_prompt and opt_img_path is not None:
            print("\n--- Generating Single Prompt Trajectories for Debugging ---")
            lr_gcov = 4400
            for p_idx in range(num_prompts):
                # 1. 为当前单一 Prompt 构造一个专属的引导场
                # 因为单 Prompt 没有冲突，所以直接关闭 learnable 和 conflict_weight
                single_guided_field = ImageGCovAGMOnlineGuidance(
                    base_model=model_fn,
                    loss_fns=[L_N_list[p_idx]],       # 🔴 关键：只传入当前这一个 CLIP Loss
                    scales=[lr_gcov],      # 🔴 使用 lr_gcov 对齐 OC 强度
                    config=config,
                    learnable=False,                  # 调试单目标不需要训练残差网络，纯 gcovA 引导极快
                    conflict_weight=0.0,              # 无冲突
                    vis_dir=None
                )
                
                # 2. 用这单独的一份引导场，重新跑一条纯净轨迹 (起点依然用最干净的 y_0)
                traj_single = generate_traj(single_guided_field, z0=y_0, u=None, N=N)
                
                # 3. 单 Prompt 独立引导结果: {id}_gcov-A_singleprompt{k}.jpg
                single_path = os.path.join(target_dir, f'{img_id}_gcov-A_singleprompt{p_idx+1}.jpg')
                save_img(inverse_scaler(traj_single[-1]), path=single_path)
                print(f"  -> Saved single effect for Prompt {p_idx+1}: {single_path}")

        # =====================================================================
        # 🔴 Debug2: 生成两个 Prompt 纯线性相加的图像 (No Residual Guidance)
        # 仅当外部传入 save_combined=True 时才运行
        # =====================================================================
        if save_combined and opt_img_path is not None:
            print("\n--- Generating Combined Trajectory ---")
            lr_gcov = 4400
            # 实例化合并引导场，严格关闭所有网络学习和冲突项
            from utils.composed_guidance import ImageGCovAGMOnlineGuidance
            combined_guided_field = ImageGCovAGMOnlineGuidance(
                base_model=model_fn,
                loss_fns=L_N_list,                          
                scales=[lr_gcov] * num_prompts,    
                config=config,
                learnable=False,                  # ✅ 彻底关闭 Residual Net
                conflict_weight=0.0,              # ✅ 设为0，不计算冲突
                vis_dir=None
            )

            # 最终推理：直接生成纯线性叠加的轨迹
            traj_combined = generate_traj(combined_guided_field, z0=y_0, u=None, N=N)

            # 两个 Prompt 纯线性相加 (No Residual Guidance): {id}_gcov-A_multiprompt.jpg
            combined_path = os.path.join(target_dir, f'{img_id}_gcov-A_multiprompt.jpg')
            save_img(inverse_scaler(traj_combined[-1]), path=combined_path)
            print(f"  -> Saved linear combined image to: {combined_path}")