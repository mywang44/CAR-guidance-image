import torch
import torch.nn as nn
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设这些是你 OC 代码里原有的函数，直接 import 进来
from .conflict_utils import (
    visualize_spatial_conflict,
    visualize_weighted_conflict,
    overlay_heatmap_on_image,
    visualize_pca_trajectory,
    visualize_pca_trajectory_with_landscape,
    visualize_gradient_angle,
    plot_gradient_map,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === [新增] 高频时间投影 ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
    """
    [关键修复]
    将 t 从 [0, 1] 映射到高频特征空间。
    这是解决 "Time Blindness" 和 "输出死鱼" 的唯一解。
    """
    def __init__(self, embedding_size=256, scale=30.0):
        super().__init__()
        # 这里的 scale=30.0 很重要，它决定了网络对时间变化的敏感度
        # 既然你不乘 999，我们就用 scale 来达成同样的效果，但数值更稳定
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: [B, 1]
        x_proj = x[:, 0:1] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# 🔴 适配梯度回归的向量场网络 (输出维度为 in_channels)
class ImageResidualEnergyNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        time_dim = base_channels * 4
        self.t_proj = GaussianFourierProjection(embedding_size=time_dim, scale=30.0)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.inc = DoubleConv(base_channels, base_channels)      
        self.down1 = Down(base_channels, base_channels * 2)      
        self.down2 = Down(base_channels * 2, base_channels * 4)  
        self.bot1 = DoubleConv(base_channels * 4, base_channels * 4)
        self.bot2 = DoubleConv(base_channels * 4, base_channels * 4)
        self.up1 = Up(base_channels * 6, base_channels * 2) 
        self.up2 = Up(base_channels * 3, base_channels)     
        
        # 🔴 核心变化：输出维度不再是 1，而是 in_channels (比如 3)
        self.outc = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x, t):
        if t.dim() == 1: t = t.view(-1, 1)
        
        t_emb = self.t_proj(t)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        x_start = self.input_proj(x)
        x1 = self.inc(x_start)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3 + t_emb
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        
        # 🔴 直接输出向量场 [B, C, H, W]，不再做 view 和 sum！
        out = self.outc(x) 
        return out

# === 下面的组件保持不变 ===

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(8, in_channels), 
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ImageGCovAGMOnlineGuidance:
    def __init__(self, base_model, loss_fns, scales, config, learnable=True, conflict_weight=0.1, vis_dir=None):
        self.flow_model = base_model
        self.classifiers = loss_fns    # 对应 2D 里的 classifiers (这里是 L_N_list)
        self.scales = scales
        self.cfg = config
        self.device = config.device
        
        # 为了兼容 2D 的调用签名，填入 Dummy targets
        self.targets = [None] * len(loss_fns) 
        self.distribution = None       # 图像场景下没有真实的 Toy GT 分布
        self.vis_dir = vis_dir
        self.conflict_weight = conflict_weight
        
        # 实例化残差网络 (默认处理3通道图像)
        self.learned_guidance_model = ImageResidualEnergyNet(in_channels=3).to(self.device)
        self._prepare_models_for_subclass()

    def __call__(self, x, t):
        """接口封装，使其能作为 dynamic 函数直接传给 generate_traj"""
        v_uncond = self.flow_model(x, t)
        g = self.compute_guidance(x, t, v_uncond)
        return v_uncond + g

    def _prepare_models_for_subclass(self):
        # Initialize last layer to zero for warm start (residual = 0 initially)
        if hasattr(self, 'learned_guidance_model'):
            nn.init.zeros_(self.learned_guidance_model.outc.weight)
            nn.init.zeros_(self.learned_guidance_model.outc.bias)

    def _compute_trajectory_conflict_mask(self, xs_stacked, ts_stacked, num_steps, batch_size):
        # ts_stacked shape: (T, B, 1)，与 xs_stacked 对齐

        conflict_list = []
        for step in range(num_steps):
            x_t = xs_stacked[step]              # [B, C, H, W]
            t_t = ts_stacked[step] * 999.0      # [B, 1]，还原到 0~999 范围

            with torch.no_grad():
                # ✅ 把 t_t 传下去
                c_t = self._compute_conflict_score(x_t, self.targets, t=t_t)
            conflict_list.append(c_t)
            
        # 将列表重新堆叠回 [T, B] 的形状，与 2D 逻辑完美对齐
        conflict = torch.stack(conflict_list, dim=0)
        
        threshold, _ = self._get_conflict_threshold_and_temperature()
        
        # 兼容 _compute_conflict_mask 逻辑
        conflict_mask = (conflict > threshold).float()
        active_ratio = conflict_mask.mean().item()
        return conflict_mask, active_ratio

    def _compute_trajectory_weights_ground_truth(self, x1, batch_size):
        """Compute target label distribution p*(i) from terminal reward."""
        t_final = torch.ones(batch_size, 1, device=self.device) * 999.0
        with torch.no_grad():
            # 图像场景无 GT Distribution，直接使用 base_log_prob Fallback 作为 Reward
            v_final = self.flow_model(x1, t_final.view(-1))
            base_log_prob = self._compute_g_cov_a_energy(
                x1, t_final, v_final,
                self.classifiers, self.targets, self.scales, self.cfg
            ).flatten()
            # =========================================================
            # 🔴 引入你的绝杀思想：计算终点 x1 的 Conflict Score 惩罚
            # =========================================================
            terminal_conflict = self._compute_conflict_score(x1, self.targets)
            
            # 用配置里的 conflict_weight 作为惩罚系数
            cw = getattr(self.cfg, "conflict_weight", 10.0) # 这个权重可能需要稍微调大点
            
            # 最终奖励 = 语义得分 - 冲突惩罚
            r1 = base_log_prob - cw * terminal_conflict
            # =========================================================

            beta = getattr(self.cfg, "energy_temperature", 1.0)
            logits = beta * r1
            logits = logits - logits.max()
            w_eff = torch.softmax(logits, dim=0)
            w_eff = w_eff.unsqueeze(-1)

        return w_eff, r1

    def _compute_trajectory_weights(self, x1, batch_size):
        """Compute effective weights based on base quality."""
        t_final = torch.ones(batch_size, 1, device=self.device) * 999.0
        with torch.no_grad():
            v_final = self.flow_model(x1, t_final.view(-1))
            base_log_prob = self._compute_g_cov_a_energy(
                x1, t_final, v_final, self.classifiers, self.targets, self.scales, self.cfg
            ).flatten()
            
            if getattr(self.cfg, "weight_zscore", True):
                med = base_log_prob.median()
                mad = (base_log_prob - med).abs().median() + 1e-8
                r1_norm = (base_log_prob - med) / mad
            else:
                r1_norm = base_log_prob
            
            beta = getattr(self.cfg, "energy_temperature", 1.0)
            scores = -beta * r1_norm
            scores = scores - scores.max() 
            w_eff = torch.softmax(scores, dim=0).unsqueeze(-1)
            
            wu = getattr(self.cfg, "weight_uniform_mix", 0.01)
            if wu > 0:
                w_eff = (1 - wu) * w_eff + wu * (1.0 / batch_size)
            
            w_eff = torch.clamp(
                w_eff,
                min=getattr(self.cfg, "weight_min", 1e-4),
                max=getattr(self.cfg, "weight_clamp_max", 1e6),
            )
        
        return w_eff, base_log_prob

    def _compute_online_loss_gradient(self, xs_stacked, ts_stacked, vs_stacked,
                                    conflict_mask, num_steps, batch_size,
                                    x1_final=None, original_images=None):
        """
        [B 版本移植] MSE Loss + 内部 backward + micro-batch 梯度累积。
        返回 (loss_float, stats_dict)，不再返回 Tensor。
        调用方需在此之前 optimizer.zero_grad()，此之后直接 optimizer.step()。
        """
        T, B, C, H, W = xs_stacked.shape
        space_dim = C * H * W
        total_samples = T * B

        xs_flat_img = xs_stacked.view(-1, C, H, W)
        ts_flat = ts_stacked.view(-1, 1)

        # ===========================================================
        # 1. 计算 Target Gradient (Teacher) —— 在 x1_final 上求能量梯度
        # ===========================================================
        if x1_final is None:
            # 兜底：取轨迹终点
            x1_final = xs_stacked[-1]

        x1_req = x1_final.detach().requires_grad_(True)

        with torch.enable_grad():
            # 用 A 原有的能量函数得到总能量标量，再对 x1 求梯度
            t_final = torch.ones(B, device=self.device) * 999.0
            v_final_dummy = torch.zeros_like(x1_req)  # 终点不需要 v_uncond 预测

            # =======================================================
            # 🔴 K 次平均降噪：真正想降的是 teacher 梯度 ∇E(x1) 的噪声。
            # CLIP 的 DiffAugment 随机增强让每次 ∇E(x1) 都带噪，
            # 这里对【梯度】做 K 次平均（而非对能量），才是降 teacher 噪声。
            # 注意：同一个 x1_req 求 K 次导，需 retain_graph=True。
            # =======================================================
            K = getattr(self.cfg, "reward_avg_K", 8)
            grad_energy = 0.0
            for _k in range(K):
                energy = self._compute_g_cov_a_energy(
                    x1_req, t_final, v_final_dummy,
                    self.classifiers, self.targets, self.scales, self.cfg
                )
                # energy 越大越好 → 梯度上升方向就是 target
                g = torch.autograd.grad(energy.sum(), x1_req, retain_graph=True)[0]
                grad_energy = grad_energy + g
            grad_energy = grad_energy / K   # ← 平均的是梯度，这才是你 teacher
            # target = ∇E(x1)，即让网络预测"朝能量上升的方向走"
            clip_norm = grad_energy.reshape(B, -1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)

        # --- 可选：Consistency Gradient (正交投影) ---
        total_grad = grad_energy.detach()

        # Target：让网络拟合这个梯度方向
        target_final    = total_grad                                        # [B, C, H, W]
        target_expanded = target_final.unsqueeze(0).expand(T, -1, -1, -1, -1)
        target_flat     = target_expanded.reshape(-1, space_dim)

        # ===========================================================
        # 2. 训练 Student (micro-batch + 内部 backward)
        # ===========================================================
        micro_batch  = batch_size
        num_chunks   = (total_samples + micro_batch - 1) // micro_batch
        total_loss_val = 0.0
        stats = {}

        for idx in range(0, total_samples, micro_batch):
            x_chunk    = xs_flat_img[idx : idx + micro_batch].detach()
            t_chunk    = ts_flat[idx : idx + micro_batch]
            targ_chunk = target_flat[idx : idx + micro_batch]

            pred_grad  = self.learned_guidance_model(x_chunk, t_chunk)
            pred_flat  = pred_grad.view(micro_batch, -1)
            targ_flat_view = targ_chunk.view(micro_batch, -1)

            # MSE Loss
            loss_chunk = ((pred_flat - targ_flat_view) ** 2).mean(dim=-1)

            # Conflict Mask 加权
            if conflict_mask is not None:
                mask_full  = conflict_mask.view(-1)
                mask_chunk = mask_full[idx : idx + micro_batch].to(loss_chunk.device)
                weight     = mask_chunk
            else:
                weight = torch.ones_like(loss_chunk)

            loss_scalar  = (loss_chunk * weight).sum() / (weight.sum() + 1e-8)
            loss_backward = loss_scalar / num_chunks
            loss_backward.backward()          # ← 内部 backward，梯度累积
            total_loss_val += loss_scalar.item()

            # 记录第一个 chunk 的统计信息
            if idx == 0:
                with torch.no_grad():
                    p_vec = pred_flat.view(-1)
                    t_vec = targ_flat_view.view(-1)
                    cosine  = F.cosine_similarity(p_vec, t_vec, dim=0, eps=1e-8)
                    p_norm  = p_vec.norm().item()
                    tn_norm = t_vec.norm().item()
                    stats['cosine']      = cosine.item()
                    stats['pred_norm']   = p_norm
                    stats['targ_norm']   = tn_norm
                    stats['ratio']       = p_norm / (tn_norm + 1e-8)
                    stats['mask_ratio']  = weight.mean().item()

            del x_chunk, pred_grad, loss_backward

        avg_loss = total_loss_val / num_chunks
        return avg_loss, stats

    def _log_online_training_progress_ground_truth(self, step, total_steps, loss, active_ratio, w_eff, r1, angle=None):
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            if len(w_eff_flat) > 1:
                w_mean, r_mean = w_eff_flat.mean(), r1.mean()
                w_centered = w_eff_flat - w_mean
                r_centered = r1 - r_mean
                numerator = (w_centered * r_centered).sum()
                denominator = torch.sqrt(
                    (w_centered.pow(2).sum() * r_centered.pow(2).sum()) + 1e-8
                )
                corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
            else:
                corr = 0.0

            # 🔴 新增：将 r1 转换为保留 2 位小数的字符串列表，方便直观查看
            r1_str = "[" + ", ".join([f"{val:.2f}" for val in r1.tolist()]) + "]"
        print(
            f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
            f"Active Conflict: {active_ratio:.1%} | "
            f"corr(w_eff, reward): {corr:.3f}",
            f"Angle(Base, Res): {angle:.1f}°",
            f"r1: {r1_str}", # 🔴 打印真实的 Reward 数组
            flush=True
        )
    
    def _log_online_training_progress(self, step, total_steps, loss, active_ratio, w_eff, base_log_prob):
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            neg_r1 = -base_log_prob
            
            if len(w_eff_flat) > 1:
                w_mean, r_mean = w_eff_flat.mean(), neg_r1.mean()
                w_centered = w_eff_flat - w_mean
                r_centered = neg_r1 - r_mean
                numerator = (w_centered * r_centered).sum()
                denominator = torch.sqrt((w_centered.pow(2).sum() * r_centered.pow(2).sum()) + 1e-8)
                corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
            else:
                corr = 0.0
        
        print(f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
              f"Active Conflict: {active_ratio:.1%} | "
              f"corr(w_eff, -r1): {corr:.3f}", flush=True)

    def train_model(self, z0, num_steps=100, steps=15):
        # ===== [新增] NaN/Inf 防护 =====
        if torch.isnan(z0).any() or torch.isinf(z0).any():
            print("!!! WARNING: z0 contains NaN/Inf! Clamping.")
            z0 = torch.nan_to_num(z0, nan=0.0)
            z0 = torch.clamp(z0, -5.0, 5.0)
        # ================================
        # 1. 对齐 2D 的变量名，优先从 cfg 获取超参数（如果没有则用默认值）
        # 注意：这里的 batch_size 必须等于你传进来的 z0(即 latent_batch) 的真实批量大小
        batch_size = z0.shape[0]  
        # 如果你想强制使用 cfg 里的步骤数，可以解开下面这行，但推荐使用外部传参控制
        steps = getattr(self.cfg, "guidance_train_steps", 10) 
        lr = getattr(self.cfg, "guidance_lr", 1e-3)
        log_interval = 1  # 图像训练较慢，间隔设小一点方便看进度
        
        # 2. ODE solver params (完全对齐 2D 的命名)
        num_steps = 10  # 2D 里是硬编码的 20，图像里我们用传入的 num_steps (100)
        dt = 1.0 / num_steps
        eps = 1e-3
        
        optimizer = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()
        
        # 增加和 2D 一模一样的启动日志打印
        print(f"[ImageGCovAGMOnlineGuidance] Training Online Residual model ({steps} steps)")
        
        # 下面的循环变量完全使用 steps
        for i in range(steps):
            self._epoch_angles = [] # 🕵️ 每次 Epoch 开始前清空历史夹角
            curr_x = z0.detach().clone()
            traj_xs = []
            traj_ts = []
            traj_vs = [] # 🔴 1. 新增：初始化用于存放 v_uncond 的列表
            
            for step in range(num_steps):
                # 严格对齐 generate_traj 的时间缩放逻辑
                t_norm = step / num_steps * (1. - eps) + eps
                t_tensor = torch.full((batch_size, 1), t_norm, device=self.device)
                t_model = t_tensor * 999.0
                
                traj_xs.append(curr_x.clone())
                # 【修改点】：存入 0~1 的 t_tensor，防止后续 MSE Loss 计算时网络爆炸！
                traj_ts.append(t_tensor)
                
                # 【修改点 1】：坚决把 U-Net 关进 no_grad 笼子里！
                # 既然底层算法不需要对 v_uncond 求导，建图纯粹是浪费 20GB 显存
                x_in = curr_x.detach()
                with torch.no_grad():
                    v_uncond = self.flow_model(x_in, t_model.view(-1))
                # ===== [新增] =====
                if torch.isnan(v_uncond).any():
                    v_uncond = torch.nan_to_num(v_uncond, nan=0.0)
                # ==================                
                traj_vs.append(v_uncond.detach().clone()) # 🔴 2. 新增：把每一步的 v_uncond 存进列表
                should_log_norms = (i+1) % log_interval == 0 and step == 0  
                # =========================================================
                # 🔴 决定是否在当前 Epoch 画图 (例如 Epoch 1, 一半, 最后一步)
                current_epoch = i + 1
                if (current_epoch == 1 or current_epoch == steps // 2 or current_epoch == steps) and step == 0:
                    vis_step_val = current_epoch
                else:
                    vis_step_val = None
                #=================================================================             
                g_total = self.compute_guidance(x_in, t_model, v_uncond)
                d_x = v_uncond + g_total
                
                curr_x = curr_x + d_x.detach() * dt
            
            x1 = curr_x
            
            xs_stacked = torch.stack(traj_xs, dim=0)
            ts_stacked = torch.stack(traj_ts, dim=0)
            vs_stacked = torch.stack(traj_vs, dim=0) # 🔴 3. 新增：把列表堆叠成 tensor
            
            conflict_mask, active_ratio = self._compute_trajectory_conflict_mask(
                xs_stacked, ts_stacked, num_steps, batch_size   # ✅ 新增 ts_stacked
            )
            if conflict_mask is None or active_ratio < 1e-6:
                print(f"Online-Step {i+1}/{steps} Skipped: Low Conflict ({active_ratio:.1%})", flush=True)
                continue
            
            w_eff, r1 = self._compute_trajectory_weights_ground_truth(x1, batch_size)
            optimizer.zero_grad()   # ← 必须在 compute_online_loss_gradient 之前清空！
            loss_val, stats = self._compute_online_loss_gradient(
                xs_stacked, ts_stacked, vs_stacked, conflict_mask,
                num_steps, batch_size, x1_final=x1            # ← 传入终点，用于计算 Teacher 梯度
            )

            # 🕵️ 计算当前 Epoch 的平均夹角
            avg_angle = sum(self._epoch_angles) / len(self._epoch_angles) if len(self._epoch_angles) > 0 else 0.0
            if log_interval and (i+1) % log_interval == 0:
                print(f"\n=== Online-Step {i+1}/{steps} ===")
                print(f"Loss (MSE): {loss_val:.6e}")
                print(f"Direction Cosine: {stats['cosine']:.4f} {'✅' if stats['cosine'] > 0 else '❌'}")
                print(f"Norms: Pred={stats['pred_norm']:.6f} | Targ={stats['targ_norm']:.6f} | Ratio={stats['ratio']:.4f}")
                print(f"Active Conflict: {active_ratio:.1%}")

                with torch.no_grad():
                    last_layer = self.learned_guidance_model.outc
                    b_val  = last_layer.bias.mean().item() if last_layer.bias is not None else 0.0
                    w_norm = last_layer.weight.norm().item()
                    if last_layer.weight.grad is not None:
                        grad_norm = last_layer.weight.grad.norm().item()
                        print(f"Backprop Grad Norm: {grad_norm:.6f}")
                    else:
                        print("!!! CRITICAL: No Gradient !!!")
                    print(f"[Trap Check] Bias Mean: {b_val:.4f} | Weight Norm: {w_norm:.6f}")
                print("="*40)



            # ===== [新增] 梯度裁剪，防 NaN =====
            torch.nn.utils.clip_grad_norm_(self.learned_guidance_model.parameters(), 1.0)
            # ====================================
            optimizer.step()

        self.learned_guidance_model.eval()
        # self.save_learned_guidance() # 如果需要保存模型权重，可以取消注释

    def _prepare_input_for_grad(self, x, need_higher_order=False):
        """Prepare input tensor for gradient computation."""
        if need_higher_order:
            return x if x.requires_grad else x.clone().detach().requires_grad_(True)
        return x.detach().requires_grad_(True) if not x.requires_grad else x

    @torch.enable_grad()
    def compute_guidance(self, x, t, v_uncond, need_higher_order=False):
        if not self.learned_guidance_model:
            return torch.zeros_like(x)

        x_req = self._prepare_input_for_grad(x, need_higher_order)
        t_vec = t.view(-1)
        t_model = t.view(-1, 1)

        # 1. Base guidance：CLIP 能量是标量 → 对 x 求导得到向量场
        base_energy = self._compute_g_cov_a_energy(
            x_req, t_vec, v_uncond.detach(),
            self.classifiers, self.targets, self.scales, self.cfg
        )
        g_base = torch.autograd.grad(
            base_energy.sum(), x_req,
            create_graph=need_higher_order,
            retain_graph=need_higher_order
        )[0]

        # 2. Residual guidance：网络【直接输出向量场】，不再 squeeze、不再求导、不再归一化
        # 训练时 MSE 让网络回归 ∇E(x1)（方向+大小都监督），输出本身就是正确尺度的力。
        # 【修改点】：将 t_model 逆向还原为 t_norm (0~1) 喂给残差网络
        eps = 1e-3
        t_norm = (t_model / 999.0 - eps) / (1.0 - eps)
        g_res = self.learned_guidance_model(x_req, t_norm)   # [B, C, H, W]，已是向量场
        lr_res = getattr(self.cfg, "lr_res", 30.0)            # 纯缩放系数，不带 /norm
        g_res = lr_res * g_res                                # 把弱残差放大到能影响 base 的量级

        # 3. Conflict 门控（层3：只在协方差大处施加残差）
        # g_res 不是能量、不能在能量层合成 → 在【梯度层】合成：g_base + weight * g_res
        if self.classifiers:
            with torch.no_grad():
                conflict = self._compute_conflict_score(x_req.detach(), self.targets, t=t_model)

            if conflict is not None:
                threshold, temperature = self._get_conflict_threshold_and_temperature()
                weight = torch.sigmoid((conflict - threshold) / temperature)
                weight = weight.view(-1, 1, 1, 1)
                grad = g_base + weight * g_res
            else:
                grad = g_base + g_res
        else:
            grad = g_base + g_res

        return grad

    def compute_direct_conflict_score(self, x, t=None, epsilon=1e-8):
        grads = []
        need_double_grad = x.requires_grad

        with torch.enable_grad():
            x_req = x if need_double_grad else x.detach().requires_grad_(True)

            # ✅ 核心修复：重新过 flow model，让 Jacobian 接通
            if t is not None:
                t_vec = t.view(-1)
                t_4d  = t.view(-1, 1, 1, 1)
                eps   = 1e-3
                t_norm = (t_4d / 999.0 - eps) / (1.0 - eps)

                # v_t 从 x_req 计算，计算图连通
                v_t    = self.flow_model(x_req, t_vec)
                x1_est = x_req + (1.0 - t_norm) * v_t   # Jacobian 在这里接入
            else:
                # 兜底：没有 t 时退化为原来的行为
                x1_est = x_req

            for L_N in self.classifiers:
                loss        = L_N(x1_est)
                loss_scalar = loss.sum() if loss.dim() > 0 else loss
                g = torch.autograd.grad(
                    loss_scalar, x_req,
                    retain_graph=True,
                    create_graph=need_double_grad
                )[0]
                grads.append(g)

        return self.compute_conflict_score(grads, epsilon=epsilon)

    def compute_conflict_score(self, grads, epsilon=1e-8):
        if not grads:
            raise ValueError("No gradients provided.")

        first = grads[0]
        B = first.shape[0] if first.dim() > 1 else 1
        device = first.device

        if len(grads) < 2:
            return torch.zeros(B, device=device)

        pairwise_scores = []
        zero_thr = getattr(self.cfg, "zero_gradient_threshold_direct", 1e-6)

        for i in range(len(grads)):
            gi = grads[i].view(B, -1)
            norm_i = gi.norm(dim=-1, keepdim=True)
            gi_unit = gi / (norm_i + epsilon)

            for j in range(i + 1, len(grads)):
                gj = grads[j].view(B, -1)
                norm_j = gj.norm(dim=-1, keepdim=True)
                gj_unit = gj / (norm_j + epsilon)

                cos = (gi_unit * gj_unit).sum(dim=-1)

                norm_i_flat = norm_i.squeeze(-1)
                norm_j_flat = norm_j.squeeze(-1)
                near_zero = (norm_i_flat < zero_thr) | (norm_j_flat < zero_thr)

                conflict = -cos + 1.0
                conflict = torch.where(near_zero, torch.zeros_like(conflict), conflict)
                pairwise_scores.append(conflict)

        conflict_avg = torch.stack(pairwise_scores, dim=0).mean(dim=0)
        return conflict_avg

    def _compute_conflict_score(self, x: torch.Tensor, labels, t=None) -> torch.Tensor:
        return self.compute_direct_conflict_score(x, t=t)

    def _compute_g_cov_a_energy(self, x, t, v_uncond, classifiers, targets, scales, cfg):
        t_ = t.view(-1, 1, 1, 1)
        eps = 1e-3
        
        # 🔴 完美反推回 OC 里的 t_norm 
        # 因为 t_model = (t_norm * (1-eps) + eps) * 999
        # 所以 t_norm = (t_ / 999.0 - eps) / (1.0 - eps)
        t_norm = (t_ / 999.0 - eps) / (1.0 - eps)
        
        # 绝对对齐 OC 里的: x1_pred = x_t + v_t * (1.0 - t_norm)
        x1_est = x if getattr(cfg, "estimate_x1", False) else x + (1.0 - t_norm) * v_uncond
        
        scales = scales if scales and len(scales) == len(classifiers) else [1.0] * len(classifiers)

        total_obj = torch.zeros(x.shape[0], device=x.device)
        for clf, target, lam in zip(classifiers, targets, scales):
            # clf 即 L_N_list 中的函数，直接返回 Loss
            # clf 内部自带 alpha 缩放，不要重复乘
            loss = clf(x1_est)
            # 根据 2D 逻辑，要求导作为 Guidance 加入到 v_uncond。
            # 为了使 x 朝 Loss 减小的方向移动，能量应当为 -Loss。     
            # OC 里更新法则为: u += -lr_gcov * grad
            # 所以这里能量定义为: E = -lam * loss (其中 lam = lr_gcov)
            # 这样 \nabla E 就算出了 -lr_gcov * \nabla loss，绝对对齐！
            total_obj = total_obj - float(lam) * loss
        return total_obj

    def _save_gradient_heatmaps(self, g_base, g_res, step):
        """
        绘制并保存 Base 和 Residual 梯度的空间热力图 (仅取 Batch 中的第一张图)
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        if self.vis_dir is None:
            return

        os.makedirs(self.vis_dir, exist_ok=True)

        # 取 Batch 中的第一张图，并计算通道维度的 L2 范数 (即每个像素上的受力大小)
        # 形状从 [B, C, H, W] -> [H, W]
        base_mag = g_base[0].norm(dim=0).detach().cpu().numpy()
        res_mag = g_res[0].norm(dim=0).detach().cpu().numpy()

        # 归一化到 0-1 以便映射颜色
        base_norm = (base_mag - base_mag.min()) / (base_mag.max() - base_mag.min() + 1e-8)
        res_norm = (res_mag - res_mag.min()) / (res_mag.max() - res_mag.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 绘制 Base Guidance 热力图
        im0 = axes[0].imshow(base_norm, cmap='turbo')
        axes[0].set_title(f'Base Guidance (CLIP) - Step {step}')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 绘制 Residual Guidance 热力图
        im1 = axes[1].imshow(res_norm, cmap='turbo')
        axes[1].set_title(f'Residual Guidance (U-Net) - Step {step}')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'gradient_heatmap_step_{step:03d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_spatial_conflict_map(self, x, t, v_uncond, step):
        """
        可视化模型眼中真实的“空间冲突区域” (加入力量加权，过滤背景边缘噪声)
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        import torch.nn.functional as F

        if self.vis_dir is None or len(self.classifiers) < 2:
            return

        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            
            t_ = t.view(-1, 1, 1, 1)
            dt_remain = 1.0 - (t_ / 999.0)
            x1_est = x_req if getattr(self.cfg, "estimate_x1", False) else x_req + dt_remain * v_uncond.detach()
            
            grads = []
            for clf in self.classifiers[:2]: 
                loss = clf(x1_est)
                loss_scalar = loss.sum() if loss.dim() > 0 else loss
                g = torch.autograd.grad(loss_scalar, x_req, retain_graph=True)[0]
                grads.append(g)

        g1 = grads[0][0] # [C, H, W]
        g2 = grads[1][0]

        # 1. 计算每个像素点的真实受力大小
        norm1 = g1.norm(dim=0) # [H, W]
        norm2 = g2.norm(dim=0)
        
        # 2. 计算方向冲突 (和之前一样)
        g1_unit = g1 / (norm1.unsqueeze(0) + 1e-8)
        g2_unit = g2 / (norm2.unsqueeze(0) + 1e-8)
        cos_sim = (g1_unit * g2_unit).sum(dim=0) 
        spatial_conflict = -cos_sim + 1.0 # 范围 [0, 2]
        
        # =========================================================
        # 🔴 核心修复：计算加权掩码 (Magnitude Weighting)
        # 只有当两个 Prompt 在同一个像素点上都发出了较大的力时，冲突才有效！
        # =========================================================
        max_norm = max(norm1.max().item(), norm2.max().item()) + 1e-8
        
        # 将受力大小归一化到 0~1，相乘作为该点的“活跃度权重”
        weight_mask = (norm1 / max_norm) * (norm2 / max_norm)
        
        # 真正的冲突 = 方向相反程度 * 双方的力量大小
        weighted_conflict = spatial_conflict * weight_mask
        # =========================================================

        # 平滑处理
        sc_tensor = weighted_conflict.unsqueeze(0).unsqueeze(0)
        smoothed_conflict = F.avg_pool2d(sc_tensor, kernel_size=5, stride=1, padding=2)
        conflict_np = smoothed_conflict.squeeze().detach().cpu().numpy()

        os.makedirs(self.vis_dir, exist_ok=True)
        plt.figure(figsize=(6, 6))
        # 换用 'inferno' 稍微暗一点的配色，更能凸显高亮区域
        plt.imshow(conflict_np, cmap='inferno') 
        plt.title(f"Weighted Spatial Conflict - Step {step}")
        plt.colorbar()
        plt.axis('off')
        
        save_path = os.path.join(self.vis_dir, f'model_spatial_conflict_step_{step:03d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    def _get_conflict_threshold_and_temperature(self):
        threshold = getattr(self.cfg, "conflict_threshold", 0.1)
        temperature = getattr(self.cfg, "conflict_temperature", 0.1)
        return threshold, temperature