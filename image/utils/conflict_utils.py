"""
Conflict visualization utilities: spatial/weighted conflict maps,
heatmap overlay, PCA trajectory, gradient angle, and gradient magnitude maps.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_true_landscape(pca, bounds, latent_shape, loss_fn, device, resolution=25, batch_size=16):
    """
    真实采样 Loss 地形图 (Brute-force Landscape)。

    PCA 空间是 x_N (轨迹终点 latent)，inverse_transform 直接得到 x_N，
    无需运行 ODE，只需 reshape 后计算 L_N(x_N)。

    Args:
        pca: 训练好的 sklearn PCA 对象
        bounds: (x_min, x_max, y_min, y_max) 绘图边界
        latent_shape: tuple (B, C, H, W)，用于 reshape 展平的 latent
        loss_fn: callable(x_N_batch [B,C,H,W]) -> tensor [B]，返回每个样本的 Loss
        device: torch device
        resolution: 网格分辨率 (25x25=625 点，建议 20-30)
        batch_size: 批处理大小

    Returns:
        grid_x, grid_y, grid_z: 用于 contourf 绘图的网格数据
    """
    x_min, x_max, y_min, y_max = bounds

    # 1. 创建 2D 网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    flat_grid_2d = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # 2. Inverse PCA: 2D -> 高维 x_N
    flat_latents_np = pca.inverse_transform(flat_grid_2d)
    all_latents = torch.from_numpy(flat_latents_np).float().to(device)
    total_points = len(all_latents)

    # latent_shape 如 (1, 18, 32, 32)，reshape 时需要 (N, C, H, W)
    if len(latent_shape) == 4 and latent_shape[0] == 1:
        shape_for_reshape = (-1, latent_shape[1], latent_shape[2], latent_shape[3])
    else:
        shape_for_reshape = (-1,) + tuple(latent_shape)

    all_losses = []
    print(f"  True landscape: sampling {resolution}x{resolution}={total_points} points...")

    with torch.no_grad():
        for i in range(0, total_points, batch_size):
            batch_flat = all_latents[i:min(i + batch_size, total_points)]
            batch_z = batch_flat.reshape(shape_for_reshape)
            batch_loss = loss_fn(batch_z)
            if isinstance(batch_loss, torch.Tensor):
                batch_loss = batch_loss.cpu().numpy()
            else:
                batch_loss = np.array(batch_loss)
            all_losses.append(batch_loss)
            if i > 0 and i % (batch_size * 10) == 0:
                print(f"    Processed {i}/{total_points}...")

    flat_losses = np.concatenate(all_losses)
    grid_z = flat_losses.reshape(grid_x.shape)
    return grid_x, grid_y, grid_z


def visualize_spatial_conflict(grad1, grad2, save_path=None, title="Conflict Map"):
    """
    可视化两个梯度在空间上的冲突程度。

    Args:
        grad1, grad2: Tensor, shape [batch_size, C, H, W] (Latent 或 Image space 均可)
        save_path: 图片保存路径
        title: 图片标题
    Returns:
        conflict_img: numpy array of conflict map
    """
    # 1. 确保在 CPU 上计算，并取第一个样本（或对所有样本求平均）
    g1 = grad1.detach().cpu()
    g2 = grad2.detach().cpu()

    # 如果 batch_size > 1，取第一个样本
    if g1.shape[0] > 1:
        g1 = g1[0:1]  # 保持 [1, C, H, W] 形状
        g2 = g2[0:1]

    # 2. 计算 Channel 维度上的余弦相似度（与 compute_cosine_similarity 保持一致）
    # shape 变为 [1, H, W]
    # dim=1 表示沿着 Channel (RGB 或 Latent Channel) 计算向量夹角
    # 标准余弦相似度，值域 [-1, 1]
    standard_cosine_map = F.cosine_similarity(g1, g2, dim=1, eps=1e-8)

    # 3. 映射到 [0, 1]（与 compute_cosine_similarity 保持一致）
    # (x + 1) / 2
    normalized_cosine_map = (standard_cosine_map + 1.0) / 2.0

    # 4. 转换为 Conflict Score (1 - Normalized Cosine)
    # Range: [0, 1]. 0=方向一致, 0.5=正交, 1=方向相反(冲突最大)
    # 与 conflict_loss 计算保持一致：conflict_loss = (1.0 - c_sim).sum()
    conflict_map = 1.0 - normalized_cosine_map

    # 4. (可选) 如果是 Latent Space (如 64x64)，插值放大到可视尺寸 (如 256x256)
    if conflict_map.shape[-1] < 256:
        conflict_map = F.interpolate(conflict_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
        conflict_map = conflict_map.squeeze(1)

    # 5. 绘图
    conflict_img = conflict_map[0].numpy()

    plt.figure(figsize=(10, 8))

    # 使用 'turbo' 色图：蓝色=无冲突，红色=高冲突
    # 值域更新为 [0, 1]，与 conflict_loss 计算保持一致
    plt.imshow(conflict_img, cmap='turbo', vmin=0, vmax=1)
    plt.colorbar(label='Conflict Score (0=Aligned, 1=Opposite)')
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

    return conflict_img


def visualize_weighted_conflict(grad1, grad2, save_path=None, title="Weighted Conflict Map"):
    """
    可视化加权冲突图：梯度幅值 * 方向冲突

    Args:
        grad1, grad2: Tensor, shape [batch_size, C, H, W]
        save_path: 图片保存路径
        title: 图片标题
    Returns:
        weighted_conflict_img: numpy array of weighted conflict map
    """
    g1 = grad1.detach().cpu()
    g2 = grad2.detach().cpu()

    # 如果 batch_size > 1，取第一个样本
    if g1.shape[0] > 1:
        g1 = g1[0:1]  # 保持 [1, C, H, W] 形状
        g2 = g2[0:1]

    # 1. 计算冲突 (方向)，与 compute_cosine_similarity 保持一致
    # 标准余弦相似度，值域 [-1, 1]
    standard_cosine_map = F.cosine_similarity(g1, g2, dim=1, eps=1e-8)
    # 映射到 [0, 1]
    normalized_cosine_map = (standard_cosine_map + 1.0) / 2.0
    # 转换为冲突分数，值域 [0, 1]
    direction_conflict = 1.0 - normalized_cosine_map  # [1, H, W]

    # 2. 计算强度 (幅值)
    mag1 = torch.norm(g1, dim=1)
    mag2 = torch.norm(g2, dim=1)
    magnitude_weight = (mag1 * mag2).sqrt()  # 几何平均

    # 3. 加权冲突 = 方向冲突 * 梯度强度
    weighted_conflict = direction_conflict * magnitude_weight

    # 归一化以便显示 (0-1)
    if weighted_conflict.max() > weighted_conflict.min():
        weighted_conflict = (weighted_conflict - weighted_conflict.min()) / (weighted_conflict.max() - weighted_conflict.min() + 1e-8)
    else:
        weighted_conflict = torch.zeros_like(weighted_conflict)

    # 插值放大
    if weighted_conflict.shape[-1] < 256:
        weighted_conflict = F.interpolate(weighted_conflict.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
        weighted_conflict = weighted_conflict.squeeze(1)

    plt.figure(figsize=(10, 8))

    # 关键修改：
    # vmin=0 固定底部
    # vmax 可以不设（自动适应），但在对比两张图时，最好两张图用同一个 vmax。
    # 例如：先跑一遍 w=0，记下 max 值，然后跑 w=0.1 时传入那个 max 值。
    plt.imshow(weighted_conflict[0].numpy(), cmap='inferno', vmin=0)

    plt.title(title)
    plt.axis('off')
    plt.colorbar(label='Weighted Conflict (Magnitude * Direction)')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

    return weighted_conflict[0].numpy()


def overlay_heatmap_on_image(original_img_tensor, heatmap_np, alpha=0.5, colormap_name='inferno', save_path=None):
    """
    将热力图叠加到原始图像上。

    Args:
        original_img_tensor (torch.Tensor): 原始图像张量。
                                            形状可以是 [1, C, H, W] 或 [C, H, W]。
                                            假设数值范围已经归一化到 [0, 1]。
        heatmap_np (numpy.ndarray): 热力图数组 (通常来自 visualize_weighted_conflict 的返回值)。
                                    形状 [H_h, W_h]，范围最好在 [0, 1] 之间以便着色。
        alpha (float): 热力图的透明度，范围 [0, 1]。值越大热力图越明显。默认 0.5。
        colormap_name (str): matplotlib 的色图名称，如 'inferno', 'jet', 'turbo'。
        save_path (str, optional): 保存叠加图像的路径。

    Returns:
        blended_img_np (numpy.ndarray): 叠加后的图像数组 [H, W, C]，范围 [0, 1]。
    """
    # --- 1. 处理原始图像 ---
    # 确保是 [C, H, W]
    if original_img_tensor.dim() == 4:
        img_tensor = original_img_tensor.squeeze(0)
    else:
        img_tensor = original_img_tensor

    # 转为 Numpy [H, W, C] 且确保在 CPU
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    # 确保范围在 [0, 1] (防止之前处理有误)
    img_np = np.clip(img_np, 0.0, 1.0)
    H, W, C = img_np.shape

    # --- 2. 处理热力图 ---
    # 转为 Tensor 以便使用插值
    heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0).float()  # [1, 1, Hh, Wh]

    # 插值放大到原图尺寸 [H, W]
    heatmap_resized = F.interpolate(heatmap_tensor, size=(H, W), mode='bilinear', align_corners=False)
    heatmap_resized_np = heatmap_resized.squeeze().numpy()  # [H, W]

    # --- 3. 热力图着色 (Gray -> RGB) ---
    # 确保热力图在 [0, 1] 用于着色映射 (Min-Max 归一化)
    # 注意：这里为了可视化叠加效果，我们进行相对归一化，让当前图中最热的地方最亮
    h_min, h_max = heatmap_resized_np.min(), heatmap_resized_np.max()
    if h_max > h_min:
        heatmap_norm = (heatmap_resized_np - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_resized_np)

    # 获取 matplotlib 色图
    cmap = plt.get_cmap(colormap_name)
    # cmap(heatmap_norm) 返回的是 RGBA [H, W, 4] 的数组
    # 我们只需要 RGB 通道 [:, :, :3]
    heatmap_rgb = cmap(heatmap_norm)[..., :3]

    # --- 4. Alpha 混合 (Blending) ---
    # blended = original * (1 - alpha) + heatmap * alpha
    blended_img_np = img_np * (1 - alpha) + heatmap_rgb * alpha
    blended_img_np = np.clip(blended_img_np, 0.0, 1.0)

    # --- 5. 保存或显示 ---
    if save_path:
        # matplotlib 保存时会自动处理 [0,1] 的 float 数组
        plt.imsave(save_path, blended_img_np)

    return blended_img_np

def visualize_pca_trajectory(history_z0, history_grad_gcov, history_grad_res, save_path=None):
    """
    使用PCA将优化轨迹和梯度方向降维到2D平面进行可视化。
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: sklearn not available, skipping PCA visualization")
        return

    # 1. 准备数据
    X = np.array(history_z0)
    G_gcov = -np.array(history_grad_gcov)
    G_res = -np.array(history_grad_res)

    if len(X) < 2:
        print("Warning: Not enough data points for PCA visualization")
        return

    # 2. 训练 PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 3. 投影梯度
    X_plus_gcov = pca.transform(X + G_gcov)
    X_plus_res = pca.transform(X + G_res)
    G_gcov_2d = X_plus_gcov - X_2d
    G_res_2d = X_plus_res - X_2d

    # 4. 计算自适应箭头长度 (Critical Fix)
    # 计算当前视野的跨度
    x_span = X_2d[:, 0].max() - X_2d[:, 0].min()
    y_span = X_2d[:, 1].max() - X_2d[:, 1].min()
    max_span = max(x_span, y_span)
    if max_span == 0: max_span = 1.0
    
    # 设定箭头长度为图幅宽度的 5% (您可以调整这个系数 0.05)
    vis_arrow_len = max_span * 0.05
    
    # 归一化方向，然后乘以可视长度
    def normalize_and_scale(vectors, length):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (vectors / norms) * length

    # 现在的向量长度在图上正好是 max_span * 0.05 这么长
    G_gcov_vis = normalize_and_scale(G_gcov_2d, vis_arrow_len)
    G_res_vis = normalize_and_scale(G_res_2d, vis_arrow_len)

    # 5. 绘图
    plt.figure(figsize=(12, 10))

    # 画轨迹
    plt.plot(X_2d[:, 0], X_2d[:, 1], 'k-', alpha=0.3, linewidth=2, label='Optimization Path', zorder=1)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=np.arange(len(X_2d)), cmap='viridis', 
                         s=80, zorder=3, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, label='Iteration', ax=plt.gca())
    
    # 画起点终点
    plt.scatter(X_2d[0, 0], X_2d[0, 1], s=250, c='green', marker='*', edgecolors='black', zorder=5, label='Start')
    plt.scatter(X_2d[-1, 0], X_2d[-1, 1], s=250, c='red', marker='*', edgecolors='black', zorder=5, label='End')

    # 画箭头 (Fix Quiver Parameters)
    # scale=1, scale_units='xy' 意味着：向量里的数值是多少，画在图上就是多少单位长
    # width 稍微调大一点以便看清
    
    plt.quiver(X_2d[:, 0], X_2d[:, 1], 
               G_gcov_vis[:, 0], G_gcov_vis[:, 1], 
               color='red', alpha=0.7, label='g-cov-A (Task)', 
               angles='xy', scale_units='xy', scale=1, width=0.005, zorder=4)

    plt.quiver(X_2d[:, 0], X_2d[:, 1], 
               G_res_vis[:, 0], G_res_vis[:, 1], 
               color='blue', alpha=0.7, label='Residual OC (Constraint)', 
               angles='xy', scale_units='xy', scale=1, width=0.005, zorder=4)

    # ... (其余 Title, Legend, Save 代码不变) ...
    plt.title('Optimization Trajectory & Gradient Conflict (PCA Projection)', fontsize=14, fontweight='bold')
    plt.xlabel('PC 1 (Principal Component 1)', fontsize=12)
    plt.ylabel('PC 2 (Principal Component 2)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Text explanation 保持不变
    # ...

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_pca_trajectory_with_landscape(history_z0, history_grad_gcov, history_grad_res,
                                            history_loss_values,
                                            save_path=None,
                                            use_true_landscape=False,
                                            true_landscape_context=None):
    """
    可视化 PCA 轨迹，并叠加 CLIP Loss 地形热力图 (Loss Landscape)。

    默认使用插值拟合 (快速)；可选 use_true_landscape=True 使用真实采样 (慢但准确)。

    Args:
        history_z0: List[numpy array], 轨迹点 (x_N，已 flatten)
        history_grad_gcov: List[numpy array], gcovA 梯度 (已 flatten)
        history_grad_res: List[numpy array], Residual OC 梯度 (已 flatten)
        history_loss_values: List[float], 每步对应的 Total Loss (插值用)
        save_path: 保存路径
        use_true_landscape: 若 True，使用真实采样替代插值 (需提供 true_landscape_context)
        true_landscape_context: dict，需包含 L_N_list, latent_shape, device, resolution(默认25), batch_size(默认16)
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: sklearn not available, skipping PCA landscape visualization")
        return
    if use_true_landscape and (true_landscape_context is None or 'L_N_list' not in true_landscape_context):
        print("Warning: use_true_landscape=True but true_landscape_context invalid, falling back to interpolation")
        use_true_landscape = False

    # 1. 数据准备
    X = np.array(history_z0)
    G_gcov = -np.array(history_grad_gcov)
    G_res = -np.array(history_grad_res)
    losses = np.array(history_loss_values)

    if len(X) < 2:
        print("Warning: Not enough data points for PCA landscape visualization")
        return
    if not use_true_landscape and len(losses) != len(X):
        print("Warning: history_loss_values length mismatch, falling back to PCA trajectory without landscape")
        visualize_pca_trajectory(history_z0, history_grad_gcov, history_grad_res, save_path=save_path)
        return

    # 2. PCA 降维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 3. 投影梯度
    X_plus_gcov = pca.transform(X + G_gcov)
    X_plus_res = pca.transform(X + G_res)
    G_gcov_2d = X_plus_gcov - X_2d
    G_res_2d = X_plus_res - X_2d

    # 4. 生成 Loss 地形
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    padding_x = (x_max - x_min) * 0.2
    padding_y = (y_max - y_min) * 0.2
    bounds = (x_min - padding_x, x_max + padding_x, y_min - padding_y, y_max + padding_y)

    if use_true_landscape:
        ctx = true_landscape_context
        L_N_list = ctx['L_N_list']
        latent_shape = ctx['latent_shape']
        device = ctx['device']
        resolution = ctx.get('resolution', 25)
        batch_size = ctx.get('batch_size', 16)

        def _loss_fn(batch_z):
            # batch_z [B, C, H, W], 对每个样本求所有 prompt 的 Loss 之和
            B = batch_z.shape[0]
            losses_out = []
            for b in range(B):
                xb = batch_z[b:b + 1]
                total = 0.0
                for ln in L_N_list:
                    total += ln(xb).item()
                losses_out.append(total)
            return torch.tensor(losses_out, device=batch_z.device, dtype=torch.float32)

        try:
            grid_x, grid_y, grid_z = compute_true_landscape(
                pca, bounds, latent_shape, _loss_fn, device,
                resolution=resolution, batch_size=batch_size
            )
            landscape_label = 'CLIP Loss (True Sampled)'
        except Exception as e:
            print(f"Warning: True landscape failed ({e}), falling back to interpolation")
            use_true_landscape = False

    if not use_true_landscape:
        if SCIPY_AVAILABLE:
            grid_x, grid_y = np.mgrid[
                x_min - padding_x:x_max + padding_x:200j,
                y_min - padding_y:y_max + padding_y:200j
            ]
            grid_z = griddata(X_2d, losses, (grid_x, grid_y), method='linear')
        else:
            print("Warning: scipy not available, falling back to PCA trajectory without landscape")
            visualize_pca_trajectory(history_z0, history_grad_gcov, history_grad_res, save_path=save_path)
            return
        landscape_label = 'CLIP Loss (Interpolated)'

    # 5. 自适应箭头长度
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)
    if max_span == 0:
        max_span = 1.0
    vis_arrow_len = max_span * 0.05

    def normalize_and_scale(vectors, length):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (vectors / norms) * length

    G_gcov_vis = normalize_and_scale(G_gcov_2d, vis_arrow_len)
    G_res_vis = normalize_and_scale(G_res_2d, vis_arrow_len)

    # 6. 绘图
    fig, ax = plt.subplots(figsize=(14, 11))

    # 绘制地形 (Contourf)
    if not np.all(np.isnan(grid_z)):
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=50, cmap='viridis_r', alpha=0.6)
        fig.colorbar(contour, ax=ax, label=landscape_label)
    else:
        print("Warning: Not enough points to generate landscape heatmap.")

    # 绘制轨迹
    ax.plot(X_2d[:, 0], X_2d[:, 1], 'k-', alpha=0.5, linewidth=2, label='Optimization Path', zorder=2)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c='black', s=30, zorder=3)

    # 起点终点
    ax.scatter(X_2d[0, 0], X_2d[0, 1], s=300, c='green', marker='*', edgecolors='white',
               linewidth=1.5, zorder=5, label='Start')
    ax.scatter(X_2d[-1, 0], X_2d[-1, 1], s=300, c='red', marker='*', edgecolors='white',
               linewidth=1.5, zorder=5, label='End')

    # 绘制箭头
    ax.quiver(X_2d[:, 0], X_2d[:, 1],
              G_gcov_vis[:, 0], G_gcov_vis[:, 1],
              color='red', alpha=0.9, label='g-cov-A (Task Descent)',
              angles='xy', scale_units='xy', scale=1, width=0.004, zorder=4)
    ax.quiver(X_2d[:, 0], X_2d[:, 1],
              G_res_vis[:, 0], G_res_vis[:, 1],
              color='blue', alpha=0.9, label='Residual OC (Constraint Descent)',
              angles='xy', scale_units='xy', scale=1, width=0.004, zorder=4)

    ax.set_title('Loss Landscape & Optimization Trajectory (PCA Projection)', fontsize=16, fontweight='bold')
    ax.set_xlabel('PC 1 (Principal Component 1)', fontsize=12)
    ax.set_ylabel('PC 2 (Principal Component 2)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.2)

    note_str = 'Landscape: TRUE SAMPLED' if use_true_landscape else 'Landscape: interpolated from visited points'
    text_str = (
        'Landscape Interpretation:\n'
        '• Darker/Blue areas: Higher Loss (Worse)\n'
        '• Brighter/Yellow areas: Lower Loss (Better)\n'
        '• Arrows point DOWNHILL (Optimization Direction)\n'
        f'• Note: {note_str}'
    )
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Landscape visualization saved to: {save_path}")
    else:
        plt.show()


# def visualize_pca_trajectory(history_z0, history_grad_gcov, history_grad_res, save_path=None):
#     """
#     使用PCA将优化轨迹和梯度方向降维到2D平面进行可视化。

#     Args:
#         history_z0: List of numpy arrays, 每次迭代的 z0 (已flatten)
#         history_grad_gcov: List of numpy arrays, 每次迭代的 gcovA 梯度 (已flatten)
#         history_grad_res: List of numpy arrays, 每次迭代的 residual OC 梯度 (已flatten)
#         save_path: 保存路径
#     """
#     if not SKLEARN_AVAILABLE:
#         print("Warning: sklearn not available, skipping PCA visualization")
#         return

#     # 1. 准备数据
#     X = np.array(history_z0)        # Shape: [Steps, Dim]
#     G_gcov = np.array(history_grad_gcov)
#     G_res = np.array(history_grad_res)

#     if len(X) < 2:
#         print("Warning: Not enough data points for PCA visualization")
#         return

#     # 2. 训练 PCA (只用路径点训练)
#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X)  # 得到路径在 2D 平面的坐标

#     # 3. 投影梯度
#     # 方法：计算 (z0 + grad) 在 2D 的位置，减去 z0 在 2D 的位置
#     X_plus_gcov = pca.transform(X + G_gcov)
#     X_plus_res = pca.transform(X + G_res)
#     G_gcov_2d = X_plus_gcov - X_2d
#     G_res_2d = X_plus_res - X_2d

#     # 归一化箭头长度（只关心方向，不关心绝对大小）
#     def normalize_vectors(vectors):
#         norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#         norms = np.maximum(norms, 1e-8)  # 避免除零
#         return vectors / norms

#     G_gcov_2d_norm = normalize_vectors(G_gcov_2d)
#     G_res_2d_norm = normalize_vectors(G_res_2d)

#     # 4. 绘图
#     plt.figure(figsize=(12, 10))

#     # 画轨迹 (点和线)
#     plt.plot(X_2d[:, 0], X_2d[:, 1], 'k-', alpha=0.3, linewidth=2, label='Optimization Path', zorder=1)
#     scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=np.arange(len(X_2d)), cmap='viridis',
#                          s=80, zorder=3, edgecolors='black', linewidths=0.5)
#     plt.colorbar(scatter, label='Iteration', ax=plt.gca())

#     # 画起点和终点
#     plt.scatter(X_2d[0, 0], X_2d[0, 1], s=200, c='green', marker='*',
#                edgecolors='black', linewidths=1.5, zorder=5, label='Start')
#     plt.scatter(X_2d[-1, 0], X_2d[-1, 1], s=200, c='red', marker='*',
#                edgecolors='black', linewidths=1.5, zorder=5, label='End')

#     # 画梯度箭头
#     # 自动调整箭头大小
#     scale = np.max(np.abs(X_2d)) * 0.15

#     # 红色箭头 = gcovA (主任务方向)
#     plt.quiver(X_2d[:, 0], X_2d[:, 1],
#                G_gcov_2d_norm[:, 0], G_gcov_2d_norm[:, 1],
#                color='red', alpha=0.7, label='g-cov-A (Task)',
#                scale=scale, scale_units='xy', width=0.003, zorder=2)

#     # 蓝色箭头 = Residual OC (约束方向)
#     plt.quiver(X_2d[:, 0], X_2d[:, 1],
#                G_res_2d_norm[:, 0], G_res_2d_norm[:, 1],
#                color='blue', alpha=0.7, label='Residual OC (Constraint)',
#                scale=scale, scale_units='xy', width=0.003, zorder=2)

#     plt.title('Optimization Trajectory & Gradient Conflict (PCA Projection)', fontsize=14, fontweight='bold')
#     plt.xlabel('PC 1 (Principal Component 1)', fontsize=12)
#     plt.ylabel('PC 2 (Principal Component 2)', fontsize=12)
#     plt.legend(loc='best', fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.3)

#     # 添加解释文本
#     plt.text(0.02, 0.98,
#             'Arrow Interpretation:\n• Red: g-cov-A direction (Task)\n• Blue: Residual OC direction (Constraint)\n• Angle <90°: Cooperative\n• Angle ~90°: Orthogonal (Ideal)\n• Angle >90°: Conflicting',
#             transform=plt.gca().transAxes, fontsize=9,
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#         print(f"PCA trajectory visualization saved to: {save_path}")
#     else:
#         plt.show()
#         plt.close()


def visualize_gradient_angle(history_angles_deg, save_path=None):
    """
    绘制 gcovA 梯度与 OC 梯度之间夹角随迭代的变化曲线。

    解读：
    - 0°: 两方向完全一致（协作）
    - 90°: 两方向正交（Residual OC 理想状态，互不抵消）
    - 180°: 两方向完全相反（冲突，互相抵消）

    Args:
        history_angles_deg: List[float], 每步的夹角（度），可能含 nan
        save_path: 保存路径
    """
    iterations = np.arange(1, len(history_angles_deg) + 1)
    angles = np.array(history_angles_deg, dtype=float)
    valid = ~np.isnan(angles)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, angles, 'o-', color='steelblue', linewidth=2, markersize=6, label='Angle (gcovA vs OC)')

    # 标注理想区域：90° 附近
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Ideal (90°, Orthogonal)')
    plt.axhspan(80, 100, alpha=0.1, color='green', label='Good zone (~90°)')

    # 冲突区域：接近 180°
    plt.axhline(y=180, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Conflict (180°)')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title('Angle between g-cov-A Gradient and OC Gradient', fontsize=14, fontweight='bold')
    plt.ylim(-5, 185)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gradient angle curve saved to: {save_path}")
    else:
        plt.show()
        plt.close()


def plot_gradient_map(grad_tensor, title="Gradient", save_path=None, size=256):
    """
    梯度空间热力图：显示梯度在空间上的强度分布。

    解读：高亮区域表示该梯度想要改变/保护的位置。
    - gcovA 热力图集中在五官/表情区域 → 主任务在改表情
    - OC 热力图集中在背景 → 约束在保护背景
    - 两者高亮重叠 → 在同一位置较劲

    Args:
        grad_tensor: [B, C, H, W]
        title: 图标题
        save_path: 保存路径
        size: 若 H,W < size 则插值到此尺寸便于查看
    """
    # 取绝对值 -> 对通道求平均 -> 取第一个样本
    heatmap = grad_tensor.abs().mean(dim=1).detach().cpu()  # [B, H, W]
    if heatmap.dim() == 3:
        heatmap = heatmap[0]  # [H, W]
    heatmap = heatmap.numpy()

    # 若尺寸过小则插值放大
    if heatmap.shape[0] < size or heatmap.shape[1] < size:
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_t = F.interpolate(heatmap_t, size=(size, size), mode='bilinear', align_corners=False)
        heatmap = heatmap_t.squeeze().numpy()

    # 归一化到 0-1
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='jet')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.colorbar(label='Magnitude (normalized)')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
