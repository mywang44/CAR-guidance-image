import gc
import io
import os
import time

import numpy as np
import logging
import lpips
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

from .flowgrad_utils import get_img, embed_to_latent, clip_semantic_loss, save_img, generate_traj, flowgrad_optimization
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

@torch.no_grad()
def generate_traj_oc(dynamic, z0, u, N):
    traj = []

    # Initial sample
    z = z0.detach().clone()
    traj.append(z.detach().clone().cpu())
    batchsize = z0.shape[0]

    dt = 1./N
    eps = 1e-3
    pred_list = []
    for i in range(N):
        z += u/N
        t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.-eps) + eps
        pred = dynamic(z, t*999)
        #print('compare',torch.sum(dynamic(z, t*(N-1))),torch.sum(u))
        z = z.detach().clone() + pred * dt

        traj.append(z.detach().clone())
        pred_list.append(pred.detach().clone().cpu())

    return traj

def dflow_optimization(z0, dynamic, N, L_N,  number_of_iterations, alpha):
    device = z0.device
    shape = z0.shape
    batch_size = z0.shape[0]
    # z0.requires_grad = True

    dt = 1./N
    eps = 1e-3 # default: 1e-3

    L_best = 0

    def grad_calculate(z0):
        z_traj, non_uniform_set = generate_traj(dynamic, z0, N=N, straightness_threshold=0)
        t_s = time.time()
        inputs = torch.zeros(z_traj[-1].shape, device=device)
        inputs.data = z_traj[-1].to(device).detach().clone()
        inputs.requires_grad = True
        loss = -L_N(inputs)
        lam = torch.autograd.grad(loss, inputs)[0]
        lam = lam.detach().clone()
        eps = 1e-3  # default: 1e-3
        g_old = None
        d = []
        for j in range(N-1, -1, -1):
            inputs = torch.zeros(lam.shape, device=device)
            inputs.data = z_traj[j].to(device).detach().clone()
            inputs.requires_grad = True
            t = (torch.ones((batch_size, )) * j / N * (1.-eps) + eps) * 999
            func = lambda x: (x.contiguous().reshape(shape) + \
                dynamic(x.contiguous().reshape(shape), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)
            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
            lam = vjp.detach().clone().contiguous().reshape(shape)
            del inputs
            if j == 0:
                break
        return lam
    L_best = 0

    # optimizer = torch.optim.LBFGS([z0], lr=alpha, max_iter=number_of_iterations, history_size=10, line_search_fn='strong_wolfe')
    # optimizer.step(closure)

    for i in range(number_of_iterations):
        z_traj, _ = generate_traj(dynamic, z0, N=N, straightness_threshold=0)
        loss = -L_N(z_traj[-1])
        if loss.detach().cpu().numpy() > L_best:
            z_best = z0
            L_best = loss.detach().cpu().numpy()
        z0 = z0 + alpha*grad_calculate(z0)
        print(f'Iter {i}: Loss {loss.item():.4f}')
    return z_best

def flowgrad_optimization_oc_d(z0, u_ind, dynamic, generate_traj, L_N, N=100, number_of_iterations=15, straightness_threshold=None, lr=2.5,
                               weight_decay=0.995):
    """
    使用Optimal Control方法优化FlowGrad中的控制变量
    
    参数:
        z0: 初始latent code (来自embed_to_latent)
        u_ind: 需要优化的控制变量索引列表 (通常是[0,1,...,N-1])
        dynamic: score model的动力学函数 (model_fn)
        generate_traj: 生成轨迹的函数
        L_N: 损失函数 (CLIP loss + 相似度loss的组合)
        N: 时间步数量 (默认100)
        number_of_iterations: 优化迭代次数 (默认15)
        lr: 学习率 (默认2.5)
        weight_decay: 权重衰减系数 (默认0.995)
    
    返回:
        opt_u: 历史最优控制变量字典 {时间步: 控制向量}
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    
    # 初始化控制变量u: 在每个时间步上添加一个控制向量
    # u[j]表示在时间步j上添加的控制量
    u = {}
    eps = 1e-3  # 数值稳定性的小常数
    for ind in u_ind:
        u[ind] = torch.zeros_like(z0).to(z0.device)  # 初始化为0
        u[ind].requires_grad = True
        u[ind].grad = torch.zeros_like(u[ind], device=u[ind].device)

    # L_best: 记录历史最佳loss值 (用于保存最优控制变量)
    L_best = 0
    
    # 开始迭代优化
    for i in range(number_of_iterations):
        # ====== 步骤1: 前向传播 - 生成带控制变量的轨迹 ======
        # 使用当前控制变量u生成从z0到最终输出的完整轨迹
        if straightness_threshold is not None:
            z_traj, non_uniform_set = generate_traj(dynamic, z0, u=u, N=N, straightness_threshold=straightness_threshold)
        else:
            z_traj = generate_traj(dynamic, z0, u=u, N=N, straightness_threshold=straightness_threshold)
        t_s = time.time()
        
        # ====== 步骤2: 计算终点的损失和梯度 ======
        # 使用伴随变量lambda (lam)存储目标函数的梯度
        inputs = torch.zeros(z_traj[-1].shape, device=device)
        inputs.data = z_traj[-1].to(device).detach().clone()
        inputs.requires_grad = True
        loss = -L_N(inputs)  # 负号是因为我们要最大化目标函数
        lam = torch.autograd.grad(loss, inputs)[0]  # lambda_N = ∇L(z_N)
        lam = lam.detach().clone()

        # ====== 步骤3: 保存历史最优控制变量 (checkpoint机制) ======
        # 如果当前loss比历史最好的还要好，保存当前的控制变量
        # 这样即使后续迭代变差，也不会丢失好的结果
        if loss.detach().cpu().numpy() > L_best:
            opt_u = {}  # 最优控制变量
            for ind in u.keys():
                opt_u[ind] = u[ind].detach().clone()
            L_best = loss.detach().cpu().numpy()
        # ====== 步骤4: 反向传播 - 使用伴随方法计算控制变量的梯度 ======
        # 从时间步N-1反向传播到0，计算每个时间步的伴随变量lambda_j
        eps = 1e-3
        g_old = None
        d = []
        for j in range(N-1, -1, -1):  # 从后向前遍历时间步
            if straightness_threshold is not None:
                if j in non_uniform_set['indices']:
                    assert j in u_ind
                else:
                    continue

            # 计算lambda_j: 使用Vector-Jacobian Product (VJP)
            # lambda_j = (∂z_{j+1}/∂z_j)^T @ lambda_{j+1}
            inputs = torch.zeros(lam.shape, device=device)
            inputs.data = z_traj[j].to(device).detach().clone()
            inputs.requires_grad = True
            t = (torch.ones((batch_size, )) * j / N * (1.-eps) + eps) * 999
            
            # 定义状态转移函数: z_{j+1} = z_j + u[j] + dynamic(z_j + u[j], t) * dt
            if straightness_threshold is not None:
                func = lambda x: (x.contiguous().reshape(shape) + u[j].detach().clone() + \
                    dynamic(x.contiguous().reshape(shape) + u[j].detach().clone(), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)
            else:
                func = lambda x: (x.contiguous().reshape(shape) + u[j].detach().clone() + \
                    dynamic(x.contiguous().reshape(shape) + u[j].detach().clone(), t.detach().clone()) / N).view(-1)
            
            # 计算VJP: 得到lambda_j
            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
            lam = vjp.detach().clone().contiguous().reshape(shape)
            
            # 控制变量u[j]的梯度就是伴随变量lambda_j
            # 这是optimal control理论的核心: ∂L/∂u[j] = lambda_j
            u[j].grad = lam.detach().clone()
            del inputs
            if j == 0: break
        
        # ====== 步骤5: 梯度重分配 (用于非均匀时间步) ======
        # 如果使用straightness_threshold，某些时间步可能被跳过
        # 将跳过的时间步的梯度设置为最近的有效时间步的梯度
        if straightness_threshold is not None:
            for j in range(len(non_uniform_set['indices'])):
                start = non_uniform_set['indices'][j]
                try:
                    end = non_uniform_set['indices'][j+1]
                except Exception:
                    end = N
                for k in range(start, end):
                    if k in u_ind:
                        u[k].grad = u[start].grad.detach().clone()
        # ====== 步骤6: 更新控制变量 ======
        # 使用梯度上升更新u (因为我们要最大化目标函数)
        # u_new = weight_decay * u_old + lr * grad_u
        for ind in u.keys():
            u[ind] = u[ind]*weight_decay + batch_size*lr*u[ind].grad
    # 返回历史最优控制变量，而不是最后一次迭代的结果
    return opt_u


def flowgrad_optimization_oc_d_multiprompt(z0, u_ind, dynamic, generate_traj, L_N_list, N=100, number_of_iterations=15, 
                                           straightness_threshold=None, lr=2.5, weight_decay=0.995,
                                           use_L_best=True):  # True: 返回历史最优控制变量; False: 返回最后一步
    """
    多Prompt优化：同时将图像align到多个text prompt
    
    核心思想：为每个prompt维护独立的控制变量 u1, u2, ..., un
    前向传播时使用它们的和: u_total = u1 + u2 + ... + un
    反向传播时分别计算每个prompt的梯度并独立更新
    
    参数:
        z0: 初始latent code
        u_ind: 需要优化的控制变量索引列表
        dynamic: score model的动力学函数
        generate_traj: 生成轨迹的函数
        L_N_list: 损失函数列表 [L_N1, L_N2, ...] 对应多个prompt
        N: 时间步数量 (默认100)
        number_of_iterations: 优化迭代次数 (默认15)
        lr: 学习率 (默认2.5)
        weight_decay: 权重衰减系数 (默认0.995)
    
    返回:
        opt_u_list: 每个prompt的历史最优控制变量列表 [{时间步: 控制向量}, ...]
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    num_prompts = len(L_N_list)
    
    # 为每个prompt初始化独立的控制变量
    # u_list[i][j] 表示第i个prompt在时间步j上的控制向量
    u_list = []
    for prompt_idx in range(num_prompts):
        u = {}
        for ind in u_ind:
            u[ind] = torch.zeros_like(z0).to(z0.device)
            u[ind].requires_grad = True
            u[ind].grad = torch.zeros_like(u[ind], device=u[ind].device)
        u_list.append(u)
    
    # 记录每个prompt的历史最佳loss和最优控制变量
    L_best_list = [0] * num_prompts
    opt_u_list = [{} for _ in range(num_prompts)]
    
    eps = 1e-3
    
    # 开始迭代优化
    for i in range(number_of_iterations):
        # ====== 步骤1: 前向传播 - 使用所有控制变量的和 ======
        # u_total[j] = u1[j] + u2[j] + ... + un[j]
        u_total = {}
        for ind in u_ind:
            u_total[ind] = sum(u_list[prompt_idx][ind] for prompt_idx in range(num_prompts))
        
        # 使用总控制变量生成轨迹
        if straightness_threshold is not None:
            z_traj, non_uniform_set = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
        else:
            z_traj = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
        
        # ====== 步骤2: 为每个prompt分别计算loss和梯度 ======
        loss_values = []  # 存储每个prompt的loss值
        for prompt_idx in range(num_prompts):
            L_N = L_N_list[prompt_idx]
            u = u_list[prompt_idx]
            
            # 计算当前prompt的终点损失
            inputs = torch.zeros(z_traj[-1].shape, device=device)
            inputs.data = z_traj[-1].to(device).detach().clone()
            inputs.requires_grad = True
            loss = -L_N(inputs)
            lam = torch.autograd.grad(loss, inputs, retain_graph=(prompt_idx < num_prompts - 1))[0]
            lam = lam.detach().clone()
            loss_values.append(loss.detach().cpu().numpy())
            
            # ====== 步骤3: 反向传播 - 计算当前prompt的控制变量梯度 ======
            for j in range(N-1, -1, -1):
                if straightness_threshold is not None:
                    if j in non_uniform_set['indices']:
                        assert j in u_ind
                    else:
                        continue
                
                # 计算lambda_j
                inputs = torch.zeros(lam.shape, device=device)
                inputs.data = z_traj[j].to(device).detach().clone()
                inputs.requires_grad = True
                t = (torch.ones((batch_size, )) * j / N * (1.-eps) + eps) * 999
                
                # 注意：这里使用u_total，因为轨迹是用总控制变量生成的
                if straightness_threshold is not None:
                    func = lambda x: (x.contiguous().reshape(shape) + u_total[j].detach().clone() + \
                                     dynamic(x.contiguous().reshape(shape) + u_total[j].detach().clone(), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)
                else:
                    func = lambda x: (x.contiguous().reshape(shape) + u_total[j].detach().clone() + \
                                     dynamic(x.contiguous().reshape(shape) + u_total[j].detach().clone(), t.detach().clone()) / N).view(-1)
                
                output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
                lam = vjp.detach().clone().contiguous().reshape(shape)
                
                # 当前prompt的控制变量梯度
                u[j].grad = lam.detach().clone()
                del inputs
                if j == 0: break
            
            # ====== 步骤4: 梯度重分配 ======
            if straightness_threshold is not None:
                for j in range(len(non_uniform_set['indices'])):
                    start = non_uniform_set['indices'][j]
                    try:
                        end = non_uniform_set['indices'][j+1]
                    except:
                        end = N
                    
                    for k in range(start, end):
                        if k in u_ind:
                            u[k].grad = u[start].grad.detach().clone()
        
        # ====== 步骤5: 更新所有prompt的控制变量 ======
        for prompt_idx in range(num_prompts):
            u = u_list[prompt_idx]
            for ind in u.keys():
                u[ind] = u[ind]*weight_decay + batch_size*lr*u[ind].grad
        
        # ====== 步骤6: 保存历史最优控制变量（在更新后保存）======
        for prompt_idx in range(num_prompts):
            if loss_values[prompt_idx] > L_best_list[prompt_idx]:
                opt_u_list[prompt_idx] = {}
                for ind in u_list[prompt_idx].keys():
                    opt_u_list[prompt_idx][ind] = u_list[prompt_idx][ind].detach().clone()
                L_best_list[prompt_idx] = loss_values[prompt_idx]
        
        # 打印优化进度
        losses_str = ', '.join([f'L{idx+1}: {L_best_list[idx]:.4f}' for idx in range(num_prompts)])
        print(f'Iteration {i+1}/{number_of_iterations}: {losses_str}')
    
    if use_L_best:
        return opt_u_list
    return u_list


def flowgrad_optimization_multiprompt(z0, u_ind, dynamic, generate_traj, L_N_list, N=100, number_of_iterations=15,
                                      straightness_threshold=None, lr=2.0, vis_dir=None):
    """
    基于原始 FlowGrad 实现的 Multi-Prompt 版本
    - 使用 PyTorch SGD 优化器
    - 为每个 Prompt 维护独立的控制变量 u
    - 每个时间步的 u_total = sum(u_prompts)
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    num_prompts = len(L_N_list)
    eps = 1e-3

    # ==========================================
    # 1. 初始化: 为每个 Prompt 创建独立的 u 和 optimizer
    # ==========================================
    u_list = []
    optimizer_list = []
    
    for _ in range(num_prompts):
        u = {}
        # 初始化控制变量 u (全部为 0)
        for ind in u_ind:
            u[ind] = torch.zeros_like(z0).to(device)
            u[ind].requires_grad = True
            u[ind].grad = torch.zeros_like(u[ind], device=device)
        u_list.append(u)
        
        # 每个 Prompt 一个 SGD 优化器
        optimizer_list.append(torch.optim.SGD([u[key] for key in u_ind], lr=lr))

    L_best_list = [1e6] * num_prompts
    opt_u_list = [{} for _ in range(num_prompts)]

    history_z0, history_grad_gcov, history_grad_res = [], [], []

    print(f"\n=== FlowGrad Multi-Prompt Optimization (SGD, lr={lr}) ===")

    for i in range(number_of_iterations):
        # 清零所有优化器的梯度
        for optimizer in optimizer_list:
            optimizer.zero_grad()

        # ==========================================
        # 2. 前向传播: 使用所有 u 的和生成轨迹
        # ==========================================
        u_total = {}
        for ind in u_ind:
            # u_total[j] = u1[j] + u2[j] + ...
            u_total[ind] = sum(u_list[p_idx][ind] for p_idx in range(num_prompts))
            
        if straightness_threshold is not None:
            z_traj, non_uniform_set = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
        else:
            z_traj = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
            non_uniform_set = None

        # ==========================================
        # 3. 反向传播: 对每个 Prompt 独立计算梯度
        # ==========================================
        current_losses = []
        grads_at_zN = []  # 每个 prompt 在 z_N 处的梯度 (用于可视化)
        
        for p_idx in range(num_prompts):
            L_N = L_N_list[p_idx]
            u = u_list[p_idx]
            
            # A. 计算终点梯度 (lambda_N)
            inputs = torch.zeros(z_traj[-1].shape, device=device)
            inputs.data = z_traj[-1].to(device).detach().clone()
            inputs.requires_grad = True
            
            loss = L_N(inputs) # 这是一个需要最小化的 Loss
            lam = torch.autograd.grad(loss, inputs)[0]
            lam = lam.detach().clone()
            grads_at_zN.append(lam.clone())
            
            current_losses.append(loss.item())

            # B. BP 循环 (使用 VJP 计算 lambda_j)
            for j in range(N-1, -1, -1):
                # 跳过非关键帧 (如果启用了非均匀采样)
                if straightness_threshold is not None:
                    if j not in non_uniform_set['indices']:
                        continue
                
                # 准备 VJP 输入
                inputs = torch.zeros(lam.shape, device=device)
                inputs.data = z_traj[j].to(device).detach().clone()
                inputs.requires_grad = True
                
                t_val = (torch.ones((1, ), device=device) * j / N * (1.-eps) + eps) * 999
                
                # 定义 VJP 函数
                # 注意：这里必须使用 u_total[j]，因为前向传播用的是总和
                if straightness_threshold is not None:
                    dt = non_uniform_set['length'][j] / N
                else:
                    dt = 1.0 / N
                    
                u_total_j = u_total[j].detach().clone()
                
                func = lambda x: (x.contiguous().reshape(shape) + u_total_j + \
                                                                    dynamic(x.contiguous().reshape(shape) + u_total_j, t_val.detach().clone()) * dt).view(-1)                
                # 计算 VJP
                output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().reshape(-1))
                lam = vjp.detach().clone().contiguous().reshape(shape)
                
                # 赋值梯度
                if j in u_ind:
                    # u[j].grad = dL/du[j] = lambda_j
                    u[j].grad = lam.detach().clone()
                
                del inputs
                if j == 0: break

            # C. 梯度重分配 (Gradient Re-assignment for non-uniform steps)
            if straightness_threshold is not None:
                for jj in range(len(non_uniform_set['indices'])):
                    start = non_uniform_set['indices'][jj]
                    try:
                        end = non_uniform_set['indices'][jj+1]
                    except:
                        end = N
                    
                    for k in range(start, end):
                        if k in u_ind:
                            u[k].grad = u[start].grad.detach().clone()

        # === 可视化: plot_gradient_map (每5步或最后一步) ===
        if vis_dir is not None and (i % 5 == 0 or i == number_of_iterations - 1):
            os.makedirs(vis_dir, exist_ok=True)
            for p_idx in range(num_prompts):
                plot_gradient_map(
                    grads_at_zN[p_idx],
                    title=f'Focus of Prompt {p_idx+1} — Iter {i+1}',
                    save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_grad_prompt{p_idx+1}_heatmap.png'),
                    size=256
                )

        # === 收集数据用于 PCA 可视化 ===
        if i % 2 == 0:
            with torch.no_grad():
                current_zN = z_traj[-1].detach().cpu().flatten().numpy()
                history_z0.append(current_zN)
                g0 = grads_at_zN[0].detach().cpu().flatten().numpy()
                history_grad_gcov.append(g0)
                # grad_res: 多 prompt 时取 prompt1 或其余之和，单 prompt 时用零向量
                if num_prompts >= 2:
                    g_rest = sum(g.detach().cpu().flatten().numpy() for g in grads_at_zN[1:])
                    history_grad_res.append(g_rest)
                else:
                    history_grad_res.append(np.zeros_like(g0))

        # ==========================================
        # 4. 更新与保存: 执行 SGD Step
        # ==========================================
        for p_idx in range(num_prompts):
            optimizer = optimizer_list[p_idx]
            optimizer.step() # 执行 u = u - lr * grad
            
            # 保存历史最优
            if current_losses[p_idx] < L_best_list[p_idx]:
                opt_u_list[p_idx] = {}
                for ind in u_list[p_idx].keys():
                    opt_u_list[p_idx][ind] = u_list[p_idx][ind].detach().clone()
                L_best_list[p_idx] = current_losses[p_idx]

        loss_str = ", ".join([f"L{idx}:{l:.4f}" for idx, l in enumerate(current_losses)])
        print(f"Iter {i}: {loss_str}")

    # === PCA 可视化 ===
    if vis_dir is not None and num_prompts >= 2 and SKLEARN_AVAILABLE and len(history_z0) > 2:
        visualize_pca_trajectory(
            history_z0, history_grad_gcov, history_grad_res,
            save_path=os.path.join(vis_dir, 'pca_trajectory.png')
        )

    # 返回所有 Prompt 的最优控制变量列表
    return opt_u_list

def flowgrad_optimization_gcovA_multiprompt(z0, u_ind, dynamic, generate_traj, L_N_list, N=100, 
                                             number_of_iterations=15,
                                             straightness_threshold=None, lr=1.0, weight_decay=0.995):
    """
    基于 nabla_xt_J_x1 (Lookahead Guidance) 的优化方法。
    核心逻辑：
    在每个时间步 t，预测 x1_pred = x_t + v(x_t) * (1-t)，
    然后计算 u_t = - lr * ∇x_t Loss(x1_pred)。
    
    控制变量跨迭代累积（类似OC方法），并使用 weight_decay 防止发散。
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    num_prompts = len(L_N_list)
    
    # 初始化每个 prompt 的控制变量
    opt_u_list = [{} for _ in range(num_prompts)]
    for p_idx in range(num_prompts):
        for ind in u_ind:
            opt_u_list[p_idx][ind] = torch.zeros_like(z0)

    eps = 1e-3

    print(f"\n=== GcovA (Lookahead/Nabla_xt_J_x1) Optimization ===")
    print(f"    lr={lr}, weight_decay={weight_decay}, iterations={number_of_iterations}")
    
    for i in range(number_of_iterations):
        # 1. 组合当前的控制变量 (如果是第一次迭代，全为0)
        u_total = {}
        for ind in u_ind:
            u_total[ind] = sum(opt_u_list[p_idx][ind] for p_idx in range(num_prompts)).to(device)
            
        # 2. 生成基础轨迹 (No Grad)
        # 我们需要先拿到轨迹上的点 x_t，然后再在这些点上开启梯度计算
        with torch.no_grad():
            if straightness_threshold is not None:
                z_traj, non_uniform_set = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
            else:
                z_traj = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)

        # 抑制 L_N 在每个时间步的打印（避免打印几千次 regu/reward）
        # L_N_list 中是 bound method (clip_loss.L_N)，需要通过 __self__ 访问实例
        for p_idx in range(num_prompts):
            L_N_list[p_idx].__self__.verbose = False

        # 3. 遍历轨迹上的每个时间点，计算前瞻梯度
        # 注意：这里不需要倒序 (N-1 -> 0)，顺序遍历即可，因为每个点的计算是独立的
        num_keyframes = 0
        for j in range(N):
            # 如果是非均匀采样，跳过非关键帧
            if straightness_threshold is not None:
                if j not in non_uniform_set['indices']:
                    continue
            num_keyframes += 1
            
            # 准备时间变量 t (归一化到 [0,1])
            t_norm = j / N
            # 许多 dynamic 函数需要 t 扩展为 batch 并缩放 (例如到 999 或 1000)
            t_val = (torch.ones((batch_size,), device=device) * t_norm * (1.-eps) + eps) * 999
            
            # === 核心修改开始：Lookahead Gradient 计算 ===
            # 获取当前状态 x_t，并确保在与 z0 相同的 device 上
            # 注意：generate_traj 在第一个时间步会将状态保存在 CPU 上 (traj.append(z.cpu()))
            # 因此这里需要显式地将轨迹点移动回 `device`，否则会出现 CPU / CUDA 混用错误。
            x_t = z_traj[j].to(device).detach().clone()
            
            with torch.enable_grad():
                x_t.requires_grad_(True)
                
                # A. 计算当前速度 v(x_t)
                # 注意：这里我们要让梯度通过 dynamic 模型反向传播
                v_t = dynamic(x_t, t_val)
                
                # B. 单步预测终点 x1_pred (One-step Projection)
                # 公式: x1 = x_t + v_t * (1 - t)
                # 对应 snippet: x1_pred = x_t + self.model(...) * (1 - t)
                dt_to_end = 1.0 - t_norm
                x1_pred = x_t + v_t * dt_to_end
                
                # C. 对每个 Prompt 计算 Loss 并求导
                for p_idx in range(num_prompts):
                    L_N = L_N_list[p_idx]
                    
                    # 计算 Loss (假设 L_N 是要最小化的，如 CLIP distance)
                    loss = L_N(x1_pred)
                    
                    # 计算梯度: grad = ∇x_t Loss
                    # 对于多个 prompt，前面的 prompt 需要 retain_graph=True，
                    # 因为 x1_pred 的计算图需要被多次反向传播。
                    # 最后一个 prompt 不需要保留计算图。
                    is_last_prompt = (p_idx == num_prompts - 1)
                    grads = torch.autograd.grad(loss, x_t, retain_graph=not is_last_prompt)[0]
                    
                    # D. 累积更新控制变量 u[j]
                    # u 叠加在速度场上，所以 u 的方向就是为了修正 x1 而需要的速度变化方向。
                    # 使用累积更新（类似OC方法）: u = u * weight_decay + (-lr * grad)
                    # 单步前瞻梯度比伴随法（adjoint）梯度弱，必须跨迭代累积才能产生足够的编辑效果。
                    if j in u_ind:
                        opt_u_list[p_idx][j] = opt_u_list[p_idx][j].to(device) * weight_decay + (- lr * grads.detach())
            
            # === 核心修改结束 ===

        # 恢复 L_N 的打印
        for p_idx in range(num_prompts):
            L_N_list[p_idx].__self__.verbose = False

        # 迭代结束后，用终点做一次带打印的 loss 评估作为汇总
        with torch.no_grad():
            x1_final = z_traj[-1].to(device)
            print(f"\n--- Iteration {i+1} Summary (keyframes={num_keyframes}) ---")
            for p_idx in range(num_prompts):
                loss_val = L_N_list[p_idx](x1_final)
                print(f"  Prompt {p_idx+1} total loss: {loss_val.item():.4f}")

        # 4. 梯度重分配 (处理非均匀采样被跳过的点)
        if straightness_threshold is not None:
            indices = non_uniform_set['indices']
            for jj in range(len(indices)):
                start = indices[jj]
                end = indices[jj+1] if jj+1 < len(indices) else N
                
                # 将计算出的 start 点的控制量复制给后续被跳过的点
                for k in range(start, end):
                    if k in u_ind:
                        for p_idx in range(num_prompts):
                            opt_u_list[p_idx][k] = opt_u_list[p_idx][start].detach().clone()
        
        print(f"Iteration {i+1} complete.")

    return opt_u_list


# def flowgrad_optimization_gcovA_multiprompt(z0, u_ind, dynamic, generate_traj, L_N_list, N=100, 
#                                              number_of_iterations=15,
#                                              straightness_threshold=None, lr=1.0, weight_decay=0.995):
#     """
#     Fixed GcovA Optimization with Correct Straightness Broadcasting.
#     """
#     device = z0.device
#     shape = z0.shape
#     batch_size = shape[0]
#     num_prompts = len(L_N_list)
    
#     # 初始化每个 prompt 的控制变量
#     opt_u_list = [{} for _ in range(num_prompts)]
#     for p_idx in range(num_prompts):
#         for ind in u_ind:
#             opt_u_list[p_idx][ind] = torch.zeros_like(z0)

#     eps = 1e-3

#     print(f"\n=== GcovA (Lookahead) Opt [Fixed Straightness] ===")
#     print(f"    lr={lr}, decay={weight_decay}, thres={straightness_threshold}")
    
#     for i in range(number_of_iterations):
#         # 1. 组合当前的控制变量
#         u_total = {}
#         for ind in u_ind:
#             u_total[ind] = sum(opt_u_list[p_idx][ind] for p_idx in range(num_prompts)).to(device)
            
#         # 2. 生成基础轨迹 (No Grad)
#         with torch.no_grad():
#             result = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
#             if straightness_threshold is not None:
#                 z_traj, non_uniform_set = result
#             else:
#                 z_traj = result
#                 non_uniform_set = None

#         # 抑制详细打印
#         for p_idx in range(num_prompts):
#             L_N_list[p_idx].__self__.verbose = False

#         # ==========================================
#         # 3. [FIXED] 遍历关键帧并广播梯度
#         # ==========================================
#         # 确定循环的索引集合
#         if straightness_threshold is not None:
#             loop_indices = non_uniform_set['indices']
#         else:
#             loop_indices = range(N)

#         for j in loop_indices:
#             # A. 确定当前关键帧覆盖的时间范围 [start, end)
#             start = j
#             if straightness_threshold is not None:
#                 length = non_uniform_set['length'][start]
#                 end = start + length
#             else:
#                 end = start + 1
            
#             # B. 准备数据
#             t_norm = start / N
#             t_val = (torch.ones((batch_size,), device=device) * t_norm * (1.-eps) + eps) * 999
            
#             # 必须 detach 并 clone，并在 device 上开启梯度
#             x_t = z_traj[start].to(device).detach().clone()
            
#             with torch.enable_grad():
#                 x_t.requires_grad_(True)
                
#                 # C. 前瞻预测 (One-step Lookahead)
#                 v_t = dynamic(x_t, t_val)
#                 x1_pred = x_t + v_t * (1.0 - t_norm)
                
#                 # D. 计算梯度并广播更新
#                 for p_idx in range(num_prompts):
#                     L_N = L_N_list[p_idx]
#                     loss = L_N(x1_pred)
                    
#                     is_last_prompt = (p_idx == num_prompts - 1)
#                     grads = torch.autograd.grad(loss, x_t, retain_graph=not is_last_prompt)[0]
#                     current_grad = grads.detach()
                    
#                     # [关键修复]: 将当前计算出的 Lookahead 梯度，应用到该段内的所有 u[k]
#                     # 这实现了 Zero-Order Hold (零阶保持) 的控制策略
#                     for k in range(start, end):
#                         if k in u_ind:
#                             # 累积更新：每个点都更新自己的历史动量
#                             opt_u_list[p_idx][k] = opt_u_list[p_idx][k].to(device) * weight_decay + (- lr * current_grad)
            
#         # ==========================================
#         # (旧代码中的 Step 4 梯度重分配已不再需要，因为上面循环内部已经处理了填充)
#         # ==========================================

#         # 迭代总结打印
#         with torch.no_grad():
#             x1_final = z_traj[-1].to(device)
#             # print(f"--- Iter {i+1} Summary ---")
#             # for p_idx in range(num_prompts):
#             #     loss_val = L_N_list[p_idx](x1_final)
#             #     print(f"  P{p_idx+1}: {loss_val.item():.4f}")
        
#         print(f"Iteration {i+1} complete.")

#     return opt_u_list

# def compute_cosine_similarity(grad_a, grad_b, epsilon=1e-8):
#     """Helper to compute cosine similarity for conflict loss."""
#     grad_a_flat = grad_a.reshape(grad_a.shape[0], -1)
#     grad_b_flat = grad_b.reshape(grad_b.shape[0], -1)
#     norm_a = grad_a_flat.norm(dim=-1, keepdim=True)
#     norm_b = grad_b_flat.norm(dim=-1, keepdim=True)
#     unit_a = grad_a_flat / (norm_a + epsilon)
#     unit_b = grad_b_flat / (norm_b + epsilon)
#     return (unit_a * unit_b).sum(dim=-1)/2.0 
# # 这个版本返回的conflict score的范围不对


def compute_cosine_similarity(grad_a, grad_b, epsilon=1e-8):
    """
    计算归一化的余弦相似度，带有数值安全保护。
    Returns: [0, 1]
    最后计算的 1-cos sim 在0.5左右
    """
    # 1. 展平 [B, C, H, W] -> [B, -1]
    grad_a_flat = grad_a.flatten(start_dim=1)
    grad_b_flat = grad_b.flatten(start_dim=1)
    
    # 2. 这种写法(norm + eps)天然抑制了小梯度的噪声
    norm_a = grad_a_flat.norm(dim=1, keepdim=True)
    norm_b = grad_b_flat.norm(dim=1, keepdim=True)
    
    # 计算单位向量
    unit_a = grad_a_flat / (norm_a + epsilon)
    unit_b = grad_b_flat / (norm_b + epsilon)
    
    # 3. 计算点积 (标准 Cosine)
    # 理论范围 [-1, 1]，但实际上由于 eps 的存在，模长 < 1，所以点积绝对值一定 < 1
    cos_sim = (unit_a * unit_b).sum(dim=-1) 
    
    # 4. === 关键一步：数值钳制 ===
    # 防止浮点误差导致 1.00000001 的情况
    cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)

    # 5. 线性映射到 [0, 1]
    # (x + 1) / 2
    normalized_score = (cos_sim + 1.0) / 2.0
    
    return normalized_score

import torch
import torch.nn.functional as F

# def compute_cosine_similarity(grad_a, grad_b, epsilon=1e-8):
#     """
#     计算归一化的余弦相似度。
    
#     Returns:
#         Tensor shape [Batch_size], 值域 [0, 1]
#         1.0: 方向完全一致 (No Conflict)
#         0.5: 方向正交
#         0.0: 方向完全相反 (Max Conflict)
        
#         如果两个梯度完全一致 (c_sim = 1.0):
#         loss = 1.0 - 1.0 = 0 (无惩罚，正确)
#         如果两个梯度完全相反 (c_sim = 0.0):
#         loss = 1.0 - 0.0 = 1.0 (最大惩罚，正确)
#         最后计算的 1-cos sim 在0.8左右
#     """
#     # 1. 拉平 (Flatten) [B, C, H, W] -> [B, -1]
#     grad_a_flat = grad_a.flatten(start_dim=1)
#     grad_b_flat = grad_b.flatten(start_dim=1)
    
#     # 2. 计算标准余弦相似度 [-1, 1]
#     # cos_sim = (A . B) / (|A| * |B|)
#     standard_cosine = F.cosine_similarity(grad_a_flat, grad_b_flat, dim=1, eps=epsilon)
    
#     # 3. 映射到 [0, 1]
#     # (x + 1) / 2
#     normalized_cosine = (standard_cosine + 1.0) / 2.0
    
#     return normalized_cosine

def flowgrad_optimization_conflict_multiprompt(z0, u_ind, dynamic, generate_traj, L_N_list, N=100,
                                                number_of_iterations=15, straightness_threshold=None,
                                                lr=2.5, weight_decay=0.995, conflict_threshold=0.5,
                                                conflict_weight=0.0, vis_dir=None, original_img=None,
                                                use_true_landscape=False,  # conflict_weight > 0 to enable minimization; use_true_landscape for Loss Landscape
                                                use_L_best=True):  # True: 返回 L_best 对应的控制变量; False: 返回最后一步
    """
    Revised Conflict-based Optimization:
    1. Minimizes the conflict score explicitly by adding grad(conflict) to the adjoint state.
    2. Still uses thresholding to decide update frequency (optional, kept based on original logic).
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    num_prompts = len(L_N_list)
    
    print(f"\n=== Corrected Conflict-based Multi-Prompt Optimization ===")
    print(f"Conflict weight: {conflict_weight} (Gradient penalty added to lam)")
    
    # Initialize u
    u_list = []
    for prompt_idx in range(num_prompts):
        u = {}
        for ind in u_ind:
            u[ind] = torch.zeros_like(z0).to(z0.device)
            u[ind].requires_grad = True # Important for optimizer if used, or manual update
        u_list.append(u)
    
    L_best_list = [float('inf')] * num_prompts # Initialize with inf for minimization logic (or 0 if maximizing score)
    # L_best checkpointing: 用于保存最优控制变量（按 curr_total_metric 最小）
    L_best = float('inf')
    best_u_list = []
    for p_idx in range(num_prompts):
        best_u = {}
        for ind in u_ind:
            best_u[ind] = torch.zeros_like(z0).to(z0.device)
        best_u_list.append(best_u)
    # Assuming L_N returns a loss to MINIMIZE based on context of 'gradient descent' usage below
    # If L_N returns a score to MAXIMIZE (like CLIP score), logic needs to invert.
    # Code below assumes L_N is a LOSS (e.g., CLIP distance).

    eps = 1e-3
    history_z0, history_grad_gcov, history_grad_res = [], [], []
    history_loss_values = []

    for i in range(number_of_iterations):
        # ====== Step 1: Forward Pass ======
        u_total = {}
        for ind in u_ind:
            # Sum control variables from all prompts
            u_stack = torch.stack([u_list[p][ind] for p in range(num_prompts)])
            u_total[ind] = torch.sum(u_stack, dim=0)
        
        # generate_traj 当 straightness_threshold is None 时只返回 traj，否则返回 (traj, non_uniform_set)
        result = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
        if straightness_threshold is not None:
            z_traj, non_uniform_set = result
        else:
            z_traj = result
            non_uniform_set = None

        # 在进入循环前或在循环内部包裹上下文管理器
        # 启用 Math Attention, 虽然慢一点，但支持二阶求导
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        # ====== Step 2: Terminal Loss & Conflict Minimization ======
        # We need to compute gradients of the loss w.r.t x1, AND gradients of the conflict w.r.t x1
        
            inputs = z_traj[-1].detach().clone()
            inputs.requires_grad = True
            
            # 1. Calculate individual gradients (nabla_x L_i)
            grads_x1 = []
            losses = []
            
            for prompt_idx in range(num_prompts):
                # 注意：这里必须开启 create_graph=True，否则无法进行二阶求导
                loss = L_N_list[prompt_idx](inputs) 
                g = torch.autograd.grad(loss, inputs, create_graph=True)[0] 
                grads_x1.append(g)
                losses.append(loss.item())

            # 2. Compute Conflict Loss and add its gradient to lambda
            # Conflict Loss = Sum of (1 - CosSim(g_i, g_j))
            total_conflict_grad = torch.zeros_like(inputs)
            avg_conflict_score = 0
            
            if conflict_weight > 0 and num_prompts > 1:
                conflict_loss = torch.tensor(0.0, device=device)
                pair_count = 0
                for p1 in range(num_prompts):
                    for p2 in range(p1 + 1, num_prompts):
                        # Cosine similarity between the gradients at terminal state
                        # Note: We are differentiating through the gradient! (Double backward)
                        c_sim = compute_cosine_similarity(grads_x1[p1], grads_x1[p2])
                        conflict_loss = conflict_loss + (1.0 - c_sim).sum()
                        pair_count += 1
                
                if pair_count > 0:
                    conflict_loss = conflict_loss / pair_count
                    avg_conflict_score = conflict_loss.item()
                    # Gradient of conflict w.r.t inputs (x1)
                    # This tells us: "How should x1 change to make the gradients of prompts more aligned?"
                    total_conflict_grad = torch.autograd.grad(conflict_loss, inputs, retain_graph=True)[0]
                    
                    # === 可视化冲突热力图 ===
                    # 仅在特定迭代次数保存，避免拖慢速度
                    if vis_dir is not None and (i % 5 == 0 or i == number_of_iterations - 1):
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # 在第一次迭代时保存原始图像
                        if i == 0 and original_img is not None:
                            save_img(original_img, path=os.path.join(vis_dir, 'original_image.png'))
                        
                        # 可视化 Prompt 0 和 Prompt 1 的冲突（如果有多个 Prompt）
                        if num_prompts >= 2:
                            # 可视化单纯的方向冲突
                            conflict_map_np = visualize_spatial_conflict(
                                grads_x1[0], grads_x1[1],
                                save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_conflict_map.png'),
                                title=f'Conflict Map (Iter {i+1}, Score: {avg_conflict_score:.4f})'
                            )
                            
                            # 可视化加权冲突
                            weighted_conflict_np = visualize_weighted_conflict(
                                grads_x1[0], grads_x1[1],
                                save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_weighted_conflict_map.png'),
                                title=f'Weighted Conflict Map (Iter {i+1})'
                            )
                            
                            # 如果提供了原始图像，生成叠加图像
                            if original_img is not None:
                                # 方向冲突热力图叠加
                                overlay_heatmap_on_image(
                                    original_img_tensor=original_img,
                                    heatmap_np=conflict_map_np,
                                    alpha=0.6,
                                    colormap_name='turbo',
                                    save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_conflict_overlay.png')
                                )
                                # 加权冲突热力图叠加
                                overlay_heatmap_on_image(
                                    original_img_tensor=original_img,
                                    heatmap_np=weighted_conflict_np,
                                    alpha=0.6,
                                    colormap_name='inferno',
                                    save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_weighted_overlay.png')
                                )
            # === [NEW] L_best Checkpointing Logic ===
            # 计算当前的总指标：平均 Task Loss + 加权 Conflict Score
            curr_avg_loss = sum(losses) / len(losses)
            # 如果 conflict_weight=0, 则只看 Task Loss
            curr_total_metric = curr_avg_loss + (conflict_weight * avg_conflict_score)
            
            is_best = False
            if curr_total_metric < L_best:
                L_best = curr_total_metric
                is_best = True
                # Deep Copy 当前所有的控制变量到 best_u_list 中
                for p in range(num_prompts):
                    for k in u_ind:
                        best_u_list[p][k] = u_list[p][k].detach().clone()
            # ========================================

            # === plot_gradient_map: 两个 prompt 梯度之和的 total grad 热力图 ===
            if vis_dir is not None and (i % 5 == 0 or i == number_of_iterations - 1):
                os.makedirs(vis_dir, exist_ok=True)
                total_grad = sum(grads_x1)
                plot_gradient_map(
                    total_grad,
                    title=f'Focus of Total Gradient (All Prompts) — Iter {i+1}',
                    save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_grad_total_heatmap.png'),
                    size=256
                )

            # === 收集数据用于 PCA 可视化 ===
            if i % 2 == 0:
                with torch.no_grad():
                    current_zN = z_traj[-1].detach().cpu().flatten().numpy()
                    history_z0.append(current_zN)
                    g0 = grads_x1[0].detach().cpu().flatten().numpy()
                    history_grad_gcov.append(g0)
                    if num_prompts >= 2:
                        g_rest = sum(g.detach().cpu().flatten().numpy() for g in grads_x1[1:])
                        history_grad_res.append(g_rest)
                    else:
                        history_grad_res.append(np.zeros_like(g0))
                    # 收集 Loss 用于 Loss Landscape 可视化
                    cw = conflict_weight if conflict_weight is not None else 0.0
                    current_total_loss = sum(losses) / len(losses)
                    if cw > 0 and num_prompts > 1:
                        current_total_loss = current_total_loss + cw * avg_conflict_score
                    history_loss_values.append(current_total_loss)

            print(f'Iter {i+1}: Losses={[f"{l:.4f}" for l in losses]}, Conflict Score={avg_conflict_score:.4f}')

            # 3. Assemble final lambda for each prompt
            # The adjoint equation is: d_lambda/dt = - lambda * df/dx
            # Terminal condition: lambda(1) = dL/dx(1) + alpha * d(Conflict)/dx(1)
            lam_list = []
            for p_idx in range(num_prompts):
                # Base gradient + weighted conflict gradient
                # We detach here because we start the backward integration
                final_lam = grads_x1[p_idx] + conflict_weight * total_conflict_grad
                lam_list.append(final_lam.detach())

        # ====== Step 3: Backward Propagation (Adjoint Method) ======
        # This remains largely the same, but uses the modified lam_list
        grad_at_step = {j: {} for j in u_ind}
        
        for prompt_idx in range(num_prompts):
            lam = lam_list[prompt_idx]
            
            for j in range(N-1, -1, -1):
                if straightness_threshold is not None and j not in non_uniform_set['indices']:
                    continue
                
                # Standard FlowGrad VJP step (ensure all tensors on same device)
                curr_z = z_traj[j].detach().clone().to(device)
                curr_z.requires_grad = True
                t_val = (torch.ones((batch_size,), device=device) * j / N * (1.-eps) + eps) * 999
                
                if straightness_threshold is not None:
                    dt = non_uniform_set['length'][j] / N
                else:
                    dt = 1.0 / N
                dt = float(dt) if hasattr(dt, 'item') else dt
                u_j = u_total[j].detach().to(device)
                lam = lam.to(device)
                    
                func = lambda x: (x.reshape(shape) + u_j + \
                                 dynamic(x.reshape(shape) + u_j, t_val) * dt).view(-1)

                output, vjp = torch.autograd.functional.vjp(func, curr_z.view(-1), lam.view(-1))
                lam = vjp.reshape(shape).detach()
                
                if j in u_ind:
                    # Determine gradient for u[j]
                    # In FlowGrad OC, grad_u = lam (because u is additive)
                    # For maximization: u_new = u + lr * lam
                    # For minimization: u_new = u - lr * lam
                    # Assuming we want to MINIMIZE loss:
                    # Update direction should be negative gradient. 
                    # If the provided OC code used '+', it might have been maximizing a score.
                    grad_at_step[j][prompt_idx] = lam.clone()

        # Propagate gradients to skipped indices (when using non-uniform sampling)
        # grad_at_step is only populated for j in non_uniform_set['indices']; copy to all u_ind
        if straightness_threshold is not None:
            indices = non_uniform_set['indices']
            # Handle indices before first keyframe: [0, indices[0])
            if len(indices) > 0 and indices[0] > 0:
                start = indices[0]
                for k in range(0, start):
                    if k in u_ind:
                        for prompt_idx in range(num_prompts):
                            grad_at_step[k][prompt_idx] = grad_at_step[start][prompt_idx].clone()
            for jj in range(len(indices)):
                start = indices[jj]
                end = indices[jj + 1] if jj + 1 < len(indices) else N
                for k in range(start, end):
                    if k in u_ind and k != start:
                        for prompt_idx in range(num_prompts):
                            grad_at_step[k][prompt_idx] = grad_at_step[start][prompt_idx].clone()

        # ====== Step 4: Update Controls ======
        # Here we can still use the conflict threshold logic to decide *when* to update,
        # but the *direction* of the update now inherently includes conflict minimization info.
        
        for j in u_ind:
             for prompt_idx in range(num_prompts):
                # Simple Gradient Descent with Momentum/Decay
                # u[k+1] = u[k] * decay - lr * grad (for minimization)
                grad = grad_at_step[j][prompt_idx]
                
                # Update logic
                u_list[prompt_idx][j] = u_list[prompt_idx][j] * weight_decay - lr * grad

    # === PCA 可视化 (含 Loss Landscape，可选 True Sampling) ===
    if vis_dir is not None and num_prompts >= 2 and SKLEARN_AVAILABLE and len(history_z0) > 2:
        try:
            true_ctx = None
            if use_true_landscape:
                true_ctx = {
                    'L_N_list': L_N_list,
                    'latent_shape': tuple(z0.shape),
                    'device': device,
                    'resolution': 25,
                    'batch_size': 16,
                }
            visualize_pca_trajectory_with_landscape(
                history_z0, history_grad_gcov, history_grad_res, history_loss_values,
                save_path=os.path.join(vis_dir, 'pca_trajectory_landscape.png'),
                use_true_landscape=use_true_landscape,
                true_landscape_context=true_ctx
            )
        except Exception as e:
            print(f"PCA landscape failed ({e}), falling back to basic PCA")
            visualize_pca_trajectory(
                history_z0, history_grad_gcov, history_grad_res,
                save_path=os.path.join(vis_dir, 'pca_trajectory.png')
            )

    if use_L_best:
        return best_u_list
    return u_list


def dflow_edit(config, text_prompts, alpha, model_path, data_loader):
    clip_scores = []
    lpips_scores = []
    id_scores = []
    clip_scores_gd = []
    lpips_scores_gd = []
    id_scores_gd = []
    for batch in data_loader:
        images = batch[:,0,:,:,:]
        batch_size = images.shape[0]
        # Create data normalizer and its inverse
        scaler = datasets.get_data_scaler(config)
        inverse_scaler = datasets.get_data_inverse_scaler(config)

        # Initialize model
        score_model = mutils.create_model(config)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(model=score_model, ema=ema, step=0)

        state = restore_checkpoint(model_path, state, device=config.device)
        ema.copy_to(score_model.parameters())

        model_fn = mutils.get_model_fn(score_model, train=False)

        # Load the image to edit
        # img = get_img('demo/celeba.jpg')
        # images = img
        # print('o_img shape',img.shape)
        original_img = images  

        log_folder = os.path.join('output', 'figs')
        print('Images will be saved to:', log_folder)
        # if not os.path.exists(log_folder): os.makedirs(log_folder)
        save_img(original_img, path=os.path.join(log_folder, 'original.png'))

        # Get latent code of the image and save reconstruction
        for text_prompt in text_prompts:
            original_img = original_img.to(config.device)
            clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
            clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
            # id_loss = IDLoss(device=config.device)

            lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

            t_s = time.time()
            latent = embed_to_latent(model_fn, scaler(original_img))
            traj = generate_traj(model_fn, latent, N=100)

            # Edit according to text prompt
            print('optimization starts')
            z0_d = dflow_optimization_lbfgs(latent, model_fn, N=100, L_N=clip_loss_1.L_N,  max_iter=5, lr=1)

            traj_oc = generate_traj(model_fn, z0=z0_d, N=100)

            print('dif', (z0_d-latent).sum())

            save_img(inverse_scaler(traj_oc[-1]), path=os.path.join(log_folder, 'optimized_dflow.png'))

            clip_scores.append(clip_loss_1.L_N(traj_oc[-1]).detach().cpu().numpy().sum())
            lpips_scores.append(lpips_f(traj_oc[-1], traj[-1]).detach().cpu().numpy().mean())
            # id_scores.append(1. - id_loss(traj[-1], traj_oc[-1]).detach().cpu().numpy().mean())

            print('text prompt', text_prompt)

            print('total_clip_loss',sum(clip_scores)/len(clip_scores))
            print('total_lpips_f',sum(lpips_scores)/len(lpips_scores))
            print('total_id',sum(id_scores)/len(id_scores))
            print('num', len(clip_scores)/5)

    return sum(clip_scores)/len(clip_scores), sum(lpips_scores)/len(lpips_scores), sum(id_scores)/len(id_scores)#,sum(clip_scores_gd)/len(clip_scores_gd), sum(lpips_scores_gd)/len(lpips_scores_gd),sum(id_scores_gd)/len(id_scores_gd)

# define a context manager which can count time cost
from contextlib import contextmanager

@contextmanager
def timer(name):
   print('running', name, '...')
   start_time = time.time()
   yield
   elapsed_time = time.time() - start_time
   print(f'\t{name} costs: {elapsed_time:.4f} s')


def dflow_edit_single(config, text_prompt, alpha, model_path, image_path, output_folder='output'):
    image = get_img(image_path)  
    batch_size = 1

    ts = time.time()
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    model_fn = mutils.get_model_fn(score_model, train=False)

    log_folder = os.path.join(output_folder, 'figs')
    print('Images will be saved to:', log_folder)
    if not os.path.exists(log_folder): os.makedirs(log_folder)
    save_img(image, path=os.path.join(log_folder, 'original.png'))

    original_img = image.to(config.device)
    with timer('clip loss'):
        clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  
    with timer('clip loss 1'):
        clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
    with timer('lpips'):
        lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

    # TODO: this step is very slow, consider preprocessing all images
    with timer('embed'):
        latent = embed_to_latent(model_fn, scaler(original_img))
    # torch.save(latent, 'latent.pt')
    save_img(inverse_scaler(latent), path=os.path.join(log_folder, 'latent.png'))
    # latent = get_img(os.path.join(log_folder, 'latent.png')).to(config.device)

    N = 5
    with timer('generate traj'):
        traj = generate_traj(model_fn, latent, N=N)
    recover_image = inverse_scaler(traj[-1])
    save_img(recover_image, path=os.path.join(log_folder, f'recover_{N}.png'))
    # torch.save(traj, 'traj.pt')
    # traj = torch.load('traj.pt')
    # for x in traj:
    #    x.to(config.device)

    lr = 1
    lbfgs_max_iter = 20
    opt_max_step = 5
    # Edit according to text prompt
    with timer('optimization'):
        z1_d, z0_d = dflow_optimization_lbfgs(latent, model_fn, N=N, L_N=clip_loss_1.L_N,  max_iter=lbfgs_max_iter, max_step=opt_max_step, lr=lr)
    save_img(inverse_scaler(z0_d), path=os.path.join(log_folder, f'z0_d_{N}.png'))

    # with timer('generate traj with z0_d'):
    #   traj_oc = generate_traj(model_fn, z0=z0_d, N=N)

    print('dif', (z0_d-latent).sum())

    save_img(inverse_scaler(z1_d), path=os.path.join(log_folder, 'optimized_dflow.png'))

    clip_loss = clip_loss_1.L_N(z1_d).detach().cpu().numpy()
    lpips_score = lpips_f(z1_d, traj[-1]).detach().cpu().numpy()

    print('text prompt', text_prompt)

    print('clip loss', clip_loss)
    print('lpips score', lpips_score)
    print('total time', time.time() - ts)

def flowgrad_edit_batch(config, model_path, image_paths, text_prompt, output_dir, method='ocflow', alpha=0.7):
    """
    单Prompt图像编辑
    参数:
        config: 配置对象
        model_path: 模型checkpoint路径
        image_paths: 输入图像路径列表
        text_prompt: 文本提示
        output_dir: 输出目录
        method: 方法名称，用于文件命名 (默认'ocflow')
        alpha: CLIP loss权重参数 (默认0.7)
    """
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    model_fn = mutils.get_model_fn(score_model, train=False)

    N = 100
    batch_size = 1
    metrics = {}  
    for img_path in tqdm(image_paths):
        target_dir = f'examples/{output_dir}'
        if img_path.startswith('examples/original'):
            opt_img_path = img_path.replace('examples/original', target_dir)
        else:
            import os
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            os.makedirs(target_dir, exist_ok=True)
            opt_img_path = os.path.join(target_dir, f'{name}_{method}{ext}')

        # Load the image to edit
        image = get_img(img_path)
        original_img = image.to(config.device)
        clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)

        t_s = time.time()
        latent = embed_to_latent(model_fn, scaler(original_img))
        traj = generate_traj(model_fn, latent, N=N)

        # Edit according to text prompt
        print(f'optimization starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        u_opt = flowgrad_optimization_oc_d(
            latent, u_ind, model_fn, generate_traj, L_N=clip_loss.L_N, N=N,
            number_of_iterations=15, lr=2.5, straightness_threshold=None
        )

        traj_oc = generate_traj(model_fn, z0=latent, u=u_opt, N=N)
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

        with torch.no_grad():
            clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
        # id_loss = IDLoss(device=config.device)

        lpips_f = lpips.LPIPS(net='alex').to(config.device)

        clip_loss_val = clip_loss_1.L_N(traj_oc[-1]).item()
        lpips_score = lpips_f(traj_oc[-1], traj[-1]).item()
        # id_loss = 1. - id_loss(traj[-1], traj_oc[-1]).detach().cpu().numpy()
        print(f'clip loss: {clip_loss_val:.4f}, lpips score: {lpips_score:.4f}, total time: {time.time() - t_s:.4f} s')

        metrics[opt_img_path] = {
            'clip_loss': clip_loss_val,
            'lpips_score': lpips_score,
            'method': method,
        }

    os.makedirs(target_dir, exist_ok=True)
    torch.save(metrics, f'{target_dir}/metrics_{method}.pt')
    return metrics

def flowgrad_edit_batch_multiprompt(config, model_path, image_paths, text_prompts, output_dir, method='multiprompt', alpha=0.7,
                                    use_L_best=True):
    """
    多Prompt图像编辑：同时将图像align到多个text prompt
    参数:
        config: 配置对象
        model_path: 模型checkpoint路径
        image_paths: 输入图像路径列表
        text_prompts: 文本提示列表，如 ['A photo of a smiling face.', 'A photo of a young face.']
        output_dir: 输出目录
        method: 方法名称，用于文件命名 (默认'multiprompt')
        alpha: CLIP loss权重参数 (默认0.7)
    返回:
        metrics: 评估指标字典
    """
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    model_fn = mutils.get_model_fn(score_model, train=False)

    N = 100
    batch_size = 1
    num_prompts = len(text_prompts)
    metrics = {}

    print(f"Multi-prompt editing with {num_prompts} prompts (method: {method}):")
    for idx, prompt in enumerate(text_prompts):
        print(f"  Prompt {idx+1}: {prompt}")

    lpips_f = lpips.LPIPS(net='alex').to(config.device)

    for img_path in tqdm(image_paths):
        target_dir = f'examples/{output_dir}'
        if img_path.startswith('examples/original'):
            opt_img_path = img_path.replace('examples/original', target_dir)
        else:
            import os
            filename = os.path.basename(img_path)
            os.makedirs(target_dir, exist_ok=True)
            name, ext = os.path.splitext(filename)
            opt_img_path = os.path.join(target_dir, f'{name}_{method}{ext}')

        # Load the image to edit
        image = get_img(img_path)
        original_img = image.to(config.device)

        # 为每个prompt创建CLIP loss函数
        clip_loss_list = []
        for prompt in text_prompts:
            clip_loss = clip_semantic_loss(prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)
            clip_loss_list.append(clip_loss)

        t_s = time.time()
        latent = embed_to_latent(model_fn, scaler(original_img))
        traj = generate_traj(model_fn, latent, N=N)  # 这是基准轨迹

        # 多Prompt优化
        print(f'\nMulti-prompt optimization starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        L_N_list = [clip_loss.L_N for clip_loss in clip_loss_list]
        opt_u_list = flowgrad_optimization_oc_d_multiprompt(
            latent, u_ind, model_fn, generate_traj, L_N_list=L_N_list,
            N=N, number_of_iterations=15, lr=2.5, straightness_threshold=None,
            use_L_best=use_L_best
        )

        u_total = {}
        for ind in u_ind:
            u_total[ind] = sum(opt_u_list[prompt_idx][ind] for prompt_idx in range(num_prompts))

        traj_oc = generate_traj(model_fn, z0=latent, u=u_total, N=N)
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

        # 同时保存每个单独prompt的结果（用于对比）
        for prompt_idx in range(num_prompts):
            traj_single = generate_traj(model_fn, z0=latent, u=opt_u_list[prompt_idx], N=N)
            name, ext = os.path.splitext(opt_img_path)
            single_path = f'{name}_prompt{prompt_idx+1}{ext}'
            save_img(inverse_scaler(traj_single[-1]), path=single_path)

        # 计算评估指标
        with torch.no_grad():
            clip_losses = []
            for prompt_idx in range(num_prompts):
                clip_loss_eval = clip_semantic_loss(text_prompts[prompt_idx], original_img, config.device,
                                                    alpha=1., inverse_scaler=inverse_scaler)
                clip_loss_val = clip_loss_eval.L_N(traj_oc[-1].to(config.device)).item()
                clip_losses.append(clip_loss_val)

        img_recon = inverse_scaler(traj[-1]).clamp(0, 1)
        img_edit = inverse_scaler(traj_oc[-1]).clamp(0, 1)
        img_recon_norm = img_recon * 2 - 1
        img_edit_norm = img_edit * 2 - 1
        lpips_score = lpips_f(img_edit_norm, img_recon_norm).item()

        print(f'\n=== Results ===')
        for idx, (prompt, loss) in enumerate(zip(text_prompts, clip_losses)):
            print(f'Prompt {idx+1} "{prompt}": CLIP loss = {loss:.4f}')
        print(f'LPIPS score: {lpips_score:.4f}')
        print(f'Total time: {time.time() - t_s:.4f} s')

        metrics[opt_img_path] = {
            'clip_losses': clip_losses,
            'clip_loss_avg': sum(clip_losses) / len(clip_losses),
            'lpips_score': lpips_score,
            'prompts': text_prompts,
            'method': method,
        }

    os.makedirs(target_dir, exist_ok=True)
    torch.save(metrics, f'{target_dir}/metrics_{method}.pt')
    return metrics

def flowgrad_edit_batch_flowgrad_multiprompt(config, model_path, image_paths, text_prompts, output_dir,
                                             method='flowgrad_multiprompt', alpha=0.7, lr=0.01):
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
    metrics = {}

    print(f"\n{'='*60}")
    print(f"FlowGrad (Original SGD) Multi-prompt editing with {num_prompts} prompts:")
    print(f"Using PyTorch SGD optimizer with lr={lr}")
    print(f"{'='*60}\n")

    lpips_f = lpips.LPIPS(net='alex').to(config.device)

    for img_path in tqdm(image_paths):
        target_dir = f'examples/{output_dir}'
        if img_path.startswith('examples/original'):
            opt_img_path = img_path.replace('examples/original', target_dir)
        else:
            import os
            filename = os.path.basename(img_path)
            os.makedirs(target_dir, exist_ok=True)
            name, ext = os.path.splitext(filename)
            opt_img_path = os.path.join(target_dir, f'{name}_{method}{ext}')

        image = get_img(img_path)
        original_img = image.to(config.device)

        clip_loss_list = []
        for prompt in text_prompts:
            clip_loss = clip_semantic_loss(prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)
            clip_loss_list.append(clip_loss)

        t_s = time.time()
        latent = embed_to_latent(model_fn, scaler(original_img))
        traj = generate_traj(model_fn, latent, N=N)

        print(f'\nFlowGrad optimization starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        L_N_list = [clip_loss.L_N for clip_loss in clip_loss_list]
        opt_u_list = flowgrad_optimization_multiprompt(
            latent, u_ind, model_fn, generate_traj, L_N_list=L_N_list,
            N=N, number_of_iterations=15, lr=lr, straightness_threshold=None
        )

        u_total = {}
        for ind in u_ind:
            u_total[ind] = sum(opt_u_list[prompt_idx][ind] for prompt_idx in range(num_prompts)
                              if ind in opt_u_list[prompt_idx])
            if ind not in u_total:
                u_total[ind] = torch.zeros_like(latent).to(latent.device)

        traj_oc = generate_traj(model_fn, z0=latent, u=u_total, N=N)
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

        for prompt_idx in range(num_prompts):
            if opt_u_list[prompt_idx]:
                traj_single = generate_traj(model_fn, z0=latent, u=opt_u_list[prompt_idx], N=N)
            else:
                traj_single = [t.clone() for t in traj]

            name, ext = os.path.splitext(opt_img_path)
            single_path = f'{name}_prompt{prompt_idx+1}{ext}'
            save_img(inverse_scaler(traj_single[-1]), path=single_path)

        with torch.no_grad():
            clip_losses = []
            for prompt_idx in range(num_prompts):
                clip_loss_eval = clip_semantic_loss(text_prompts[prompt_idx], original_img, config.device,
                                                    alpha=1., inverse_scaler=inverse_scaler)
                clip_loss_val = clip_loss_eval.L_N(traj_oc[-1].to(config.device)).item()
                clip_losses.append(clip_loss_val)

        img_recon = inverse_scaler(traj[-1].to(config.device)).clamp(0, 1)
        img_edit = inverse_scaler(traj_oc[-1].to(config.device)).clamp(0, 1)
        img_recon_norm = img_recon * 2 - 1
        img_edit_norm = img_edit * 2 - 1
        lpips_score = lpips_f(img_edit_norm, img_recon_norm).item()

        print(f'\n=== Results ===')
        for idx, (prompt, loss) in enumerate(zip(text_prompts, clip_losses)):
            print(f'Prompt {idx+1} "{prompt}": CLIP loss = {loss:.4f}')
        print(f'LPIPS score: {lpips_score:.4f}')
        print(f'Total time: {time.time() - t_s:.4f} s')

        metrics[opt_img_path] = {
            'clip_losses': clip_losses,
            'clip_loss_avg': sum(clip_losses) / len(clip_losses),
            'lpips_score': lpips_score,
            'prompts': text_prompts,
            'method': method,
            'lr': lr,
        }

    os.makedirs(target_dir, exist_ok=True)
    torch.save(metrics, f'{target_dir}/metrics_{method}.pt')
    return metrics


def flowgrad_edit_batch_gcovA_multiprompt(config, model_path, image_paths, text_prompts, output_dir,
                                          method='gcovA_multiprompt', alpha=0.7, lr=2.5):
    """
    GcovA版本多Prompt图像编辑：直接使用-L_N的梯度作为控制方向
    核心思想：
        1. 计算每个prompt的-L_N梯度（反向传播到每个时间步）
        2. 直接使用这些梯度作为控制变量u
    参数:
        config, model_path, image_paths, text_prompts, output_dir, method, alpha, lr
    返回:
        metrics: 评估指标字典
    """
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    model_fn = mutils.get_model_fn(score_model, train=False)

    N = 100
    batch_size = 1
    num_prompts = len(text_prompts)
    metrics = {}

    print(f"\n{'='*60}")
    print(f"GcovA Multi-prompt (Accumulated Gradient Control) with {num_prompts} prompts:")
    print(f"Using accumulated -L_N gradient (no weight_decay), lr={lr}")
    for idx, prompt in enumerate(text_prompts):
        print(f"  Prompt {idx+1}: {prompt}")
    print(f"{'='*60}\n")

    lpips_f = lpips.LPIPS(net='alex').to(config.device)

    for img_path in tqdm(image_paths):
        target_dir = f'examples/{output_dir}'
        if img_path.startswith('examples/original'):
            opt_img_path = img_path.replace('examples/original', target_dir)
        else:
            import os
            filename = os.path.basename(img_path)
            os.makedirs(target_dir, exist_ok=True)
            name, ext = os.path.splitext(filename)
            opt_img_path = os.path.join(target_dir, f'{name}_{method}{ext}')

        image = get_img(img_path)
        original_img = image.to(config.device)

        clip_loss_list = []
        for prompt in text_prompts:
            clip_loss = clip_semantic_loss(prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)
            clip_loss_list.append(clip_loss)

        t_s = time.time()
        latent = embed_to_latent(model_fn, scaler(original_img))
        traj = generate_traj(model_fn, latent, N=N)

        print(f'\nGcovA (Accumulated Gradient Control) starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        L_N_list = [clip_loss.L_N for clip_loss in clip_loss_list]
        opt_u_list = flowgrad_optimization_gcovA_multiprompt(
            latent, u_ind, model_fn, generate_traj, L_N_list=L_N_list,
            N=N, number_of_iterations=15, straightness_threshold=None, lr=lr
        )

        u_total = {}
        for ind in u_ind:
            u_total[ind] = sum(opt_u_list[prompt_idx][ind] for prompt_idx in range(num_prompts)
                              if ind in opt_u_list[prompt_idx])
            if ind not in u_total or u_total[ind] is None:
                u_total[ind] = torch.zeros_like(latent).to(latent.device)

        traj_oc = generate_traj(model_fn, z0=latent, u=u_total, N=N)
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

        for prompt_idx in range(num_prompts):
            if opt_u_list[prompt_idx]:
                traj_single = generate_traj(model_fn, z0=latent, u=opt_u_list[prompt_idx], N=N)
            else:
                traj_single = traj
            name, ext = os.path.splitext(opt_img_path)
            single_path = f'{name}_prompt{prompt_idx+1}{ext}'
            save_img(inverse_scaler(traj_single[-1]), path=single_path)

        with torch.no_grad():
            clip_losses = []
            for prompt_idx in range(num_prompts):
                clip_loss_eval = clip_semantic_loss(text_prompts[prompt_idx], original_img, config.device,
                                                    alpha=1., inverse_scaler=inverse_scaler)
                clip_loss_val = clip_loss_eval.L_N(traj_oc[-1].to(config.device)).item()
                clip_losses.append(clip_loss_val)

        img_recon = inverse_scaler(traj[-1]).clamp(0, 1)
        img_edit = inverse_scaler(traj_oc[-1]).clamp(0, 1)
        img_recon_norm = img_recon * 2 - 1
        img_edit_norm = img_edit * 2 - 1
        lpips_score = lpips_f(img_edit_norm, img_recon_norm).item()

        print(f'\n=== Results (GcovA - Direct Gradient Control) ===')
        for idx, (prompt, loss) in enumerate(zip(text_prompts, clip_losses)):
            print(f'Prompt {idx+1} "{prompt}": CLIP loss = {loss:.4f}')
        print(f'LPIPS score: {lpips_score:.4f}')
        print(f'Total time: {time.time() - t_s:.4f} s')

        metrics[opt_img_path] = {
            'clip_losses': clip_losses,
            'clip_loss_avg': sum(clip_losses) / len(clip_losses),
            'lpips_score': lpips_score,
            'prompts': text_prompts,
            'method': method,
        }

    os.makedirs(target_dir, exist_ok=True)
    torch.save(metrics, f'{target_dir}/metrics_{method}.pt')
    return metrics

def flowgrad_edit_batch_conflict_multiprompt(config, model_path, image_paths, text_prompts, output_dir,
                                             method='conflict_multiprompt', alpha=0.7, conflict_threshold=0.5,
                                             conflict_weight=0.0, lr=2.5, use_true_landscape=False,
                                             use_L_best=True):
    """
    基于冲突检测的多Prompt图像编辑：只在prompt之间存在冲突时才更新控制变量
    参数:
        config, model_path, image_paths, text_prompts, output_dir, method, alpha,
        conflict_threshold: 冲突阈值 (默认0.5)，值越大需要更大分歧才认为是冲突
    返回:
        metrics: 评估指标字典
    """
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    model_fn = mutils.get_model_fn(score_model, train=False)

    N = 100
    batch_size = 1
    num_prompts = len(text_prompts)
    metrics = {}

    print(f"\n{'='*60}")
    print(f"Conflict-based Multi-prompt editing with {num_prompts} prompts:")
    print(f"Conflict threshold: {conflict_threshold}")
    for idx, prompt in enumerate(text_prompts):
        print(f"  Prompt {idx+1}: {prompt}")
    print(f"{'='*60}\n")

    lpips_f = lpips.LPIPS(net='alex').to(config.device)

    for img_path in tqdm(image_paths):
        target_dir = f'examples/{output_dir}'
        if img_path.startswith('examples/original'):
            opt_img_path = img_path.replace('examples/original', target_dir)
            base, ext = os.path.splitext(opt_img_path)
            opt_img_path = f'{base}_cw{conflict_weight}{ext}'
        else:
            import os
            filename = os.path.basename(img_path)
            os.makedirs(target_dir, exist_ok=True)
            name, ext = os.path.splitext(filename)
            opt_img_path = os.path.join(target_dir, f'{name}_{method}_cw{conflict_weight}{ext}')

        image = get_img(img_path)
        original_img = image.to(config.device)

        clip_loss_list = []
        for prompt in text_prompts:
            clip_loss = clip_semantic_loss(prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)
            clip_loss_list.append(clip_loss)

        t_s = time.time()
        latent = embed_to_latent(model_fn, scaler(original_img))
        traj = generate_traj(model_fn, latent, N=N)

        print(f'\nConflict-based optimization starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        L_N_list = [clip_loss.L_N for clip_loss in clip_loss_list]
        
        # 设置可视化目录：在输出目录下创建 conflict_maps_cw{conflict_weight} 子目录（始终启用，用于 plot_gradient_map、visualize_pca_trajectory 等）
        vis_dir = os.path.join(target_dir, f'conflict_maps_cw{conflict_weight}')
        
        opt_u_list = flowgrad_optimization_conflict_multiprompt(
            latent, u_ind, model_fn, generate_traj, L_N_list=L_N_list,
            N=N, number_of_iterations=15, lr=lr, straightness_threshold=None,
            conflict_threshold=conflict_threshold, conflict_weight=conflict_weight,
            vis_dir=vis_dir, original_img=original_img,
            use_true_landscape=use_true_landscape,
            use_L_best=use_L_best
        )

        u_total = {}
        for ind in u_ind:
            u_total[ind] = sum(opt_u_list[prompt_idx][ind] for prompt_idx in range(num_prompts))

        traj_oc = generate_traj(model_fn, z0=latent, u=u_total, N=N)
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

        for prompt_idx in range(num_prompts):
            traj_single = generate_traj(model_fn, z0=latent, u=opt_u_list[prompt_idx], N=N)
            name, ext = os.path.splitext(opt_img_path)
            single_path = f'{name}_prompt{prompt_idx+1}{ext}'
            save_img(inverse_scaler(traj_single[-1]), path=single_path)

        with torch.no_grad():
            clip_losses = []
            for prompt_idx in range(num_prompts):
                clip_loss_eval = clip_semantic_loss(text_prompts[prompt_idx], original_img, config.device,
                                                    alpha=1., inverse_scaler=inverse_scaler)
                clip_loss_val = clip_loss_eval.L_N(traj_oc[-1].to(config.device)).item()
                clip_losses.append(clip_loss_val)

        img_recon = inverse_scaler(traj[-1]).clamp(0, 1)
        img_edit = inverse_scaler(traj_oc[-1]).clamp(0, 1)
        img_recon_norm = img_recon * 2 - 1
        img_edit_norm = img_edit * 2 - 1
        lpips_score = lpips_f(img_edit_norm, img_recon_norm).item()

        print(f'\n=== Results ===')
        for idx, (prompt, loss) in enumerate(zip(text_prompts, clip_losses)):
            print(f'Prompt {idx+1} "{prompt}": CLIP loss = {loss:.4f}')
        print(f'LPIPS score: {lpips_score:.4f}')
        print(f'Total time: {time.time() - t_s:.4f} s')

        metrics[opt_img_path] = {
            'clip_losses': clip_losses,
            'clip_loss_avg': sum(clip_losses) / len(clip_losses),
            'lpips_score': lpips_score,
            'prompts': text_prompts,
            'method': method,
            'conflict_threshold': conflict_threshold,
        }

    os.makedirs(target_dir, exist_ok=True)
    torch.save(metrics, f'{target_dir}/metrics_{method}_cw{conflict_weight}.pt')
    return metrics


def flowgrad_optimization_hybrid_multiprompt(z0, u_ind, dynamic, generate_traj, L_N_list, 
                                             N=100, number_of_iterations=15, 
                                             straightness_threshold=None, 
                                             lr_gcov=2.5, lr_res=2.5, 
                                             weight_decay=0.995):
    """
    Hybrid FlowGrad Optimization: gcovA (Greedy) + Residual OC (Global)
    
    控制变量由两部分组成: u_total = u_gcovA + u_res
    1. u_gcovA: 使用 Lookahead Gradient (∇x_t Loss) 更新
    2. u_res:   使用 Adjoint Method (OC) 更新，作为残差项修补 gcovA 忽略的全局动力学
    
    参数:
        lr_gcov: gcovA 部分的学习率 (通常较小，如 1.0)
        lr_res:  residual OC 部分的学习率 (通常较大，如 2.5)
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    num_prompts = len(L_N_list)
    eps = 1e-3
    
    # ==========================================
    # 1. 初始化两组控制变量
    # ==========================================
    # Group A: gcovA (Greedy/Lookahead)
    opt_u_gcov = [{} for _ in range(num_prompts)]
    for p_idx in range(num_prompts):
        for ind in u_ind:
            opt_u_gcov[p_idx][ind] = torch.zeros_like(z0).to(device)

    # Group B: Residual (Optimal Control / Adjoint)
    opt_u_res = [{} for _ in range(num_prompts)]
    grad_res_buffer = [{} for _ in range(num_prompts)] # 缓存 OC 的梯度
    for p_idx in range(num_prompts):
        for ind in u_ind:
            opt_u_res[p_idx][ind] = torch.zeros_like(z0).to(device)
            # u_res 需要梯度信息用于 SGD/Momentum 更新 (这里简化为手动更新)
            opt_u_res[p_idx][ind].requires_grad = True 

    print(f"\n=== Hybrid Optimization: gcovA (LR={lr_gcov}) + Residual OC (LR={lr_res}) ===")
    
    for i in range(number_of_iterations):
        # ==========================================
        # 2. 组合总控制变量 u_total
        # ==========================================
        u_total = {}
        for ind in u_ind:
            # Sum gcovA parts
            sum_gcov = sum(opt_u_gcov[p][ind] for p in range(num_prompts))
            # Sum Residual parts
            sum_res = sum(opt_u_res[p][ind] for p in range(num_prompts))
            
            u_total[ind] = (sum_gcov + sum_res).to(device)

        # ==========================================
        # 3. 生成轨迹 (Base Trajectory)
        # ==========================================
        # 注意：我们需要轨迹点 x_t 来计算两套梯度
        # generate_traj 当 straightness_threshold is None 时只返回 traj，否则返回 (traj, non_uniform_set)
        with torch.no_grad():
            result = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
            if straightness_threshold is not None:
                z_traj, non_uniform_set = result
            else:
                z_traj = result
                non_uniform_set = None
        
        # 临时关闭 loss 的 verbose
        for p_idx in range(num_prompts):
            if hasattr(L_N_list[p_idx], '__self__'):
                L_N_list[p_idx].__self__.verbose = False

        # ==========================================
        # 4. 计算 gcovA 梯度 (Lookahead)
        #    并立即更新 u_gcov
        # ==========================================
        # 这一步不需要反向传播整个轨迹，只需要局部计算
        for j in range(N):
            if straightness_threshold is not None and j not in non_uniform_set['indices']:
                continue
            
            t_norm = j / N
            t_val = (torch.ones((batch_size,), device=device) * t_norm * (1.-eps) + eps) * 999
            x_t = z_traj[j].to(device).detach().clone()
            
            with torch.enable_grad():
                x_t.requires_grad_(True)
                # Lookahead Prediction
                v_t = dynamic(x_t, t_val)
                x1_pred = x_t + v_t * (1.0 - t_norm)
                
                for p_idx in range(num_prompts):
                    loss = L_N_list[p_idx](x1_pred)
                    # 计算 ∇x_t Loss
                    is_last = (p_idx == num_prompts - 1)
                    grads = torch.autograd.grad(loss, x_t, retain_graph=not is_last)[0]
                    
                    # Update gcovA term
                    if j in u_ind:
                        # Update rule: decay + negative gradient
                        opt_u_gcov[p_idx][j] = opt_u_gcov[p_idx][j] * weight_decay + (-lr_gcov * grads.detach())

        # ==========================================
        # 5. 计算 Residual OC 梯度 (Adjoint Method)
        # ==========================================
        # A. 计算终点 Lambda (Terminal Condition)
        x_N = z_traj[-1].to(device).detach().clone()
        x_N.requires_grad = True
        
        lam_list = []
        loss_vals = []
        for p_idx in range(num_prompts):
            loss = L_N_list[p_idx](x_N)
            # 注意：我们要最小化 Loss，所以 lambda = +grad(Loss)
            # OC 中通常最大化 J，所以 lambda = grad(J)。如果 Loss 是越小越好，
            # 那么 lambda 指向 Loss 增加的方向。我们需要 u 往 lambda 相反方向走。
            lam = torch.autograd.grad(loss, x_N, retain_graph=(p_idx < num_prompts-1))[0]
            lam_list.append(lam.detach())
            loss_vals.append(loss.item())

        # B. 反向传播 Lambda (Adjoint Equation)
        # 为每个 Prompt 独立计算 Adjoint
        for p_idx in range(num_prompts):
            lam = lam_list[p_idx]
            
            for j in range(N-1, -1, -1):
                if straightness_threshold is not None and j not in non_uniform_set['indices']:
                    continue
                
                # 准备 VJP 上下文
                curr_z = z_traj[j].to(device).detach().clone()
                curr_z.requires_grad = True
                t_val = (torch.ones((batch_size,), device=device) * j / N * (1.-eps) + eps) * 999
                
                if straightness_threshold is not None:
                    dt = non_uniform_set['length'][j] / N
                else:
                    dt = 1.0 / N
                
                u_total_j = u_total[j].detach().clone() # 使用当前的总 u
                
                # 定义局部动力学: x_{j+1} approx x_j + u_total + f(...) * dt
                func = lambda x: (x.contiguous().reshape(shape) + u_total_j + \
                                  dynamic(x.contiguous().reshape(shape) + u_total_j, t_val) * dt).view(-1)
                
                # VJP 计算 lambda_j
                output, vjp = torch.autograd.functional.vjp(func, curr_z.view(-1), lam.view(-1))
                lam = vjp.reshape(shape).detach()
                
                # 保存梯度 for u_res
                # Optimal Control 理论: dH/du = lambda. 
                # 我们要 Minimize Loss => u 应沿 -lambda 方向更新
                if j in u_ind:
                    grad_res_buffer[p_idx][j] = lam.clone()
                    
                del curr_z
                if j == 0: break
                
            # 梯度重分配 (Gradient Re-assignment)
            if straightness_threshold is not None:
                indices = non_uniform_set['indices']
                for jj in range(len(indices)):
                    start = indices[jj]
                    end = indices[jj+1] if jj+1 < len(indices) else N
                    for k in range(start, end):
                        if k in u_ind:
                            grad_res_buffer[p_idx][k] = grad_res_buffer[p_idx][start].clone()

        # ==========================================
        # 6. 更新 u_res
        # ==========================================
        for p_idx in range(num_prompts):
            for ind in u_ind:
                grad = grad_res_buffer[p_idx][ind]
                # Update rule: decay + negative gradient (minimization)
                # 注意：grad 是 dL/du，所以我们要减去它
                opt_u_res[p_idx][ind] = opt_u_res[p_idx][ind] * weight_decay - lr_res * grad

        # 打印进度
        # 恢复 verbose
        for p_idx in range(num_prompts):
            if hasattr(L_N_list[p_idx], '__self__'):
                L_N_list[p_idx].__self__.verbose = False
                
        loss_str = ", ".join([f"L{idx}:{l:.4f}" for idx, l in enumerate(loss_vals)])
        print(f"Iter {i+1}: {loss_str}")

    return opt_u_gcov, opt_u_res

#-----------------------------------------------
def flowgrad_optimization_hybrid_conflict(z0, u_ind, dynamic, generate_traj, L_N_list, 
                                          N=100, number_of_iterations=15, 
                                          straightness_threshold=None, 
                                          lr_gcov=1.0, lr_res=2.5, 
                                          weight_decay=0.995,
                                          conflict_weight=0.1,
                                          vis_dir=None,
                                          original_img=None,
                                          use_true_landscape=False,  # True: 真实采样 Loss 地形 (慢但准确)
                                          use_L_best=True):  # True: 返回 L_best 对应的控制变量; False: 返回最后一步
    """
    Hybrid FlowGrad Optimization with Conflict Awareness
    
    机制:
    1. u_gcovA: 贪婪地优化每个 Prompt 的 CLIP Loss (快速语义对齐)
    2. u_res:   通过 Adjoint Method 优化 (CLIP Loss + Conflict Score)
                -> 既修补全局动力学，又负责拉齐不同 Prompt 的梯度方向
    """
    device = z0.device
    shape = z0.shape
    batch_size = shape[0]
    num_prompts = len(L_N_list)
    eps = 1e-3
    
    # 初始化两组控制变量
    opt_u_gcov = [{} for _ in range(num_prompts)]
    opt_u_res = [{} for _ in range(num_prompts)]
    grad_res_buffer = [{} for _ in range(num_prompts)]
    
    for p_idx in range(num_prompts):
        for ind in u_ind:
            opt_u_gcov[p_idx][ind] = torch.zeros_like(z0).to(device)
            opt_u_res[p_idx][ind] = torch.zeros_like(z0).to(device)
            opt_u_res[p_idx][ind].requires_grad = True 

    print(f"\n=== Hybrid Conflict Opt: gcovA(LR={lr_gcov}) + Res(LR={lr_res}, Conflict={conflict_weight}) ===")
    
    # L_best checkpointing: 用于保存最优控制变量（按 curr_total_metric 最小）
    L_best = float('inf')
    best_u_gcov = [{} for _ in range(num_prompts)]
    best_u_res = [{} for _ in range(num_prompts)]
    for p_idx in range(num_prompts):
        for ind in u_ind:
            best_u_gcov[p_idx][ind] = torch.zeros_like(z0).to(device)
            best_u_res[p_idx][ind] = torch.zeros_like(z0).to(device)
    
    # 初始化数据收集列表（用于PCA可视化）
    history_z0 = []  # 存储每次迭代的 z0
    history_grad_gcov = []  # 存储 gcovA 的总梯度
    history_grad_res = []  # 存储 residual OC 的总梯度
    history_loss_values = []  # 存储每步的 CLIP Loss (用于 Loss Landscape 可视化)
    # 梯度夹角记录（gcovA vs OC，用于几何角度可视化）
    history_angle_gcov_oc = []  # 单位：度
    
    for i in range(number_of_iterations):
        # 1. 组合总控制变量 u_total
        u_total = {}
        for ind in u_ind:
            sum_gcov = sum(opt_u_gcov[p][ind] for p in range(num_prompts))
            sum_res = sum(opt_u_res[p][ind] for p in range(num_prompts))
            u_total[ind] = (sum_gcov + sum_res).to(device)

        # 2. 生成轨迹 (Base Trajectory)
        # generate_traj 当 straightness_threshold is None 时只返回 traj，否则返回 (traj, non_uniform_set)
        with torch.no_grad():
            result = generate_traj(dynamic, z0, u=u_total, N=N, straightness_threshold=straightness_threshold)
            if straightness_threshold is not None:
                z_traj, non_uniform_set = result
            else:
                z_traj = result
                non_uniform_set = None

        for p_idx in range(num_prompts):
            if hasattr(L_N_list[p_idx], '__self__'): L_N_list[p_idx].__self__.verbose = False

        # ==========================================
        # ==========================================
        # 3. 计算 gcovA 梯度 
        # ==========================================
        # 收集 gcovA 梯度用于PCA可视化（聚合所有时间步和prompt的梯度）
        grad_gcov_list = []
        
        # 确定循环的索引集合
        if straightness_threshold is not None:
            loop_indices = non_uniform_set['indices']
        else:
            loop_indices = range(N)

        for j in loop_indices:
            # 确定当前关键帧覆盖的时间范围 [start, end)
            start = j
            if straightness_threshold is not None:
                length = non_uniform_set['length'][start]
                end = start + length
            else:
                end = start + 1
            
            t_norm = start / N
            t_val = (torch.ones((batch_size,), device=device) * t_norm * (1.-eps) + eps) * 999
            x_t = z_traj[start].to(device).detach().clone()
            
            with torch.enable_grad():
                x_t.requires_grad_(True)
                v_t = dynamic(x_t, t_val)
                x1_pred = x_t + v_t * (1.0 - t_norm)
                
                for p_idx in range(num_prompts):
                    loss = L_N_list[p_idx](x1_pred)
                    is_last = (p_idx == num_prompts - 1)
                    grads = torch.autograd.grad(loss, x_t, retain_graph=not is_last)[0]
                    current_grad = grads.detach()
                    
                    # [关键修复]: 将梯度应用到该时间段内的所有点
                    for k in range(start, end):
                        if k in u_ind:
                            # 累积更新：u = u * decay - lr * grad
                            opt_u_gcov[p_idx][k] = opt_u_gcov[p_idx][k] * weight_decay + (-lr_gcov * current_grad)
                    
                    # 仅收集一次梯度用于可视化
                    if j == loop_indices[0]: 
                         grad_gcov_list.append(current_grad)
        
        if grad_gcov_list:
            grad_gcov_total = sum(grad_gcov_list) / len(grad_gcov_list)
        else:
            grad_gcov_total = torch.zeros_like(z0)

        # ==========================================
        # 4. 计算 Conflict-Aware Residual 梯度
        # ==========================================
        
        # 强制使用 Math Attention 以支持二阶导数 (Conflict 梯度需要)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x_N = z_traj[-1].to(device).detach().clone()
            x_N.requires_grad = True
            
            # A. 计算每个 Prompt 的基础梯度
            grads_xN = []
            loss_vals = []
            for p_idx in range(num_prompts):
                loss = L_N_list[p_idx](x_N)
                # create_graph=True 是必须的，因为我们要对梯度求导
                g = torch.autograd.grad(loss, x_N, create_graph=True)[0]
                grads_xN.append(g)
                loss_vals.append(loss.item())

            # B. 计算 Conflict Gradient
            total_conflict_grad = torch.zeros_like(x_N)
            avg_conflict_score = 0.0
            
            if conflict_weight > 0 and num_prompts > 1:
                conflict_loss = torch.tensor(0.0, device=device)
                pair_count = 0
                for p1 in range(num_prompts):
                    for p2 in range(p1 + 1, num_prompts):
                        # compute_cosine_similarity 需要你之前定义的那个函数
                        c_sim = compute_cosine_similarity(grads_xN[p1], grads_xN[p2])
                        conflict_loss = conflict_loss + (1.0 - c_sim).sum()
                        pair_count += 1
                
                if pair_count > 0:
                    conflict_loss = conflict_loss / pair_count
                    avg_conflict_score = conflict_loss.item()
                    # 计算 Conflict 对 x_N 的梯度
                    total_conflict_grad = torch.autograd.grad(conflict_loss, x_N, retain_graph=True)[0]
                    
                    # === 可视化冲突热力图 ===
                    # 仅在特定迭代次数保存，避免拖慢速度
                    if vis_dir is not None and (i % 5 == 0 or i == number_of_iterations - 1):
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # 在第一次迭代时保存原始图像
                        if i == 0 and original_img is not None:
                            save_img(original_img, path=os.path.join(vis_dir, 'original_image.png'))
                        
                        # 可视化 Prompt 0 和 Prompt 1 的冲突（如果有多个 Prompt）
                        if num_prompts >= 2:
                            # grads_xN[p_idx] 的形状是 [batch_size, C, H, W]，直接传入即可
                            # 可视化单纯的方向冲突
                            conflict_map_np = visualize_spatial_conflict(
                                grads_xN[0], grads_xN[1],
                                save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_conflict_map.png'),
                                title=f'Conflict Map (Iter {i+1}, Score: {avg_conflict_score:.4f})'
                            )
                            
                            # 可视化加权冲突
                            weighted_conflict_np = visualize_weighted_conflict(
                                grads_xN[0], grads_xN[1],
                                save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_weighted_conflict_map.png'),
                                title=f'Weighted Conflict Map (Iter {i+1})'
                            )
                            
                            # 如果提供了原始图像，生成叠加图像
                            if original_img is not None:
                                # 方向冲突热力图叠加
                                overlay_heatmap_on_image(
                                    original_img_tensor=original_img,
                                    heatmap_np=conflict_map_np,
                                    alpha=0.6,
                                    colormap_name='turbo',
                                    save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_conflict_overlay.png')
                                )
                                # 加权冲突热力图叠加
                                overlay_heatmap_on_image(
                                    original_img_tensor=original_img,
                                    heatmap_np=weighted_conflict_np,
                                    alpha=0.6,
                                    colormap_name='inferno',
                                    save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_weighted_overlay.png')
                                )

            # === L_best Checkpointing ===
            curr_avg_loss = sum(loss_vals) / len(loss_vals)
            curr_total_metric = curr_avg_loss + (conflict_weight * avg_conflict_score)
            
            is_best = False
            if curr_total_metric < L_best:
                L_best = curr_total_metric
                is_best = True
                for p in range(num_prompts):
                    for k in u_ind:
                        best_u_gcov[p][k] = opt_u_gcov[p][k].detach().clone()
                        best_u_res[p][k] = opt_u_res[p][k].detach().clone()

            lam_list = []
            for p_idx in range(num_prompts):
                scale_factor = 1.0 / num_prompts if num_prompts > 0 else 1.0
                final_lam = grads_xN[p_idx] + (conflict_weight * total_conflict_grad) * scale_factor
                lam_list.append(final_lam.detach())

        # D. 反向传播 Lambda (Adjoint Method)
        for p_idx in range(num_prompts):
            lam = lam_list[p_idx]
            # 倒序遍历
            for j in range(N-1, -1, -1):
                # 如果启用了 straightness，只处理关键帧
                if straightness_threshold is not None and j not in non_uniform_set['indices']:
                    continue
                
                curr_z = z_traj[j].to(device).detach().clone()
                curr_z.requires_grad = True
                t_val = (torch.ones((batch_size,), device=device) * j / N * (1.-eps) + eps) * 999
                
                # 动态步长计算
                if straightness_threshold is not None: 
                    dt = non_uniform_set['length'][j] / N
                else: 
                    dt = 1.0 / N
                
                u_total_j = u_total[j].detach().clone()
                func = lambda x: (x.contiguous().reshape(shape) + u_total_j + \
                                  dynamic(x.contiguous().reshape(shape) + u_total_j, t_val) * dt).view(-1)
                
                output, vjp = torch.autograd.functional.vjp(func, curr_z.view(-1), lam.view(-1))
                lam = vjp.reshape(shape).detach()
                
                if j in u_ind:
                    grad_res_buffer[p_idx][j] = lam.clone()
                del curr_z
                if j == 0: break
                
            # [FIXED] 梯度重分配 (Step 4 的广播已经存在，这里保持原样即可)
            if straightness_threshold is not None:
                indices = non_uniform_set['indices']
                for jj in range(len(indices)):
                    start = indices[jj]
                    try: 
                        end = indices[jj+1]
                    except: 
                        end = N
                    for k in range(start, end):
                        if k in u_ind: 
                            grad_res_buffer[p_idx][k] = grad_res_buffer[p_idx][start].clone()

        # 5. 更新 u_res
        grad_res_list = []
        for p_idx in range(num_prompts):
            for ind in u_ind:
                grad = grad_res_buffer[p_idx][ind]
                opt_u_res[p_idx][ind] = opt_u_res[p_idx][ind] * weight_decay - lr_res * grad
                grad_res_list.append(grad)
        
        if grad_res_list:
            grad_res_total = sum(grad_res_list) / len(grad_res_list)
        else:
            grad_res_total = torch.zeros_like(z0)
        
        # 聚合 residual OC 梯度：对所有时间步和prompt求平均
        if grad_res_list:
            grad_res_total = sum(grad_res_list) / len(grad_res_list)
        else:
            grad_res_total = torch.zeros_like(z0)

        # === 几何角度：gcovA 与 OC 梯度的夹角（每步都记录）===
        with torch.no_grad():
            flat_gcov = grad_gcov_total.view(-1)
            flat_res = grad_res_total.view(-1)
            # 避免全零向量
            norm_g = flat_gcov.norm()
            norm_r = flat_res.norm()
            if norm_g > 1e-10 and norm_r > 1e-10:
                cosine_sim = F.cosine_similarity(
                    flat_gcov.unsqueeze(0), flat_res.unsqueeze(0), dim=1
                ).item()
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                angle_deg = torch.acos(torch.tensor(cosine_sim, device=device)).item() * (180.0 / 3.14159265)
            else:
                angle_deg = float('nan')
            history_angle_gcov_oc.append(angle_deg)
        # =====================================================

        # === 空间热力图：gcovA 与 OC 梯度的空间分布 ===
        if vis_dir is not None and (i % 5 == 0 or i == number_of_iterations - 1):
            os.makedirs(vis_dir, exist_ok=True)
            plot_gradient_map(
                grad_gcov_total,
                title=f'Focus of g-cov-A (Task) — Iter {i+1}',
                save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_grad_gcov_heatmap.png'),
                size=256
            )
            plot_gradient_map(
                grad_res_total,
                title=f'Focus of OC Term (Constraint) — Iter {i+1}',
                save_path=os.path.join(vis_dir, f'iter_{i+1:03d}_grad_oc_heatmap.png'),
                size=256
            )
        # ================================================

        # === 收集数据用于PCA可视化 ===
        if i % 2 == 0:  
            with torch.no_grad():
                # [FIX]: 收集最终生成的图像 Latent (x_N)，而不是初始 Latent z0
                current_xN = z_traj[-1].detach().cpu().flatten().numpy()
                history_z0.append(current_xN) 
                
                # 对于梯度，建议收集 x_N 处的梯度 (lambda)，这比 u 的平均梯度更有物理意义
                # 这里为了简单，我们取 Step 4 计算出的 grads_xN (主任务) 和 total_conflict_grad (约束)
                # 注意：需要把 tensor 从 GPU 拉回 CPU 并 flatten
                g_main_flat = grads_xN[0].detach().cpu().flatten().numpy() 
                g_res_flat = total_conflict_grad.detach().cpu().flatten().numpy()
                
                history_grad_gcov.append(g_main_flat)  # Main Gradient
                history_grad_res.append(g_res_flat)    # Conflict Gradient
                
                # 收集 Loss 值用于 Loss Landscape 可视化 (CLIP Loss 平均 + Conflict 项)
                cw = conflict_weight if conflict_weight is not None else 0.0
                current_total_loss = sum(loss_vals) / len(loss_vals)
                if cw > 0 and num_prompts > 1:
                    current_total_loss = current_total_loss + cw * avg_conflict_score
                history_loss_values.append(current_total_loss)
        # ============================

        # 恢复 verbose 并打印
        for p_idx in range(num_prompts):
            if hasattr(L_N_list[p_idx], '__self__'): L_N_list[p_idx].__self__.verbose = False
                
        loss_str = ", ".join([f"L{idx}:{l:.4f}" for idx, l in enumerate(loss_vals)])
        angle_str = f"{history_angle_gcov_oc[-1]:.1f}°" if not np.isnan(history_angle_gcov_oc[-1]) else "N/A"
        print(f"Iter {i+1}: {loss_str}, Conflict: {avg_conflict_score:.4f}, Angle(gcovA,OC): {angle_str}")
    
    # === PCA可视化 (含 Loss Landscape) ===
    if vis_dir is not None and SKLEARN_AVAILABLE and len(history_z0) > 2:
        try:
            true_ctx = None
            if use_true_landscape:
                true_ctx = {
                    'L_N_list': L_N_list,
                    'latent_shape': tuple(z0.shape),
                    'device': device,
                    'resolution': 25,
                    'batch_size': 16,
                }
            visualize_pca_trajectory_with_landscape(
                history_z0, history_grad_gcov, history_grad_res, history_loss_values,
                save_path=os.path.join(vis_dir, 'pca_trajectory_landscape.png'),
                use_true_landscape=use_true_landscape,
                true_landscape_context=true_ctx
            )
        except Exception as e:
            print(f"PCA landscape failed ({e}), falling back to basic PCA")
            visualize_pca_trajectory(
                history_z0, history_grad_gcov, history_grad_res,
                save_path=os.path.join(vis_dir, 'pca_trajectory.png')
            )
    # ================

    # === 梯度夹角曲线可视化 (gcovA vs OC) ===
    if vis_dir is not None and len(history_angle_gcov_oc) > 0:
        visualize_gradient_angle(history_angle_gcov_oc, save_path=os.path.join(vis_dir, 'gradient_angle.png'))
    # ========================================
    
    
    # use_L_best: 返回 L_best 对应的控制变量，否则返回最后一步
    if use_L_best:
        return best_u_gcov, best_u_res
    return opt_u_gcov, opt_u_res

def flowgrad_edit_batch_hybrid_multiprompt(config, model_path, image_paths, text_prompts, output_dir, 
                                           method='hybrid_multiprompt', alpha=0.7, 
                                           lr_gcov=1.0, lr_res=2.5, conflict_weight=None,
                                           use_true_landscape=False,
                                           use_L_best=True):
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
    metrics = {}
    lpips_f = lpips.LPIPS(net='alex').to(config.device)

    print(f"\n{'='*60}")
    print(f"Hybrid Multi-prompt (gcovA + Residual OC):")
    for idx, prompt in enumerate(text_prompts):
        print(f"  Prompt {idx+1}: {prompt}")
    print(f"{'='*60}\n")

    for img_path in tqdm(image_paths):
        # ... (路径处理代码) ...
        target_dir = f'examples/{output_dir}'
        if img_path.startswith('examples/original'):
            opt_img_path = img_path.replace('examples/original', target_dir)
            # 如果传入了 conflict_weight，在文件名中加入 _cw{conflict_weight}
            if conflict_weight is not None:
                base, ext = os.path.splitext(opt_img_path)
                opt_img_path = f'{base}_cw{conflict_weight}{ext}'
        else:
            import os
            filename = os.path.basename(img_path)
            os.makedirs(target_dir, exist_ok=True)
            name, ext = os.path.splitext(filename)
            # 如果传入了 conflict_weight，在文件名中加入 _cw{conflict_weight}
            if conflict_weight is not None:
                opt_img_path = os.path.join(target_dir, f'{name}_{method}_cw{conflict_weight}{ext}')
            else:
                opt_img_path = os.path.join(target_dir, f'{name}_{method}{ext}')

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
        
        # # 调用混合优化器：若传入 conflict_weight 则使用 Conflict 版本，否则用普通 Hybrid
        # if conflict_weight is not None:
        #     # 设置可视化目录（含 conflict_weight=0 时也启用，用于 plot_gradient_map、visualize_pca_trajectory）
        #     vis_dir = os.path.join(target_dir, f'conflict_maps_cw{conflict_weight}')
        #     opt_u_gcov, opt_u_res = flowgrad_optimization_hybrid_conflict(
        #         latent, u_ind, model_fn, generate_traj, L_N_list=L_N_list,
        #         N=N, number_of_iterations=15, straightness_threshold=None,
        #         lr_gcov=lr_gcov, lr_res=lr_res, conflict_weight=conflict_weight,
        #         vis_dir=vis_dir, original_img=original_img,
        #         use_true_landscape=use_true_landscape,
        #         use_L_best=use_L_best
        #     )
        # else:
        #     opt_u_gcov, opt_u_res = flowgrad_optimization_hybrid_multiprompt(
        #         latent, u_ind, model_fn, generate_traj, L_N_list=L_N_list,
        #         N=N, number_of_iterations=15, straightness_threshold=None,
        #         lr_gcov=lr_gcov, lr_res=lr_res
        #     )
        
        # # 合并所有控制变量生成最终轨迹
        # u_total = {}
        # for ind in u_ind:
        #     sum_gcov = sum(opt_u_gcov[p][ind] for p in range(num_prompts))
        #     sum_res = sum(opt_u_res[p][ind] for p in range(num_prompts))
        #     u_total[ind] = sum_gcov + sum_res
        
        # traj_oc = generate_traj(model_fn, z0=latent, u=u_total, N=N)

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
            
            # # 保存每个 Prompt 的总贡献 (gcov + res)
            # for p_idx in range(num_prompts):
            #     u_p = {k: opt_u_gcov[p_idx][k] + opt_u_res[p_idx][k] for k in u_ind}
            #     traj_single = generate_traj(model_fn, z0=latent, u=u_p, N=N)
                
            #     name, ext = os.path.splitext(opt_img_path)
            #     single_path = f'{name}_prompt{p_idx+1}{ext}'
            #     save_img(inverse_scaler(traj_single[-1]), path=single_path)




        # # =========================================================
        # # 🔴 Debug1: 仿照 OC 范式，单独保存每个 Prompt 的独立引导结果
        # # =========================================================
        # if opt_img_path is not None:
        #     print("\n--- Generating Single Prompt Trajectories for Debugging ---")
        #     lr_gcov = 3400
        #     for p_idx in range(num_prompts):
        #         # 1. 为当前单一 Prompt 构造一个专属的引导场
        #         # 因为单 Prompt 没有冲突，所以直接关闭 learnable 和 conflict_weight
        #         single_guided_field = ImageGCovAGMOnlineGuidance(
        #             base_model=model_fn,
        #             loss_fns=[L_N_list[p_idx]],       # 🔴 关键：只传入当前这一个 CLIP Loss
        #             scales=[lr_gcov],      # 🔴 使用 lr_gcov 对齐 OC 强度
        #             config=config,
        #             learnable=False,                  # 调试单目标不需要训练残差网络，纯 gcovA 引导极快
        #             conflict_weight=0.0,              # 无冲突
        #             vis_dir=None
        #         )
                
        #         # 2. 用这单独的一份引导场，重新跑一条纯净轨迹 (起点依然用最干净的 y_0)
        #         traj_single = generate_traj(single_guided_field, z0=y_0, u=None, N=N)
                
        #         # 3. 拼接文件名并保存 (严格复刻原代码范式)
        #         name, ext = os.path.splitext(opt_img_path)
        #         single_path = f'{name}_prompt{p_idx+1}{ext}'
        #         save_img(inverse_scaler(traj_single[-1]), path=single_path)
        #         print(f"  -> Saved single effect for Prompt {p_idx+1}: {single_path}")

        # =====================================================================
        # 🔴 Debug2: 生成两个 Prompt 纯线性相加的图像 (No Residual Guidance)
        # =====================================================================
        # print("\n--- Generating Combined Trajectory ---")
        # lr_gcov = 4400
        # # 实例化合并引导场，严格关闭所有网络学习和冲突项
        # from utils.composed_guidance import ImageGCovAGMOnlineGuidance
        # combined_guided_field = ImageGCovAGMOnlineGuidance(
        #     base_model=model_fn,
        #     loss_fns=L_N_list,                          
        #     scales=[lr_gcov] * num_prompts,    
        #     config=config,
        #     learnable=False,                  # ✅ 彻底关闭 Residual Net
        #     conflict_weight=0.0,              # ✅ 设为0，不计算冲突
        #     vis_dir=None
        # )

        # # 最终推理：直接生成纯线性叠加的轨迹
        # traj_combined = generate_traj(combined_guided_field, z0=y_0, u=None, N=N)

        # # 保存合并后的图片
        # name, ext = os.path.splitext(opt_img_path)
        # combined_path = f'{name}_prompt_combined{ext}'
        # save_img(inverse_scaler(traj_combined[-1]), path=combined_path)
        # print(f"  -> Saved linear combined image to: {combined_path}")
                    
        # # =====================================================================
        # # 🔵 PCGrad Baseline：直接替换 combined_guided_field 的构造
        # # =====================================================================
        
        # from utils.composed_guidance import ImagePCGradGuidance
        
        # print("\n--- [PCGrad Baseline] Building PCGrad Guidance Field ---")
        # lr_gcov = 4400  # 与原 combined baseline 保持相同的 scale，确保公平对比
        
        # pcgrad_guided_field = ImagePCGradGuidance(
        #     base_model=model_fn,
        #     loss_fns=L_N_list,                      # 与原代码完全相同
        #     scales=[lr_gcov] * num_prompts,          # 与原代码完全相同
        #     config=config,
        #     random_proj=True,                        # 开启随机投影顺序（论文默认）
        # )
        
        # # PCGrad 不需要 train_model()，直接推理
        # print("[PCGrad Baseline] No training needed. Running inference directly.")
        # traj_pcgrad = generate_traj(pcgrad_guided_field, z0=y_0, u=None, N=N)
        
        # # 保存结果
        # name, ext = os.path.splitext(opt_img_path)
        # pcgrad_path = f'{name}_prompt_pcgrad{ext}'
        # save_img(inverse_scaler(traj_pcgrad[-1]), path=pcgrad_path)
        # print(f"  -> [PCGrad] Saved to: {pcgrad_path}")

        # # =========================================================
        # # 🔵 FK-ODE Baseline (Fair comparison: FK resampling + ODE)
        # # =========================================================
        # print(f"\n{'-'*60}")
        # print(f"Running FK-ODE Baseline (no GLASS, no gradients):")
        # print(f"{'-'*60}")

        # fk_K = 8                    # 粒子数
        # fk_n_checkpoints = 5        # resampling 检查点数
        # fk_temperature = 5.0        # reward 温度
        # fk_noise_level = 0.1        # 初始扰动

        # # 1. 初始化 K 个粒子（围绕 y_0 加微小扰动）
        # y_0_batch_fk = y_0.repeat(fk_K, 1, 1, 1)
        # z_fk = torch.randn_like(y_0_batch_fk)
        # particles = math.sqrt(1 - fk_noise_level**2) * y_0_batch_fk + fk_noise_level * z_fk

        # # 2. 设定 resampling 的检查点（均匀分布在 ODE 轨迹上）
        # checkpoint_steps = [int(N * (i+1) / fk_n_checkpoints) for i in range(fk_n_checkpoints)]
        
        # # 3. 前向 ODE + FK resampling
        # dt = 1.0 / N
        # eps = 1e-3
        
        # with torch.no_grad():
        #     for i in range(N):
        #         t = torch.ones(fk_K, device=config.device) * (i / N * (1. - eps) + eps)
        #         pred = model_fn(particles, t * 999)
        #         particles = particles + pred * dt

        #         # 在检查点处做 FK resampling
        #         if (i + 1) in checkpoint_steps:
        #             # (a) Denoiser: 预测终点 x̂₁ = x_t + v * (1 - t)
        #             t_current = i / N * (1. - eps) + eps
        #             x1_hat = particles + pred * (1.0 - t_current)
                    
        #             # (b) Reward: 评估每个粒子的 CLIP 分数
        #             log_r = torch.zeros(fk_K, device=config.device)
        #             for clip_loss in clip_loss_list:
        #                 loss_k = clip_loss.L_N(x1_hat)   # (K,)
        #                 log_r = log_r - fk_temperature * loss_k
                    
        #             # (c) Resample: 按 reward 重采样
        #             weights = torch.softmax(log_r, dim=0)
        #             indices = torch.multinomial(weights, fk_K, replacement=True)
        #             particles = particles[indices]
                    
        #             # 打印 ESS
        #             ess = 1.0 / (weights ** 2).sum().item()
        #             print(f"  Checkpoint step {i+1}/{N}: "
        #                   f"ESS={ess:.1f}/{fk_K}, "
        #                   f"reward: min={log_r.min():.2f} max={log_r.max():.2f}")

        # # 4. 选最佳粒子
        # with torch.no_grad():
        #     final_rewards = torch.zeros(fk_K, device=config.device)
        #     for clip_loss in clip_loss_list:
        #         final_rewards -= clip_loss.L_N(particles)
        #     best_idx = final_rewards.argmax().item()
        #     best_particle = particles[best_idx:best_idx+1]

        # # 5. 保存
        # fk_ode_path = os.path.join(target_dir, f'{name}_fk_ode_K{fk_K}{ext}')
        # save_img(inverse_scaler(best_particle), path=fk_ode_path)
        # print(f"  -> Saved FK-ODE baseline to: {fk_ode_path}")

        # # 6. Metrics
        # with torch.no_grad():
        #     for p_idx, prompt in enumerate(text_prompts):
        #         loss_val = clip_loss_list[p_idx].L_N(best_particle).mean().item()
        #         print(f"  Prompt {p_idx+1} '{prompt}': CLIP loss = {loss_val:.4f}")
            
        #     img_recon = inverse_scaler(traj[-1].to(config.device)).clamp(0, 1)
        #     img_edit = inverse_scaler(best_particle).clamp(0, 1)
        #     lpips_score = lpips_f(img_edit * 2 - 1, img_recon * 2 - 1).item()
        #     print(f"  LPIPS: {lpips_score:.4f}")

        # # 计算 Metrics (LPIPS 修正版)
        # with torch.no_grad():
        #     clip_losses = []
        #     for p_idx in range(num_prompts):
        #         loss_val = clip_loss_list[p_idx].L_N(traj_oc[-1].to(config.device)).item()
        #         clip_losses.append(loss_val)
            
        #     img_recon = inverse_scaler(traj[-1].to(config.device)).clamp(0, 1)
        #     img_edit = inverse_scaler(traj_oc[-1].to(config.device)).clamp(0, 1)
        #     img_recon_norm = img_recon * 2 - 1
        #     img_edit_norm = img_edit * 2 - 1
            
        #     lpips_score = lpips_f(img_edit_norm, img_recon_norm).item()
            
        #     print(f'\n=== Results ===')
        #     for idx, (prompt, loss) in enumerate(zip(text_prompts, clip_losses)):
        #         print(f'Prompt {idx+1} "{prompt}": CLIP loss = {loss:.4f}')
        #     print(f'LPIPS score: {lpips_score:.4f}')
        #     print(f'Total time: {time.time() - t_s:.4f} s')

        #     metrics[opt_img_path] = {
        #         'clip_losses': clip_losses,
        #         'lpips_score': lpips_score,
        #         'method': method
        #     }
    
    os.makedirs(target_dir, exist_ok=True)
    torch.save(metrics, f'{target_dir}/metrics_{method}.pt')
    return metrics