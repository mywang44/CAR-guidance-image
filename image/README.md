# OC-Flow

This code follows from [FlowGrad](https://github.com/gnobitab/FlowGrad).

## Controlling Rectified Flow on CelebA-HQ

We provide the scripts for applying OC-Flow to control the output of pre-trained Rectified Flow model on CelebA-HQ.

The pre-trained generative model can be downloaded from [Rectified Flow CelebA-HQ](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view?usp=sharing) 
Just put it in ``` ./ ```

### Dependencies
The following packages are required,

```
torch, numpy, lpips, clip, ml_collections, absl-py 
```

We also provide a build_env.sh script to install the dependencies.

If you hit `ImportError: ...libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`,
it's a missing runtime `libittnotify.so` in some environments. The provided
`build_env.sh` includes a small workaround that makes `import torch` work.

### Run

We provide a demo image ``` ./demo/celeba.jpg ``` for running our model.

```
python main_data.py
```

### Dataset

The full Celeba-hq-1024 dataset can be downloaded from [kaggle celeba-hq dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)

### 📊 2D 算法向高维图像迁移的诊断报告

**一、 迁移工程状态总结**

* **工程复现状态：完美跑通。** 包含 MSE Loss 计算、梯度求解、冲突分数（Conflict Score）掩码、以及在线优化循环在内的所有底层工程逻辑，已完全对齐原有的 2D 最优控制（OC）逻辑。
* **现象诊断：底层机制有效，但高维特征失效。** 能够观察到 Loss 稳定下降、相关系数（Corr）极高（常驻 0.8~0.9+），说明网络确实在学习。但由于高维空间的天然壁垒，Residual Guidance 仅能改变全局色调，无法进行局部面部特征等结构性修复。

**二、 核心失效原因及确凿证据**

#### 🛑 致命原因 1：维度稀释导致“原生梯度”极度微弱 (Dimensionality Dilution)

在 2D 空间中，目标能量差异和空间距离的比例相对正常，能自然产生有效的修正梯度。但在 19 万维（256x256x3）的图像空间中，稀疏样本导致计算出的理论梯度极小，完全无法与 Base Guidance 抗衡。

* **确凿证据（测算日志）**：
我们在 `[Theory Check]` 中精准测量了潜空间样本的物理状态：
* `Avg dX` (图像间平均欧氏距离)：高达 **198.18**
* `Avg dE` (能量/目标差异)：仅为 **6.48**
* `Min Slope` (理论平均梯度/坡度)：仅为 **0.0327**


* **结论**：未经人工倍率（如 `lr_res=100`）强行补偿的残差网络，其原生推力极度微弱。即便强行放大以对齐 Base Guidance（~45.0），放大的也是微观上的“平滑均值”，而非精准的结构发力点。

#### 🛑 致命原因 2：极度欠定系统导致“高频噪声过拟合” (Spectral Bias & Overfitting)

原算法试图在 Online 阶段（仅 50 步，Batch Size = 4~6）让一个从零初始化的 CNN/U-Net 学会解决复杂的语义冲突。这在图像任务中是典型的“极度欠定系统”，网络根本无法在几张图中顿悟出“人脸结构”，而是选择了走捷径——死记硬背这几张图的高频噪声指纹来强行降低 Loss。

* **确凿证据（热力图 Heatmap 对比）**：
* **Base Guidance (CLIP)**：热力图极其精准地点亮了眼睛、嘴巴等语义关键区域，证明 CLIP 拥有强大的预训练常识。
* **Residual Guidance (U-Net)**：热力图呈现出大面积的 0.5 均值绿屏，并伴随随机分布的细碎高频斑点。完全没有聚焦在面部轮廓上。


* **结论**：网络求导产生的并不是“语义修复方向”，而是“高频对抗噪声”。反映到图像上，就是肉眼看不出结构变化，只感觉被糊了一层有色滤镜。