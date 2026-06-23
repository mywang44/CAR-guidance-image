#!/bin/bash

# ==========================================
# 1. 声明并导出编译器环境变量 (全局生效)
# ==========================================
# 使用你专门准备的 compiler_env 里的 GCC 11
export CC=/home/users/xuehui/miniconda3/envs/compiler_env/bin/x86_64-conda-linux-gnu-cc
export CXX=/home/users/xuehui/miniconda3/envs/compiler_env/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=/home/users/xuehui/miniconda3/envs/compiler_env/bin/x86_64-conda-linux-gnu-c++

# 打印一下确认是否生效（可选，方便在 nohup.out 以外的终端直接看到）
echo "Current CC: $CC"
echo "Current CXX: $CXX"

# 运行脚本 - 选择不同的method参数
# 可选方法:
#   - ocflow: OC-Flow方法（论文主方法，手动梯度更新+权重衰减）
#   - flowgrad: 原始FlowGrad方法（PyTorch SGD优化器）
#   - ocfm: 直接优化初始点z0的方法
#   - multiprompt: 多Prompt方法，同时将图像align到多个text prompt

# # 使用 ocfm_multiprompt 方法运行
# # flowgrad + oc + multiprompt
# CUDA_VISIBLE_DEVICES=0 nohup python ./main_data.py \
#   --method=ocfm_multiprompt \
#   > batch_size64_result_ocfm_multiprompt.log 2>&1 &

# # flowgrad + oc + multiprompt + conflict
# # gcar_ocfm_multiprompt
# CUDA_VISIBLE_DEVICES=0 nohup python ./main_data.py \
#   --method=gcar_ocfm_multiprompt \
#   --conflict_threshold=0.2 \
#   --conflict_weight=0.0 \
#   --conflict_lr=2.5 \
#   > batch_size64_result_gcar_ocfm_multiprompt_cw00.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python ./main_data.py \
#   --method=gcar_ocfm_multiprompt \
#   --conflict_threshold=0.2 \
#   --conflict_weight=0.03 \
#   --conflict_lr=2.5 \
#   > batch_size64_result_gcar_ocfm_multiprompt_cw003.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python ./main_data.py \
#   --method=gcar_ocfm_multiprompt \
#   --conflict_threshold=0.2 \
#   --conflict_weight=0.08 \
#   --conflict_lr=2.5 \
#   > batch_size64_result_gcar_ocfm_multiprompt_cw008.log 2>&1 &

# # # flowgrad + multiprompt
# CUDA_VISIBLE_DEVICES=0 nohup python ./main_data.py \
#   --method=flowgrad_multiprompt \
#   --flowgrad_lr=2.5 \
#   > batch_size64_result_flowgrad_multiprompt.log 2>&1 &

# # gcovA + multiprompt
# CUDA_VISIBLE_DEVICES=1 nohup python ./main_data.py \
#   --method=gcovA_multiprompt \
#   > batch_size64_result_gcovA_multiprompt.log 2>&1 &

# # gcar (Trained Residual Guidance)
# CUDA_VISIBLE_DEVICES=2 nohup python ./main_data.py \
#   --method=gcar_gcovA_multiprompt \
#   --conflict_weight=0.0 \
#   --conflict_lr=2.5 \
#   > batch_size64_result_gcar_gcovA_multiprompt_cw00.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python ./main_data.py \
#   --method=gcar_gcovA_multiprompt \
#   --conflict_weight=0.03 \
#   --conflict_lr=2.5 \
#   > batch_size64_result_gcar_gcovA_multiprompt_cw003.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python ./main_data.py \
#   --method=gcar_gcovA_multiprompt \
#   --conflict_weight=0.08 \
#   --conflict_lr=2.5 \
#   > batch_size64_result_gcar_gcovA_multiprompt_cw008.log 2>&1 &
  

# gcovA + multiprompt
CUDA_VISIBLE_DEVICES=1 nohup python ./main_data.py \
  --method=gcovA_multiprompt \
  > 5_batch_size64_result_gcovA_multiprompt.log 2>&1 &

# gcar (Trained Residual Guidance)
CUDA_VISIBLE_DEVICES=1 nohup python ./main_data.py \
  --method=gcar_gcovA_multiprompt \
  --conflict_weight=0.0 \
  --conflict_lr=2.5 \
  > 5_batch_size64_result_gcar_gcovA_multiprompt_cw00.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python ./main_data.py \
  --method=gcar_gcovA_multiprompt \
  --conflict_weight=0.03 \
  --conflict_lr=2.5 \
  > 5_batch_size64_result_gcar_gcovA_multiprompt_cw003.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python ./main_data.py \
  --method=gcar_gcovA_multiprompt \
  --conflict_weight=0.08 \
  --conflict_lr=2.5 \
  > 5_batch_size64_result_gcar_gcovA_multiprompt_cw008.log 2>&1 &
  

