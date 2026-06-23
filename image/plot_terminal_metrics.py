#!/usr/bin/env python3
"""
绘制 conflict_multiprompt 优化过程中，terminal 时刻的 -L_N_1、-L_N_2 和 terminal conflict_score 随迭代的变化。
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_size64_result_conflict_multiprompt.log")

# Iter 1 terminal: L_N_1=-0.1519, L_N_2=-0.1645
PATTERN_LN = re.compile(r"\s*Iter\s+(\d+)\s+terminal:\s+L_N_1=([-\d.]+),\s+L_N_2=([-\d.]+)")
# Iter 1 terminal conflict_score=0.9003, blend=0.9003, ...
PATTERN_CONFLICT = re.compile(r"\s*Iter\s+(\d+)\s+terminal\s+conflict_score=([\d.]+)")


def parse_terminal_metrics(path):
    """解析 log，返回 iters, L_N_1, L_N_2, conflict_score（与 iter 一一对应）。"""
    iters = []
    L_N_1 = []
    L_N_2 = []
    conflict_scores = []
    with open(path, "r") as f:
        for line in f:
            m_ln = PATTERN_LN.match(line.strip())
            m_c = PATTERN_CONFLICT.match(line.strip())
            if m_ln:
                iters.append(int(m_ln.group(1)))
                # 日志里打印的是 L_N_1=...（即 -L_N 的数值），用户要画 -L_N_1、-L_N_2，即直接使用解析值
                L_N_1.append(float(m_ln.group(2)))
                L_N_2.append(float(m_ln.group(3)))
            if m_c:
                conflict_scores.append(float(m_c.group(2)))
    # 保证顺序一致：按 iter 对齐（同一 iter 先出现 L_N 行再出现 conflict 行）
    n = len(iters)
    if len(conflict_scores) < n:
        conflict_scores.extend([np.nan] * (n - len(conflict_scores)))
    conflict_scores = conflict_scores[:n]
    return np.array(iters), np.array(L_N_1), np.array(L_N_2), np.array(conflict_scores)


def main():
    if not os.path.isfile(LOG_PATH):
        print(f"File not found: {LOG_PATH}")
        return
    iters, L_N_1, L_N_2, conflict_scores = parse_terminal_metrics(LOG_PATH)
    if len(iters) == 0:
        print("No terminal metrics found in log.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(iters, L_N_1, "b-o", markersize=5, label="-L_N_1 (reward 1)")
    ax1.plot(iters, L_N_2, "g-s", markersize=5, label="-L_N_2 (reward 2)")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("-L_N (terminal reward)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(iters, conflict_scores, "r-^", markersize=5, label="terminal conflict_score")
    ax2.set_ylabel("Terminal conflict_score", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 1.05)

    plt.title("Terminal metrics: -L_N_1, -L_N_2, conflict_score (batch_size64_result_conflict_multiprompt.log)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(LOG_PATH), "terminal_metrics_conflict_multiprompt.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
