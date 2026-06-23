#!/usr/bin/env python3
"""
Plot conflict_score during OC optimization for two prompt pairs:
- sad & angry:  batch_size64_result_conflict_multiprompt.log
- smiling & curly: smile_curly_batch_size64_result_conflict_multiprompt.log
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Log paths and labels (用户指定：batch_size64 = smiling&curly 输出，smile_curly = angry&sad 输出)
LOG_FILES = [
    ("batch_size64_result_conflict_multiprompt.log", "smiling & curly"),
    ("smile_curly_batch_size64_result_conflict_multiprompt.log", "angry & sad"),
]

PATTERN = re.compile(
    r"\s*Step\s+(\d+),\s+Prompt pair.*?conflict_score=([\d.]+),\s+avg_conflict=([\d.]+).*"
)


def parse_conflict_log(path):
    """Parse a conflict_multiprompt log file; return steps and conflict_scores (and avg_conflict)."""
    steps = []
    conflict_scores = []
    avg_conflicts = []
    with open(path, "r") as f:
        for line in f:
            m = PATTERN.match(line.strip())
            if m:
                step = int(m.group(1))
                cs = float(m.group(2))
                ac = float(m.group(3))
                steps.append(step)
                conflict_scores.append(cs)
                avg_conflicts.append(ac)
    return np.array(steps), np.array(conflict_scores), np.array(avg_conflicts)


def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    threshold = 0.5
    for log_name, label in LOG_FILES:
        path = os.path.join(base, log_name)
        if not os.path.isfile(path):
            print(f"Skip (not found): {path}")
            continue
        steps, conflict_scores, avg_conflicts = parse_conflict_log(path)
        # Use sequential index as x (optimization step index across iterations)
        x = np.arange(len(conflict_scores))
        ax.plot(x, conflict_scores, label=label, alpha=0.85)
        print(f"{label}: {len(conflict_scores)} points, conflict_score in [{conflict_scores.min():.3f}, {conflict_scores.max():.3f}]")

    ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax.set_xlabel("Optimization step (trajectory step × iterations)", fontsize=12)
    ax.set_ylabel("Conflict score (1 - cos_sim)", fontsize=12)
    ax.set_title("Conflict score during OC optimization (conflict_multiprompt)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)  # auto upper limit to show full range
    plt.tight_layout()
    out = os.path.join(base, "conflict_score_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
