#!/usr/bin/env python3
"""
Script to parse training logs and plot Loss (MSE) and Active Conflict Ratio.
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Raw log data
log_data = """
=== Step 1/100 Training Monitor ===
Loss (MSE): 1.706885e-03
Mean Reward: -1.3883 (Should Increase)
Direction Cosine (mean/median): 0.0000 / 0.0000 :x:
RMS: Pred=0.000000e+00 | Targ=1.040537e-04 | Ratio=0.0000
Norms: Pred=0.000000 | Targ=0.113014 | Ratio=0.0000
Backprop Grad Norm: 1.263416
Active Conflict Ratio (global): 30.00%
Mask ones count: 18 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.3149e-02 |g2|=2.3149e-02
[conflict determinism] c1=0.6940 c2=0.6940

=== Step 2/100 Training Monitor ===
Loss (MSE): 2.565223e-03
Mean Reward: -1.3924 (Should Increase)
Direction Cosine (mean/median): 0.0007 / 0.0005 :white_check_mark:
RMS: Pred=1.037320e-04 | Targ=1.040537e-04 | Ratio=0.9969
Norms: Pred=0.112665 | Targ=0.113014 | Ratio=0.9969
Backprop Grad Norm: 152.316910
Active Conflict Ratio (global): 20.00%
Mask ones count: 12 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.4882e-02 |g2|=2.4882e-02
[conflict determinism] c1=0.7951 c2=0.7951

=== Step 3/100 Training Monitor ===
Loss (MSE): 1.945830e-03
Mean Reward: -1.3863 (Should Increase)
Direction Cosine (mean/median): 0.0084 / 0.0078 :white_check_mark:
RMS: Pred=1.138391e-04 | Targ=1.040537e-04 | Ratio=1.0940
Norms: Pred=0.123642 | Targ=0.113014 | Ratio=1.0940
Backprop Grad Norm: 128.637329
Active Conflict Ratio (global): 20.00%
Mask ones count: 12 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.5018e-02 |g2|=2.5018e-02
[conflict determinism] c1=0.8335 c2=0.8335

=== Step 4/100 Training Monitor ===
Loss (MSE): 2.367455e-03
Mean Reward: -1.3929 (Should Increase)
Direction Cosine (mean/median): 0.0067 / 0.0065 :white_check_mark:
RMS: Pred=1.152614e-04 | Targ=1.040537e-04 | Ratio=1.1077
Norms: Pred=0.125187 | Targ=0.113014 | Ratio=1.1077
Backprop Grad Norm: 150.694199
Active Conflict Ratio (global): 25.00%
Mask ones count: 15 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.4886e-02 |g2|=2.4886e-02
[conflict determinism] c1=0.6423 c2=0.6423

=== Step 5/100 Training Monitor ===
Loss (MSE): 2.576921e-03
Mean Reward: -1.3996 (Should Increase)
Direction Cosine (mean/median): 0.0038 / 0.0036 :white_check_mark:
RMS: Pred=9.297938e-05 | Targ=1.040537e-04 | Ratio=0.8936
Norms: Pred=0.100986 | Targ=0.113014 | Ratio=0.8936
Backprop Grad Norm: 128.487534
Active Conflict Ratio (global): 28.33%
Mask ones count: 17 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=1.9411e-02 |g2|=1.9411e-02
[conflict determinism] c1=0.8152 c2=0.8152

=== Step 6/100 Training Monitor ===
Loss (MSE): 9.771182e-04
Mean Reward: -1.3879 (Should Increase)
Direction Cosine (mean/median): 0.0022 / 0.0021 :white_check_mark:
RMS: Pred=7.430005e-05 | Targ=1.040537e-04 | Ratio=0.7141
Norms: Pred=0.080698 | Targ=0.113014 | Ratio=0.7141
Backprop Grad Norm: 57.497284
Active Conflict Ratio (global): 21.67%
Mask ones count: 13 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.0912e-02 |g2|=2.0912e-02
[conflict determinism] c1=0.8604 c2=0.8604

=== Step 7/100 Training Monitor ===
Loss (MSE): 2.290357e-03
Mean Reward: -1.3936 (Should Increase)
Direction Cosine (mean/median): 0.0015 / 0.0013 :white_check_mark:
RMS: Pred=9.124940e-05 | Targ=1.040537e-04 | Ratio=0.8769
Norms: Pred=0.099107 | Targ=0.113014 | Ratio=0.8769
Backprop Grad Norm: 103.318115
Active Conflict Ratio (global): 25.00%
Mask ones count: 15 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.8637e-02 |g2|=2.8637e-02
[conflict determinism] c1=0.6100 c2=0.6100

=== Step 8/100 Training Monitor ===
Loss (MSE): 1.707489e-03
Mean Reward: -1.3959 (Should Increase)
Direction Cosine (mean/median): 0.0039 / 0.0037 :white_check_mark:
RMS: Pred=8.942248e-05 | Targ=1.040537e-04 | Ratio=0.8594
Norms: Pred=0.097123 | Targ=0.113014 | Ratio=0.8594
Backprop Grad Norm: 85.280487
Active Conflict Ratio (global): 20.00%
Mask ones count: 12 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.0532e-02 |g2|=2.0532e-02
[conflict determinism] c1=0.6220 c2=0.6220

=== Step 9/100 Training Monitor ===
Loss (MSE): 2.103124e-03
Mean Reward: -1.3902 (Should Increase)
Direction Cosine (mean/median): 0.0085 / 0.0080 :white_check_mark:
RMS: Pred=7.257668e-05 | Targ=1.040537e-04 | Ratio=0.6975
Norms: Pred=0.078827 | Targ=0.113014 | Ratio=0.6975
Backprop Grad Norm: 93.854904
Active Conflict Ratio (global): 18.33%
Mask ones count: 11 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=1.9699e-02 |g2|=1.9699e-02
[conflict determinism] c1=1.0682 c2=1.0682

=== Step 10/100 Training Monitor ===
Loss (MSE): 2.183956e-03
Mean Reward: -1.3817 (Should Increase)
Direction Cosine (mean/median): 0.0180 / 0.0173 :white_check_mark:
RMS: Pred=4.587578e-05 | Targ=1.040537e-04 | Ratio=0.4409
Norms: Pred=0.049826 | Targ=0.113014 | Ratio=0.4409
Backprop Grad Norm: 62.728222
Active Conflict Ratio (global): 21.67%
Mask ones count: 13 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.1348e-02 |g2|=2.1348e-02
[conflict determinism] c1=0.7774 c2=0.7774

=== Step 11/100 Training Monitor ===
Loss (MSE): 2.084793e-03
Mean Reward: -1.3791 (Should Increase)
Direction Cosine (mean/median): 0.0179 / 0.0169 :white_check_mark:
RMS: Pred=5.069040e-05 | Targ=1.040537e-04 | Ratio=0.4872
Norms: Pred=0.055056 | Targ=0.113014 | Ratio=0.4872
Backprop Grad Norm: 48.546337
Active Conflict Ratio (global): 26.67%
Mask ones count: 16 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.2692e-02 |g2|=2.2692e-02
[conflict determinism] c1=0.6320 c2=0.6320

=== Step 12/100 Training Monitor ===
Loss (MSE): 2.048122e-03
Mean Reward: -1.3987 (Should Increase)
Direction Cosine (mean/median): 0.0113 / 0.0107 :white_check_mark:
RMS: Pred=7.656561e-05 | Targ=1.040537e-04 | Ratio=0.7358
Norms: Pred=0.083159 | Targ=0.113014 | Ratio=0.7358
Backprop Grad Norm: 71.828453
Active Conflict Ratio (global): 26.67%
Mask ones count: 16 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.3744e-02 |g2|=2.3744e-02
[conflict determinism] c1=1.3041 c2=1.3041

=== Step 13/100 Training Monitor ===
Loss (MSE): 1.770347e-03
Mean Reward: -1.4016 (Should Increase)
Direction Cosine (mean/median): 0.0084 / 0.0084 :white_check_mark:
RMS: Pred=8.716836e-05 | Targ=1.040537e-04 | Ratio=0.8377
Norms: Pred=0.094675 | Targ=0.113014 | Ratio=0.8377
Backprop Grad Norm: 85.050316
Active Conflict Ratio (global): 23.33%
Mask ones count: 14 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.8165e-02 |g2|=2.8165e-02
[conflict determinism] c1=0.6178 c2=0.6178

=== Step 14/100 Training Monitor ===
Loss (MSE): 2.065337e-03
Mean Reward: -1.4139 (Should Increase)
Direction Cosine (mean/median): 0.0075 / 0.0077 :white_check_mark:
RMS: Pred=7.718197e-05 | Targ=1.040537e-04 | Ratio=0.7418
Norms: Pred=0.083829 | Targ=0.113014 | Ratio=0.7418
Backprop Grad Norm: 80.295761
Active Conflict Ratio (global): 25.00%
Mask ones count: 15 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=1.8815e-02 |g2|=1.8815e-02
[conflict determinism] c1=1.0277 c2=1.0277

=== Step 15/100 Training Monitor ===
Loss (MSE): 1.900495e-03
Mean Reward: -1.3931 (Should Increase)
Direction Cosine (mean/median): 0.0080 / 0.0082 :white_check_mark:
RMS: Pred=5.653563e-05 | Targ=1.040537e-04 | Ratio=0.5433
Norms: Pred=0.061404 | Targ=0.113014 | Ratio=0.5433
Backprop Grad Norm: 37.467354
Active Conflict Ratio (global): 20.00%
Mask ones count: 12 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.8999e-02 |g2|=2.8999e-02
[conflict determinism] c1=0.7797 c2=0.7797

=== Step 16/100 Training Monitor ===
Loss (MSE): 1.167982e-03
Mean Reward: -1.3869 (Should Increase)
Direction Cosine (mean/median): 0.0071 / 0.0068 :white_check_mark:
RMS: Pred=5.294836e-05 | Targ=1.040537e-04 | Ratio=0.5089
Norms: Pred=0.057508 | Targ=0.113014 | Ratio=0.5089
Backprop Grad Norm: 35.454632
Active Conflict Ratio (global): 16.67%
Mask ones count: 10 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=3.2801e-02 |g2|=3.2801e-02
[conflict determinism] c1=0.5442 c2=0.5442

=== Step 17/100 Training Monitor ===
Loss (MSE): 1.272381e-03
Mean Reward: -1.3891 (Should Increase)
Direction Cosine (mean/median): 0.0054 / 0.0047 :white_check_mark:
RMS: Pred=6.301222e-05 | Targ=1.040537e-04 | Ratio=0.6056
Norms: Pred=0.068439 | Targ=0.113014 | Ratio=0.6056
Backprop Grad Norm: 67.877579
Active Conflict Ratio (global): 21.67%
Mask ones count: 13 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.9008e-02 |g2|=2.9008e-02
[conflict determinism] c1=0.5355 c2=0.5355

=== Step 18/100 Training Monitor ===
Loss (MSE): 1.835595e-03
Mean Reward: -1.3978 (Should Increase)
Direction Cosine (mean/median): 0.0054 / 0.0048 :white_check_mark:
RMS: Pred=6.227367e-05 | Targ=1.040537e-04 | Ratio=0.5985
Norms: Pred=0.067636 | Targ=0.113014 | Ratio=0.5985
Backprop Grad Norm: 71.956390
Active Conflict Ratio (global): 25.00%
Mask ones count: 15 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.3918e-02 |g2|=2.3918e-02
[conflict determinism] c1=0.5806 c2=0.5806

=== Step 19/100 Training Monitor ===
Loss (MSE): 1.259593e-03
Mean Reward: -1.3889 (Should Increase)
Direction Cosine (mean/median): 0.0073 / 0.0071 :white_check_mark:
RMS: Pred=4.850012e-05 | Targ=1.040537e-04 | Ratio=0.4661
Norms: Pred=0.052677 | Targ=0.113014 | Ratio=0.4661
Backprop Grad Norm: 51.937336
Active Conflict Ratio (global): 16.67%
Mask ones count: 10 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.2692e-02 |g2|=2.2692e-02
[conflict determinism] c1=0.8794 c2=0.8794

=== Step 20/100 Training Monitor ===
Loss (MSE): 1.623402e-03
Mean Reward: -1.3952 (Should Increase)
Direction Cosine (mean/median): 0.0101 / 0.0098 :white_check_mark:
RMS: Pred=3.699636e-05 | Targ=1.040537e-04 | Ratio=0.3556
Norms: Pred=0.040182 | Targ=0.113014 | Ratio=0.3556
Backprop Grad Norm: 34.153553
Active Conflict Ratio (global): 16.67%
Mask ones count: 10 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=3.3049e-02 |g2|=3.3049e-02
[conflict determinism] c1=0.8719 c2=0.8719

=== Step 21/100 Training Monitor ===
Loss (MSE): 1.926624e-03
Mean Reward: -1.3886 (Should Increase)
Direction Cosine (mean/median): 0.0089 / 0.0083 :white_check_mark:
RMS: Pred=4.292786e-05 | Targ=1.040537e-04 | Ratio=0.4126
Norms: Pred=0.046625 | Targ=0.113014 | Ratio=0.4126
Backprop Grad Norm: 34.865135
Active Conflict Ratio (global): 25.00%
Mask ones count: 15 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.2897e-02 |g2|=2.2897e-02
[conflict determinism] c1=0.8604 c2=0.8604

=== Step 22/100 Training Monitor ===
Loss (MSE): 1.609875e-03
Mean Reward: -1.3891 (Should Increase)
Direction Cosine (mean/median): 0.0072 / 0.0072 :white_check_mark:
RMS: Pred=5.302431e-05 | Targ=1.040537e-04 | Ratio=0.5096
Norms: Pred=0.057591 | Targ=0.113014 | Ratio=0.5096
Backprop Grad Norm: 47.277405
Active Conflict Ratio (global): 18.33%
Mask ones count: 11 / 60
========================================
[CLIP grad determinism] cos(g1,g2)=1.0000 |g1|=2.8142e-02 |g2|=2.8142e-02
[conflict determinism] c1=0.7405 c2=0.7405
"""

def parse_log(log_text):
    """Parse the training log and extract metrics."""
    steps = []
    losses = []
    active_ratios = []
    mean_rewards = []
    direction_cosines = []
    
    # Regex patterns
    step_pattern = r"=== Step (\d+)/\d+ Training Monitor ==="
    loss_pattern = r"Loss \(MSE\): ([\d.e+-]+)"
    active_ratio_pattern = r"Active Conflict Ratio \(global\): ([\d.]+)%"
    reward_pattern = r"Mean Reward: ([-\d.]+)"
    direction_cosine_pattern = r"Direction Cosine \(mean/median\): ([\d.]+) / [\d.]+"
    
    # Split by step blocks
    blocks = re.split(r'(?==== Step)', log_text)
    
    for block in blocks:
        if not block.strip():
            continue
            
        step_match = re.search(step_pattern, block)
        loss_match = re.search(loss_pattern, block)
        active_ratio_match = re.search(active_ratio_pattern, block)
        reward_match = re.search(reward_pattern, block)
        direction_cosine_match = re.search(direction_cosine_pattern, block)
        
        if step_match and loss_match and active_ratio_match:
            steps.append(int(step_match.group(1)))
            losses.append(float(loss_match.group(1)))
            active_ratios.append(float(active_ratio_match.group(1)))
            
            if reward_match:
                mean_rewards.append(float(reward_match.group(1)))
            if direction_cosine_match:
                direction_cosines.append(float(direction_cosine_match.group(1)))
    
    return {
        'steps': np.array(steps),
        'losses': np.array(losses),
        'active_ratios': np.array(active_ratios),
        'mean_rewards': np.array(mean_rewards),
        'direction_cosines': np.array(direction_cosines)
    }

def plot_metrics(data, save_path='training_metrics.png'):
    """Plot Loss and Active Conflict Ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss (MSE)
    ax1 = axes[0, 0]
    ax1.plot(data['steps'], data['losses'] * 1000, 'b-o', linewidth=2, markersize=4, label='Loss (MSE)')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss (MSE) x 1e-3', fontsize=12)
    ax1.set_title('Training Loss (MSE)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add smoothed line
    if len(data['losses']) > 3:
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(data['losses'] * 1000, size=3)
        ax1.plot(data['steps'], smoothed, 'r--', linewidth=2, alpha=0.7, label='Smoothed')
        ax1.legend()
    
    # Plot 2: Active Conflict Ratio
    ax2 = axes[0, 1]
    ax2.bar(data['steps'], data['active_ratios'], color='coral', alpha=0.7, edgecolor='darkred')
    ax2.axhline(y=np.mean(data['active_ratios']), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(data["active_ratios"]):.2f}%')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Active Conflict Ratio (%)', fontsize=12)
    ax2.set_title('Active Conflict Ratio per Step', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    ax2.set_ylim(0, max(data['active_ratios']) * 1.2)
    
    # Plot 3: Mean Reward
    if len(data['mean_rewards']) > 0:
        ax3 = axes[1, 0]
        ax3.plot(data['steps'], data['mean_rewards'], 'g-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_ylabel('Mean Reward', fontsize=12)
        ax3.set_title('Mean Reward (Should Increase)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        # Add trend line
        z = np.polyfit(data['steps'], data['mean_rewards'], 1)
        p = np.poly1d(z)
        ax3.plot(data['steps'], p(data['steps']), 'r--', linewidth=2, alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
        ax3.legend()
    
    # Plot 4: Direction Cosine
    if len(data['direction_cosines']) > 0:
        ax4 = axes[1, 1]
        ax4.plot(data['steps'], data['direction_cosines'], 'm-o', linewidth=2, markersize=4)
        ax4.set_xlabel('Step', fontsize=12)
        ax4.set_ylabel('Direction Cosine (mean)', fontsize=12)
        ax4.set_title('Direction Cosine (mean)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()

def print_statistics(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Steps: {len(data['steps'])}")
    
    print(f"\nLoss (MSE):")
    print(f"  - Initial: {data['losses'][0]:.6f}")
    print(f"  - Final:   {data['losses'][-1]:.6f}")
    print(f"  - Min:     {np.min(data['losses']):.6f} (Step {data['steps'][np.argmin(data['losses'])]})")
    print(f"  - Max:     {np.max(data['losses']):.6f} (Step {data['steps'][np.argmax(data['losses'])]})")
    print(f"  - Mean:    {np.mean(data['losses']):.6f}")
    print(f"  - Std:     {np.std(data['losses']):.6f}")
    
    print(f"\nActive Conflict Ratio (%):")
    print(f"  - Initial: {data['active_ratios'][0]:.2f}%")
    print(f"  - Final:   {data['active_ratios'][-1]:.2f}%")
    print(f"  - Min:     {np.min(data['active_ratios']):.2f}% (Step {data['steps'][np.argmin(data['active_ratios'])]})")
    print(f"  - Max:     {np.max(data['active_ratios']):.2f}% (Step {data['steps'][np.argmax(data['active_ratios'])]})")
    print(f"  - Mean:    {np.mean(data['active_ratios']):.2f}%")
    print(f"  - Std:     {np.std(data['active_ratios']):.2f}%")
    
    if len(data['mean_rewards']) > 0:
        print(f"\nMean Reward:")
        print(f"  - Initial: {data['mean_rewards'][0]:.4f}")
        print(f"  - Final:   {data['mean_rewards'][-1]:.4f}")
        print(f"  - Best:    {np.max(data['mean_rewards']):.4f} (Step {data['steps'][np.argmax(data['mean_rewards'])]})")
        
        # Check if reward is increasing (improvement)
        reward_change = data['mean_rewards'][-1] - data['mean_rewards'][0]
        print(f"  - Change:  {reward_change:+.4f} ({'Improved' if reward_change > 0 else 'Decreased'})")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Parse the log data
    data = parse_log(log_data)
    
    # Print statistics
    print_statistics(data)
    
    # Plot metrics
    plot_metrics(data, save_path='training_metrics.png')
