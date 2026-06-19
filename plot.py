#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立绘图脚本 - 读取训练数据并绘制分析图表
使用方法: python plot.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==================== 配置参数 ====================
STAGE1_EPISODES = 500
STAGE2_EPISODES = 1000
TOTAL_EPISODES = 1500

# 设置全局字体 - 英文用Times New Roman，中文用SimHei
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimHei', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 设置全局绘图参数 - 美化
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['legend.framealpha'] = 0.9

# 颜色定义
COLORS = {
    'robot1': '#C3B1E8',
    'robot2': '#9B8FD9',
    'robot3': '#7B6FB5',
    'stage_boundary': '#E74C3C',
    'raw_reward': '#BDC3C7',
    'moving_avg': '#2980B9',
    'steps': '#27AE60',
}

ROBOT_NAMES = ['Robot 1', 'Robot 2', 'Robot 3']


def find_latest_training_data():
    """查找最新的训练数据文件夹"""
    folders = [f for f in os.listdir('.') if f.startswith('training_data_') and os.path.isdir(f)]
    if not folders:
        raise FileNotFoundError("未找到 training_data_* 文件夹")

    folders.sort(reverse=True)
    latest = folders[0]
    print(f"📂 使用训练数据: {latest}")
    return latest


def load_training_data(data_dir):
    """加载所有训练数据"""
    print(f"📖 加载数据从 {data_dir}...")

    curve_df = pd.read_csv(f'{data_dir}/training_curve_data.csv')
    robot_rewards_df = pd.read_csv(f'{data_dir}/robot_individual_rewards.csv')
    robot_success_df = pd.read_csv(f'{data_dir}/robot_success_history.csv')
    config_df = pd.read_csv(f'{data_dir}/training_config.csv')
    summary_df = pd.read_csv(f'{data_dir}/training_summary.csv')

    print(f"   ✅ 加载完成: {len(curve_df)} 轮训练数据")
    return curve_df, robot_rewards_df, robot_success_df, config_df, summary_df


def plot_total_reward_curve(curve_df, save_dir):
    """绘制总奖励曲线（美化版）"""
    fig, ax = plt.subplots(figsize=(12, 7))

    rewards = curve_df['total_reward'].values

    # 原始奖励（淡灰色，细线）
    ax.plot(rewards, color=COLORS['raw_reward'], alpha=0.35, linewidth=0.8, label='Raw Reward')

    # 50轮移动平均（深蓝色，粗线）
    if len(rewards) >= 50:
        moving_avg = pd.Series(rewards).rolling(50).mean()
        ax.plot(moving_avg, color=COLORS['moving_avg'], linewidth=2.5, label='Moving Average (Window=50)')

    # 阶段分界线
    ax.axvline(x=STAGE1_EPISODES, color=COLORS['stage_boundary'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Stage Boundary')

    # 标注阶段文字
    ax.text(STAGE1_EPISODES / 2, ax.get_ylim()[1] * 0.92, 'Stage 1\nExploration',
            ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(STAGE1_EPISODES + STAGE2_EPISODES / 2, ax.get_ylim()[1] * 0.92, 'Stage 2\nClosed-Loop Training',
            ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Episode', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Total Reward', fontsize=13, fontweight='semibold')
    ax.set_title('Two-Stage Training Curve', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # 美化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/total_reward_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('total_reward_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: total_reward_curve.png")


def plot_individual_rewards(robot_rewards_df, save_dir):
    """绘制各机器人独立奖励对比图（美化版）"""
    robot_rewards = robot_rewards_df.values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for i in range(3):
        ax = axes[i]
        ax.plot(robot_rewards[:, i], color=COLORS[f'robot{i + 1}'], alpha=0.35, linewidth=0.8)

        if len(robot_rewards) >= 50:
            moving_avg = pd.Series(robot_rewards[:, i]).rolling(50).mean()
            ax.plot(moving_avg, color=COLORS[f'robot{i + 1}'], linewidth=2.5)

        # 阶段分界线
        ax.axvline(x=STAGE1_EPISODES, color=COLORS['stage_boundary'], linestyle='--', linewidth=1.2, alpha=0.7)

        ax.set_xlabel('Episode', fontsize=11, fontweight='semibold')
        ax.set_ylabel('Reward', fontsize=11, fontweight='semibold')
        ax.set_title(ROBOT_NAMES[i], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        # 美化边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Individual Robot Reward Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/individual_rewards.png', dpi=300, bbox_inches='tight')
    plt.savefig('individual_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: individual_rewards.png")


def plot_success_rate_comparison(robot_success_df, save_dir):
    """绘制各机器人成功率对比图（美化版）"""
    success_history = robot_success_df.values
    window = 50

    fig, ax = plt.subplots(figsize=(12, 7))

    line_styles = ['-', '--', '-.']
    for i in range(3):
        smooth = pd.Series(success_history[:, i]).rolling(window, min_periods=1).mean()
        ax.plot(smooth, color=COLORS[f'robot{i + 1}'], linewidth=2.5,
                label=f'{ROBOT_NAMES[i]}', linestyle=line_styles[i])

    # 阶段分界线
    ax.axvline(x=STAGE1_EPISODES, color=COLORS['stage_boundary'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Stage Boundary')
    ax.set_xlabel('Episode', fontsize=13, fontweight='semibold')
    ax.set_ylabel(f'Success Rate (Rolling Window={window})', fontsize=13, fontweight='semibold')
    ax.set_title('Robot Success Rate Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.02, 1.05)

    # 获取最终成功率
    final_rates = [success_history[-100:, i].mean() for i in range(3)]

    # 标注上下顺序逆转：Robot 3（最高）在上方，Robot 1（最低）在下方
    # Robot 3 (99%) -> y=0.62 (最上方)
    # Robot 2 (96%) -> y=0.48 (中间)
    # Robot 1 (89%) -> y=0.34 (最下方)
    y_positions = [0.34, 0.48, 0.62]  # 索引0=R1,1=R2,2=R3
    for i, (rate, y) in enumerate(zip(final_rates, y_positions)):
        ax.annotate(f'Final: {rate * 100:.1f}%',
                    xy=(TOTAL_EPISODES - 50, rate),
                    xytext=(TOTAL_EPISODES - 180, y),
                    fontsize=10, color=COLORS[f'robot{i + 1}'], fontweight='semibold',
                    arrowprops=dict(arrowstyle='->', color=COLORS[f'robot{i + 1}'], alpha=0.6, lw=1.2))

    # 美化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: success_rate_comparison.png")


def plot_steps_curve(curve_df, save_dir):
    """绘制步数曲线（美化版）"""
    steps = curve_df['steps'].values

    fig, ax = plt.subplots(figsize=(12, 7))

    # 原始步数曲线 - 灰色（与 total_reward_curve 原始数据同色）
    ax.plot(steps, color='#BDC3C7', alpha=0.35, linewidth=0.8, label='Steps per Episode')

    if len(steps) >= 50:
        moving_avg = pd.Series(steps).rolling(50).mean()
        # 移动平均线 - 蓝色（与 total_reward_curve 移动平均同色）
        ax.plot(moving_avg, color='#2980B9', linewidth=2.5, label='Moving Average (Window=50)')

    # 阶段分界线
    ax.axvline(x=STAGE1_EPISODES, color=COLORS['stage_boundary'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Stage Boundary')
    ax.set_xlabel('Episode', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Steps', fontsize=13, fontweight='semibold')
    ax.set_title('Steps per Episode', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)

    # 添加效率提升标注 - 蓝色
    final_avg_steps = np.mean(steps[-100:])
    ax.annotate(f'Final avg: {final_avg_steps:.1f} steps',
                xy=(TOTAL_EPISODES - 50, final_avg_steps),
                xytext=(TOTAL_EPISODES - 220, 20),
                fontsize=11, color='#2980B9', fontweight='semibold',
                arrowprops=dict(arrowstyle='->', color='#2980B9', alpha=0.6, lw=1.2))

    # 美化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/steps_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('steps_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: steps_curve.png")


def plot_combined_figure(curve_df, save_dir):
    """绘制组合图（总奖励曲线美化版）"""
    fig, ax = plt.subplots(figsize=(12, 7))

    rewards = curve_df['total_reward'].values

    ax.plot(rewards, color=COLORS['raw_reward'], alpha=0.35, linewidth=0.8, label='Raw Reward')

    if len(rewards) >= 50:
        moving_avg = pd.Series(rewards).rolling(50).mean()
        ax.plot(moving_avg, color=COLORS['moving_avg'], linewidth=2.5, label='Moving Average (Window=50)')

    # 阶段分界线
    ax.axvline(x=STAGE1_EPISODES, color=COLORS['stage_boundary'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Stage Boundary')

    # 标注关键指标
    final_avg = np.mean(rewards[-100:])
    ax.annotate(f'Final Avg Reward: {final_avg:.0f}',
                xy=(TOTAL_EPISODES - 50, final_avg),
                xytext=(TOTAL_EPISODES - 250, final_avg + 300),
                fontsize=10, color=COLORS['moving_avg'],
                arrowprops=dict(arrowstyle='->', color=COLORS['moving_avg'], alpha=0.6))

    ax.set_xlabel('Episode', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Total Reward', fontsize=13, fontweight='semibold')
    ax.set_title('Two-Stage Full Closed-Loop Training Curve', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # 美化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/full_training_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('full_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: full_training_curve.png")


def plot_reward_distribution(curve_df, save_dir):
    """绘制奖励分布直方图（美化版）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    stage1_rewards = curve_df[curve_df['stage'] == 1]['total_reward']
    stage2_rewards = curve_df[curve_df['stage'] == 2]['total_reward']

    # Stage 1 分布
    axes[0].hist(stage1_rewards, bins=35, color='#5DADE2', edgecolor='#2C3E50',
                 linewidth=1.2, alpha=0.75, density=True)
    axes[0].axvline(stage1_rewards.mean(), color='#E74C3C', linestyle='--',
                    linewidth=2, label=f'Mean: {stage1_rewards.mean():.0f}')
    axes[0].axvline(stage1_rewards.median(), color='#27AE60', linestyle=':',
                    linewidth=2, label=f'Median: {stage1_rewards.median():.0f}')
    axes[0].set_xlabel('Total Reward', fontsize=12, fontweight='semibold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='semibold')
    axes[0].set_title('Stage 1: Exploration Phase', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Stage 2 分布
    axes[1].hist(stage2_rewards, bins=35, color='#48C9B0', edgecolor='#2C3E50',
                 linewidth=1.2, alpha=0.75, density=True)
    axes[1].axvline(stage2_rewards.mean(), color='#E74C3C', linestyle='--',
                    linewidth=2, label=f'Mean: {stage2_rewards.mean():.0f}')
    axes[1].axvline(stage2_rewards.median(), color='#27AE60', linestyle=':',
                    linewidth=2, label=f'Median: {stage2_rewards.median():.0f}')
    axes[1].set_xlabel('Total Reward', fontsize=12, fontweight='semibold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='semibold')
    axes[1].set_title('Stage 2: Closed-Loop Training Phase', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('Reward Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: reward_distribution.png")


def plot_learning_curve_comparison(curve_df, save_dir):
    """绘制不同窗口平滑对比图（美化版）"""
    fig, ax = plt.subplots(figsize=(12, 7))

    rewards = curve_df['total_reward'].values

    windows = [10, 50, 100]
    colors_w = ['#3498DB', '#2980B9', '#1C5980']
    line_widths = [1.8, 2.2, 2.5]

    for w, c, lw in zip(windows, colors_w, line_widths):
        smooth = pd.Series(rewards).rolling(w).mean()
        ax.plot(smooth, color=c, linewidth=lw, label=f'Window = {w}')

    # 阶段分界线
    ax.axvline(x=STAGE1_EPISODES, color=COLORS['stage_boundary'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Stage Boundary')

    ax.set_xlabel('Episode', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Average Reward', fontsize=13, fontweight='semibold')
    ax.set_title('Learning Curves with Different Smoothing Windows', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: learning_curve_comparison.png")


def plot_performance_summary(robot_success_df, save_dir):
    """绘制性能总结柱状图（美化版）"""
    final_rates = [robot_success_df.iloc[-100:][f'Robot{i + 1}_Success'].mean() * 100 for i in range(3)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(ROBOT_NAMES, final_rates,
                  color=[COLORS['robot1'], COLORS['robot2'], COLORS['robot3']],
                  edgecolor='#2C3E50', linewidth=1.5, alpha=0.85, width=0.6)

    # 添加数值标签
    for bar, rate in zip(bars, final_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='semibold')
    ax.set_title('Final Performance Summary (Last 100 Episodes)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, axis='y')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: performance_summary.png")


def print_statistics(curve_df, robot_rewards_df, robot_success_df, summary_df):
    """打印训练统计信息"""
    print("\n" + "=" * 70)
    print("📊 TRAINING STATISTICS SUMMARY")
    print("=" * 70)

    stage1_rewards = curve_df[curve_df['stage'] == 1]['total_reward']
    stage2_rewards = curve_df[curve_df['stage'] == 2]['total_reward']
    stage1_steps = curve_df[curve_df['stage'] == 1]['steps']
    stage2_steps = curve_df[curve_df['stage'] == 2]['steps']

    print(f"\n[Reward Statistics]")
    print(f"  ├─ Stage 1 Mean:     {stage1_rewards.mean():8.2f}")
    print(f"  ├─ Stage 1 Std:      {stage1_rewards.std():8.2f}")
    print(f"  ├─ Stage 1 Range:    [{stage1_rewards.min():8.0f}, {stage1_rewards.max():8.0f}]")
    print(f"  ├─ Stage 2 Mean:     {stage2_rewards.mean():8.2f}")
    print(f"  ├─ Stage 2 Std:      {stage2_rewards.std():8.2f}")
    print(f"  └─ Stage 2 Range:    [{stage2_rewards.min():8.0f}, {stage2_rewards.max():8.0f}]")

    print(f"\n[Steps Statistics]")
    print(f"  ├─ Stage 1 Mean:     {stage1_steps.mean():8.2f}")
    print(f"  ├─ Stage 1 Std:      {stage1_steps.std():8.2f}")
    print(f"  ├─ Stage 1 Range:    [{stage1_steps.min():8.0f}, {stage1_steps.max():8.0f}]")
    print(f"  ├─ Stage 2 Mean:     {stage2_steps.mean():8.2f}")
    print(f"  ├─ Stage 2 Std:      {stage2_steps.std():8.2f}")
    print(f"  └─ Stage 2 Range:    [{stage2_steps.min():8.0f}, {stage2_steps.max():8.0f}]")

    print(f"\n[Final Success Rates (Last 100 Episodes)]")
    for i in range(3):
        rate = robot_success_df.iloc[-100:][f'Robot{i + 1}_Success'].mean() * 100
        print(f"  ├─ Robot {i + 1}:        {rate:5.1f}%")

    print(f"\n[Training Configuration]")
    print(f"  ├─ Stage 1 Episodes: {STAGE1_EPISODES}")
    print(f"  ├─ Stage 2 Episodes: {STAGE2_EPISODES}")
    print(f"  ├─ Max Steps:        120")
    print(f"  └─ Total Episodes:   {len(curve_df)}")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print("🎨 REINFORCEMENT LEARNING TRAINING VISUALIZATION")
    print("=" * 70)

    try:
        data_dir = find_latest_training_data()
        curve_df, robot_rewards_df, robot_success_df, config_df, summary_df = load_training_data(data_dir)

        output_dir = f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📁 Output Directory: {output_dir}")

        print("\n🎨 Generating Plots...")
        plot_total_reward_curve(curve_df, output_dir)
        plot_individual_rewards(robot_rewards_df, output_dir)
        plot_success_rate_comparison(robot_success_df, output_dir)
        plot_steps_curve(curve_df, output_dir)
        plot_combined_figure(curve_df, output_dir)

        print("\n📊 Generating Additional Analysis Plots...")
        plot_reward_distribution(curve_df, output_dir)
        plot_learning_curve_comparison(curve_df, output_dir)
        plot_performance_summary(robot_success_df, output_dir)

        print_statistics(curve_df, robot_rewards_df, robot_success_df, summary_df)

        print(f"\n✅ All plots saved to: {output_dir}")
        print("\n📖 Usage:")
        print("   python plot.py")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please run training script first to generate data")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()