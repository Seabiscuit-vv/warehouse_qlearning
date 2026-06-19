#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从断点继续训练并生成路径效果图
使用方法: python path.py
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from new_small_maze import Maze, UNIT, X_Block, Y_Block
import time
import os
import random

# ==================== 配置 ====================
CHECKPOINT_DIR = "checkpoint_stage2_900"  # 从哪个断点继续
CONTINUE_EPISODES = 50  # 继续训练50轮
SAVE_NEW_CHECKPOINT = True  # 是否保存新断点

# 动作空间（5个动作：上、下、右、左、等待）
ACTIONS = [0, 1, 2, 3, 4]

# 颜色定义
COLORS = {
    'robot1': '#C3B1E8',
    'robot2': '#9B8FD9',
    'robot3': '#7B6FB5',
    'shelf': '#C8C5C0',
    'operation_desk': '#F0B8C8',
    'origin_marker': '#D3D3D3',
    'target1': '#F0C49A',
    'target2': '#E8AE7A',
    'target3': '#D9985E',
    'human': '#A8CFE8',
}

TARGETS = {1: (380, 200), 2: (40, 200), 3: (160, 200)}
ORIGINS = {1: (70, 50), 2: (210, 50), 3: (350, 50)}

OPERATION_DESKS = [
    (20, 0, 120, 40), (160, 0, 260, 40), (300, 0, 400, 40),
]


class QLearningTable:
    def __init__(self, actions, q_table=None):
        self.actions = actions  # actions = [0,1,2,3,4]
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.alpha = 0.08
        self.gamma = 0.94

    def choose_action(self, observation, epsilon):
        self.check_state_exist(observation)
        if np.random.uniform() < epsilon:
            return np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            max_q = state_action.max()
            best_actions = state_action[state_action == max_q].index
            return np.random.choice(best_actions)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r if s_ == 'terminal' else r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 使用 self.actions 作为列名，确保列数一致
            new_row = pd.DataFrame([[0] * len(self.actions)], index=[state], columns=self.actions)
            self.q_table = pd.concat([self.q_table, new_row])


def load_qtable_from_checkpoint(checkpoint_dir):
    """从断点加载Q表并包装成QLearningTable对象"""
    RL_list = []
    for i in range(3):
        q_table_path = f'{checkpoint_dir}/robot_{i + 1}_qtable.pkl'
        if os.path.exists(q_table_path):
            with open(q_table_path, 'rb') as f:
                q_table = pickle.load(f)
            print(f"✅ 加载 Robot{i + 1} Q表成功，状态数: {len(q_table)}")
            # 使用 ACTIONS 确保动作空间一致
            rl = QLearningTable(ACTIONS, q_table)
            RL_list.append(rl)
        else:
            print(f"⚠️ 未找到 Robot{i + 1} Q表: {q_table_path}")
            RL_list.append(QLearningTable(ACTIONS))
    return RL_list


def continue_training(env, RL_list, episodes=50):
    """继续训练"""
    print(f"\n🔄 继续训练 {episodes} 轮...")

    # 必须调用此方法初始化 render_speed 属性
    env.set_stage_speed(2)

    for episode in range(episodes):
        # 随机起点
        positions = []
        for i in range(3):
            x = random.randint(5, 15) * UNIT
            y = random.randint(5, 15) * UNIT
            positions.append([x, y, x + 20, y + 20])

        for i, pos in enumerate(positions):
            env.canvas.coords([env.rect1, env.rect2, env.rect3][i], *pos)

        obs = positions.copy()
        env.resetHuman()

        has_cargo = [False, False, False]
        arrived_home = [False, False, False]

        step_count = 0
        epsilon = 0.1

        while step_count < 120 and not all(arrived_home):
            step_count += 1
            env.render(episode, 0, step_count, "TRAINING")
            env.move_humans_fixed()

            for i in range(3):
                if arrived_home[i]:
                    continue

                state_str = f"{obs[i]}_{has_cargo[i]}"
                action = RL_list[i].choose_action(state_str, epsilon)

                current_target = 'origin' if has_cargo[i] else 'target'
                next_obs, done_type = env.step_unified_v2(i + 1, action, current_target)

                reward = -1.0
                if done_type in ['hit', 'robot_block', 'human_block']:
                    reward -= 10.0

                if done_type == 'arrive':
                    reward += 500.0
                    if not has_cargo[i]:
                        has_cargo[i] = True
                    else:
                        arrived_home[i] = True

                if done_type == 'arrive' and has_cargo[i]:
                    next_state = 'terminal'
                else:
                    next_state = f"{next_obs}_{has_cargo[i]}"

                RL_list[i].learn(state_str, action, reward, next_state)

                if next_obs != 'terminal':
                    obs[i] = next_obs

        if (episode + 1) % 10 == 0:
            print(f"   完成 {episode + 1}/{episodes} 轮")

    print(f"✅ 继续训练完成")
    return RL_list


def save_checkpoint(RL_list, checkpoint_name):
    """保存新断点"""
    os.makedirs(checkpoint_name, exist_ok=True)
    for i, rl in enumerate(RL_list):
        with open(f'{checkpoint_name}/robot_{i + 1}_qtable.pkl', 'wb') as f:
            pickle.dump(rl.q_table, f)
    print(f"💾 新断点已保存: {checkpoint_name}")


def run_demo_with_trajectory(env, RL_list):
    """运行演示并记录轨迹"""
    print("\n🎮 运行演示并记录轨迹...")

    # 必须调用此方法初始化 render_speed 属性
    env.set_stage_speed(2)

    trajectories = {1: {'forward': [], 'backward': []},
                    2: {'forward': [], 'backward': []},
                    3: {'forward': [], 'backward': []}}
    start_positions = {}
    cargo_positions = {}
    home_positions = {}

    # 固定起点（从原点出发，便于论文截图）
    fixed_starts = [
        [70, 50, 90, 70],
        [210, 50, 230, 70],
        [350, 50, 370, 70]
    ]

    for i, pos in enumerate(fixed_starts):
        env.canvas.coords([env.rect1, env.rect2, env.rect3][i], *pos)
        start_positions[i + 1] = (pos[0], pos[1])
        print(f"   Robot{i + 1} 起点: ({pos[0]}, {pos[1]})")

    env.resetHuman()
    obs = fixed_starts.copy()

    has_cargo = [False, False, False]
    arrived_home = [False, False, False]

    step_count = 0
    ep_rewards = [0, 0, 0]

    print("\n📦 机器人开始执行任务...")
    print("   Robot1(紫色) -> 右下角目标 -> 返回原点")
    print("   Robot2(蓝色) -> 左下角目标 -> 返回原点")
    print("   Robot3(深紫) -> 中间目标 -> 返回原点")

    while step_count < 200 and not all(arrived_home):
        step_count += 1
        env.render(0, sum(ep_rewards), step_count, "DEMO")
        env.move_humans_fixed()

        for i in range(3):
            if arrived_home[i]:
                continue

            # 记录轨迹
            current_pos = (obs[i][0], obs[i][1])
            if not has_cargo[i]:
                trajectories[i + 1]['forward'].append(current_pos)
            else:
                trajectories[i + 1]['backward'].append(current_pos)

            state_str = f"{obs[i]}_{has_cargo[i]}"
            action = RL_list[i].choose_action(state_str, epsilon=0.0)

            current_target = 'origin' if has_cargo[i] else 'target'
            next_obs, done_type = env.step_unified_v2(i + 1, action, current_target)

            reward = -1.0
            if done_type in ['hit', 'robot_block', 'human_block']:
                reward -= 10.0

            if done_type == 'arrive':
                reward += 500.0
                if not has_cargo[i]:
                    has_cargo[i] = True
                    cargo_positions[i + 1] = (obs[i][0], obs[i][1])
                    print(f"   ✅ Robot{i + 1} 取货成功！位置: ({obs[i][0]}, {obs[i][1]})")
                else:
                    arrived_home[i] = True
                    home_positions[i + 1] = (obs[i][0], obs[i][1])
                    print(f"   🏁 Robot{i + 1} 返航成功！位置: ({obs[i][0]}, {obs[i][1]})")

            ep_rewards[i] += reward
            if next_obs != 'terminal':
                obs[i] = next_obs

        if step_count % 50 == 0:
            status = [f"R{i + 1}:{'📦' if has_cargo[i] else '🔍'}{'🏠' if arrived_home[i] else '➡'}"
                      for i in range(3)]
            print(f"   步数 {step_count}: {'  '.join(status)}")

    success_count = sum(arrived_home)
    print(f"\n📊 演示结果: {success_count}/3 机器人成功完成任务")
    print(f"   总步数: {step_count}")
    print(f"   各机器人奖励: R1={ep_rewards[0]:.0f}, R2={ep_rewards[1]:.0f}, R3={ep_rewards[2]:.0f}")

    return trajectories, start_positions, cargo_positions, home_positions


def draw_curved_arrow(ax, start, end, color, linestyle='-', alpha=0.7):
    """绘制曲线箭头（不使用scipy）"""
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2 + 30

    t = np.linspace(0, 1, 50)
    x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_x + t ** 2 * end[0]
    y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_y + t ** 2 * end[1]

    ax.plot(x, y, color=color, linewidth=2, linestyle=linestyle, alpha=alpha)

    if len(x) > 10:
        ax.annotate('', xy=(x[-5], y[-5]), xytext=(x[-15], y[-15]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=alpha))


def plot_path_illustration(save_dir='.'):
    """绘制路径示意图（与弹窗方向一致）"""
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.set_xlim(0, 420)
    ax.set_ylim(420, 0)
    ax.set_aspect('equal')

    # 网格
    for i in range(0, 421, 20):
        ax.axhline(y=i, color='#E8E0D0', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i, color='#E8E0D0', linewidth=0.5, alpha=0.5)

    # 货架
    for x in X_Block:
        for y in Y_Block:
            rect = plt.Rectangle((x, y), 20, 20, facecolor=COLORS['shelf'],
                                 edgecolor='#A0A0A0', linewidth=0.5, alpha=0.9)
            ax.add_patch(rect)

    # 操作台
    for (x1, y1, x2, y2) in OPERATION_DESKS:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             facecolor=COLORS['operation_desk'], edgecolor='#D0A0A0', linewidth=0.5)
        ax.add_patch(rect)

    # 原点标记
    origins = [(70, 50), (210, 50), (350, 50)]
    origin_labels = ['Robot 1\nStart', 'Robot 2\nStart', 'Robot 3\nStart']
    origin_colors = ['#C3B1E8', '#9B8FD9', '#7B6FB5']
    for i, ((x, y), label, color) in enumerate(zip(origins, origin_labels, origin_colors)):
        rect = plt.Rectangle((x - 10, y - 10), 20, 20, facecolor=COLORS['origin_marker'],
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        robot_rect = plt.Rectangle((x - 8, y - 8), 16, 16, facecolor=color, alpha=0.7)
        ax.add_patch(robot_rect)
        ax.text(x, y - 15, label, ha='center', fontsize=8, fontweight='bold', color=color)

    # 目标货架
    targets = [(380, 200), (40, 200), (160, 200)]
    target_colors = [COLORS['target1'], COLORS['target2'], COLORS['target3']]
    target_labels = ['Robot 1\nTarget', 'Robot 2\nTarget', 'Robot 3\nTarget']
    for i, ((x, y), color, label) in enumerate(zip(targets, target_colors, target_labels)):
        rect = plt.Rectangle((x, y), 20, 20, facecolor=color, edgecolor='#B08060', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 10, y + 25, label, ha='center', fontsize=8, fontweight='bold', color='#8B5A2B')

    # 动态行人
    human1 = plt.Rectangle((140, 0), 20, 20, facecolor=COLORS['human'], edgecolor='#80A0C0', linewidth=1)
    human2 = plt.Rectangle((260, 0), 20, 20, facecolor=COLORS['human'], edgecolor='#80A0C0', linewidth=1)
    ax.add_patch(human1)
    ax.add_patch(human2)
    ax.text(150, 30, 'Dynamic\nHuman 1', ha='center', fontsize=7, color='#5A7A9A')
    ax.text(270, 30, 'Dynamic\nHuman 2', ha='center', fontsize=7, color='#5A7A9A')

    # 路径箭头
    draw_curved_arrow(ax, (70, 50), (380, 200), '#C3B1E8', '-', 0.7)
    draw_curved_arrow(ax, (380, 200), (70, 50), '#C3B1E8', '--', 0.7)
    draw_curved_arrow(ax, (210, 50), (40, 200), '#9B8FD9', '-', 0.7)
    draw_curved_arrow(ax, (40, 200), (210, 50), '#9B8FD9', '--', 0.7)
    draw_curved_arrow(ax, (350, 50), (160, 200), '#7B6FB5', '-', 0.7)
    draw_curved_arrow(ax, (160, 200), (350, 50), '#7B6FB5', '--', 0.7)

    legend_elements = [
        Patch(facecolor='#C3B1E8', label='Robot 1 Path'),
        Patch(facecolor='#9B8FD9', label='Robot 2 Path'),
        Patch(facecolor='#7B6FB5', label='Robot 3 Path'),
        Patch(facecolor=COLORS['shelf'], label='Static Shelf'),
        Patch(facecolor=COLORS['operation_desk'], label='Operation Desk'),
        Patch(facecolor=COLORS['origin_marker'], label='Start Position'),
        Patch(facecolor=COLORS['target1'], label='Target Location'),
        Patch(facecolor=COLORS['human'], label='Dynamic Human'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

    ax.set_title('Multi-Robot Warehouse Path Planning Illustration', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/path_illustration.png', dpi=300, bbox_inches='tight')
    print(f"✅ 路径示意图已保存: path_illustration.png")
    plt.close()


def plot_final_path_result(trajectories, start_positions, cargo_positions, home_positions, save_dir='.'):
    """绘制最终路径结果图"""
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.set_xlim(0, 420)
    ax.set_ylim(420, 0)
    ax.set_aspect('equal')

    # 网格
    for i in range(0, 421, 20):
        ax.axhline(y=i, color='#E8E0D0', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i, color='#E8E0D0', linewidth=0.5, alpha=0.5)

    # 货架
    for x in X_Block:
        for y in Y_Block:
            rect = plt.Rectangle((x, y), 20, 20, facecolor=COLORS['shelf'],
                                 edgecolor='#A0A0A0', linewidth=0.5, alpha=0.9)
            ax.add_patch(rect)

    # 操作台
    for (x1, y1, x2, y2) in OPERATION_DESKS:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             facecolor=COLORS['operation_desk'], edgecolor='#D0A0A0', linewidth=0.5)
        ax.add_patch(rect)

    # 目标货架
    for i, (x, y) in TARGETS.items():
        color = [COLORS['target1'], COLORS['target2'], COLORS['target3']][i - 1]
        rect = plt.Rectangle((x, y), 20, 20, facecolor=color, edgecolor='#B08060', linewidth=1.5)
        ax.add_patch(rect)

    # 原点标记
    for i, (x, y) in ORIGINS.items():
        rect = plt.Rectangle((x - 10, y - 10), 20, 20, facecolor=COLORS['origin_marker'],
                             edgecolor='#B0B0B0', linewidth=1)
        ax.add_patch(rect)

    # 动态行人
    ax.add_patch(plt.Rectangle((140, 0), 20, 20, facecolor=COLORS['human'], edgecolor='#80A0C0', linewidth=1))
    ax.add_patch(plt.Rectangle((260, 0), 20, 20, facecolor=COLORS['human'], edgecolor='#80A0C0', linewidth=1))

    robot_colors = {1: '#C3B1E8', 2: '#9B8FD9', 3: '#7B6FB5'}
    robot_names = {1: 'Robot 1', 2: 'Robot 2', 3: 'Robot 3'}

    # 绘制轨迹
    for robot_id in [1, 2, 3]:
        color = robot_colors[robot_id]

        forward = trajectories[robot_id]['forward']
        if len(forward) > 1:
            forward_x = [p[0] + 10 for p in forward]
            forward_y = [p[1] + 10 for p in forward]
            ax.plot(forward_x, forward_y, color=color, linewidth=2, alpha=0.8,
                    label=f'{robot_names[robot_id]} Forward Path')

            # 添加方向箭头
            step = max(1, len(forward) // 20)
            for idx in range(step, len(forward), step):
                if idx < len(forward_x):
                    prev_x, prev_y = forward_x[idx - step], forward_y[idx - step]
                    curr_x, curr_y = forward_x[idx], forward_y[idx]
                    ax.annotate('', xy=(curr_x, curr_y), xytext=(prev_x, prev_y),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.6))

        backward = trajectories[robot_id]['backward']
        if len(backward) > 1:
            backward_x = [p[0] + 10 for p in backward]
            backward_y = [p[1] + 10 for p in backward]
            ax.plot(backward_x, backward_y, color=color, linewidth=2, linestyle='--', alpha=0.8,
                    label=f'{robot_names[robot_id]} Return Path')

    # 标注关键点
    # 起点标注 (S)
    for robot_id, (x, y) in start_positions.items():
        ax.scatter(x + 10, y + 10, c='white', s=100, marker='o', edgecolors='black', zorder=5, linewidths=2)
        ax.annotate('S', xy=(x + 10, y + 10), ha='center', va='center', fontsize=10, fontweight='bold')

    # 取货点标注 (G)
    for robot_id, (x, y) in cargo_positions.items():
        ax.scatter(x + 10, y + 10, c='gold', s=100, marker='*', edgecolors='black', zorder=5)
        ax.annotate('G', xy=(x + 10, y + 10), ha='center', va='center', fontsize=9, fontweight='bold')

    # 返航点标注 (H)
    for robot_id, (x, y) in home_positions.items():
        ax.scatter(x + 10, y + 10, c='green', s=100, marker='s', edgecolors='black', zorder=5)
        ax.annotate('H', xy=(x + 10, y + 10), ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    legend_elements = [
        Patch(facecolor='#C3B1E8', label='Robot 1 Forward Path'),
        Patch(facecolor='#9B8FD9', label='Robot 2 Forward Path'),
        Patch(facecolor='#7B6FB5', label='Robot 3 Forward Path'),
        Patch(facecolor='gray', alpha=0.5, label='Return Path (Dashed)'),
        Patch(facecolor='gold', label='Cargo Pickup (G)'),
        Patch(facecolor='green', label='Return Home (H)'),
        Patch(facecolor='white', edgecolor='black', label='Start Point (S)'),
        Patch(facecolor=COLORS['shelf'], label='Static Shelf'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

    ax.set_title('Final Path Planning Result - Robot Trajectories', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/final_path_planning_result.png', dpi=300, bbox_inches='tight')
    print(f"✅ 最终路径结果图已保存: final_path_planning_result.png")
    plt.close()


def main():
    print("=" * 60)
    print("🗺️ 从断点继续训练并生成路径效果图")
    print("=" * 60)

    # 1. 加载断点
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ 断点文件夹不存在: {CHECKPOINT_DIR}")
        print("\n可用的断点文件夹:")
        for f in os.listdir('.'):
            if f.startswith('checkpoint_'):
                print(f"   - {f}")
        return

    print(f"\n📂 加载断点: {CHECKPOINT_DIR}")
    RL_list = load_qtable_from_checkpoint(CHECKPOINT_DIR)

    # 2. 创建环境
    env = Maze()

    # 3. 继续训练
    RL_list = continue_training(env, RL_list, CONTINUE_EPISODES)

    # 4. 保存新断点（可选）
    if SAVE_NEW_CHECKPOINT:
        new_checkpoint = f"checkpoint_after_{CONTINUE_EPISODES}"
        save_checkpoint(RL_list, new_checkpoint)

    # 5. 生成路径示意图
    print("\n📊 生成路径示意图...")
    plot_path_illustration('.')

    # 6. 运行演示并记录轨迹
    try:
        trajectories, start_positions, cargo_positions, home_positions = run_demo_with_trajectory(env, RL_list)

        # 7. 绘制最终路径结果图
        print("\n📊 生成最终路径结果图...")
        plot_final_path_result(trajectories, start_positions, cargo_positions, home_positions, '.')

        print("\n✅ 完成！")
        print("   生成的文件:")
        print("   - path_illustration.png (路径示意图)")
        print("   - final_path_planning_result.png (最终路径结果图)")

        input("\n按回车键关闭窗口...")
        env.destroy()

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        env.destroy()


if __name__ == "__main__":
    main()