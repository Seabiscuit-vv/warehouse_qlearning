from new_small_maze import Maze, UNIT
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

# ==================== 训练参数 ====================
STAGE1_EPISODES = 500
STAGE2_EPISODES = 1000
MAX_STEPS = 120

STEP_PENALTY = -1.0
WALL_PENALTY = -10.0
GOAL_REWARD = 500.0

# ==================== 断点恢复设置 ====================
RESUME_TRAINING = True  # 设为True启用断点恢复
RESUME_CHECKPOINT = "checkpoint_stage2_900"  # 断点文件夹名: "checkpoint_stage1_500" 或 "checkpoint_stage2_500" 或 "checkpoint_stage2_750" 或 "checkpoint_stage2_900"


# ===================================================


class UnifiedQLearningTable:
    def __init__(self, actions, robot_id=0):
        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.alpha = 0.12
        self.gamma = 0.94
        self.robot_id = robot_id

    def choose_action(self, observation, epsilon):
        self.check_state_exist(observation)
        if np.random.uniform() < epsilon:
            state_action = self.q_table.loc[observation, :]
            max_q = state_action.max()
            if max_q > 0:
                best_actions = state_action[state_action == max_q].index
                if np.random.uniform() < 0.7:
                    return np.random.choice(best_actions)
            return np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            return np.random.choice(state_action[state_action == state_action.max()].index)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r if s_ == 'terminal' else r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.DataFrame([[0] * len(self.actions)], index=[state], columns=self.q_table.columns)
            self.q_table = pd.concat([self.q_table, new_row])

    def update_alpha(self, success_rate):
        if success_rate > 0.7:
            self.alpha = max(0.03, self.alpha * 0.98)
        elif success_rate < 0.2:
            self.alpha = min(0.15, self.alpha * 1.02)


def get_random_position_simple():
    x = random.randint(2, 19) * UNIT
    y = random.randint(2, 19) * UNIT
    return [x, y, x + 20, y + 20]


def get_position_by_robot(robot_num, episode, total_episodes):
    progress = episode / total_episodes
    x_range = (12, 18) if robot_num == 1 else ((2, 8) if robot_num == 2 else (7, 14))
    if progress > 0.6:
        x_range = (2, 18)
    y_range = (5, 16)
    x = random.randint(x_range[0], x_range[1]) * UNIT
    y = random.randint(y_range[0], y_range[1]) * UNIT
    return [x, y, x + 20, y + 20]


def save_checkpoint(checkpoint_name, RL_list, all_rewards, all_steps, all_stage_markers,
                    robot_individual_rewards, robot_success_history, episode_count):
    """保存断点"""
    os.makedirs(checkpoint_name, exist_ok=True)

    # 保存Q表
    for i, rl in enumerate(RL_list):
        with open(f'{checkpoint_name}/robot_{i + 1}_qtable.pkl', 'wb') as f:
            pickle.dump(rl.q_table, f)

    # 保存训练进度
    checkpoint_df = pd.DataFrame({
        'episode': range(len(all_rewards)),
        'total_reward': all_rewards,
        'steps': all_steps,
        'stage': all_stage_markers
    })
    checkpoint_df.to_csv(f'{checkpoint_name}/checkpoint_data.csv', index=False)

    # 保存独立数据
    robot_rewards_df = pd.DataFrame(robot_individual_rewards,
                                    columns=['Robot1_Reward', 'Robot2_Reward', 'Robot3_Reward'])
    robot_rewards_df.to_csv(f'{checkpoint_name}/robot_individual_rewards.csv', index=False)

    robot_success_df = pd.DataFrame(robot_success_history,
                                    columns=['Robot1_Success', 'Robot2_Success', 'Robot3_Success'])
    robot_success_df.to_csv(f'{checkpoint_name}/robot_success_history.csv', index=False)

    # 保存当前episode计数和alpha值
    meta_data = {
        'episode_count': episode_count,
        'alpha_r1': RL_list[0].alpha,
        'alpha_r2': RL_list[1].alpha,
        'alpha_r3': RL_list[2].alpha
    }
    meta_df = pd.DataFrame([meta_data])
    meta_df.to_csv(f'{checkpoint_name}/checkpoint_meta.csv', index=False)

    print(f" 检查点已保存: {checkpoint_name} (Episode {episode_count})")


def load_checkpoint(checkpoint_name, RL_list):
    """加载断点"""
    # 加载Q表
    for i in range(3):
        with open(f'{checkpoint_name}/robot_{i + 1}_qtable.pkl', 'rb') as f:
            RL_list[i].q_table = pickle.load(f)

    # 加载训练进度
    checkpoint_df = pd.read_csv(f'{checkpoint_name}/checkpoint_data.csv')
    all_rewards = checkpoint_df['total_reward'].tolist()
    all_steps = checkpoint_df['steps'].tolist()
    all_stage_markers = checkpoint_df['stage'].tolist()

    # 加载独立数据
    robot_rewards_df = pd.read_csv(f'{checkpoint_name}/robot_individual_rewards.csv')
    robot_individual_rewards = robot_rewards_df.values.tolist()

    robot_success_df = pd.read_csv(f'{checkpoint_name}/robot_success_history.csv')
    robot_success_history = robot_success_df.values.tolist()

    # 加载元数据
    meta_df = pd.read_csv(f'{checkpoint_name}/checkpoint_meta.csv')
    episode_count = int(meta_df['episode_count'].values[0])

    print(f" 从断点恢复: {checkpoint_name} (已完成 {episode_count} 轮)")
    print(f"   - Robot1 alpha: {RL_list[0].alpha}")
    print(f"   - Robot2 alpha: {RL_list[1].alpha}")
    print(f"   - Robot3 alpha: {RL_list[2].alpha}")

    return RL_list, all_rewards, all_steps, all_stage_markers, robot_individual_rewards, robot_success_history, episode_count


def save_training_data(all_rewards, all_steps, all_stage_markers,
                       robot_individual_rewards, robot_success_history,
                       RL_list, timestamp):
    """保存所有训练数据到文件"""
    save_dir = f'training_data_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存训练曲线数据
    training_df = pd.DataFrame({
        'episode': range(len(all_rewards)),
        'total_reward': all_rewards,
        'steps': all_steps,
        'stage': all_stage_markers
    })
    training_df.to_csv(f'{save_dir}/training_curve_data.csv', index=False)

    # 2. 保存每个机器人的Q表
    for i, rl in enumerate(RL_list):
        q_table_df = rl.q_table.copy()
        q_table_df.to_csv(f'{save_dir}/robot_{i + 1}_qtable.csv')
        with open(f'{save_dir}/robot_{i + 1}_qtable.pkl', 'wb') as f:
            pickle.dump(rl.q_table, f)

    # 3. 保存每个机器人的独立奖励
    robot_rewards_df = pd.DataFrame(robot_individual_rewards,
                                    columns=['Robot1_Reward', 'Robot2_Reward', 'Robot3_Reward'])
    robot_rewards_df.to_csv(f'{save_dir}/robot_individual_rewards.csv', index=False)

    # 4. 保存每个机器人的成功率历史
    robot_success_df = pd.DataFrame(robot_success_history,
                                    columns=['Robot1_Success', 'Robot2_Success', 'Robot3_Success'])
    robot_success_df.to_csv(f'{save_dir}/robot_success_history.csv', index=False)

    # 5. 保存训练配置
    config_data = {
        'stage1_episodes': STAGE1_EPISODES,
        'stage2_episodes': STAGE2_EPISODES,
        'max_steps': MAX_STEPS,
        'step_penalty': STEP_PENALTY,
        'wall_penalty': WALL_PENALTY,
        'goal_reward': GOAL_REWARD,
        'total_episodes': len(all_rewards),
        'timestamp': timestamp,
        'final_alpha': [rl.alpha for rl in RL_list],
        'final_qtable_shapes': [rl.q_table.shape for rl in RL_list]
    }
    config_df = pd.DataFrame([config_data])
    config_df.to_csv(f'{save_dir}/training_config.csv', index=False)

    # 6. 保存训练统计摘要
    summary_stats = {
        'robot': [],
        'q_table_size': [],
        'mean_reward_stage1': [],
        'mean_reward_stage2': [],
        'mean_steps_stage1': [],
        'mean_steps_stage2': [],
        'final_success_rate': []
    }

    for i in range(len(RL_list)):
        summary_stats['robot'].append(f'Robot_{i + 1}')
        summary_stats['q_table_size'].append(len(RL_list[i].q_table))

        stage1_mask = [m == 1 for m in all_stage_markers]
        stage2_mask = [m == 2 for m in all_stage_markers]

        stage1_rewards = [all_rewards[j] for j in range(len(all_rewards)) if stage1_mask[j]]
        stage2_rewards = [all_rewards[j] for j in range(len(all_rewards)) if stage2_mask[j]]
        stage1_steps = [all_steps[j] for j in range(len(all_steps)) if stage1_mask[j]]
        stage2_steps = [all_steps[j] for j in range(len(all_steps)) if stage2_mask[j]]

        summary_stats['mean_reward_stage1'].append(np.mean(stage1_rewards) if stage1_rewards else 0)
        summary_stats['mean_reward_stage2'].append(np.mean(stage2_rewards) if stage2_rewards else 0)
        summary_stats['mean_steps_stage1'].append(np.mean(stage1_steps) if stage1_steps else 0)
        summary_stats['mean_steps_stage2'].append(np.mean(stage2_steps) if stage2_steps else 0)

        # 计算最终成功率
        success_history = np.array(robot_success_history)[:, i]
        final_success_rate = np.mean(success_history[-100:]) if len(success_history) >= 100 else np.mean(
            success_history)
        summary_stats['final_success_rate'].append(final_success_rate)

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f'{save_dir}/training_summary.csv', index=False)

    print(f" 训练数据已保存到目录: {save_dir}")
    print(f"   - 训练曲线数据: training_curve_data.csv")
    print(f"   - 各机器人独立奖励: robot_individual_rewards.csv")
    print(f"   - 各机器人成功率历史: robot_success_history.csv")
    print(f"   - 机器人1-3 Q表: robot_[1-3]_qtable.csv/.pkl")
    print(f"   - 训练配置: training_config.csv")
    print(f"   - 训练统计摘要: training_summary.csv")

    return save_dir


def plot_comprehensive_results(save_dir, all_rewards, all_steps, all_stage_markers,
                               robot_individual_rewards, robot_success_history):
    """绘制所有图表"""

    # 1. 总奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, color='gray', alpha=0.3, label='Raw Reward')
    if len(all_rewards) >= 50:
        plt.plot(pd.Series(all_rewards).rolling(50).mean(), color='blue', linewidth=2, label='Moving Avg (50)')
    plt.axvline(x=STAGE1_EPISODES, color='red', linestyle='--', label='Stage Boundary')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve - Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/total_reward_curve.png', dpi=150)
    plt.close()

    # 2. 各机器人独立奖励对比
    robot_rewards = np.array(robot_individual_rewards)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#C3B1E8', '#9B8FD9', '#7B6FB5']
    titles = ['Robot 1 (Far Right)', 'Robot 2 (Near Left)', 'Robot 3 (Middle)']

    for i in range(3):
        axes[i].plot(robot_rewards[:, i], color=colors[i], alpha=0.3)
        if len(robot_rewards) >= 50:
            axes[i].plot(pd.Series(robot_rewards[:, i]).rolling(50).mean(),
                         color=colors[i], linewidth=2)
        axes[i].axvline(x=STAGE1_EPISODES, color='red', linestyle='--')
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Reward')
        axes[i].set_title(titles[i])
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/individual_rewards.png', dpi=150)
    plt.close()

    # 3. 各机器人成功率对比
    success_history = np.array(robot_success_history)
    plt.figure(figsize=(12, 6))
    window = 50
    for i in range(3):
        smooth = pd.Series(success_history[:, i]).rolling(window, min_periods=1).mean()
        plt.plot(smooth, color=colors[i], linewidth=2, label=titles[i])
    plt.axvline(x=STAGE1_EPISODES, color='red', linestyle='--', label='Stage Boundary')
    plt.xlabel('Episode')
    plt.ylabel(f'Success Rate (Rolling {window})')
    plt.title('Robot Success Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(f'{save_dir}/success_rate_comparison.png', dpi=150)
    plt.close()

    # 4. 步数曲线
    plt.figure(figsize=(12, 6))
    plt.plot(all_steps, color='green', alpha=0.3, label='Steps per Episode')
    if len(all_steps) >= 50:
        plt.plot(pd.Series(all_steps).rolling(50).mean(), color='darkgreen', linewidth=2, label='Moving Avg')
    plt.axvline(x=STAGE1_EPISODES, color='red', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/steps_curve.png', dpi=150)
    plt.close()

    print(f" 所有图表已保存到: {save_dir}")


# ==================== Stage 1: 去程盲探建立基准 ====================
def stage1_exploration(RL_list, env, all_rewards, all_steps, all_stage_markers,
                       robot_individual_rewards, robot_success_history, start_episode=0):
    print(" Stage 1: 盲探启动中...")
    env.set_stage_speed(1)

    stage1_success_count = 0
    stage1_reward_history = []
    stage1_step_history = []

    for episode in range(start_episode, STAGE1_EPISODES):
        positions = [get_random_position_simple() for _ in range(3)]
        for i, pos in enumerate(positions):
            env.canvas.coords([env.rect1, env.rect2, env.rect3][i], *pos)

        obs = positions.copy()
        env.resetHuman()
        ep_rewards = [0, 0, 0]
        step_count = 0
        arrived = [False, False, False]

        while step_count < MAX_STEPS:
            step_count += 1
            env.render(episode, sum(ep_rewards), step_count, "STAGE1")
            epsilon = max(0.2, 0.95 - episode / STAGE1_EPISODES * 0.7)

            for i in range(3):
                if arrived[i]: continue
                state_str = f"{obs[i]}_False"
                action = RL_list[i].choose_action(state_str, epsilon)

                next_obs, done_type = env.step_unified_v2(i + 1, action, 'target')

                reward = STEP_PENALTY
                if done_type == 'hit':
                    reward += WALL_PENALTY
                elif done_type == 'arrive':
                    reward += GOAL_REWARD
                    arrived[i] = True

                next_state_str = 'terminal' if done_type == 'arrive' else f"{next_obs}_False"
                RL_list[i].learn(state_str, action, reward, next_state_str)
                ep_rewards[i] += reward
                if next_obs != 'terminal': obs[i] = next_obs

            if all(arrived): break

        total_reward = sum(ep_rewards)
        all_rewards.append(total_reward)
        all_steps.append(step_count)
        all_stage_markers.append(1)
        robot_individual_rewards.append(ep_rewards.copy())
        robot_success_history.append([1 if arrived[i] else 0 for i in range(3)])

        if any(arrived):
            stage1_success_count += 1

        stage1_reward_history.append(total_reward)
        stage1_step_history.append(step_count)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(stage1_reward_history[-50:])
            avg_steps = np.mean(stage1_step_history[-50:])
            success_rate = (stage1_success_count / (episode + 1)) * 100
            print(f"\n Stage 1 - Episode {episode + 1}/{STAGE1_EPISODES}")
            print(f"   ├─ 最近50轮平均奖励: {avg_reward:.2f}")
            print(f"   ├─ 最近50轮平均步数: {avg_steps:.2f}")
            print(f"   ├─ 总体成功率: {success_rate:.1f}%")
            print(f"   └─ 本轮奖励: {total_reward:.2f} | 步数: {step_count}")

    # 自动保存断点1
    save_checkpoint("checkpoint_stage1_500", RL_list, all_rewards, all_steps, all_stage_markers,
                    robot_individual_rewards, robot_success_history, STAGE1_EPISODES)

    print(f"\n Stage 1 完成！总体成功率: {(stage1_success_count / STAGE1_EPISODES) * 100:.1f}%")
    return RL_list


# ==================== Stage 2: 全双向闭环大协同 ====================
def stage2_complete_loop(RL_list, env, all_rewards, all_steps, all_stage_markers,
                         robot_individual_rewards, robot_success_history, start_episode=0):
    print(" Stage 2: 启动双向大闭合训练（曼哈顿优先路权体系启动）...")
    env.set_stage_speed(2)
    success_history = [[0] * 100 for _ in range(3)]

    stage2_success_count = 0
    stage2_reward_history = []
    stage2_step_history = []
    robot_complete_count = [0, 0, 0]

    for episode in range(start_episode, STAGE2_EPISODES):
        positions = [get_position_by_robot(i + 1, episode, STAGE2_EPISODES) for i in range(3)]
        for i, pos in enumerate(positions):
            env.canvas.coords([env.rect1, env.rect2, env.rect3][i], *pos)

        obs = positions.copy()
        env.resetHuman()

        has_cargo = [False, False, False]
        arrived_home = [False, False, False]
        wait_counters = [0, 0, 0]
        last_positions = obs.copy()

        ep_rewards = [0, 0, 0]
        step_count = 0
        epsilon = max(0.1, 0.85 - episode / STAGE2_EPISODES * 0.75)

        while step_count < MAX_STEPS:
            step_count += 1
            env.render(episode + STAGE1_EPISODES, sum(ep_rewards), step_count, "STAGE2")
            env.move_humans_fixed()

            priorities = []
            for i in range(3):
                if arrived_home[i]:
                    priorities.append((9999, i))
                else:
                    tgt_t = 'origin' if has_cargo[i] else 'target'
                    dist = env.get_manhattan_distance(obs[i], i + 1, tgt_t)
                    priorities.append((dist, i))

            priorities.sort(key=lambda x: x[0])

            for _, i in priorities:
                if arrived_home[i]: continue

                if obs[i] == last_positions[i]:
                    wait_counters[i] += 1
                else:
                    wait_counters[i] = 0
                last_positions[i] = obs[i].copy()

                if wait_counters[i] >= 5:
                    escape_action = random.choice([1, 3])
                    next_obs, _ = env.step_unified_v2(i + 1, escape_action, 'origin' if has_cargo[i] else 'target')
                    if next_obs != 'terminal': obs[i] = next_obs
                    wait_counters[i] = 0
                    continue

                state_str = f"{obs[i]}_{has_cargo[i]}"
                action = RL_list[i].choose_action(state_str, epsilon)

                current_target_type = 'origin' if has_cargo[i] else 'target'
                next_obs, done_type = env.step_unified_v2(i + 1, action, current_target_type)

                reward = STEP_PENALTY
                if done_type in ['hit', 'robot_block', 'human_block']:
                    reward += WALL_PENALTY

                if done_type == 'arrive':
                    reward += GOAL_REWARD
                    if not has_cargo[i]:
                        has_cargo[i] = True
                        next_state_str = f"{next_obs}_True"
                    else:
                        arrived_home[i] = True
                        next_state_str = 'terminal'
                        robot_complete_count[i] += 1
                else:
                    next_state_str = f"{next_obs}_{has_cargo[i]}"

                RL_list[i].learn(state_str, action, reward, next_state_str)
                ep_rewards[i] += reward
                if next_obs != 'terminal': obs[i] = next_obs

            if all(arrived_home): break

        total_reward = sum(ep_rewards)
        all_rewards.append(total_reward)
        all_steps.append(step_count)
        all_stage_markers.append(2)
        robot_individual_rewards.append(ep_rewards.copy())
        robot_success_history.append([1 if arrived_home[i] else 0 for i in range(3)])

        if all(arrived_home):
            stage2_success_count += 1

        stage2_reward_history.append(total_reward)
        stage2_step_history.append(step_count)

        for i in range(3):
            success_history[i].append(1 if arrived_home[i] else 0)
            success_history[i].pop(0)

        # 自动保存断点2（第500轮时，即总第1000轮）
        if (episode + 1) == 500:
            save_checkpoint("checkpoint_stage2_500", RL_list, all_rewards, all_steps, all_stage_markers,
                            robot_individual_rewards, robot_success_history, STAGE1_EPISODES + episode + 1)

        # 自动保存断点3（第750轮时，即总第1250轮）
        if (episode + 1) == 750:
            save_checkpoint("checkpoint_stage2_750", RL_list, all_rewards, all_steps, all_stage_markers,
                            robot_individual_rewards, robot_success_history, STAGE1_EPISODES + episode + 1)

        # 自动保存断点4（第900轮时，即总第1400轮）
        if (episode + 1) == 900:
            save_checkpoint("checkpoint_stage2_900", RL_list, all_rewards, all_steps, all_stage_markers,
                            robot_individual_rewards, robot_success_history, STAGE1_EPISODES + episode + 1)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(stage2_reward_history[-50:])
            avg_steps = np.mean(stage2_step_history[-50:])
            success_rate = (stage2_success_count / (episode + 1)) * 100
            robot_success_rates = [(robot_complete_count[i] / (episode + 1)) * 100 for i in range(3)]

            print(f"\n Stage 2 - Episode {episode + 1}/{STAGE2_EPISODES}")
            print(f"   ├─ 最近50轮平均奖励: {avg_reward:.2f}")
            print(f"   ├─ 最近50轮平均步数: {avg_steps:.2f}")
            print(f"   ├─ 本轮总奖励: {total_reward:.2f} | 步数: {step_count}")
            print(f"   ├─ 完整成功率（三机全回）: {success_rate:.1f}%")
            print(
                f"   ├─ 各机器人成功率: R1={robot_success_rates[0]:.1f}%, R2={robot_success_rates[1]:.1f}%, R3={robot_success_rates[2]:.1f}%")

            for i in range(3):
                RL_list[i].update_alpha(sum(success_history[i]) / 100)

    print(f"\n Stage 2 完成！")
    print(f"   ├─ 完整成功率（三机全回）: {(stage2_success_count / STAGE2_EPISODES) * 100:.1f}%")
    print(f"   ├─ 各机器人成功率: R1={(robot_complete_count[0] / STAGE2_EPISODES) * 100:.1f}%, "
          f"R2={(robot_complete_count[1] / STAGE2_EPISODES) * 100:.1f}%, "
          f"R3={(robot_complete_count[2] / STAGE2_EPISODES) * 100:.1f}%")

    return RL_list


if __name__ == "__main__":
    env = Maze()
    RL_list = [UnifiedQLearningTable(list(range(env.n_actions)), robot_id=i) for i in range(3)]

    all_rewards, all_steps, all_stage_markers = [], [], []
    robot_individual_rewards = []
    robot_success_history = []

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(" 双阶段知识迁移闭环控制系统启动")
    print(f" 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stage 1 轮数: {STAGE1_EPISODES}")
    print(f"  Stage 2 轮数: {STAGE2_EPISODES}")
    print(f" 断点恢复: {'开启' if RESUME_TRAINING else '关闭'}")
    if RESUME_TRAINING:
        print(f" 断点路径: {RESUME_CHECKPOINT}")
    print("=" * 80)

    # 断点恢复逻辑
    start_episode_stage1 = 0
    start_episode_stage2 = 0

    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        RL_list, all_rewards, all_steps, all_stage_markers, robot_individual_rewards, robot_success_history, completed_episodes = load_checkpoint(
            RESUME_CHECKPOINT, RL_list)

        if RESUME_CHECKPOINT == "checkpoint_stage1_500":
            start_episode_stage1 = STAGE1_EPISODES  # Stage1已完成
            start_episode_stage2 = 0
            print(" 从 Stage 1 结束点恢复，开始 Stage 2 训练")
        elif RESUME_CHECKPOINT == "checkpoint_stage2_500":
            start_episode_stage1 = STAGE1_EPISODES  # Stage1已完成
            start_episode_stage2 = 500  # Stage2已完成500轮
            print(" 从 Stage 2 的500轮恢复，继续 Stage 2 训练")
        elif RESUME_CHECKPOINT == "checkpoint_stage2_750":
            start_episode_stage1 = STAGE1_EPISODES  # Stage1已完成
            start_episode_stage2 = 750  # Stage2已完成750轮
            print(" 从 Stage 2 的750轮恢复，继续 Stage 2 训练")
        elif RESUME_CHECKPOINT == "checkpoint_stage2_900":
            start_episode_stage1 = STAGE1_EPISODES  # Stage1已完成
            start_episode_stage2 = 900  # Stage2已完成900轮
            print(" 从 Stage 2 的900轮恢复，继续 Stage 2 训练")

    # 运行训练
    if not RESUME_TRAINING or start_episode_stage1 == 0:
        RL_list = stage1_exploration(RL_list, env, all_rewards, all_steps, all_stage_markers,
                                     robot_individual_rewards, robot_success_history, 0)

    RL_list = stage2_complete_loop(RL_list, env, all_rewards, all_steps, all_stage_markers,
                                   robot_individual_rewards, robot_success_history, start_episode_stage2)

    # 保存最终训练数据
    save_dir = save_training_data(all_rewards, all_steps, all_stage_markers,
                                  robot_individual_rewards, robot_success_history,
                                  RL_list, timestamp)

    # 绘制图表
    plot_comprehensive_results(save_dir, all_rewards, all_steps, all_stage_markers,
                               robot_individual_rewards, robot_success_history)

    # 额外保存曲线到当前目录
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, color='gray', alpha=0.3, label='Raw Reward')
    if len(all_rewards) >= 50:
        plt.plot(pd.Series(all_rewards).rolling(50).mean(), color='blue', linewidth=2, label='Moving Avg (50)')
    plt.axvline(x=STAGE1_EPISODES, color='red', linestyle='--')
    plt.title("Two-Stage Full Closed-Loop Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('full_training_curve.png', dpi=150)

    # 显示训练总结
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print(" 任务圆满完成，组合Q表训练完毕！")
    print(f" 总训练回合数: {len(all_rewards)}")
    print(f"  训练总耗时: {duration}")
    print(f" 所有训练数据已保存到: {save_dir}")
    print("=" * 60)

    env.destroy()