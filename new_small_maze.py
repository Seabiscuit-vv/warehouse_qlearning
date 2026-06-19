#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Warehouse Environment - Collision Avoidance & Multi-Agent Coordination Edition
"""

import numpy as np
import time
import sys
import random
from datetime import datetime

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20  # pixels
MAZE_H = 21  # grid height
MAZE_W = 21  # grid width

COLORS = {
    'bg': '#FDF8F0', 'grid_line': '#E8E0D0',
    'robot1': '#C3B1E8', 'robot2': '#9B8FD9', 'robot3': '#7B6FB5',
    'target1': '#F0C49A', 'target2': '#E8AE7A', 'target3': '#D9985E',
    'human': '#A8CFE8', 'operation_desk': '#F0B8C8', 'shelf': '#C8C5C0',
    'origin_marker': '#D3D3D3',
    'log_bg': '#FDF8F0', 'log_fg': '#6B5B4F',
    'log_frame_border': '#E8E0D0', 'log_info': '#8B9B8B',
    'log_success': '#A8B88A', 'log_error': '#D4A5A5', 'log_warning': '#E8B88A',
}

# 静态货架网格坐标
X_Block_pic = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
X_Block = [element * UNIT for element in X_Block_pic]
Y_Block_pic = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
Y_Block = [element * UNIT for element in Y_Block_pic]

origin1 = np.array([70, 50])
origin2 = np.array([210, 50])
origin3 = np.array([350, 50])


class LogPanel(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=COLORS['log_bg'], **kwargs)
        self.configure(relief=tk.GROOVE, bd=2, highlightbackground=COLORS['log_frame_border'])
        self.logs = []
        self.max_logs = 150

        self.title_label = tk.Label(self, text="🤖 仓库多智能体监控面板", font=("微软雅黑", 11, "bold"),
                                    bg=COLORS['log_bg'], fg=COLORS['robot2'])
        self.title_label.pack(pady=(10, 5))

        self.stage_frame = tk.Frame(self, bg=COLORS['log_bg'])
        self.stage_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Label(self.stage_frame, text="当前阶段:", font=("微软雅黑", 9, "bold"), bg=COLORS['log_bg'],
                 fg=COLORS['log_fg']).pack(side=tk.LEFT)
        self.stage_label = tk.Label(self.stage_frame, text="--", font=("微软雅黑", 9, "bold"), bg=COLORS['log_bg'],
                                    fg=COLORS['log_success'])
        self.stage_label.pack(side=tk.LEFT, padx=5)

        self.status_frame = tk.Frame(self, bg=COLORS['log_bg'])
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.r1_coord = self._create_coord_row("R1 坐标:", COLORS['robot1'])
        self.r2_coord = self._create_coord_row("R2 坐标:", COLORS['robot2'])
        self.r3_coord = self._create_coord_row("R3 坐标:", COLORS['robot3'])

        tk.Frame(self, height=1, bg=COLORS['log_frame_border']).pack(fill=tk.X, padx=10, pady=5)

        self.text_frame = tk.Frame(self, bg=COLORS['log_bg'])
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.scrollbar = tk.Scrollbar(self.text_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(self.text_frame, font=("Consolas", 9), bg=COLORS['log_bg'], fg=COLORS['log_fg'],
                                wrap=tk.WORD, yscrollcommand=self.scrollbar.set, relief=tk.FLAT, height=18)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.log_text.yview)

        self.log_text.tag_config("info", foreground=COLORS['log_info'])
        self.log_text.tag_config("success", foreground=COLORS['log_success'])
        self.log_text.tag_config("error", foreground=COLORS['log_error'])
        self.log_text.tag_config("warning", foreground=COLORS['log_warning'])

    def _create_coord_row(self, text, color):
        lf = tk.Frame(self.status_frame, bg=COLORS['log_bg'])
        lf.pack(fill=tk.X, pady=1)
        tk.Label(lf, text=text, font=("微软雅黑", 9, "bold"), bg=COLORS['log_bg'], fg=color).pack(side=tk.LEFT)
        lbl = tk.Label(lf, text="--", font=("Consolas", 9), bg=COLORS['log_bg'], fg=COLORS['log_fg'])
        lbl.pack(side=tk.LEFT, padx=5)
        return lbl

    def add_log(self, message, tag="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry, tag)
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()
        self.logs.append(log_entry)
        if len(self.logs) > self.max_logs:
            self.log_text.delete('1.0', '2.0')
            self.logs.pop(0)

    def update_coords(self, r1_s, r2_s, r3_s):
        if r1_s and len(r1_s) >= 2: self.r1_coord.config(text=f"({int(r1_s[0])},{int(r1_s[1])})")
        if r2_s and len(r2_s) >= 2: self.r2_coord.config(text=f"({int(r2_s[0])},{int(r2_s[1])})")
        if r3_s and len(r3_s) >= 2: self.r3_coord.config(text=f"({int(r3_s[0])},{int(r3_s[1])})")

    def set_stage(self, stage_name, stage_color):
        self.stage_label.config(text=stage_name, fg=stage_color)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r', 'w']
        self.n_actions = len(self.action_space)
        self.title('Warehouse Management Navigation')
        self.attributes('-topmost', True)
        self.lift()
        self.focus_force()
        self.after(100, lambda: self.attributes('-topmost', False))
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT + 320, MAZE_H * UNIT + 40))
        self.center_window()
        self.configure(bg=COLORS['bg'])

        self.maze_frame = tk.Frame(self, bg=COLORS['bg'])
        self.maze_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.log_panel = LogPanel(self)
        self.log_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._build_maze()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def set_stage_speed(self, stage):
        if stage == 1:
            self.render_speed = 0.0001
            self.log_panel.set_stage("Stage 1: 地图盲探模式", "#8B9B8B")
        elif stage == 2:
            self.render_speed = 0.05
            self.log_panel.set_stage("Stage 2: 全流程训练阶段", "#D9985E")

    def _build_maze(self):
        self.canvas = tk.Canvas(self.maze_frame, bg=COLORS['bg'], height=MAZE_H * UNIT, width=MAZE_W * UNIT,
                                highlightthickness=0)
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT, fill=COLORS['grid_line'])
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r, fill=COLORS['grid_line'])

        for i in range(1, 16, 7):
            self.canvas.create_rectangle(i * UNIT, 0, (i + 5) * UNIT, 2 * UNIT, fill=COLORS['operation_desk'],
                                         outline='')
        for k in range(1, 17, 5):
            for i in range(5, 18, 3):
                self.canvas.create_rectangle(k * UNIT, i * UNIT, (k + 4) * UNIT, (i + 2) * UNIT, fill=COLORS['shelf'],
                                             outline='')

        self.human1 = self.canvas.create_rectangle(7 * UNIT, 0, 8 * UNIT, 1 * UNIT, fill=COLORS['human'],
                                                   outline='white', width=1)
        self.human2 = self.canvas.create_rectangle(13 * UNIT, 0, 14 * UNIT, 1 * UNIT, fill=COLORS['human'],
                                                   outline='white', width=1)

        self.target1 = self.canvas.create_rectangle(19 * UNIT, 10 * UNIT, 20 * UNIT, 11 * UNIT, fill=COLORS['target1'],
                                                    outline='')
        self.target2 = self.canvas.create_rectangle(2 * UNIT, 10 * UNIT, 3 * UNIT, 11 * UNIT, fill=COLORS['target2'],
                                                    outline='')
        self.target3 = self.canvas.create_rectangle(8 * UNIT, 10 * UNIT, 9 * UNIT, 11 * UNIT, fill=COLORS['target3'],
                                                    outline='')

        self.org1 = self.canvas.create_rectangle(origin1[0] - 10, origin1[1] - 10, origin1[0] + 10, origin1[1] + 10,
                                                 fill=COLORS['origin_marker'], outline='#BEBEBE')
        self.org2 = self.canvas.create_rectangle(origin2[0] - 10, origin2[1] - 10, origin2[0] + 10, origin2[1] + 10,
                                                 fill=COLORS['origin_marker'], outline='#BEBEBE')
        self.org3 = self.canvas.create_rectangle(origin3[0] - 10, origin3[1] - 10, origin3[0] + 10, origin3[1] + 10,
                                                 fill=COLORS['origin_marker'], outline='#BEBEBE')

        self.rect1 = self.canvas.create_rectangle(origin1[0] - 10, origin1[1] - 10, origin1[0] + 10, origin1[1] + 10,
                                                  fill=COLORS['robot1'], outline='')
        self.rect2 = self.canvas.create_rectangle(origin2[0] - 10, origin2[1] - 10, origin2[0] + 10, origin2[1] + 10,
                                                  fill=COLORS['robot2'], outline='')
        self.rect3 = self.canvas.create_rectangle(origin3[0] - 10, origin3[1] - 10, origin3[0] + 10, origin3[1] + 10,
                                                  fill=COLORS['robot3'], outline='')
        self.canvas.pack()

    def resetRobot(self):
        self.update()
        self.canvas.coords(self.rect1, origin1[0] - 10, origin1[1] - 10, origin1[0] + 10, origin1[1] + 10)
        self.canvas.coords(self.rect2, origin2[0] - 10, origin2[1] - 10, origin2[0] + 10, origin2[1] + 10)
        self.canvas.coords(self.rect3, origin3[0] - 10, origin3[1] - 10, origin3[0] + 10, origin3[1] + 10)
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2), self.canvas.coords(self.rect3)

    def resetHuman(self):
        self.update()
        self.canvas.coords(self.human1, 7 * UNIT, 0, 8 * UNIT, 1 * UNIT)
        self.canvas.coords(self.human2, 13 * UNIT, 0, 14 * UNIT, 1 * UNIT)
        return self.canvas.coords(self.human1), self.canvas.coords(self.human2)

    def is_valid_human_position(self, x1, y1, x2, y2):
        """
        检查人类位置是否合法
        与机器人规则对齐，但人类允许在顶部边缘移动
        """
        # 1. 边界检查（x边界与机器人一致，y边界不同）
        if x1 < 0 or x2 > MAZE_W * UNIT or y2 > MAZE_H * UNIT:
            return False

        # y1 不能小于0（不能超出顶部）
        if y1 < 0:
            return False

        # 2. 货架区域检测（与机器人完全一致：检测左上角坐标）
        if int(x1) in X_Block and int(y1) in Y_Block:
            return False

        # 3. 操作台区域检测（人类不能进入操作台矩形内部）
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 检查是否在操作台内部（y在0-40，x在操作台区域）
        if 0 <= center_y <= 40:
            if (20 <= center_x <= 120) or (160 <= center_x <= 260) or (300 <= center_x <= 400):
                return False

        # 4. 不能与机器人原点重叠
        origins_coords = [
            self.canvas.coords(self.org1),
            self.canvas.coords(self.org2),
            self.canvas.coords(self.org3)
        ]
        if [x1, y1, x2, y2] in origins_coords:
            return False

        # 5. 不能与目标货架重叠
        targets_coords = [
            self.canvas.coords(self.target1),
            self.canvas.coords(self.target2),
            self.canvas.coords(self.target3)
        ]
        if [x1, y1, x2, y2] in targets_coords:
            return False

        return True

    def get_random_direction(self):
        """获取随机方向（上下左右）"""
        directions = [(UNIT, 0), (-UNIT, 0), (0, UNIT), (0, -UNIT)]
        random.shuffle(directions)
        return directions

    def try_move_human(self, human_id):
        """尝试移动单个人类（异步随机移动）"""
        # 20%概率移动
        if random.random() > 0.2:
            return

        human = self.human1 if human_id == 1 else self.human2
        current_coords = self.canvas.coords(human)

        # 获取随机方向列表
        directions = self.get_random_direction()

        # 尝试所有方向，找到第一个合法的位置
        for dx, dy in directions:
            new_coords = [
                current_coords[0] + dx,
                current_coords[1] + dy,
                current_coords[2] + dx,
                current_coords[3] + dy
            ]

            # 检查位置是否合法
            if not self.is_valid_human_position(*new_coords):
                continue

            # 检查是否与机器人重合
            robots = [
                self.canvas.coords(self.rect1),
                self.canvas.coords(self.rect2),
                self.canvas.coords(self.rect3)
            ]
            if new_coords in robots:
                continue

            # 检查是否与另一个人类重合
            other_human = self.human2 if human_id == 1 else self.human1
            if new_coords == self.canvas.coords(other_human):
                continue

            # 合法移动
            self.canvas.move(human, dx, dy)
            return

        # 如果所有方向都被阻挡，原地不动

    def move_humans_advanced(self):
        """高级人类移动：异步、全地图随机、完整避障"""
        self.try_move_human(1)
        self.try_move_human(2)

    def move_humans_fixed(self):
        """保留原函数名，调用新的移动策略"""
        self.move_humans_advanced()

    def get_manhattan_distance(self, current_pos, robot_num, target_type):
        """计算曼哈顿距离用于路权分配仲裁"""
        if target_type == 'origin':
            tgt = self.canvas.coords([self.org1, self.org2, self.org3][robot_num - 1])
        else:
            tgt = self.canvas.coords([self.target1, self.target2, self.target3][robot_num - 1])
        return abs(current_pos[0] - tgt[0]) / UNIT + abs(current_pos[1] - tgt[1]) / UNIT

    def step_unified_v2(self, robot_num, action, target_type):
        """
        全功能步进核心：集成多机互撞检测、静态/动态人机避障判断
        """
        rect = [self.rect1, self.rect2, self.rect3][robot_num - 1]
        s = self.canvas.coords(rect)

        base_action = moveAgent(s, action)
        proposed_s_ = [s[0] + base_action[0], s[1] + base_action[1], s[2] + base_action[0], s[3] + base_action[1]]

        # 1. 边界与货架硬体碰撞检测
        if (proposed_s_[0] < 0 or proposed_s_[1] < 40 or
                proposed_s_[2] > MAZE_H * UNIT or proposed_s_[3] > MAZE_W * UNIT or
                (int(proposed_s_[0]) in X_Block and int(proposed_s_[1]) in Y_Block)):
            return s, 'hit'

        # 2. 人机动态碰撞阻断
        if proposed_s_ == self.canvas.coords(self.human1) or proposed_s_ == self.canvas.coords(self.human2):
            return s, 'human_block'

        # 3. 智能体间物理撞击判定
        other_robots = [self.rect1, self.rect2, self.rect3]
        other_robots.remove(rect)
        for other in other_robots:
            if proposed_s_ == self.canvas.coords(other):
                return s, 'robot_block'

        # 允许移动
        self.canvas.move(rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(rect)

        # 4. 终点判定
        tgt_coords = self.canvas.coords([self.org1, self.org2, self.org3][robot_num - 1] if target_type == 'origin' else
                                        [self.target1, self.target2, self.target3][robot_num - 1])
        if s_ == tgt_coords:
            msg = f"Robot{robot_num} 顺利回港！" if target_type == 'origin' else f"Robot{robot_num} 成功采集到货物！"
            self.log_panel.add_log(msg, "success")
            return 'terminal', 'arrive'

        coords_args = [None, None, None]
        coords_args[robot_num - 1] = s_
        self.log_panel.update_coords(*coords_args)

        return s_, 'nothing'

    def render(self, episode=None, total_reward=None, step_count=None, current_state=None):
        if episode is not None:
            self.log_panel.title_label.config(
                text=f"🤖 仓库多智能体监控面板\nEpisode: {episode}\nStep: {step_count}\nReward: {int(total_reward)}"
            )
        time.sleep(self.render_speed)
        self.update()


def moveAgent(s, action):
    base_action = np.array([0, 0])
    if action == 0 and s[1] > UNIT:
        base_action[1] -= UNIT  # Up
    elif action == 1 and s[1] < (MAZE_H - 1) * UNIT:
        base_action[1] += UNIT  # Down
    elif action == 2 and s[0] < (MAZE_W - 1) * UNIT:
        base_action[0] += UNIT  # Right
    elif action == 3 and s[0] > UNIT:
        base_action[0] -= UNIT  # Left
    return base_action