import sys, os, random, numpy as np, cv2, torch, torch.nn as nn
from collections import deque
from ski_game_dqn import SkiEnv
import math
import random
import time
import cv2
import torch
import numpy as np
from DQN_agent import DQN
import matplotlib.pyplot as plt

def process_img(image):
    image= cv2.resize(image, (84, 84))
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV)
    return image[1]

class params():
    def __init__(self):
        self.gamma = 0.99
        self.action_dim = 5
        self.obs_dim = (4,84,84) #80*80*3
        self.capacity = 50000
        self.cuda = 'cuda:0'
        self.Frames = 4
        self.episodes = int(1e8)
        self.updatebatch= 512
        self.test_episodes= 10
        self.epsilon = 0.1
        self.Q_NETWORK_ITERATION = 50
        self.learning_rate = 0.001

env = SkiEnv()
arg = params()
agent = DQN(env,arg)

def load_state():
    modelpath = 'ski_dqn.pkl'
    state_file = torch.load(modelpath)
    agent.Net.load_state_dict(state_file)
    agent.targetNet.load_state_dict(state_file)

def _process_frame(raw_frame):
    """环境返回的原始帧 -> 80×80 二值图"""
    return process_img(raw_frame)

def _make_init_obs(raw_frame):
    """把单帧变成 4×80×80 的初始观测"""
    frame = _process_frame(raw_frame)
    return np.repeat(frame[np.newaxis], 4, axis=0)   # shape=(4,80,80)

def _roll_obs(prev_obs, new_frame):
    """新帧替换最旧的一帧，保持 4 帧堆叠"""
    new_frame = _process_frame(new_frame)[np.newaxis]  # (1,80,80)
    return np.concatenate([new_frame, prev_obs[:-1]], axis=0)

def training():
    reward_curve = []
    fig, ax = plt.subplots()

    for episode in range(arg.episodes):
        obs = _make_init_obs(env.reset())
        done = False

        # 存放整条 trajectory
        traj = dict(obs=[], action=[], reward=[], next_obs=[], done=[])

        # 探索策略退火
        if episode % 10_000 == 0 and episode > 100:
            arg.epsilon /= math.sqrt(10)

        while not done:
            # 前 300 幕纯随机探索
            if episode < 300:
                action = random.randint(0, 1)
            else:
                action = agent.get_action(obs)

            frame, reward, done = env.step(action)
            if done:
                reward -= 101

            next_obs = _roll_obs(obs, frame)

            traj['obs'].append(obs)
            traj['action'].append(action)
            traj['reward'].append(reward)
            traj['next_obs'].append(next_obs)
            traj['done'].append(done)

            obs = next_obs

        # 打印正向突破
        if sum(traj['reward']) > 0:
            print(f'Breakthrough! total reward = {sum(traj["reward"])}')

        # 存入经验池
        agent.Buffer.store_data(traj, len(traj['obs']))

        # 训练
        if agent.Buffer.ptr > 500:
            batch = agent.Buffer.sample(arg.updatebatch)
            agent.update(batch)

        # 定期评估 & 保存
        if episode % 50 == 0:
            if episode % 300 == 0:
                torch.save(agent.Net.state_dict(), f'ski_dqn_{episode}.pkl')

            avg_reward = test_performance()
            reward_curve.append(avg_reward)
            print(f'Episode {episode:>6} | test avg reward = {avg_reward:.2f}')

            ax.clear()
            ax.plot(reward_curve, 'g-')
            plt.savefig('reward.jpg')

# ---------- 评估 ----------
def test_performance():
    """跑 `arg.test_episodes` 幕，返回平均 episode 奖励"""
    rewards = []
    for _ in range(arg.test_episodes):
        obs = _make_init_obs(env.reset())
        done = False
        ep_r = 0
        while not done:
            action = agent.greedy_action(obs)
            frame, r, done = env.step(action)
            obs = _roll_obs(obs, frame)
            ep_r += r
        rewards.append(ep_r - 101) # 与训练端保持一致
    return np.mean(rewards)