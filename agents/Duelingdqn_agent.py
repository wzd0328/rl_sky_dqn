import random
import numpy as np
import torch
import torch.nn as nn
from networks.QNet import DuelingQNet
from utils.replay_memory import Replay_Memory


class DuelingDQN():
    def __init__(self, env, args):
        self.arg = args
        self.env = env
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim

        # ε-greedy 参数
        self.epsilon_start = args.epsilon  # 初始探索率
        self.epsilon_end = args.epsilon_min     # 最小探索率
        self.epsilon_decay = args.epsilon_decay  # 衰减步数
        self.epsilon = self.epsilon_start
        self.total_steps = 0  # 计数步数

        # 经验回放
        self.Buffer = Replay_Memory(args)

        # 网络
        self.Net = DuelingQNet(args.obs_dim, self.action_dim).to(args.cuda)
        self.targetNet = DuelingQNet(args.obs_dim, self.action_dim).to(args.cuda)
        self.targetNet.load_state_dict(self.Net.state_dict())

        # 优化器 & 损失函数
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=args.learning_rate)
        self.loss_func = nn.MSELoss()
        self.learnstep = 0

    def update_epsilon(self):
        """指数衰减 ε"""
        self.epsilon = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.total_steps / self.epsilon_decay)

    def get_action(self, obs):
        """选择动作 (ε-greedy)"""
        self.total_steps += 1
        self.update_epsilon()

        if random.random() > self.epsilon:
            return self.greedy_action(obs)
        else:
            return random.randint(0, self.action_dim - 1)

    def greedy_action(self, obs):
        """贪心选择动作"""
        obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
        if len(obs.shape) == 3:  # 如果缺少 batch 维度
            obs = obs.unsqueeze(0)
        q_values = self.Net(obs)
        action = np.argmax(q_values.detach().cpu().numpy(), axis=-1)
        return action

    def update(self, data):
        """更新网络参数"""
        action, obs, next_obs, done, reward = \
            data['action'], data['obs'], data['next_obs'], data['done'], data['reward']

        if self.learnstep % self.arg.Q_NETWORK_ITERATION == 0:
            self.targetNet.load_state_dict(self.Net.state_dict())
        self.learnstep += 1

        # 转 tensor
        obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.arg.cuda, dtype=torch.float32)
        action = torch.tensor(action, device=self.arg.cuda, dtype=torch.long)
        reward = torch.tensor(reward, device=self.arg.cuda, dtype=torch.float32)
        done = torch.tensor(done, device=self.arg.cuda, dtype=torch.float32)

        # Q值估计
        q_eval = self.Net(obs).gather(1, action)
        q_next = self.targetNet(next_obs).detach()
        q_target = reward + self.arg.gamma * q_next.max(1)[0].view(action.shape[0], 1) * (1 - done)

        # 计算损失并反向传播
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
