import random
import torch
import numpy as np 
from networks.NoisyQNet import NoisyQ_net  # 假设你已经修改或创建了包含NoisyQ_net的QNet.py文件
from utils.replay_memory import Replay_Memory
import torch.nn as nn

class NoisyDQN():
    def __init__(self, env, arg):
        self.arg = arg
        self.Buffer = Replay_Memory(arg)
        self.Net = NoisyQ_net(arg.Frames, arg.action_dim).to(self.arg.cuda)
        self.targetNet = NoisyQ_net(arg.Frames, arg.action_dim).to(self.arg.cuda)
        self.learnstep = 0
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=arg.learning_rate)
        self.loss_func = nn.MSELoss()

    def get_action(self, obs):
        if random.random() > self.arg.epsilon:
            return self.greedy_action(obs)
        else:
            return random.randint(0, self.arg.action_dim - 1)

    def greedy_action(self, obs):
        with torch.no_grad():  # 在推理时不需要计算梯度
            obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            q_values = self.Net(obs)
            action = np.argmax(q_values.cpu().numpy(), axis=-1)
        return action[0]  # 返回单个整数动作

    def update(self, data):
        action, obs, next_obs, done, reward = data['action'], data['obs'], data['next_obs'], data['done'], data['reward']
        
        if self.learnstep % self.arg.Q_NETWORK_ITERATION == 0:
            self.targetNet.load_state_dict(self.Net.state_dict())
        self.learnstep += 1
        
        # q_eval
        obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.arg.cuda, dtype=torch.float32)
        action = torch.tensor(action, device=self.arg.cuda, dtype=torch.long)
        reward = torch.tensor(reward, device=self.arg.cuda, dtype=torch.float32)
        
        q_eval = self.Net(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q_next = self.targetNet(next_obs).detach()
        q_target = reward + self.arg.gamma * q_next.max(1)[0].view(action.shape[0])
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()