import sys, os, random, numpy as np, cv2, torch, torch.nn as nn
from collections import deque
from ski_game_dqn import SkiEnv

# -------------------- 超参数 --------------------
ACTIONS      = SkiEnv.ACTIONS
GAMMA        = 0.99
OBSERVE      = 5000
EXPLORE      = 500000
REPLAY_MEMORY= 50000
BATCH_SIZE   = 32
UPDATE_TIME  = 100
INITIAL_EPS  = 0.1
FINAL_EPS    = 0.0001
FRAME_PER_ACTION = 1
MODEL_FILE   = 'ski_dqn.pth'
# ----------------------------------------------

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4, 2), nn.ReLU(),
            nn.Conv2d(32,64, 4, 2, 1),  nn.ReLU(),
            nn.Conv2d(64,64, 3, 1, 1),  nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(64*10*10, 256), nn.ReLU(),
            nn.Linear(256, ACTIONS))
    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.view(x.size(0), -1))

class DQNAgent:
    def __init__(self):
        self.replay = deque(maxlen=REPLAY_MEMORY)
        self.time_step = 0
        self.epsilon = INITIAL_EPS # epsilon的概率随机选择一个动作，1-epsilon

        self.Q = DeepNet().cuda() # 当前值网络
        self.Q_target = DeepNet().cuda() # 目标值网络
        if os.path.exists(MODEL_FILE):
            self.Q.load_state_dict(torch.load(MODEL_FILE))
            self.Q_target.load_state_dict(self.Q.state_dict())
            print('loaded checkpoint')
        # self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer= torch.optim.Adam(self.Q.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if self.time_step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                return random.randint(0, ACTIONS-1)
            else:
                state = torch.from_numpy(state).cuda().unsqueeze(0).float()
                with torch.no_grad():
                    return int(self.Q(state).argmax(1))
        else:
            return 0   # 默认直行

    def perceive(self, s, a, r, s_, done):
        self.replay.append((s, a, r, s_, done))
        if len(self.replay) > BATCH_SIZE and self.time_step > OBSERVE:
            self.train()
        if self.time_step % UPDATE_TIME == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
            torch.save(self.Q.state_dict(), MODEL_FILE)
        # epsilon 退火
        if self.epsilon > FINAL_EPS and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPS - FINAL_EPS) / EXPLORE
        self.time_step += 1

    def train(self):
        # 从replay memory中随机抽取batch_size个数据
        batch = random.sample(self.replay, BATCH_SIZE)
        s  = torch.from_numpy(np.array([b[0] for b in batch])).cuda().float() # 当前状态
        a  = torch.LongTensor([b[1] for b in batch]).cuda() # 动作
        r  = torch.FloatTensor([b[2] for b in batch]).cuda() # 奖励
        s_ = torch.from_numpy(np.array([b[3] for b in batch])).cuda().float() # 下一个状态
        done= torch.BoolTensor([b[4] for b in batch]).cuda()

        # r_batch存储reward
        r_batch = np.zeros([BATCH_SIZE, 1])
        
        q_eval = self.Q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.Q_target(s_).max(1)[0]
            q_target = r + GAMMA * q_next * (~done)
        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.time_step % 1000 == 0:
            print(f'step={self.time_step}  loss={loss.item():.4f}  eps={self.epsilon:.4f}')

# -------------------- 主循环 --------------------
env   = SkiEnv()
agent = DQNAgent()
s = env.reset()
while True:
    a = agent.act(s)
    s_, r, done = env.step(a)
    agent.perceive(s, a, r, s_, done)
    s = s_
    if done:
        s = env.reset()