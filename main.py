import sys, os, random, numpy as np, cv2, torch, torch.nn as nn
from collections import deque
from ski_env import make_skiing_env
import time
import cv2
import torch
import numpy as np
from DQN_agent import DQN
import matplotlib.pyplot as plt

def process_img(image):
    """处理图像为84x84的二值图"""
    if isinstance(image, tuple):  # 如果是(obs, info)元组
        image = image[0]
    image = cv2.resize(image, (84, 84))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 修改为RGB转灰度
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = image / 255.0  # 归一化到0-1
    return image

class params():
    def __init__(self):
        self.gamma = 0.99
        self.action_dim = 5  # 5种动作
        self.obs_dim = (4, 84, 84)  # 4帧堆叠，84x84
        self.capacity = 50000  # 增大经验池容量
        self.cuda = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.Frames = 4
        self.episodes = int(1e8)  # 减少总幕数进行测试
        self.updatebatch = 512  # 增大批次大小
        self.test_episodes = 10  # 减少测试幕数
        self.epsilon = 0.1  # 初始探索率
        self.epsilon_min = 0.001  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.Q_NETWORK_ITERATION = 50  # 目标网络更新频率
        self.learning_rate = 0.001  # 调整学习率

def load_state(arg, agent):
    """加载预训练模型"""
    modelpath = 'ski_dqn_best.pkl'
    if os.path.exists(modelpath):
        state_file = torch.load(modelpath, map_location=arg.cuda)
        agent.Net.load_state_dict(state_file)
        agent.targetNet.load_state_dict(state_file)
        print("模型加载成功!")

def _process_frame(frame):
    """处理单帧图像"""
    if isinstance(frame, tuple):  # 如果是reset返回的元组
        frame = frame[0] if isinstance(frame[0], np.ndarray) else frame
    return process_img(frame)

def _make_init_obs(raw_frame):
    """创建初始观测（4帧堆叠）"""
    frame = _process_frame(raw_frame)
    return np.stack([frame] * 4, axis=0)  # shape=(4,84,84)

def _roll_obs(prev_obs, new_frame):
    """滚动更新观测帧"""
    new_frame = _process_frame(new_frame)[np.newaxis]  # (1,84,84)
    return np.concatenate([new_frame, prev_obs[:-1]], axis=0)

def training(arg, agent, env):
    """训练函数"""
    reward_curve = []
    fig, ax = plt.subplots()
    
    # 记录最佳表现
    best_reward = -float('inf')

    for episode in range(arg.episodes):
        # 重置环境
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            raw_frame, info = reset_result
        else:
            raw_frame, info = reset_result, {}
            
        obs = _make_init_obs(raw_frame)
        done = False
        total_reward = 0
        step_count = 0

        # 存放轨迹数据
        traj = dict(obs=[], action=[], reward=[], next_obs=[], done=[])

        while not done:
            # ε-贪婪策略选择动作
            if random.random() < arg.epsilon:
                action = random.randint(0, arg.action_dim - 1)
            else:
                action = agent.get_action(obs)

            # 执行动作
            step_result = env.step(action)
            if len(step_result) == 5:  # Gymnasium返回5个值
                next_frame, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_frame, reward, done, info = step_result
            
            if done:
                reward -= 100  # 终止时的惩罚

            total_reward += reward
            step_count += 1

            # 处理下一帧观测
            next_obs = _roll_obs(obs, next_frame)

            # 存储经验
            traj['obs'].append(obs.copy())
            traj['action'].append(action)
            traj['reward'].append(reward)
            traj['next_obs'].append(next_obs.copy())
            traj['done'].append(done)

            obs = next_obs

            # 限制单幕步数，防止无限循环
            if step_count > 1000:
                break

        # 存入经验池
        if len(traj['obs']) > 0:
            agent.Buffer.store_data(traj, len(traj['obs']))

        # 衰减探索率
        arg.epsilon = max(arg.epsilon_min, arg.epsilon * arg.epsilon_decay)

        # 训练网络（当经验池有足够数据时）
        if hasattr(agent.Buffer, 'ptr') and agent.Buffer.ptr > 1000:
            for _ in range(10):  # 每幕训练多次
                batch = agent.Buffer.sample(arg.updatebatch)
                if batch is not None:
                    loss = agent.update(batch)
                    if loss is not None and episode % 100 == 0:
                        print(f"Episode {episode}, Loss: {loss:.4f}")

        # 定期评估和保存
        if episode % 50 == 0:
            original_render_mode = env.render_mode
            env.close()
            test_env = make_skiing_env("Skiing-rgb-v0", render_mode=None)  # 测试时不显示窗口
            
            avg_reward = test_performance(arg, agent, test_env)
            reward_curve.append(avg_reward)
            
            print(f'Episode {episode:>6} | '
                  f'Test Reward: {avg_reward:.2f} | '
                  f'Epsilon: {arg.epsilon:.3f} | '
                  f'Steps: {step_count}')

            # 更新最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.Net.state_dict(), 'ski_dqn_best.pkl')
                print(f"新的最佳模型已保存! 奖励: {best_reward:.2f}")

            # 关闭测试环境，重新打开训练环境
            test_env.close()
            if original_render_mode is not None:
                env = make_skiing_env("Skiing-rgb-v0", render_mode=original_render_mode)
            else:
                env = make_skiing_env("Skiing-rgb-v0", render_mode=None)

            # 更新奖励曲线图
            ax.clear()
            ax.plot(reward_curve, 'b-', linewidth=2)
            ax.set_xlabel('Episode (x50)')
            ax.set_ylabel('Average Reward')
            ax.set_title('Training Progress')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('reward_curve.jpg', dpi=150, bbox_inches='tight')

    # 训练结束保存最终模型
    torch.save(agent.Net.state_dict(), 'ski_dqn_final.pkl')
    print("训练完成!最终模型已保存。")

def test_performance(arg, agent, test_env=None):
    """测试智能体性能"""
    if test_env is None:
        test_env = make_skiing_env("Skiing-rgb-v0", render_mode=None)
    
    rewards = []
    
    for _ in range(arg.test_episodes):
        reset_result = test_env.reset()
        if isinstance(reset_result, tuple):
            raw_frame, info = reset_result
        else:
            raw_frame, info = reset_result, {}
            
        obs = _make_init_obs(raw_frame)
        done = False
        ep_reward = 0
        step_count = 0
        
        while not done:
            action = agent.greedy_action(obs)  # 贪婪策略
            
            step_result = test_env.step(action)
            if len(step_result) == 5:
                next_frame, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_frame, reward, done, info = step_result
                
            obs = _roll_obs(obs, next_frame)
            ep_reward += reward
            step_count += 1
            
            # 限制测试步数
            if step_count > 500:
                break
                
        rewards.append(ep_reward)
    
    if test_env != env:  # 如果是临时创建的测试环境，关闭它
        test_env.close()
    
    return np.mean(rewards)

def demo_play(arg, agent, env):
    """演示模式：使用训练好的模型进行游戏（带窗口显示）"""
    load_state(arg, agent)
    arg.epsilon = 0.0  # 完全贪婪
    
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        raw_frame, info = reset_result
    else:
        raw_frame, info = reset_result, {}
        
    obs = _make_init_obs(raw_frame)
    done = False
    total_reward = 0
    step_count = 0
    
    print("开始演示! 按Ctrl+C退出")
    
    try:
        while not done:
            action = agent.greedy_action(obs)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_frame, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_frame, reward, done, info = step_result
                
            obs = _roll_obs(obs, next_frame)
            total_reward += reward
            step_count += 1
            
            # 显示当前信息
            if step_count % 10 == 0:
                print(f"步数: {step_count}, 累计奖励: {total_reward:.1f}")
            
            # 控制演示速度
            time.sleep(0.05)  # 20 FPS
            
            if done:
                print(f"演示结束! 总步数: {step_count}, 最终得分: {total_reward}")
                break
                
            # 限制演示时间
            if step_count > 1000:
                print(f"演示达到最大步数! 最终得分: {total_reward}")
                break
                
    except KeyboardInterrupt:
        print(f"\n演示被用户中断! 最终得分: {total_reward}")
    
    # 询问是否重新开始演示
    restart = input("是否重新开始演示? (y/n): ").strip().lower()
    if restart == 'y':
        demo_play(arg, agent, env)

def test_with_display(arg, agent):
    """带窗口显示的测试模式"""
    load_state(arg, agent)
    
    # 创建带窗口的测试环境
    test_env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
    
    rewards = []
    
    for episode in range(arg.test_episodes):
        reset_result = test_env.reset()
        if isinstance(reset_result, tuple):
            raw_frame, info = reset_result
        else:
            raw_frame, info = reset_result, {}
            
        obs = _make_init_obs(raw_frame)
        done = False
        ep_reward = 0
        step_count = 0
        
        print(f"开始测试第 {episode + 1}/{arg.test_episodes} 幕...")
        
        while not done:
            action = agent.greedy_action(obs)
            
            step_result = test_env.step(action)
            if len(step_result) == 5:
                next_frame, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_frame, reward, done, info = step_result
                
            obs = _roll_obs(obs, next_frame)
            ep_reward += reward
            step_count += 1
            
            # 慢速显示，便于观察
            time.sleep(0.03)
            
            # 限制测试步数
            if step_count > 500:
                break
                
        rewards.append(ep_reward)
        print(f"第 {episode + 1} 幕结束: 奖励={ep_reward:.1f}, 步数={step_count}")
        
        # 幕间暂停
        if episode < arg.test_episodes - 1:
            print("准备下一幕...")
            time.sleep(2)
    
    test_env.close()
    
    avg_reward = np.mean(rewards)
    print(f"\n测试完成! 平均奖励: {avg_reward:.2f}")
    return avg_reward

if __name__ == '__main__':
    # 选择模式
    # 初始化环境和智能体 - 训练时不需要窗口，测试和演示时需要
    mode = input("选择模式 (1-训练, 2-测试, 3-演示): ").strip()
    arg = params()
    if mode == "1":
        print("开始训练模式...")
        # 训练模式：使用rgb_array模式提高效率
        env = make_skiing_env("Skiing-rgb-v0", render_mode=None)  # 无窗口渲染
        agent = DQN(env, arg)
        load_state(arg, agent)  # 加载已有模型继续训练
        training(arg, agent, env)
    elif mode == "2":
        print("开始测试模式（带窗口显示）...")
        env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
        agent = DQN(env, arg)
        test_with_display(arg, agent)
    elif mode == "3":
        print("开始演示模式...")
        env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
        agent = DQN(env, arg)
        demo_play(arg, agent, env)
    else:
        print("开始训练模式...")
        # 训练模式：使用rgb_array模式提高效率
        env = make_skiing_env("Skiing-rgb-v0", render_mode=None)  # 无窗口渲染
        agent = DQN(env, arg)
        load_state(arg, agent)  # 加载已有模型继续训练
        training(arg, agent, env)
    
    env.close()