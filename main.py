import sys
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pygame
import time

from envs.ski_env import make_skiing_env
from utils.human_player import HumanPlayer
def process_img(image):
    """处理图像为的二值图"""
    if isinstance(image, tuple):  # 如果是(obs, info)元组
        image = image[0]
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 修改为RGB转灰度
    _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    # image = image / 255.0  # 归一化到0-1
    # 显示image
    cv2.imshow('Processed Frame', image)
    # cv2.imwrite('./results/processed_frame.png', image)  # 保存为图像文件
    return image

class params():
    def __init__(self):
        self.gamma = 0.99
        self.action_dim = 3  # 3种动作
        self.obs_dim = (4, 128, 128)  # 4帧堆叠
        self.capacity = 10000  # 增大经验池容量
        self.cuda = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.Frames = 4
        self.episodes = int(1e2)  # 减少总幕数进行测试
        self.updatebatch = 512  # 增大批次大小
        self.test_episodes = 10  # 减少测试幕数
        self.epsilon = 0.001  # 初始探索率
        self.epsilon_min = 0.0001  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.Q_NETWORK_ITERATION = 100  # 目标网络更新频率
        self.learning_rate = 0.0001  # 调整学习率
        self.agent_type = "DQN"  # 初始化

def create_agent(env, args, agent_type="DQN"):
    # 使用 args 中定义的 obs_dim（因为是图像堆叠）
    act_dim = args.action_dim # 这个仍然需要，但DQN内部也会从arg中获取
    if agent_type == "DQN":
        from agents.DQN_agent import DQN
        agent = DQN(env, args) # 修正：DQN 的 __init__ 签名是 (self, env, arg)
    elif agent_type == "NoisyDQN":
        from agents.noisydqn_agent import NoisyDQN
        agent = NoisyDQN(env, args) 

    else:
        raise ValueError(f"未知的 Agent 类型: {agent_type}")
    return agent

def load_state(path, agent):
    """加载预训练模型"""
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=agent.arg.cuda)
        agent.Net.load_state_dict(state_dict)
        agent.targetNet.load_state_dict(state_dict)
        print(f"模型加载成功: {path}")
    else:
        print(f"未找到模型文件: {path}")

def _process_frame(frame):
    """处理单帧图像"""
    if isinstance(frame, tuple):  # 如果是reset返回的元组
        frame = frame[0] if isinstance(frame[0], np.ndarray) else frame
    return process_img(frame)

def _make_init_obs(raw_frame):
    """创建初始观测（4帧堆叠）"""
    frame = _process_frame(raw_frame)
    return np.stack([frame] * 4, axis=0)  # shape=(4,H,W)

def _roll_obs(prev_obs, new_frame):
    """滚动更新观测帧"""
    new_frame = _process_frame(new_frame)[np.newaxis]  # (1,H,W)
    return np.concatenate([new_frame, prev_obs[:-1]], axis=0)

def training(arg, agent, env, save_path, final_path, reward_curve_path):
    """训练函数（支持自定义保存路径）"""
    reward_curve = []
    fig, ax = plt.subplots(figsize=(10, 6))

    best_reward = -float('inf')

    for episode in range(arg.episodes):
        print(f"Starting episode {episode}")

        # 重置环境
        # print(env._use_images)
        # reset_result = env.reset(options={"use_images": False})
        # print(reset_result[0])
        # print(env._use_images)
        reset_result = env.reset()

        if isinstance(reset_result, tuple):
            raw_frame, info = reset_result
        else:
            raw_frame, info = reset_result, {}

        # 初始化观测值
        obs = _make_init_obs(raw_frame)
        done = False
        total_reward = 0
        step_count = 0

        traj = dict(obs=[], action=[], reward=[], next_obs=[], done=[])

        # 开始训练过程
        while not done:
            if random.random() < arg.epsilon:
                action = random.randint(0, arg.action_dim - 1)
                # print("随机选择:", action)
            else:
                action = agent.get_action(obs)
                # print("贪心选择:", action)

            step_result = env.step(action)

            if len(step_result) == 5:
                next_frame, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_frame, reward, done, info = step_result

            if done:
                reward -= 100  # 终止惩罚

            total_reward += reward
            step_count += 1

            next_obs = _roll_obs(obs, next_frame)

            traj['obs'].append(obs.copy())
            traj['action'].append(action)
            traj['reward'].append(reward)
            traj['next_obs'].append(next_obs.copy())
            traj['done'].append(done)

            obs = next_obs

            if step_count > 10000:
                break

        if len(traj['obs']) > 0:
            agent.Buffer.store_data(traj, len(traj['obs']))

        arg.epsilon = max(arg.epsilon_min, arg.epsilon * arg.epsilon_decay)

        # 如果 buffer 大小超过 1000，则进行更新
        if hasattr(agent.Buffer, 'ptr') and agent.Buffer.ptr > 1000:
            for _ in range(10):
                batch = agent.Buffer.sample(arg.updatebatch)
                if batch is not None:
                    loss = agent.update(batch)
                    if loss is not None and episode % 100 == 0:
                        print(f"Episode {episode}, Loss: {loss:.4f}")

        # 每 50 轮进行一次性能测试和模型保存
        if episode % 50 == 0:
            original_render_mode = env.render_mode
            env.close()
            test_env = make_skiing_env("Skiing-rgb-v0", render_mode="rgb_array")  # 测试时不渲染窗口

            avg_reward = test_performance(arg, agent, test_env, model_path=None)  # 已加载
            reward_curve.append(avg_reward)

            print(f'Episode {episode:>6} | Test Reward: {avg_reward:.2f} | '
                  f'Epsilon: {arg.epsilon:.3f} | Steps: {step_count}')

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.Net.state_dict(), save_path)
                print(f"🏆 新的最佳模型已保存! 奖励: {best_reward:.2f}")

            test_env.close()
            env = make_skiing_env("Skiing-rgb-v0", render_mode=original_render_mode)

            ax.clear()
            ax.plot(reward_curve, 'b-', linewidth=2)
            ax.set_xlabel('Episode (x50)')
            ax.set_ylabel('Average Reward')
            ax.set_title(f'Training Progress - {arg.agent_type}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(reward_curve_path, dpi=150, bbox_inches='tight')

    torch.save(agent.Net.state_dict(), final_path)
    print(f"✅ 训练完成! 最终模型已保存至: {final_path}")

def test_performance(arg, agent, test_env=None, model_path=None):
    """测试性能（支持加载指定模型）"""
    if model_path:
        load_state(model_path, agent)

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
            action = agent.greedy_action(obs)
            step_result = test_env.step(action)
            if len(step_result) == 5:
                _, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                _, reward, done, _ = step_result
            obs = _roll_obs(obs, step_result[0])
            ep_reward += reward
            step_count += 1
            if step_count > 1000:
                break
        rewards.append(ep_reward)

    if test_env:
        test_env.close()

    return np.mean(rewards)

def demo_play(arg, agent, env, model_path=None):
    """演示模式（带窗口）"""
    if model_path:
        load_state(model_path, agent)  # 传入路径
    arg.epsilon = 0.0

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        raw_frame, info = reset_result
    else:
        raw_frame, info = reset_result, {}
    obs = _make_init_obs(raw_frame)
    done = False
    total_reward = 0
    step_count = 0

    print("🎮 开始演示! 按 ESC 退出，R 重新开始")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and done:
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        raw_frame, info = reset_result
                    else:
                        raw_frame, info = reset_result, {}
                    obs = _make_init_obs(raw_frame)
                    done = False
                    total_reward = 0
                    step_count = 0
                elif event.type == pygame.MOUSEBUTTONDOWN and done:
                    mouse_pos = event.pos
                    restart_button = pygame.Rect(env._screen_width // 2 - 100, env._screen_height // 2 + 30, 200, 50)
                    quit_button = pygame.Rect(env._screen_width // 2 - 100, env._screen_height // 2 + 100, 200, 50)
                    if restart_button.collidepoint(mouse_pos):
                        reset_result = env.reset()
                        if isinstance(reset_result, tuple):
                            raw_frame, info = reset_result
                        else:
                            raw_frame, info = reset_result, {}
                        obs = _make_init_obs(raw_frame)
                        done = False
                        total_reward = 0
                        step_count = 0
                    elif quit_button.collidepoint(mouse_pos):
                        return

            if not done:
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

                if step_count % 50 == 0:
                    print(f"步数: {step_count}, 累计奖励: {total_reward:.1f}")

                time.sleep(0.01)

                if done:
                    final_score = info.get("score", total_reward)
                    print(f"🎯 演示结束! 总步数: {step_count}, 最终得分: {final_score}")
    except KeyboardInterrupt:
        print(f"\n⏹️ 演示被用户中断! 最终得分: {total_reward}")
def human_play_mode(env):
    """人工玩法模式"""
    # 创建人工玩家
    human_player = HumanPlayer()
    
    # 重置环境
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        raw_frame, info = reset_result
    else:
        raw_frame, info = reset_result, {}
    
    done = False
    total_reward = 0
    step_count = 0
    
    try:
        while True:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("游戏退出")
                        return
                    elif event.key == pygame.K_r and done:
                        # 重新开始游戏
                        reset_result = env.reset()
                        if isinstance(reset_result, tuple):
                            raw_frame, info = reset_result
                        else:
                            raw_frame, info = reset_result, {}
                        human_player.reset()
                        done = False
                        total_reward = 0
                        step_count = 0
                        print("游戏重新开始!")
                # 处理鼠标点击（用于游戏结束界面的按钮）
                elif event.type == pygame.MOUSEBUTTONDOWN and done:
                    mouse_pos = event.pos
                    restart_button = pygame.Rect(env._screen_width // 2 - 100, env._screen_height // 2 + 30, 200, 50)
                    quit_button = pygame.Rect(env._screen_width // 2 - 100, env._screen_height // 2 + 100, 200, 50)
                    if restart_button.collidepoint(mouse_pos):
                        # 触发重新开始
                        reset_result = env.reset()
                        if isinstance(reset_result, tuple):
                            raw_frame, info = reset_result
                        else:
                            raw_frame, info = reset_result, {}
                        human_player.reset()
                        done = False
                        total_reward = 0
                        step_count = 0
                        print("游戏重新开始!")
                    elif quit_button.collidepoint(mouse_pos):
                        # 触发退出游戏
                        print("游戏退出")
                        return
            
            if not done:
                # 获取人工玩家动作
                action = human_player.get_action_from_keyboard()
                
                # 执行动作
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_frame, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_frame, reward, done, info = step_result
                
                total_reward += reward
                step_count += 1
                
                # 显示实时信息
                if step_count % 50 == 0:
                    print(f"得分: {info.get('score', 0):.1f}, 速度: {info.get('speed', 0):.1f}, 旗帜数量: {info.get('flag_count', 0)}")
                
                # 限制最大步数
                if step_count > 5000:
                    done = True
                    # 设置游戏结束状态
                    if hasattr(env, '_game_over'):
                        env._game_over = True
                
            # 渲染游戏
            env.render()
            time.sleep(0.05 if not done else 0.01)  # 控制游戏速度
            
    except KeyboardInterrupt:
        print("游戏退出")
    finally:
        env.close()

def test_with_display(arg, agent, model_path=None):
    """带窗口的测试模式"""
    load_state(model_path, agent)
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

        print(f"▶️ 开始测试第 {episode + 1}/{arg.test_episodes} 幕...")

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
            time.sleep(0.03)
            if step_count > 500:
                break

        rewards.append(ep_reward)
        print(f"🔚 第 {episode + 1} 幕结束: 奖励={ep_reward:.1f}, 步数={step_count}")
        if episode < arg.test_episodes - 1:
            time.sleep(2)

    test_env.close()
    avg_reward = np.mean(rewards)
    print(f"\n📊 测试完成! 平均奖励: {avg_reward:.2f}")
    return avg_reward

if __name__ == '__main__':
    # ==================== 配置区域====================
    AGENT_TYPE = "DQN"  # 可选: "DQN", "NoisyDQN"
    MODEL_SAVE_PATH = f"models/ski_{AGENT_TYPE.lower()}_best_flag.pkl"
    MODEL_FINAL_PATH = f"models/ski_{AGENT_TYPE.lower()}_final.pkl"
    REWARD_CURVE_PATH = f"results/reward_curve_{AGENT_TYPE.lower()}.jpg"

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    arg = params()
    arg.agent_type = AGENT_TYPE

    mode = input("选择模式 (1-训练, 2-测试, 3-演示, 4-游玩): ").strip()

    # try:
    if mode == "1":  # 训练
        print(f"🚀 开始训练 {AGENT_TYPE}...")
        # env = make_skiing_env("Skiing-rgb-v0", render_mode="rgb_array")  # 无窗口渲染，使用图像
        env = make_skiing_env("Skiing-rgb-v0", render_mode="human")  # 有窗口渲染，便于调试
        agent = create_agent(env, arg, AGENT_TYPE)
        load_state(MODEL_SAVE_PATH, agent)
        training(arg, agent, env, MODEL_SAVE_PATH, MODEL_FINAL_PATH, REWARD_CURVE_PATH)

    elif mode == "2":  # 测试（带显示）
        print(f"🧪 开始测试 {AGENT_TYPE}（带窗口）...")
        env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
        agent = create_agent(env, arg, AGENT_TYPE)
        test_with_display(arg, agent, MODEL_SAVE_PATH)

    elif mode == "3":  # 演示
        print(f"🎬 开始演示 {AGENT_TYPE}...")
        env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
        agent = create_agent(env, arg, AGENT_TYPE)
        demo_play(arg, agent, env, MODEL_SAVE_PATH)

    elif mode == "4":  # 人工游玩
        print("🎮 人工玩家模式...")
        env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
        human_play_mode(env)

    else:
        print("⚠️ 无效输入，启动训练模式...")
        env = make_skiing_env("Skiing-rgb-v0", render_mode="rgb_array")  # 无窗口渲染，使用图像
        # env = make_skiing_env("Skiing-rgb-v0", render_mode="human")  # 有窗口渲染，便于调试
        agent = create_agent(env, arg, AGENT_TYPE)
        load_state(MODEL_SAVE_PATH, agent)
        training(arg, agent, env, MODEL_SAVE_PATH, MODEL_FINAL_PATH, REWARD_CURVE_PATH)

    env.close()
    # except Exception as e:
    #     print(f"❌ 程序异常: {e}")
    # finally:
    #     pygame.quit()
    #     cv2.destroyAllWindows()