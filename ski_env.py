#
# 滑雪小游戏环境（RGB图像版本）
# 返回RGB图像作为观测值，适合用于视觉输入的强化学习算法
#

from enum import IntEnum
from typing import Dict, Optional, Tuple, Union
import gymnasium
import numpy as np
import pygame
import os
from pathlib import Path

# 常量定义
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PLAYER_WIDTH = 30
PLAYER_HEIGHT = 30
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 40
INITIAL_PLAYER_SPEED = 2
MAX_PLAYER_SPEED = 10
ACCELERATION_RATE = 0.001
OBSTACLE_GENERATION_RATE_INITIAL = 0.02
OBSTACLE_GENERATION_RATE_MAX = 0.1
OBSTACLE_VEL_Y = 3
PLAYER_ANGLE_CHANGE = 15  # 角度变化量

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BACKGROUND_COLOR = (135, 206, 235)  # 天蓝色背景，模拟天空
SNOW_COLOR = (255, 250, 250)  # 雪地颜色

class Actions(IntEnum):
    """滑雪者的可能动作"""
    STRAIGHT, LEFT_15, LEFT_45, RIGHT_15, RIGHT_45 = 0, 1, 2, 3, 4

class SkiingRGBEnv(gymnasium.Env):
    """滑雪小游戏环境（RGB图像版本）
    
    这个版本返回RGB图像作为观测值，可以直接用process_image函数处理
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        screen_size: Tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT),
        render_mode: Optional[str] = None,
        use_images: bool = True,
        debug: bool = False,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._debug = debug
        self._use_images = use_images
        
        # 动作空间：5种动作
        self.action_space = gymnasium.spaces.Discrete(5)
        
        # 观察空间：RGB图像 (高度, 宽度, 3通道)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, 
            shape=(screen_size[1], screen_size[0], 3),  # (height, width, channels)
            dtype=np.uint8
        )
            
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        
        # 游戏状态变量
        self._player_x = 0
        self._player_y = 0
        self._player_angle = 0  # 角度，0表示垂直向下
        self._player_speed = INITIAL_PLAYER_SPEED
        self._obstacles = []  # 障碍物列表，每个障碍物是(x, y)元组
        self._score = 0  # 分数基于垂直下滑距离
        self._distance = 0  # 下滑总距离
        self._obstacle_generation_rate = OBSTACLE_GENERATION_RATE_INITIAL
        
        if not pygame.get_init():
            pygame.init()

        if not pygame.font.get_init():
            pygame.font.init()

        self._font = pygame.font.Font(None, 36)

        # 渲染相关 - 必须初始化，因为我们要返回渲染的图像
        self._fps_clock = pygame.time.Clock()
        self._display = None
        self._surface = pygame.Surface(screen_size)
        
        # 初始化pygame显示（即使render_mode为None也要初始化，因为我们需要渲染图像）
        self._make_display()
        
        if use_images:
            self._images = self._load_images()
        else:
            self._images = None
        
    def step(self, action: Union[Actions, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一个动作，更新游戏状态"""
        terminal = False
        reward = 0.1  # 存活奖励
        
        # 处理玩家动作
        if action == Actions.STRAIGHT:
            self._player_angle = 0
        elif action == Actions.LEFT_15:
            self._player_angle = -15
        elif action == Actions.LEFT_45:
            self._player_angle = -45
        elif action == Actions.RIGHT_15:
            self._player_angle = 15
        elif action == Actions.RIGHT_45:
            self._player_angle = 45
        
        # 根据角度计算水平移动
        angle_rad = np.radians(self._player_angle)
        horizontal_move = self._player_speed * np.sin(angle_rad)
        
        # 更新玩家位置
        self._player_x += horizontal_move
        self._player_y += self._player_speed  # 垂直方向始终向下
        
        # 增加下滑距离和分数
        self._distance += self._player_speed
        self._score = self._distance
        
        # 根据距离增加速度
        self._player_speed = min(MAX_PLAYER_SPEED, INITIAL_PLAYER_SPEED + self._distance * ACCELERATION_RATE)
        
        # 根据距离增加障碍物生成率
        self._obstacle_generation_rate = min(
            OBSTACLE_GENERATION_RATE_MAX, 
            OBSTACLE_GENERATION_RATE_INITIAL + self._distance * 0.0001
        )
        
        # 生成新障碍物
        if np.random.random() < self._obstacle_generation_rate:
            self._generate_obstacle()
        
        # 移动障碍物（向上移动，因为玩家在向下滑）
        new_obstacles = []
        for obstacle in self._obstacles:
            x, y = obstacle
            y -= OBSTACLE_VEL_Y  # 障碍物向上移动
            
            # 只保留还在屏幕内的障碍物
            if y > -OBSTACLE_HEIGHT:
                new_obstacles.append((x, y))
        self._obstacles = new_obstacles
        
        # 检查游戏结束条件
        # 1. 超出屏幕左右边界
        if self._player_x < 0 or self._player_x > self._screen_width - PLAYER_WIDTH:
            terminal = True
            reward = -1.0
            if self._debug:
                print("游戏结束：超出屏幕边界")
        
        # 2. 碰到障碍物
        player_rect = pygame.Rect(self._player_x, self._player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
        for obstacle in self._obstacles:
            obstacle_rect = pygame.Rect(obstacle[0], obstacle[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
            if player_rect.colliderect(obstacle_rect):
                terminal = True
                reward = -1.0
                if self._debug:
                    print("游戏结束：碰到障碍物")
                break
        
        # 3. 超出屏幕底部（正常情况不会发生，因为玩家向下滑）
        if self._player_y > self._screen_height:
            terminal = True
            reward = -1.0
            if self._debug:
                print("游戏结束：超出屏幕底部")
        
        # 获取观察值（RGB图像）
        obs = self._get_observation()
        
        # 如果是human模式，额外渲染到窗口
        if self.render_mode == "human":
            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])
        
        info = {"score": self._score, "distance": self._distance, "speed": self._player_speed}
        
        return obs, reward, terminal, False, info
    
    def reset(self, seed=None, options=None):
        """重置游戏状态"""
        super().reset(seed=seed)
        
        # 重置玩家状态
        self._player_x = self._screen_width // 2 - PLAYER_WIDTH // 2
        self._player_y = 10
        self._player_angle = 0
        self._player_speed = INITIAL_PLAYER_SPEED
        
        # 清空障碍物
        self._obstacles = []
        
        # 重置分数和距离
        self._score = 0
        self._distance = 0
        self._obstacle_generation_rate = OBSTACLE_GENERATION_RATE_INITIAL
        
        # 生成初始障碍物
        for _ in range(5):
            self._generate_obstacle()
        
        # 获取观察值（RGB图像）
        obs = self._get_observation()
        
        # 如果是human模式，渲染初始状态
        if self.render_mode == "human":
            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])
        
        info = {"score": self._score, "distance": self._distance, "speed": self._player_speed}
        
        return obs, info
    
    def render(self) -> None:
        """渲染游戏画面 - 返回RGB数组"""
        if self.render_mode == "rgb_array":
            return self._get_observation()
        else:
            # human模式已经在step和reset中处理
            if self._display is None:
                self._make_display()
            return self._get_observation()
    
    def close(self):
        """关闭环境"""
        if hasattr(self, '_display') and self._display is not None:
            pygame.display.quit()
        if pygame.get_init():
            pygame.quit()
        super().close()
    
    def _generate_obstacle(self):
        """生成一个新的障碍物"""
        # 确保障碍物不会生成在玩家初始位置附近
        min_x = 0
        max_x = self._screen_width - OBSTACLE_WIDTH
        
        # 障碍物从屏幕底部生成
        x = np.random.randint(min_x, max_x + 1)
        y = self._screen_height  # 从屏幕底部开始
        
        self._obstacles.append((x, y))
    
    def _get_observation(self) -> np.ndarray:
        """获取观察值 - 返回RGB图像数组"""
        # 绘制当前游戏状态到surface
        self._draw_surface()
        
        # 将pygame surface转换为numpy数组
        rgb_array = pygame.surfarray.array3d(self._surface)
        
        # 转置数组从 (width, height, 3) 到 (height, width, 3)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        
        return rgb_array
    
    def _load_images(self):
        """加载游戏图像资源"""
        images = {}
        
        try:
            # 创建玩家图像（简单的彩色矩形）
            player_surface = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(player_surface, RED, (0, 0, PLAYER_WIDTH, PLAYER_HEIGHT))
            pygame.draw.circle(player_surface, BLUE, (PLAYER_WIDTH//2, PLAYER_HEIGHT//2), PLAYER_WIDTH//3)
            images["player"] = player_surface
            
            # 创建障碍物图像
            obstacle_surface = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(obstacle_surface, GREEN, (0, 0, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
            pygame.draw.rect(obstacle_surface, BLACK, (5, 5, OBSTACLE_WIDTH-10, OBSTACLE_HEIGHT-10), 2)
            images["obstacle"] = obstacle_surface
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            self._use_images = False
        
        return images
    
    def _make_display(self) -> None:
        """初始化pygame显示"""
        if self.render_mode == "human":
            self._display = pygame.display.set_mode((self._screen_width, self._screen_height))
            pygame.display.set_caption("滑雪小游戏")
        else:
            # 对于rgb_array模式，我们不需要实际显示窗口，但需要初始化pygame
            if not pygame.get_init():
                pygame.init()
    
    def _draw_surface(self) -> None:
        """绘制游戏画面到surface"""
        # 绘制背景
        self._surface.fill(BACKGROUND_COLOR)
        
        # 绘制雪地（底部）
        pygame.draw.rect(self._surface, SNOW_COLOR, (0, self._screen_height * 0.7, self._screen_width, self._screen_height * 0.3))
        
        # 绘制障碍物
        for obstacle in self._obstacles:
            x, y = obstacle
            if self._use_images and self._images:
                # 使用图像绘制障碍物
                self._surface.blit(self._images["obstacle"], (x, y))
            else:
                # 使用矩形绘制障碍物
                pygame.draw.rect(self._surface, GREEN, (x, y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
                pygame.draw.rect(self._surface, BLACK, (x+5, y+5, OBSTACLE_WIDTH-10, OBSTACLE_HEIGHT-10), 2)
        
        # 绘制玩家
        if self._use_images and self._images:
            # 旋转玩家图像以匹配角度
            rotated_player = pygame.transform.rotate(self._images["player"], -self._player_angle)
            rect = rotated_player.get_rect(center=(self._player_x + PLAYER_WIDTH//2, self._player_y + PLAYER_HEIGHT//2))
            self._surface.blit(rotated_player, rect)
        else:
            # 使用矩形和三角形绘制玩家
            player_rect = pygame.Rect(self._player_x, self._player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
            pygame.draw.rect(self._surface, RED, player_rect)
            
            # 根据角度绘制方向指示器
            center_x = self._player_x + PLAYER_WIDTH // 2
            center_y = self._player_y + PLAYER_HEIGHT // 2
            
            # 绘制方向三角形
            angle_rad = np.radians(self._player_angle)
            end_x = center_x + 15 * np.sin(angle_rad)
            end_y = center_y - 15 * np.cos(angle_rad)  # 负号因为y轴向下
        
            pygame.draw.line(self._surface, BLUE, (center_x, center_y), (end_x, end_y), 3)
            pygame.draw.circle(self._surface, BLUE, (int(end_x), int(end_y)), 5)
        
        # 绘制分数
        score_text = self._font.render(f"距离: {int(self._distance)}", True, BLACK)
        self._surface.blit(score_text, (10, 10))
        
        # 绘制速度
        speed_text = self._font.render(f"速度: {self._player_speed:.1f}", True, BLACK)
        self._surface.blit(speed_text, (10, 50))
    
    def _update_display(self) -> None:
        """更新显示（仅用于human模式）"""
        if self.render_mode == "human" and self._display is not None:
            pygame.event.get()
            self._display.blit(self._surface, [0, 0])
            pygame.display.update()


# 创建环境的函数
def make_skiing_env(env_name, **kwargs):
    """创建滑雪环境（模仿gymnasium的make函数）"""
    if env_name == "Skiing-rgb-v0":
        return SkiingRGBEnv(**kwargs)
    else:
        raise ValueError(f"未知的环境名称: {env_name}")


# 处理图像的函数（与Flappy Bird相同的处理流程）
def process_img(image):
    """处理RGB图像，返回二值化后的灰度图像"""
    import cv2
    
    # 确保图像是numpy数组
    if isinstance(image, tuple):
        # 如果传入的是(obs, info)元组，取第一个元素
        image = image[0]
    
    # 调整大小到84x84
    image = cv2.resize(image, (84, 84))
    
    # 转换为灰度图
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 二值化处理（阈值199，大于阈值的设为1，其他设为0）
    _, binary_image = cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV)
    
    return binary_image


# 测试代码
# if __name__ == "__main__":
#     # 测试环境
#     env = make_skiing_env("Skiing-rgb-v0", render_mode="human", debug=True)
    
#     # 测试process_img函数
#     obs, info = env.reset()
#     print(f"原始观测形状: {obs.shape}")  # 应该是(512, 288, 3)
    
#     processed_obs = process_img(obs)
#     print(f"处理后的观测形状: {processed_obs.shape}")  # 应该是(84, 84)
    
#     # 模拟DQN的帧堆叠
#     obs_stack = np.expand_dims(processed_obs, axis=0)  # 添加批次维度
#     obs_stack = np.repeat(obs_stack, 4, axis=0)  # 堆叠4帧
#     print(f"堆叠后的观测形状: {obs_stack.shape}")  # 应该是(4, 84, 84)
    
#     # 运行几帧测试
#     running = True
#     for i in range(100):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
        
#         # 处理图像
#         processed_obs = process_img(obs)
        
#         if terminated or truncated:
#             obs, info = env.reset()
#             processed_obs = process_img(obs)
        
#         # 处理退出事件
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 break
        
#         if not running:
#             break
    
#     env.close()