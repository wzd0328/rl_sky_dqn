from enum import IntEnum
from typing import Dict, Optional, Tuple, Union
import gymnasium
import numpy as np
import pygame
import os
from pathlib import Path

# 常量定义
SCREEN_WIDTH = 628
SCREEN_HEIGHT = 700
PLAYER_WIDTH = 80
PLAYER_HEIGHT = 100
OBSTACLE_WIDTH = 60
OBSTACLE_HEIGHT = 80
INITIAL_PLAYER_SPEED = 4
MAX_PLAYER_SPEED = 15
ACCELERATION_RATE = 0.001
OBSTACLE_GENERATION_RATE_INITIAL = 0.02
OBSTACLE_GENERATION_RATE_MAX = 0.1
# OBSTACLE_VEL_Y = 3
PLAYER_ANGLE_CHANGE = 15  # 角度变化量
ACTION_CONSISTENCY_REWARD = 0.02  # 动作平滑性奖励

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BACKGROUND_COLOR = (255, 206, 235)  # 背景颜色
SNOW_COLOR = (255, 250, 250)  # 雪地颜色

class Actions(IntEnum):
    """滑雪者的可能动作-5种"""
    STRAIGHT, LEFT_15, LEFT_45, RIGHT_15, RIGHT_45, KEEP_CURRENT = 0, 1, 2, 3, 4, 5

class SkiingRGBEnv(gymnasium.Env):
    """滑雪小游戏环境-返回RGB图像
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
        
        # 动作空间：6种动作
        self.action_space = gymnasium.spaces.Discrete(6)

        # 动作平滑性奖励
        self._action_consistency_reward = ACTION_CONSISTENCY_REWARD
        
        # 观察空间：RGB图像 (高度, 宽度, 3通道)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, 
            shape=(screen_size[1], screen_size[0], 3),  # (height, width, channels)
            dtype=np.uint8
        )
            
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]

        # 背景滚动相关属性
        self._bg_scroll_offset = 0  # 背景滚动偏移量
        self._bg_scroll_speed = INITIAL_PLAYER_SPEED
        
        # 游戏状态变量
        self._player_x = 0
        self._player_y = 0
        self._player_angle = 0  # 角度，0表示垂直向下
        self._player_speed = INITIAL_PLAYER_SPEED
        self._obstacles = []  # 障碍物列表，每个障碍物是(x, y)元组
        self._score = 0  # 分数为垂直下滑距离
        self._distance = 0  # 下滑总距离
        self._obstacle_generation_rate = OBSTACLE_GENERATION_RATE_INITIAL
        self._game_over = False
        
        if not pygame.get_init():
            pygame.init()

        if not pygame.font.get_init():
            pygame.font.init()

        if render_mode == "human":
            # 加载并播放背景音乐
            pygame.mixer.music.load('./assets/music/troublemaker.mp3')  # 确保路径正确
            pygame.mixer.music.play(-1)  # -1表示循环播放

        self._font = pygame.font.Font(None, 36)
        # 预加载游戏结束字体
        self._game_over_font_large = pygame.font.Font(None, 72)
        self._game_over_font_medium = pygame.font.Font(None, 36)
        self._game_over_font_small = pygame.font.Font(None, 24)

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
        # 如果游戏已经结束，直接返回当前状态
        if self._game_over:
            obs = self._get_observation()
            return obs, 0, True, False, {"score": self._score, "game_over": True}
        
        terminal = False
        reward = 0.1  # 存活奖励
        action_bonus = 0.0  # 动作一致性奖励

        self._bg_scroll_offset += self._bg_scroll_speed
        
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
        elif action == Actions.KEEP_CURRENT:
            # 保持当前角度不变, 给予小奖励
            action_bonus = self._action_consistency_reward
            reward += action_bonus
        
        # 根据角度计算水平移动
        angle_rad = np.radians(self._player_angle)
        horizontal_move = self._player_speed * np.sin(angle_rad)

        # 更新玩家位置
        self._player_x += horizontal_move
        
        # 增加下滑距离和分数
        self._distance += self._player_speed * 0.1
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
            x, y, obs_type = obstacle
            y -= self._player_speed  # 障碍物向上移动，使用玩家速度衡量
            
            # 只保留还在屏幕内的障碍物
            if y > self._player_y:
                new_obstacles.append((x, y, obs_type))
        self._obstacles = new_obstacles
        
        # 检查游戏结束条件
        # 1. 超出屏幕左右边界
        if self._player_x < PLAYER_WIDTH // 3 or self._player_x > self._screen_width - PLAYER_WIDTH // 3:
            terminal = True
            reward = -10.0
            self._game_over = True
            if self._debug:
                print("游戏结束：超出屏幕边界")
        
        # 2. 碰到障碍物
        else:
            def circle_collision(center1_x, center1_y, radius1, center2_x, center2_y, radius2):
                """圆形碰撞检测"""
                distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
                return distance < (radius1 + radius2)

            # 使用圆形碰撞检测
            player_center_x = self._player_x
            player_center_y = self._player_y
            player_radius = min(PLAYER_WIDTH, PLAYER_HEIGHT) // 2 * 0.7  # 使用70%的半径，让碰撞更合理

            for obstacle in self._obstacles:
                obs_center_x, obs_center_y, obs_type = obstacle
                obs_radius = min(OBSTACLE_WIDTH, OBSTACLE_HEIGHT) // 2 * 0.7
                
                if circle_collision(player_center_x, player_center_y, player_radius,
                                obs_center_x, obs_center_y, obs_radius) and player_center_y < obs_center_y:
                    terminal = True
                    reward = -10.0
                    self._game_over = True
                    self._final_score = self._score
                    if self._debug:
                        print("游戏结束：碰到障碍物（圆形检测）")
                    break
            # player_rect = pygame.Rect(self._player_x - PLAYER_WIDTH // 2, self._player_y - PLAYER_HEIGHT // 2, PLAYER_WIDTH, PLAYER_HEIGHT)
            # for obstacle in self._obstacles:
            #     obstacle_rect = pygame.Rect(obstacle[0]- OBSTACLE_WIDTH // 2, obstacle[1] - OBSTACLE_HEIGHT // 2, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
            #     if player_rect.colliderect(obstacle_rect):
            #         terminal = True
            #         reward = -10.0
            #         self._game_over = True
            #         if self._debug:
            #             print("游戏结束：碰到障碍物")
            #         break
        
        # 获取观察值（RGB图像）
        obs = self._get_observation()
        
        # 如果是human模式，额外渲染到窗口
        if self.render_mode == "human":
            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])
        
        info = {"score": self._score, "distance": self._distance, "speed": self._player_speed, "game_over": self._game_over}
        
        return obs, reward, terminal, False, info
    
    def reset(self, seed=None, options=None):
        """重置游戏状态"""
        super().reset(seed=seed)
        if options is not None:
            self._use_images = options["use_images"]
        # 重置游戏状态
        self._game_over = False

        self._bg_scroll_offset = 0

        # 重置玩家状态
        self._player_x = self._screen_width // 2
        self._player_y = 120
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
        half_width = OBSTACLE_WIDTH // 2
        half_height = OBSTACLE_HEIGHT // 2
        
        # 障碍物中心坐标的有效范围
        min_x = half_width
        max_x = self._screen_width - half_width
        min_y = half_height
        max_y = self._screen_height - half_height
        
        # 新障碍物从屏幕底部生成
        y = self._screen_height - half_height
        
        # 安全间距
        safety_margin = PLAYER_WIDTH + 5
        
        # 最大尝试次数
        max_attempts = 50
        attempts = 0

        obs_type = np.random.randint(0, 2) # 随机选择障碍物类型（0或1）
        
        while attempts < max_attempts:
            # 随机生成障碍物中心x坐标
            x = np.random.randint(min_x, max_x + 1)
            
            # 创建带安全间距的检测区域
            detection_rect = pygame.Rect(
                x - half_width - safety_margin,      # 左边界（带间距）
                y - half_height - safety_margin,     # 上边界（带间距）
                OBSTACLE_WIDTH + 2 * safety_margin,  # 宽度（带间距）
                OBSTACLE_HEIGHT + 2 * safety_margin  # 高度（带间距）
            )
            
            # 检查是否与现存障碍物重叠（考虑安全间距）
            overlap = False
            for existing_obstacle in self._obstacles:
                existing_x, existing_y, _ = existing_obstacle
                existing_detection_rect = pygame.Rect(
                    existing_x - half_width - safety_margin,
                    existing_y - half_height - safety_margin,
                    OBSTACLE_WIDTH + 2 * safety_margin,
                    OBSTACLE_HEIGHT + 2 * safety_margin
                )
                
                if detection_rect.colliderect(existing_detection_rect):
                    overlap = True
                    break
            
            # 如果没有重叠，添加障碍物
            if not overlap:
                self._obstacles.append((x, y, obs_type))
                return
            
            attempts += 1
        
        # 备用策略：如果无法找到理想位置，尝试在现有障碍物之间寻找空隙
        if attempts >= max_attempts and self._obstacles:
            # 对现有障碍物按x坐标排序
            sorted_obstacles = sorted(self._obstacles, key=lambda obs: obs[0])
            
            # 查找障碍物之间的空隙
            for i in range(len(sorted_obstacles)):
                if i == 0:
                    # 检查最左边的空隙
                    left_boundary = half_width
                    right_boundary = sorted_obstacles[0][0] - half_width
                    gap = right_boundary - left_boundary
                    
                    if gap >= OBSTACLE_WIDTH + 2 * safety_margin:
                        x = left_boundary + gap // 2
                        self._obstacles.append((x, y, obs_type))
                        return
                
                if i == len(sorted_obstacles) - 1:
                    # 检查最右边的空隙
                    left_boundary = sorted_obstacles[i][0] + half_width
                    right_boundary = self._screen_width - half_width
                    gap = right_boundary - left_boundary
                    
                    if gap >= OBSTACLE_WIDTH + 2 * safety_margin:
                        x = left_boundary + gap // 2
                        self._obstacles.append((x, y, obs_type))
                        return
                else:
                    # 检查中间的空隙
                    left_obstacle = sorted_obstacles[i]
                    right_obstacle = sorted_obstacles[i + 1]
                    
                    left_boundary = left_obstacle[0] + half_width
                    right_boundary = right_obstacle[0] - half_width
                    gap = right_boundary - left_boundary
                    
                    if gap >= OBSTACLE_WIDTH + 2 * safety_margin:
                        x = left_boundary + gap // 2
                        self._obstacles.append((x, y, obs_type))
                        return
    
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
            # # 创建玩家图像（简单的彩色矩形）
            # player_surface = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT), pygame.SRCALPHA)
            # pygame.draw.rect(player_surface, RED, (0, 0, PLAYER_WIDTH, PLAYER_HEIGHT))
            # pygame.draw.circle(player_surface, BLUE, (PLAYER_WIDTH//2, PLAYER_HEIGHT//2), PLAYER_WIDTH//3)
            # images["player"] = player_surface
            
            # # 创建障碍物图像
            # obstacle_surface = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT), pygame.SRCALPHA)
            # pygame.draw.rect(obstacle_surface, GREEN, (0, 0, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
            # pygame.draw.rect(obstacle_surface, BLACK, (5, 5, OBSTACLE_WIDTH-10, OBSTACLE_HEIGHT-10), 2)
            # images["obstacle"] = obstacle_surface

            # 加载玩家5种动作图像
            assets_dir = Path("./assets/figure")
            player_images = {}
            actions = ["straight", "left_15", "left_45", "right_15", "right_45"]
            for action in actions:
                img_path = assets_dir / f"{action}.png"
                if img_path.exists():
                    player_surface = pygame.image.load(str(img_path)).convert_alpha()
                    # 调整大小到标准尺寸
                    if action == "straight":
                        player_surface = pygame.transform.scale(player_surface, (PLAYER_WIDTH - 15, PLAYER_HEIGHT))
                    else:
                        player_surface = pygame.transform.scale(player_surface, (PLAYER_WIDTH, PLAYER_HEIGHT))
                    player_images[action] = player_surface
            images["player_straight"] = player_images["straight"]
            images["player_left_15"] = player_images["left_15"]
            images["player_left_45"] = player_images["left_45"]
            images["player_right_15"] = player_images["right_15"]
            images["player_right_45"] = player_images["right_45"]

            # 加载障碍物图像
            obstacle_images = []
            obstacle_names = ["stone.png", "tree.png"]
            for obs_name in obstacle_names:
                obstacle_path = assets_dir / obs_name
                if obstacle_path.exists():
                    obstacle_surface = pygame.image.load(str(obstacle_path)).convert_alpha()
                    obstacle_surface = pygame.transform.scale(obstacle_surface, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
                    obstacle_images.append(obstacle_surface)
            images["obstacle"] = obstacle_images

            # 加载背景图像
            bg_path = assets_dir / "snow.jpg"
            if bg_path.exists():
                bg_surface = pygame.image.load(str(bg_path)).convert()
                bg_surface = pygame.transform.scale(bg_surface, (self._screen_width, self._screen_height))
                images["background"] = bg_surface
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            self._use_images = False
        
        return images
    
    def _make_display(self) -> None:
        """初始化pygame显示"""
        if self.render_mode == "human" or self._use_images:
            self._display = pygame.display.set_mode((self._screen_width, self._screen_height))
            pygame.display.set_caption("滑雪小游戏")
        else:
            # 对于rgb_array模式，我们不需要实际显示窗口，但需要初始化pygame
            if not pygame.get_init():
                pygame.init()
    
    def _draw_surface(self) -> None:
        """绘制游戏画面到surface"""
        # 绘制背景
        if self._use_images and self._images.get("background"):
            bg_surface = self._images["background"]
            bg_surface1 = pygame.transform.flip(bg_surface, False, True)
            frameRect = bg_surface.get_rect()
            i = self._bg_scroll_offset % frameRect.height
            self._surface.blit(bg_surface1, (0, -i))
            self._surface.blit(bg_surface, (0, frameRect.height-i))
            # pygame.display.flip()
            # self._surface.blit(self._images["background"], (0, 0))
        else:
            self._surface.fill(SNOW_COLOR)
            # 绘制雪地（底部）
            # pygame.draw.rect(self._surface, SNOW_COLOR, (0, self._screen_height * 0.7, self._screen_width, self._screen_height * 0.3))
        
        if not self._game_over:
            # 绘制障碍物
            for obstacle in self._obstacles:
                x, y, obs_type = obstacle
                if self._use_images and self._images:
                    # 使用图像绘制障碍物，坐标为中心
                    obstacle_img = self._images["obstacle"][obs_type]
                    img_rect = obstacle_img.get_rect(center=(x, y))
                    self._surface.blit(obstacle_img, img_rect)
                else:
                    # 使用矩形绘制障碍物
                    half_width = OBSTACLE_WIDTH // 2
                    half_height = OBSTACLE_HEIGHT // 2
                    obstacle_rect = pygame.Rect(x - half_width, y - half_height, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
                    pygame.draw.rect(self._surface, GREEN, obstacle_rect)
                    pygame.draw.rect(self._surface, BLACK, obstacle_rect.inflate(-10, -10), 2)
            
            # 绘制玩家
            if self._use_images and self._images:
                if self._player_angle == 0:
                    player_img = self._images["player_straight"]
                elif self._player_angle == -15:
                    player_img = self._images["player_left_15"]
                elif self._player_angle == -45:
                    player_img = self._images["player_left_45"]
                elif self._player_angle == 15:
                    player_img = self._images["player_right_15"]
                elif self._player_angle == 45:
                    player_img = self._images["player_right_45"]
                else:
                    player_img = self._images["player_straight"]
                player_rect = player_img.get_rect(center=(self._player_x, self._player_y))
                self._surface.blit(player_img, player_rect)
            else:
                player_surface = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT), pygame.SRCALPHA)
                pygame.draw.rect(player_surface, RED, (0, 0, PLAYER_WIDTH, PLAYER_HEIGHT))
                pygame.draw.circle(player_surface, BLUE, (PLAYER_WIDTH//2, PLAYER_HEIGHT//2), PLAYER_WIDTH//3)
                # 旋转玩家图像以匹配角度
                rotated_player = pygame.transform.rotate(player_surface, -self._player_angle)
                rect = rotated_player.get_rect(center=(self._player_x, self._player_y))
                self._surface.blit(rotated_player, rect)
                # # 使用矩形和三角形绘制玩家
                # player_rect = pygame.Rect(self._player_x, self._player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
                # pygame.draw.rect(self._surface, RED, player_rect)
                
                # # 根据角度绘制方向指示器
                # center_x = self._player_x + PLAYER_WIDTH // 2
                # center_y = self._player_y + PLAYER_HEIGHT // 2
                
                # # 绘制方向三角形
                # angle_rad = np.radians(self._player_angle)
                # end_x = center_x + 15 * np.sin(angle_rad)
                # end_y = center_y - 15 * np.cos(angle_rad)  # 负号因为y轴向下
            
                # pygame.draw.line(self._surface, BLUE, (center_x, center_y), (end_x, end_y), 3)
                # pygame.draw.circle(self._surface, BLUE, (int(end_x), int(end_y)), 5)
            
            # 绘制分数
            score_text = self._font.render(f"Score: {int(self._distance)}", True, BLACK)
            self._surface.blit(score_text, (10, 10))
            
            # 绘制速度
            speed_text = self._font.render(f"Speed: {self._player_speed:.1f}", True, BLACK)
            self._surface.blit(speed_text, (10, 50))
        else:
            # 游戏结束状态，绘制结束界面
            self._draw_game_over_screen()
    
    def _draw_game_over_screen(self) -> None:
        """绘制游戏结束界面"""
        # 半透明遮罩
        overlay = pygame.Surface((self._screen_width, self._screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # 半透明黑色
        self._surface.blit(overlay, (0, 0))
        
        # 游戏结束标题
        game_over_text = self._game_over_font_large.render("Game Over", True, WHITE)
        game_over_rect = game_over_text.get_rect(center=(self._screen_width // 2, self._screen_height // 2 - 80))
        self._surface.blit(game_over_text, game_over_rect)
        
        # 最终得分
        score_text = self._game_over_font_medium.render(f"Final Score: {int(self._score)}", True, WHITE)
        score_rect = score_text.get_rect(center=(self._screen_width // 2, self._screen_height // 2 - 20))
        self._surface.blit(score_text, score_rect)
        
        # 重新开始按钮
        restart_button = pygame.Rect(self._screen_width // 2 - 100, self._screen_height // 2 + 30, 200, 50)
        pygame.draw.rect(self._surface, GREEN, restart_button, border_radius=10)
        pygame.draw.rect(self._surface, WHITE, restart_button, 2, border_radius=10)
        
        restart_text = self._game_over_font_small.render("Restart", True, BLACK)
        restart_text_rect = restart_text.get_rect(center=restart_button.center)
        self._surface.blit(restart_text, restart_text_rect)
        
        # 退出按钮
        quit_button = pygame.Rect(self._screen_width // 2 - 100, self._screen_height // 2 + 100, 200, 50)
        pygame.draw.rect(self._surface, RED, quit_button, border_radius=10)
        pygame.draw.rect(self._surface, WHITE, quit_button, 2, border_radius=10)
        
        quit_text = self._game_over_font_small.render("Exit", True, WHITE)
        quit_text_rect = quit_text.get_rect(center=quit_button.center)
        self._surface.blit(quit_text, quit_text_rect)
        
        # 提示信息
        hint_text = self._game_over_font_small.render("Or Enter R/Esc", True, WHITE)
        hint_rect = hint_text.get_rect(center=(self._screen_width // 2, self._screen_height // 2 + 170))
        self._surface.blit(hint_text, hint_rect)

    def _update_display(self) -> None:
        """更新显示（仅用于human模式）"""
        if self.render_mode == "human" and self._display is not None:
            # # 处理事件，包括游戏结束时的按钮点击
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         self.close()
            #         return
            #     elif event.type == pygame.MOUSEBUTTONDOWN and self._game_over:
            #         # 处理游戏结束时的鼠标点击
            #         mouse_pos = pygame.mouse.get_pos()
            #         restart_button = pygame.Rect(self._screen_width // 2 - 100, self._screen_height // 2 + 30, 200, 50)
            #         quit_button = pygame.Rect(self._screen_width // 2 - 100, self._screen_height // 2 + 100, 200, 50)
                    
            #         if restart_button.collidepoint(mouse_pos):
            #             # 触发重新开始（通过外部控制）
            #             pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
            #         elif quit_button.collidepoint(mouse_pos):
            #             # 触发退出
            #             pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            pygame.event.get()
            self._display.blit(self._surface, [0, 0])
            pygame.display.update()


# 创建环境的函数
def make_skiing_env(env_name, **kwargs):
    """创建滑雪环境"""
    if env_name == "Skiing-rgb-v0":
        return SkiingRGBEnv(**kwargs)
    else:
        raise ValueError(f"未知的环境名称: {env_name}")


# # 处理图像的函数（与Flappy Bird相同的处理流程）
# def process_img(image):
#     """处理RGB图像，返回二值化后的灰度图像"""
#     import cv2
    
#     # 确保图像是numpy数组
#     if isinstance(image, tuple):
#         # 如果传入的是(obs, info)元组，取第一个元素
#         image = image[0]
    
#     # 调整大小
#     image = cv2.resize(image, (128, 128))
    
#     # 转换为灰度图
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
#     # 二值化处理（阈值199，大于阈值的设为1，其他设为0）
#     _, binary_image = cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV)
    
#     return binary_image


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