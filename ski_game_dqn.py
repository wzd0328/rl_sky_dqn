import pygame
import sys
import random
import cv2
import numpy as np

WIDTH, HEIGHT = 400, 600           # 画面稍窄，方便卷积
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ski-DQN")
clock = pygame.time.Clock()

class Skier(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((30, 50))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect(center=(WIDTH//2, HEIGHT-80))
        self.speed_x = 0
        self.speed_y = 4          # 基础下落速度

    def steer(self, dx):
        self.speed_x = dx

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        # 限制左右边界
        self.rect.x = max(0, min(WIDTH-self.rect.width, self.rect.x))

class Tree(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill((0, 150, 0))
        self.rect = self.image.get_rect(topleft=(x, y))

    def update(self, dy):
        self.rect.y += dy

class SkiEnv:
    """
    DQN 专用接口，模仿 gym：
    obs = env.reset()        -> 4×80×80  uint8
    obs,reward,done = env.step(action)  action in 0~4
    """
    ACTIONS = 5
    ANGLE_MAP = [-45, -15, 0, 15, 45]   # 对应动作 0~4

    def __init__(self):
        self.skier = Skier()
        self.trees = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.skier)
        self.score = 0
        self.base_speed = 4
        self._spawn_trees()
        self.frame = 0

    def _spawn_trees(self):
        # 初始 8 棵树
        for _ in range(8):
            t = Tree(random.randint(0, WIDTH-40), random.randint(-200, -40))
            self.trees.add(t)
            self.all_sprites.add(t)

    def _get_observation(self):
        # 渲染→灰度→resize→二值→归一 0/1
        arr = pygame.surfarray.array3d(screen)
        arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (80, 80))
        _, bw = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)
        return bw.astype(np.uint8)

    def reset(self):
        self.__init__()
        first_obs = self._get_observation()
        self.obs_buffer = np.stack([first_obs]*4, axis=0)
        return self.obs_buffer

    def step(self, action):
        assert 0 <= action < self.ACTIONS
        # 根据动作计算横向速度
        angle = self.ANGLE_MAP[action]
        dx = int(6 * np.tan(np.radians(angle)))
        self.skier.steer(dx)

        # 更新精灵
        speed = self.base_speed + self.score//500   # 随得分加速
        self.skier.update()
        for t in self.trees:
            t.update(speed)
            if t.rect.top > HEIGHT:
                t.rect.bottom = random.randint(-200, -40)
                t.rect.x = random.randint(0, WIDTH-40)

        # 碰撞检测
        done = False
        reward = 1
        if pygame.sprite.spritecollideany(self.skier, self.trees):
            reward = -1000
            done = True
        if self.skier.rect.top < 0 or self.skier.rect.bottom > HEIGHT:
            reward = -1000
            done = True

        self.score += 1

        # 渲染（训练时可注释掉 pygame.display.flip() 加速）
        screen.fill(WHITE)
        self.all_sprites.draw(screen)
        pygame.display.flip()
        clock.tick(120)          # 训练时帧率可再提高

        # 构造下一状态
        obs = self._get_observation()
        self.obs_buffer = np.append(self.obs_buffer[1:], obs[np.newaxis, :, :], axis=0)
        return self.obs_buffer, reward, done

# 仅供测试
if __name__ == "__main__":
    env = SkiEnv()
    obs = env.reset()
    for _ in range(1000):
        a = random.randint(0, 4)
        obs, r, done = env.step(a)
        if done:
            obs = env.reset()