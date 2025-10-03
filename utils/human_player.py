import pygame

class HumanPlayer:
    """人工玩家类，处理键盘输入和动作转换"""
    def __init__(self):
        self.current_action = 2  # 初始动作为直行
        self.last_key_time = 0
        self.key_cooldown = 200  # 按键冷却时间(毫秒)

    def get_action_from_keyboard(self):
        """从键盘输入获取动作"""
        current_time = pygame.time.get_ticks()
        
        # 检查按键冷却
        if current_time - self.last_key_time < self.key_cooldown:
            return 2  # 保持当前角度
        
        # 获取按键状态
        keys = pygame.key.get_pressed()
        left_pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right_pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        
        # 检测新按键
        if left_pressed:
            self.last_key_time = current_time
            self.current_action = 0
            print(f"动作变更向左", self.last_key_time)
            return self.current_action
        
        if right_pressed:
            self.last_key_time = current_time
            self.current_action = 1
            print(f"动作变更向右")
            return self.current_action
        
        return 2  # 保持当前角度
    
    def reset(self):
        """重置玩家状态"""
        self.current_action = 2
        self.last_key_time = 0