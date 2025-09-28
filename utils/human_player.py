import pygame

class HumanPlayer:
    """人工玩家类，处理键盘输入和动作转换"""
    def __init__(self):
        self.current_action = 0  # 初始动作为直行
        self.action_names = ["直行", "左偏15°", "左偏45°", "右偏15°", "右偏45°"]
        self.last_key_time = 0
        self.key_cooldown = 200  # 按键冷却时间(毫秒)
        
        # 动作转换映射表
        self.action_transitions = {
            # 当前动作: {左键按下: 新动作, 右键按下: 新动作}
            0: {"left": 1, "right": 3},  # 直行 → 左15/右15
            1: {"left": 2, "right": 0},  # 左15 → 左45/直行
            2: {"left": 2, "right": 1},  # 左45 → 左45/左15
            3: {"left": 0, "right": 4},  # 右15 → 直行/右45
            4: {"left": 3, "right": 4},  # 右45 → 右15/右45
        }
    
    def get_action_from_keyboard(self):
        """从键盘输入获取动作"""
        current_time = pygame.time.get_ticks()
        
        # 检查按键冷却
        if current_time - self.last_key_time < self.key_cooldown:
            return self.current_action
        
        # 获取按键状态
        keys = pygame.key.get_pressed()
        left_pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right_pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        
        # 检测新按键
        if left_pressed:
            self.last_key_time = current_time
            self.current_action = self.action_transitions[self.current_action]["left"]
            print(f"动作变更: {self.action_names[self.current_action]}")
            return self.current_action
        
        if right_pressed:
            self.last_key_time = current_time
            self.current_action = self.action_transitions[self.current_action]["right"]
            print(f"动作变更: {self.action_names[self.current_action]}")
            return self.current_action
        
        return self.current_action
    
    def reset(self):
        """重置玩家状态"""
        self.current_action = 0
        self.last_key_time = 0