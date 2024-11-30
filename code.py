import numpy as np
import random

# 环境参数
class Environment:
    def __init__(self, num_vulnerabilities, initial_state):
        self.num_vulnerabilities = num_vulnerabilities  # 漏洞数量
        self.state = initial_state  # 系统的初始状态
        self.target_system = self._generate_system()  # 生成目标系统
        self.max_steps = 100  # 最大攻击步骤

    def _generate_system(self):
        # 模拟一个具有多个漏洞的系统，漏洞的严重性随机生成
        return np.random.randint(1, 10, size=self.num_vulnerabilities)

    def reset(self):
        self.state = np.zeros(self.num_vulnerabilities)  # 初始化系统状态
        return self.state

    def step(self, action):
        # 执行攻击动作，返回新的状态和奖励
        reward = 0
        if action < len(self.target_system):
            # 如果选择的漏洞有效
            if self.target_system[action] > 0:
                reward = 10  # 成功攻击到漏洞，获得奖励
                self.target_system[action] = 0  # 漏洞被利用，修复
            else:
                reward = -5  # 无效攻击，惩罚
        self.state = np.copy(self.target_system)  # 更新系统状态
        return self.state, reward

# Q-Learning 参数
class QLearningAgent:
    def __init__(self, num_vulnerabilities, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_vulnerabilities = num_vulnerabilities
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索与利用的平衡
        self.q_table = np.zeros(num_vulnerabilities)  # 初始化Q值表

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.num_vulnerabilities))  # 随机选择动作（探索）
        else:
            return np.argmax(self.q_table)  # 选择Q值最高的动作（利用）

    def learn(self, state, action, reward, next_state):
        # Q-learning 更新公式
        best_next_action = np.argmax(self.q_table)
        td_target = reward + self.gamma * self.q_table[best_next_action]
        self.q_table[action] += self.alpha * (td_target - self.q_table[action])

# 主程序
def run_simulation():
    num_vulnerabilities = 5  # 假设有5个漏洞
    initial_state = np.zeros(num_vulnerabilities)  # 初始系统状态
    env = Environment(num_vulnerabilities, initial_state)  # 创建环境
    agent = QLearningAgent(num_vulnerabilities)  # 创建Q-Learning智能体

    num_episodes = 500  # 训练轮数
    for episode in range(num_episodes):
        state = env.reset()  # 重置环境
        total_reward = 0

        for step in range(env.max_steps):
            action = agent.choose_action(state)  # 选择攻击动作
            next_state, reward = env.step(action)  # 执行动作并得到新的状态和奖励
            agent.learn(state, action, reward, next_state)  # 智能体学习
            total_reward += reward  # 累计奖励
            if np.sum(env.target_system) == 0:  # 如果所有漏洞都被利用，结束本轮
                print(f"Episode {episode + 1}: All vulnerabilities exploited.")
                break
            state = next_state

        if (episode + 1) % 50 == 0:  # 每50轮打印一次
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print("Training finished.")

    # 测试阶段：评估代理表现
    state = env.reset()
    total_reward = 0
    for step in range(env.max_steps):
        action = agent.choose_action(state)  # 测试时仍然使用策略选择动作
        next_state, reward = env.step(action)
        total_reward += reward
        if np.sum(env.target_system) == 0:
            print(f"Test: All vulnerabilities exploited.")
            break
        state = next_state
    print(f"Test Total Reward: {total_reward}")

# 运行模拟
run_simulation()
