import ale_py  # Đăng ký tự động các env Atari
import gym

# Tạo môi trường để test
env = gym.make("PongNoFrameskip-v4")
print("Môi trường được tạo thành công!")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Đóng env để tránh leak
env.close()