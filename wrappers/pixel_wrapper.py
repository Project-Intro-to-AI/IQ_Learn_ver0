import gym
import numpy as np
import cv2  # uv add opencv-python nếu chưa

class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, pixels_only=False):
        super().__init__(env)
        self.pixels_only = pixels_only
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):
        frame = self.env.render(mode='rgb_array')
        frame = cv2.resize(frame, (84, 84))  # Resize to 84x84 RGB
        frame = frame.astype(np.uint8)  # Ensure dtype
        if self.pixels_only:
            return frame  # Pixel only (H, W, C) for CNN (transpose in forward)
        else:
            # Concat state + pixel flattened (if hybrid)
            pixel_flat = frame.reshape(-1)
            return np.concatenate([obs, pixel_flat])  # State + flattened pixel