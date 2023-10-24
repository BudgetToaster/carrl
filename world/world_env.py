import numpy as np
import pygame
import torch
from gymnasium import spaces

import map_processor
from world import World
import gymnasium as gym

edges = map_processor.to_edges('squiggle.png')
vertices = map_processor.edges_img_to_vertices(edges)
vertices = map_processor.cut_corners(vertices, 20)


class WorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "fps": 20}

    def __init__(self, render_mode=None, obs_points=36, device='cpu'):
        self.device = device

        self._obs_points = obs_points
        # Shape is for vision + log speed + throttle + turning angle
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_points + 3,))

        # Turning actions (left/right), brake action, throttle action
        self.action_nvec = [5, 3, 5]
        self.action_space = spaces.MultiDiscrete(nvec=self.action_nvec)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.world = World(1)
        self.world.add_edges(vertices, 0.7, -1220 * 0.7, -1000 * 0.7)

        self.window = None
        self.clock = None

        self.reward_total = 0

    def _get_obs(self):
        if len(self.world.vehicles) == 0:
            return np.zeros(self.observation_space.shape)
        car = self.world.vehicles[0]
        speed = car.get_forward_speed()
        return np.concatenate([
            np.array(self.world.get_vision(car, self._obs_points), dtype=np.float32),
            np.array([np.sign(speed) * np.log1p(np.abs(speed))], dtype=np.float32),
            np.array([car.throttle], dtype=np.float32),
            np.array([car.steer_angle], dtype=np.float32)
        ])

    def _get_info(self):
        return {}

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if not self.window:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((640, 480))

        self.world.draw(self.window)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.world = World(1)
        self.world.add_edges(vertices, 0.7, -1220 * 0.7, -1000 * 0.7)

        self.reward_total = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def step(self, action):
        # Turning actions (left/right), brake action, throttle action
        steer = action[0] / (self.action_nvec[0] - 1) * 2 - 1
        steer *= np.pi / 6
        brake = action[1] / (self.action_nvec[1] - 1)
        if action[2] == 0:
            throttle = -0.1
        else:
            throttle = ((action[2] - 1) / (self.action_nvec[1] - 2))**1.5

        if len(self.world.vehicles) > 0:
            car = self.world.vehicles[0]
            car.throttle = throttle
            car.steer_angle = steer
            car.brake = brake
        self.world.update(1 / 20)
        self.world.update(1 / 20)

        terminated = len(self.world.vehicles) == 0
        reward = (self.world.vehicles[0].get_forward_speed() / 10) if not terminated else -10
        observation = self._get_obs()
        info = self._get_info()

        self.reward_total += reward

        return observation, reward, terminated, False, info
