import numpy as np
import pyglet
import torch
from gym import spaces

from world import World
import gym


class WorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "fps": 20}

    def __init__(self, render_mode=None, obs_points=36, action_pts=None):
        self._obs_points = obs_points
        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(0, 1, shape=(obs_points,), dtype=float),
                "speed": spaces.Box(-100, 100, shape=(1,), dtype=float),
                "acceleration": spaces.Box(-100, 100, shape=(1,), dtype=float),
                "throttle": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "turning_angle": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=float)
            }
        )

        if action_pts is None:
            action_pts = [20, 10, 10]
        self.action_pts = action_pts
        assert len(action_pts) == 3

        # Turning actions (left/right), brake action, throttle action
        self.action_space = spaces.MultiDiscrete(action_pts)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.world = World(1)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        car = self.world.vehicles[0]
        return {
            "vision": np.array(self.world.get_vision(car, self._obs_points), dtype=np.float32),
            "speed": np.array(car.get_forward_speed(), dtype=np.float32),
            "throttle": np.array(car.throttle, dtype=np.float32),
            "turning_angle": np.array(car.steer_angle, dtype=np.float32)
        }

    def _get_info(self):
        return {}

    def _render_frame(self):
        if not self.window:
            self.window = pyglet.window.Window(640, 480)

        @self.window.event
        def on_draw():
            self.window.clear()
            self.world.draw(self.window)
            pyglet.app.exit()

        pyglet.app.run(0)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.world = World(1)
        n = 5
        s = 100
        for x in range(-n, n+1):
            shift = self.np_random.random() * 100 - 50
            for y in range(-n, n+1):
                lanes = 2
                ix = x * s
                iy = y * s
                self.world.make_building(ix + 3.65 / 2 * lanes,
                                    iy + 3.65 / 2 * lanes + shift,
                                    ix + s - 3.65 / 2 * lanes,
                                    iy + s - 3.65 / 2 * lanes + shift)
        self.x_bounds = (-n * s - 3.65 / 2 * lanes, (n + 1) * s + 3.65 / 2 * lanes)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Turning actions (left/right), brake action, throttle action
        steer = action[0] / (self.action_pts[0] - 1) * 2 - 1
        steer *= np.pi / 6
        brake = action[1] / (self.action_pts[1] - 1)
        throttle = action[2] / (self.action_pts[2] - 1)

        self.world.update(1/20)

        terminated = len(self.world.vehicles) == 0
        reward = np.log1p(self.world.vehicles[0].get_forward_speed()) if not terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info