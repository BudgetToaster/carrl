import time

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import PPO
from tqdm import tqdm

from world.world_env import WorldEnv


def test_ppo():
    device = 'cuda'

    n_envs = 16
    lr = 3e-4

    vec_env = make_vec_env(lambda: WorldEnv('human', device=device, obs_points=8), n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        vec_env,
        lr,
        n_steps=4096,
        gamma=0.95,
        clip_range=.2,
        batch_size=128
    )

    pbar = tqdm()
    best_reward = 0
    def cb(locals, globals):
        nonlocal best_reward
        best_reward = max(vec_env.get_attr('reward_total', list(range(n_envs))))
        pbar.set_postfix({'r': best_reward})
        pbar.update()

    model.learn(10000000, cb)
    model.save("ppo_car")


    env = WorldEnv('human', device=device)
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, _, info = env.step(action)
        env.render()
        time.sleep(1/10)
        if term:
            env.reset()


if __name__ == "__main__":
    test_ppo()