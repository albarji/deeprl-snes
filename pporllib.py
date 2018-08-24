# Agent that learns how to play a SNES game by using RLLib implementation of PPO

import retro
import gym
import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env
from ray.tune.logger import pretty_print
import numpy as np
import cv2
from collections import deque


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers. This object should only be
        converted to numpy array before being passed to the model. You'd not
        believe how complex the previous solution was.

        Source: https://github.com/openai/sonic-on-ray/blob/master/sonic_on_ray/sonic_on_ray.py
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        """Stack the k last frames.
        Returns a lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames

        Source: https://github.com/openai/sonic-on-ray/blob/master/sonic_on_ray/sonic_on_ray.py
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(shp[0], shp[1], shp[2] * k),
                                                dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        Source: https://github.com/openai/sonic-on-ray/blob/master/sonic_on_ray/sonic_on_ray.py
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 80
        self.height = 80
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 1),
                                                dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO. This is incredibly important
    and effects performance a lot.
    """
    def __init__(self, env, rewardscaling=1):
        self.rewardscaling = rewardscaling
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return reward * self.rewardscaling


class SnesDiscretizer(gym.ActionWrapper):
    """Wrap a gym-retro environment and make it use discrete actions for a SNES game.

    This encodes the prior knowledge that combined actions like UP + RIGHT might be valuable,
    but actions like A + B + RIGHT + L do not make sense (in general). It also helps making the
    environment much simpler.
    """
    def __init__(self, env):
        super(SnesDiscretizer, self).__init__(env)
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        actions = [
            ['DOWN'], ['LEFT'], ['RIGHT'], ['UP'],  # Basic directions
            ['DOWN', 'RIGHT'], ['RIGHT', 'UP'], ['UP', 'LEFT'], ['LEFT', 'DOWN'],  # Diagonals
            ['A'], ['B'], ['X'], ['Y'], ['L'], ['R'], ['SELECT'], ['START']  # Buttons
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


def make_env(game, state, rewardscaling=1):
    """Creates the SNES environment"""
    env = retro.make(game=game, state=state)
    env = RewardScaler(env, rewardscaling)
    env = SnesDiscretizer(env)
    env = WarpFrame(env)
    env = FrameStack(env)
    return env


def register_snes(game, state):
    """Registers a given SNES game as a ray environment

    The environment is registered with name 'snes_env'
    """
    register_env("snes_env", lambda env_config: make_env(game=game, state=state))


def train(checkpoint):
    """Trains a policy network"""
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    print("PPO config:", config)
    agent = ppo.PPOAgent(config=config, env="snes_env")
    try:
        agent.restore(checkpoint)
        print(f"Resumed checkpoint {checkpoint}")
    except:
        print("Restarted policy network from scratch")

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = agent.train()
        print(pretty_print(result))

        if i % 10 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)


def test(checkpoint, num_steps=10000):
    """Tests and renders a previously trained model"""
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1
    agent = ppo.PPOAgent(config=config, env="snes_env")
    agent.restore(checkpoint)
    env = agent.local_evaluator.env
    steps = 0
    while steps < (num_steps or steps + 1):
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            env.render()
            steps += 1
            state = next_state
        print("Episode reward", reward_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that learns how to play a SNES game by using RLLib.')
    parser.add_argument('game', type=str, help='Game to play. Must be a valid Gym Retro game')
    parser.add_argument('state', type=str, help='State (level) of the game to play')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file in which to save learning progress')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no policy updates, render environment)')

    args = parser.parse_args()

    register_snes(args.game, args.state)
    if args.test:
        test(checkpoint=args.checkpoint)
    else:
        train(checkpoint=args.checkpoint)
