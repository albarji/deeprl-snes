import numpy as np
import cv2
from collections import deque
import gym
import moviepy.editor as mpy
import yaml
from skimage import color
import string
import random

# Module configuration


def load_actions_config(actionsfile):
    """Loads an checks sanity of actions configurations"""
    with open(actionsfile) as f:
        config = yaml.load(f)
    assert "gamepads" in config
    assert "games" in config
    for game in config["games"]:
        assert "gamepad" in config["games"][game]
        assert "actions" in config["games"][game]
    return config


ACTIONS_FILE = "games.yaml"
ACTIONS_CONFIG = load_actions_config(ACTIONS_FILE)


# Environment wrappers


class SkipFrames(gym.Wrapper):
    """Gym wrapper that skips n-1 out every n frames.

    This helps training since frame-precise actions are not really needed.
    In the skipped frames the last performed action is repeated, or if a
    pad_action is provided, such action is used.
    """
    def __init__(self, env, skip=4, pad_action=None):
        gym.Wrapper.__init__(self, env)
        self._skip = skip
        self._pad_action = pad_action

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        obs = done = info = None
        for i in range(self._skip):
            if i == 0 or self._pad_action is None:
                doact = action
            else:
                doact = self._pad_action
            obs, reward, done, info = self.env.step(doact)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()


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
    def __init__(self, env, togray=True):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        Optionally also transform to a single grayscale channel.

        Source: https://github.com/openai/sonic-on-ray/blob/master/sonic_on_ray/sonic_on_ray.py
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.togray = togray
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.height, self.width, 1 if togray else 3),
            dtype=np.uint8
        )

    def observation(self, frame):
        if self.togray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        if self.togray:
            return frame[:, :, None]
        else:
            return frame[:, :, :]


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


class RewardClipper(gym.RewardWrapper):
    """Clips the rewards to {-1, 0, 1}

    Can be useful to prevent the agent getting crazy about very large rewards.
    """
    def reward(self, reward):
        return np.sign(reward)


class RewardTimeDump(gym.RewardWrapper):
    """Adds a small negative reward per step.

    This kind of penalty can be useful to enforce finding fast solutions. But it should not
    be large enough to make the agent converge to dying fast.
    """
    def __init__(self, env, penalty=1e-3):
        self.penalty = penalty
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return reward - self.penalty


class NoopResetEnv(gym.Wrapper):
    """Performs no-action a random number of frames at the beginning of each episode

    Reference: https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/atari_wrappers.py#L79
    """
    def __init__(self, env, noop_max=30, noop_action=0):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = noop_action

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        """Normal step"""
        return self.env.step(ac)


class ButtonsRemapper(gym.ActionWrapper):
    """Wrap a gym-retro environment and make it use discrete actions according to a given button remap

    As initialization parameters one must provide the list of buttons available in the console
    gamepad, and a list of actions that will be considered in the game. Each action is a list of
    buttons being pushed simultaneously.
    """
    def __init__(self, env, game):
        super(ButtonsRemapper, self).__init__(env)
        if game not in ACTIONS_CONFIG["games"]:
            raise ValueError(f"ERROR: unknown game {game}, could not discretize actions. "
                             f"Please add game configuration to {ACTIONS_FILE}")
        gameconfig = ACTIONS_CONFIG["games"][game]
        gamepad, actions = gameconfig["gamepad"], gameconfig["actions"]
        buttonmap = ACTIONS_CONFIG["gamepads"][gamepad]
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttonmap))
            for button in action:
                arr[buttonmap.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class MovieRecorder(gym.ObservationWrapper):
    """Gym wrapper that records gameplay into video files."""
    MODES = {"all", "best"}

    def __init__(self, env, fileprefix, mode="all"):
        """Creates a movie recorder wrapper

            env: environment to wrap
            fileprefix: prefix for created movie files, to be followed by reward obtained
            mode: "all" to save a movie for every episode
                  "best" to save only when a better result is obtained
        """
        gym.Wrapper.__init__(self, env)
        assert mode in self.MODES
        self.mode = mode
        self.fileprefix = fileprefix
        self.best_reward = -np.infty
        self.frames = []
        self.episode_reward = 0

    def observation(self, frame):
        self.frames.append(frame)
        return frame

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        return self.observation(observation), reward, done, info

    def reset(self):
        if len(self.frames) > 0:
            if self.mode == "all" or (self.mode == "best" and self.episode_reward > self.best_reward):
                clip = mpy.ImageSequenceClip(self.frames, fps=60)
                filename = f"{self.fileprefix}_reward{self.episode_reward}_{id_generator()}.mp4"
                clip.write_videofile(filename)
            if self.best_reward < self.episode_reward:
                self.best_reward = self.episode_reward
            del self.frames
            self.frames = []
        self.episode_reward = 0
        return self.env.reset()


class ProcessedMovieRecorder(MovieRecorder):
    """Specialized movie recorder for processed frames"""
    def reset(self):
        # Color correction in case the processed frames were turned to grayscale values
        self.frames = [color.gray2rgb(f) for f in self.frames]
        return super(ProcessedMovieRecorder, self).reset()


# General functions

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """Generates a random id for a filename"""
    return ''.join(random.choice(chars) for _ in range(size))


def saveanimation(rawframes, filename):
    """Saves a sequence of frames as an animation

    The filename must include an appropriate video extension
    """
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)
