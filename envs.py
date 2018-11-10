import numpy as np
import cv2
from collections import deque
import gym
import moviepy.editor as mpy


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


GENESIS_BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
SNES_BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


class ButtonsRemapper(gym.ActionWrapper):
    """Wrap a gym-retro environment and make it use discrete actions according to a given button remap

    As initialization parameters one must provide the list of buttons available in the console
    gamepad, and a list of actions that will be considered in the game. Each action is a list of
    buttons being pushed simultaneously.
    """
    def __init__(self, env, buttonmap, actions):
        super(ButtonsRemapper, self).__init__(env)
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttonmap))
            for button in action:
                arr[buttonmap.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class GradiusDiscretizer(ButtonsRemapper):
    """Wrap a gym-retro environment and make it use discrete actions for a SNES game.

    This encodes the prior knowledge that combined actions like UP + RIGHT might be valuable,
    but actions like A + B + RIGHT + L do not make sense (in general). It also helps making the
    environment much simpler.
    """
    def __init__(self, env):
        actions = [
            [],  # No-op
            ['DOWN'], ['LEFT'], ['RIGHT'], ['UP'],  # Basic directions
            ['DOWN', 'RIGHT'], ['RIGHT', 'UP'], ['UP', 'LEFT'], ['LEFT', 'DOWN'],  # Diagonals
            ['A'], ['B'], ['X'], ['Y'], ['L'], ['R'], ['SELECT'], ['START']  # Buttons
        ]
        super(GradiusDiscretizer, self).__init__(env, SNES_BUTTONS, actions)


class ColumnsGenesisDiscretizer(ButtonsRemapper):
    """Wrap a gym-retro environment and make it use discrete actions for the Columns Genesis game.

    This encodes the prior knowledge that in Columns it only makes sense to use LEFT, RIGHT, DOWN
    and column swap (A)
    """
    def __init__(self, env):
        actions = [
            [],  # No-op
            ['LEFT'], ['RIGHT'],  # Move column around
            # ['DOWN'],  # Speed up column fall to get additional points
                         # It is better not to allow this action because the agent gets stuck
                         # in the easy strategy of just dropping columns fast
            ['A']  # Column jewels swap
        ]
        super(ColumnsGenesisDiscretizer, self).__init__(env, GENESIS_BUTTONS, actions)


class ComixZoneDiscretizer(ButtonsRemapper):
    """Wrap a gym-retro environment and make it use discrete actions for the Comix Zone game"""
    def __init__(self, env):
        actions = [
            [],  # No-op
            ['LEFT'], ['RIGHT'],  # Move around
            ['B'], ['B', 'LEFT'], ['B', 'RIGHT'],  # Jump at different angles
            ['DOWN'],  # Crouch
            ['A'], ['A', 'DOWN'], ['A', 'LEFT'], ['A', 'RIGHT'], ['A', 'UP'],  # Attacks
            ['X'], ['Y'], ['Z'],  # Use items
        ]
        super(ComixZoneDiscretizer, self).__init__(env, GENESIS_BUTTONS, actions)


class SuperMarioWorldDiscretizer(ButtonsRemapper):
    """Wrap a gym-retro environment and make it use discrete actions for the Super Mario World game"""
    def __init__(self, env):
        actions = [
            [],  # No-op
            ['LEFT'], ['RIGHT'],  # Move around
            ['A'], ['A', 'LEFT'], ['A', 'RIGHT'],  # Spin jumps
            ['B'], ['B', 'LEFT'], ['B', 'RIGHT'],  # Normal jumps
            ['X'], ['X', 'LEFT'], ['X', 'RIGHT'],  # Run
            ['DOWN'],  # Crouch
            ['SELECT']  # Drops reserve item
        ]
        super(SuperMarioWorldDiscretizer, self).__init__(env, SNES_BUTTONS, actions)


class DonkeyKongCountryDiscretizer(ButtonsRemapper):
    """Wrap a gym-retro environment and make it use discrete actions for the Donkey Kong Country game"""
    def __init__(self, env):
        actions = [
            [],  # No-op
            ['A'],  # Switch monkeys
            ['B'],  # Jump/swim
            ['Y'],  # Pick/run
            ['LEFT'], ['RIGHT'],  # Move around
            ['Y'], ['Y', 'LEFT'], ['Y', 'RIGHT'],  # Run
            ['DOWN']  # Crouch
        ]
        super(DonkeyKongCountryDiscretizer, self).__init__(env, SNES_BUTTONS, actions)


def discretize_actions(env, game):
    """Modifies a gym environment to discretize the set of possible actions

    To do so we use prior knowledge on the actions that make sense in the game.
    """
    discretizers = {
        "Columns-Genesis": ColumnsGenesisDiscretizer,
        "ComixZone-Genesis": ComixZoneDiscretizer,
        "GradiusIII-Snes": GradiusDiscretizer,
        "SuperMarioWorld-Snes": SuperMarioWorldDiscretizer,
        "DonkeyKongCountry-Snes": DonkeyKongCountryDiscretizer
    }
    if game in discretizers:
        return discretizers[game](env)
    else:
        raise ValueError(f"ERROR: unknown game {game}, could not discretize actions")


def saveanimation(rawframes, filename):
    """Saves a sequence of frames as an animation

    The filename must include an appropriate video extension
    """
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)
