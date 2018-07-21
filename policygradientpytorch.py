# Agent that learns how to play a SNES game by using a simple Policy Gradient method

import retro
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
import moviepy.editor as mpy
import argparse
from skimage import color


# Environment definitions
eps = np.finfo(np.float32).eps.item()

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepro(image):
    """ prepro uint8 frame into tensor image"""
    image = image[::4, ::4, :]  # downsample by factor of 4
    image = color.rgb2gray(image)  # turn to grayscale
    return np.expand_dims(image, axis=0)  # Put channels first


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward

    Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    return discounted_r


class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""

    action_shape = []

    def __init__(self, env):
        super(Policy, self).__init__()

        self.action_shape = env.action_space.n

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dense = nn.Linear(768, 512)  # SNES results in 640 pixels here
        self.head = nn.Linear(512, self.action_shape)

    def forward(self, x):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu(self.bn2((self.conv2(x))))
        x = F.relu(self.bn3((self.conv3(x))))
        x = F.relu(self.dense(x.view(x.size(0), -1)))
        return F.sigmoid(self.head(x))

    def select_action(self, state):
        """Returns the action selected by the policy, as well as the logprobs of each eaction"""
        state = state.float().unsqueeze(0)
        probs = self(state)
        m = Bernoulli(probs)
        actions = m.sample()
        return actions.tolist()[0], m.log_prob(actions)


class ReplayMemory:
    """Implements an experience replay memory"""
    def __init__(self, size=1000000):
        self.size = size
        self.memory = []

    def append(self, observation, action, reward):
        """Adds an experience to the memory"""
        if len(self.memory) >= self.size:
            self.pop()
        self.memory.append((observation, action, reward))

    def extend(self, iterable):
        """Adds an iterable of experiences to the memory

        Each experience must be in the format
            (observation, action, reward)
        """
        for exp in iterable:
            self.append(*exp)

    def pop(self):
        """Removes the oldest memory"""
        self.memory.pop()

    def minibatch(self, size=128):
        """Returns a random minibatch of memories

        Memories are returned as an iterable of tuples in the form
            (observation, action, reward)
        """
        return np.random.choice(self.memory, size=size, replace=False)


def saveanimation(rawframes, filename):
    """Saves a sequence of frames as an animation

    The filename must include an appropriate video extension
    """
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)


def train(game, state=None, render=False, checkpoint='policygradient.pt', saveanimations=False, memorysize=1000000,
          episodesteps=10000, maxsteps=5000000, test=False):
    env = retro.make(game=game, state=state)
    try:
        policy = torch.load(checkpoint)
        print("Resumed checkpoint {}".format(checkpoint))
    except:
        policy = Policy(env)
        print("Created policy network from scratch")
    print(policy)
    policy.to(device)
    print("device: {}".format(device))
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-4)

    episode = 0
    totalsteps = 0
    memory = ReplayMemory(memorysize)
    episoderewards = []
    while totalsteps < maxsteps:
        # Run episode
        observation = env.reset()
        x = prepro(observation)
        rewards = []
        rawframes = []
        logprobs = []
        for _ in range(episodesteps):
            if render:
                env.render()
            x = torch.tensor(x).to(device)
            action, logp = policy.select_action(x)
            observation, reward, done, info = env.step(action)
            memory.append(x, action, reward)
            x = prepro(observation)
            rewards.append(reward)
            rawframes.append(observation)
            logprobs.append(logp)
            if done:
                break
        episoderewards.append(np.sum(rewards))
        totalsteps += len(rewards)

        print(f"Episode {episode} end, {totalsteps} steps performed. 100-episodes average reward "
              f"{np.mean(episoderewards[-100:]):.0f}")
        # TODO: update network using a random batch of samples from the memory
        drewards = discount_rewards(rewards)
        # Update policy network
        if not test:
            policy_loss = [-log_prob * reward for log_prob, reward in zip(logprobs, drewards)]
            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

            # Save policy network from time to time
            if not episode % 10:
                torch.save(policy, checkpoint)

        # Save animation (if requested)
        if saveanimations:
            saveanimation(rawframes, "{}_episode{}.mp4".format(checkpoint, episode))

        episode += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that learns how to play a SNES game by using a simple Policy '
                                                 'Gradient method.')
    parser.add_argument('game', type=str, help='Game to play. Must be a valid Gym Retro game')
    parser.add_argument('state', type=str, help='State (level) of the game to play')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file in which to save learning progress')
    parser.add_argument('--render', action='store_true', help='Render game while playing')
    parser.add_argument('--saveanimations', action='store_true', help='Save mp4 files with playthroughs')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no policy updates)')

    args = parser.parse_args()
    train(args.game, args.state, render=args.render, saveanimations=args.saveanimations, checkpoint=args.checkpoint)
