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


# Environment definitions
eps = np.finfo(np.float32).eps.item()

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepro(image):
    """ prepro uint8 frame into tensor image"""
    image = image[::4, ::4, :]  # downsample by factor of 4
    return np.rollaxis(image, 2, 0)  # Put channels first


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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(640, self.action_shape)  # SNES results in 640 pixels here

        self.saved_log_probs = []

    def forward(self, x):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu(self.bn2((self.conv2(x))))
        x = F.relu(self.bn3((self.conv3(x))))
        return F.sigmoid(self.head(x.view(x.size(0), -1)))

    def select_action(self, state):
        state = state.float().unsqueeze(0)
        probs = self(state)
        m = Bernoulli(probs)
        actions = m.sample()
        self.saved_log_probs.append(m.log_prob(actions))
        return actions.tolist()[0]


def runepisode(env, policy, steps=5000, render=False):
    observation = env.reset()
    x = prepro(observation)
    observations = []
    rewards = []
    rawframes = []

    for _ in range(steps):
        if render:
            env.render()
        x = torch.tensor(x).to(device)
        action = policy.select_action(x)
        observation, reward, done, info = env.step(action)
        x = prepro(observation)
        observations.append(x)
        rewards.append(reward)
        rawframes.append(observation)
        if done:
            break

    return rewards, observations, rawframes


def saveanimation(rawframes, filename):
    """Saves a sequence of frames as an animation

    The filename must include an appropriate video extension
    """
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)


def train(game, state=None, render=False, checkpoint='policygradient.pt', saveanimations=False):
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
    while True:
        # Gather samples
        rewards, observations, rawframes = runepisode(env, policy, render=render)
        print("Total reward for episode {}: {}".format(episode, np.sum(rewards)))
        drewards = discount_rewards(rewards)
        # Update policy network
        policy_loss = [-log_prob * reward for log_prob, reward in zip(policy.saved_log_probs, drewards)]
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.saved_log_probs[:]

        episode += 1
        # Save policy network from time to time
        if not episode % 10:
            torch.save(policy, checkpoint)
        # Save animation (if requested)
        if saveanimations:
            saveanimation(rawframes, "{}_episode{}.mp4".format(checkpoint, episode))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that learns how to play a SNES game by using a simple Policy '
                                                 'Gradient method.')
    parser.add_argument('game', type=str, help='Game to play. Must be a valid Gym Retro game')
    parser.add_argument('state', type=str, help='State (level) of the game to play')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file in which to save learning progress')
    parser.add_argument('--render', action='store_true', help='Render game while playing')
    parser.add_argument('--saveanimations', action='store_true', help='Save mp4 files with playthroughs')

    args = parser.parse_args()
    train(args.game, args.state, render=args.render, saveanimations=args.saveanimations, checkpoint=args.checkpoint)
