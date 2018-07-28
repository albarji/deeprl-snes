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
import skimage
from skimage import color
from collections import deque


# Environment definitions
eps = np.finfo(np.float32).eps.item()

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepro(image):
    """ prepro uint8 frame into tensor image"""
    image = image[::4, ::4, :]  # downsample by factor of 4
    image = color.rgb2gray(image)  # turn to grayscale
    return image - 0.5  # 0-center


def discount_rewards(r, terminals, gamma=0.99):
    """Take 1D float array of rewards and compute clipped discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    r = np.sign(r)  # Clip rewards
    for t in reversed(range(0, len(r))):
        if terminals[t]:
            running_add = r[t]
        else:
            running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""

    action_shape = []

    def __init__(self, env, windowlength=4):
        super(Policy, self).__init__()

        self.action_shape = env.action_space.n

        self.conv1 = nn.Conv2d(windowlength, 32, kernel_size=8, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.dense = nn.Linear(3840, 512)
        self.actionshead = nn.Linear(512, self.action_shape)
        self.valuehead = nn.Linear(512, 1)

    def forward(self, x):
        x = F.selu(self.bn1((self.conv1(x))))
        x = F.selu(self.bn2((self.conv2(x))))
        x = F.selu(self.bn3((self.conv3(x))))
        x = F.selu(self.dense(x.view(x.size(0), -1)))
        return F.sigmoid(self.actionshead(x)), self.valuehead(x)

    def _outdist(self, state):
        """Computes the probatility distribution of activating each output unit, given an input state"""
        probs, _ = self(state.float().unsqueeze(0))
        return Bernoulli(probs)

    def select_action(self, state):
        """Selects an action following the policy

        Returns the selected action and the log probabilities of that action being selected.
        """
        m = self._outdist(state)
        action = m.sample()
        return action, m.log_prob(action)

    def action_logprobs_value(self, state, action):
        """Returns the logprobabilities of performing a given action at given state under this policy

        Also returns the value of the current state, for convenience.
        """
        probs, value = self(state.float().unsqueeze(0))
        m = Bernoulli(probs)
        return m.log_prob(action), value

    def entropy(self, state):
        """Returns the entropy of the policy for a given state"""
        return torch.sum(self._outdist(state).entropy())


def saveanimation(rawframes, filename):
    """Saves a sequence of frames as an animation

    The filename must include an appropriate video extension
    """
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)


def runepisode(env, policy, episodesteps, render, windowlength=4):
    """Runs an episode under the given policy

    Returns the episode history: an array of tuples in the form
        (observation, processed observation, logprobabilities, action, reward, terminal)
    """
    observation = env.reset()
    x = prepro(observation)
    statesqueue = deque([x for _ in range(windowlength)], maxlen=windowlength)
    xbatch = np.stack(statesqueue, axis=0)
    history = []
    for _ in range(episodesteps):
        if render:
            env.render()
        st = torch.tensor(xbatch).to(device)
        action, p = policy.select_action(st)
        newobservation, reward, done, info = env.step(action.tolist()[0])
        history.append((observation, xbatch, p, action, reward, done))
        if done:
            break
        observation = newobservation
        x = prepro(observation)
        statesqueue.append(x)
        xbatch = np.stack(statesqueue, axis=0)
    return history


def loadnetwork(env, checkpoint, restart):
    """Loads the policy network from a checkpoint"""
    if restart:
        policy = Policy(env)
        print("Restarted policy network from scratch")
    else:
        try:
            policy = torch.load(checkpoint)
            print(f"Resumed checkpoint {checkpoint}")
        except:
            policy = Policy(env)
            print(f"Checkpoint {checkpoint} not found, created policy network from scratch")
    policy.to(device)
    return policy


def train(game, state=None, render=False, checkpoint='policygradient.pt', episodesteps=10000, maxsteps=50000000,
          restart=False, batchsize=1, optimizersteps=10, epscut=0.1, valuecoef=1, entcoef=0.01):
    """Trains a policy network"""
    env = retro.make(game=game, state=state)
    policy = loadnetwork(env, checkpoint, restart)
    print(policy)
    print("device: {}".format(device))
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    episode = 0
    totalsteps = 0
    networkupdates = 0
    episoderewards = []
    while totalsteps < maxsteps:
        # Run batch of episodes
        states = []
        rewards = []
        probs = []
        actions = []
        terminals = []
        #TODO: implement a real batch-size operation, such in OpenAI's PPO. This will allow for larger episodes
        # This involves:
        # - Gather history of steps of size T (e.g. 128 x 8 = 1024)
        # - Compute Advantage estimates for history
        # - For each optimizer epoch (K epochs)
        #   - Randomly shuffle the gathered history
        #   - For each minibatch of size M < T over the shuffled history (e.g. M = 32)
        #       - Perform 1 Adam step minimizing the total loss
        for _ in range(batchsize):
            history = runepisode(env, policy, episodesteps, render)
            _, st, ps, ac, rew, ter = zip(*history)
            episoderewards.append(np.sum(rew))
            totalsteps += len(history)
            print(f"Episode {episode} end, {totalsteps} steps performed. 100-episodes average reward "
                  f"{np.mean(episoderewards[-100:]):.0f}")

            states.extend(st)
            rewards.extend(rew)
            probs.extend(ps)
            actions.extend(ac)
            terminals.extend(ter)

            del history, st, ps, ac, rew, ter
            episode += 1

        # Proximal Policy Optimization update
        states = [torch.tensor(st).to(device) for st in states]
        probs = [torch.exp(p.detach()) for p in probs]
        drewards = discount_rewards(rewards, terminals)
        for _ in range(optimizersteps):
            optimizer.zero_grad()
            logprobs, values = zip(*[policy.action_logprobs_value(st, ac) for st, ac in zip(states, actions)])
            advantages = [reward - value.tolist()[0][0] for reward, value in zip(drewards, values)]
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + eps)
            # Policy Gradients loss (advantages)
            newprobs = [torch.exp(logprob) for logprob in logprobs]
            probratios = [newprob/prob for prob, newprob in zip(probs, newprobs)]
            clippings = [torch.min(rt * adv, torch.clamp(rt, 1-epscut, 1+epscut) * adv)
                         for rt, adv in zip(probratios, advantages)]
            pgloss = torch.cat(clippings).mean()
            # Entropy loss
            entropyloss = torch.mean(torch.stack([policy.entropy(st) for st in states]))
            # Value estimation loss
            # TODO: in OpenAI PPO a clipping is also performed in this loss
            valueloss = torch.mean(torch.stack([(value - reward)**2 for value, reward in zip(values, drewards)]))
            # Total loss
            loss = pgloss - valuecoef * valueloss + entcoef * entropyloss
            print(f"loss {loss} (pg {pgloss} value {valueloss} entropy {entropyloss})")
            # Maximize loss == minimize - loss
            loss = -loss
            loss.backward()
            optimizer.step()

            del values, newprobs, probratios, clippings

        del states, rewards, probs, actions, terminals

        # Save policy network from time to time
        networkupdates += 1
        if not networkupdates % 10:
            torch.save(policy, checkpoint)


def test(game, state=None, render=False, checkpoint='policygradient.pt', saveanimations=False, episodesteps=10000):
    """Tests a previously trained network"""
    env = retro.make(game=game, state=state)
    policy = loadnetwork(env, checkpoint, False)
    print(policy)
    print("device: {}".format(device))

    episode = 0
    episoderewards = []
    while True:
        # Run episode
        history = runepisode(env, policy, episodesteps, render)
        observations, states, _, _, rewards, _ = zip(*history)
        episoderewards.append(np.sum(rewards))
        print(f"Episode {episode} end, reward {np.sum(rewards)}. 5-episodes average reward "
              f"{np.mean(episoderewards[-5:]):.0f}")

        # Save animation (if requested)
        if saveanimations:
            saveanimation(list(observations), f"{checkpoint}_episode{episode}.mp4")
            saveanimation([skimage.img_as_ubyte(color.gray2rgb(st[-1] + 0.5)) for st in states],
                          f"{checkpoint}_processed_episode{episode}.mp4")

        episode += 1
        del history, observations, states, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that learns how to play a SNES game by using a simple Policy '
                                                 'Gradient method.')
    parser.add_argument('game', type=str, help='Game to play. Must be a valid Gym Retro game')
    parser.add_argument('state', type=str, help='State (level) of the game to play')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file in which to save learning progress')
    parser.add_argument('--render', action='store_true', help='Render game while playing')
    parser.add_argument('--saveanimations', action='store_true', help='Save mp4 files with playthroughs')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no policy updates)')
    parser.add_argument('--restart', action='store_true', help='Ignore existing checkpoint file, restart from scratch')
    parser.add_argument('--batchsize', type=int, default=1, help='Number of episodes in each updating batch')
    parser.add_argument('--optimizersteps', type=int, default=4, help='Number of optimizer steps in each PPO update')
    parser.add_argument('--episodesteps', type=int, default=10000, help='Max number of steps to run in each episode')

    args = parser.parse_args()
    if args.test:
        test(args.game, args.state, render=args.render, saveanimations=args.saveanimations,
             checkpoint=args.checkpoint, episodesteps=args.episodesteps)
    else:
        train(args.game, args.state, render=args.render, checkpoint=args.checkpoint, restart=args.restart,
              batchsize=args.batchsize, optimizersteps=args.optimizersteps, episodesteps=args.episodesteps)
