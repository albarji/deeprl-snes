# Agent that learns how to play a SNES game by using Proximal Policy Optimization

import retro
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import moviepy.editor as mpy
import argparse
import skimage
from skimage import color
from collections import deque


# Environment definitions
eps = np.finfo(np.float32).eps.item()

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    return env


def prepro(image):
    """ prepro uint8 frame into tensor image"""
    image = image[::4, ::4, :]  # downsample by factor of 4
    image = color.rgb2gray(image)  # turn to grayscale
    return image - 0.5  # 0-center


def discount_rewards(r, gamma=0.99):
    """Take 1D float array of rewards and compute clipped discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
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
        return F.softmax(self.actionshead(x)), self.valuehead(x)

    def _outdist(self, state):
        """Computes the probatility distribution of activating each output unit, given an input state"""
        probs, _ = self(state.float().unsqueeze(0))
        return Categorical(probs)

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
        m = Categorical(probs)
        return m.log_prob(action), value

    def value(self, state):
        """Estimates the value of the given state"""
        _, value = self(state.float().unsqueeze(0))
        return value

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


def experiencegenerator(env, policy, episodesteps=None, render=False, windowlength=4, verbose=True):
    """Generates experience from the environment.

    If the environment episode ends, it is resetted to continue acquiring experience.

    Yields experiences as tuples in the form:
        (observation, processed observation, logprobabilities, action, reward,
         new observation, new processed observation, terminal)
    """
    # Generate experiences indefinitely
    episode = 0
    totalsteps = 0
    episoderewards = []
    while True:
        # Reinitialize environment
        observation = env.reset()
        x = prepro(observation)
        statesqueue = deque([x for _ in range(windowlength)], maxlen=windowlength)
        xbatch = np.stack(statesqueue, axis=0)
        step = 0

        # Steps
        rewards = 0
        while True:
            if render:
                env.render()
            st = torch.tensor(xbatch).to(device)
            action, p = policy.select_action(st)
            newobservation, reward, done, info = env.step(action.tolist()[0])
            x = prepro(newobservation)
            statesqueue.append(x)
            newxbatch = np.stack(statesqueue, axis=0)
            yield (observation, xbatch, p, action, reward, newobservation, newxbatch, done)
            rewards += reward

            step += 1
            if done or (episodesteps is not None and step >= episodesteps):
                break
            observation = newobservation
            xbatch = newxbatch

        totalsteps += step
        episoderewards.append(rewards)
        episode += 1
        if verbose:
            print(f"Episode {episode} end, {totalsteps} total steps performed. Reward {rewards:.2f}, "
                  f"100-episodes average reward {np.mean(episoderewards[-100:]):.2f}")


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


def ppostep(policy, optimizer, states, actions, baseprobs, values, advantages, epscut, gradclip,
            valuecoef, entcoef):
    """Performs a step of Proximal Policy Optimization

    Arguments:
        - policy: policy network to optimize.
        - optimizer: pytorch optimizer algorithm to use.
        - states: iterable of gathered experience states
        - actions: iterable of performed actions in the states
        - basepros: current base probabilities of performing those actions
        - values: estimated values for each state
        - advantages: estimated advantage values for those actions
        - epscut: epsilon cut for policy gradient update
        - gradclip: maximum norm for clipping gradients
        - valuecoef: weight of value function loss
        - entcoef: weight of entropy function loss

    Returns the value of the losses computed in the optimization step.

    Reference: https://arxiv.org/pdf/1707.06347.pdf
    """
    optimizer.zero_grad()
    # Compute action probabilities and state values for current network parameters
    logprobs, newvalues = zip(*[policy.action_logprobs_value(st, ac) for st, ac in zip(states, actions)])
    newvalues = torch.stack([x[0][0] for x in newvalues])
    # Policy Gradients loss (advantages)
    newprobs = [torch.exp(logp) for logp in logprobs]
    probratios = [newprob / prob for prob, newprob in zip(baseprobs, newprobs)]
    clippings = [torch.min(rt * adv, torch.clamp(rt, 1 - epscut, 1 + epscut) * adv)
                 for rt, adv in zip(probratios, advantages)]
    pgloss = -torch.cat(clippings).mean()
    # Entropy loss
    entropyloss = - entcoef * torch.mean(torch.stack([policy.entropy(st) for st in states]))
    # Value estimation loss
    # (newvalue - [advantage + oldvalue])^2, that is, make new value closer to estimated error in old estimate
    # A clipped version of the loss is also included, thus imposing a kind of Huber loss
    returns = advantages + values
    valuelosses = (newvalues - returns) ** 2
    newvalues_clipped = values + torch.clamp(newvalues - values, epscut)
    valuelosses_clipped = (newvalues_clipped - returns) ** 2
    valueloss = valuecoef * 0.5 * torch.mean(torch.max(valuelosses, valuelosses_clipped))
    # Total loss
    loss = pgloss + valueloss + entropyloss
    # Optimizer step, with clipped gradients to a max norm to avoid exploding gradients
    loss.backward()
    torch.nn.utils.clip_grad_norm(policy.parameters(), gradclip)
    optimizer.step()

    return loss, pgloss, valueloss, entropyloss


def generalized_advantage_estimation(values, rewards, lastvalue, gamma, lam):
    """Computes a Generalized Advantage Estimator (GAE)

    This estimator allows computing advantages for any state, even if the current episode
    is unfinished. To do so, value function estimates are exploited.

    Arguments:
        - values: estimated values for each state
        - rewards: iterable of obtained rewards for those states
        - lastvalue: estimated value after all the steps above have been performed
        - gamma: rewards discount parameter
        - lam: GAE discount parameter.
            For lam close to 1 we have high-variance low-bias estimates
            For lam close to 0 we have low-variance high-bias estimates

    Reference: https://arxiv.org/pdf/1506.02438.pdf
    """
    allvalues = np.asarray(values.tolist() + [lastvalue])
    deltas = rewards + gamma * allvalues[1:] - allvalues[:-1]
    advantages = discount_rewards(deltas, gamma*lam)
    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + eps)
    return advantages.astype(np.float32)


def adjust_learning_rate(optimizer, lr):
    """Readjusts the learning rate of a given optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(game, state=None, render=False, checkpoint='policygradient.pt', episodesteps=10000, maxsteps=50000000,
          restart=False, minibatchsize=32, nminibatches=32, optimizersteps=4, epscut_start=0.1, epscut_end=0,
          gradclip=0.5, valuecoef=1, entcoef=0.01, gamma=0.99, lam=0.95, lr_start=0.00025,
          lr_end=0, rewardscaling=1):
    """Trains a policy network"""
    env = make_env(game=game, state=state, rewardscaling=rewardscaling)
    policy = loadnetwork(env, checkpoint, restart)
    print(policy)
    print("device: {}".format(device))
    optimizer = optim.Adam(policy.parameters(), lr=lr_start)
    expgen = experiencegenerator(env, policy, episodesteps=episodesteps, render=render)

    totalsteps = 0
    networkupdates = 0
    while totalsteps < maxsteps:
        # Annealings
        epscut = np.interp(totalsteps, [0, maxsteps], [epscut_start, epscut_end])
        lr = np.interp(totalsteps, [0, maxsteps], [lr_start, lr_end])
        adjust_learning_rate(optimizer, lr)

        # Gather experiences
        samples = [next(expgen) for _ in range(minibatchsize*nminibatches)]
        totalsteps += minibatchsize * nminibatches
        _, states, logprobs, actions, rewards, _, newstates, terminals = zip(*samples)
        states = [torch.tensor(st).to(device) for st in states]
        probs = [torch.exp(p.detach()) for p in logprobs]
        values = torch.stack([policy.value(st).detach()[0][0] for st in states])
        lastvalue = policy.value(torch.tensor(newstates[-1]).to(device)).detach()[0][0]
        # Compute advantages
        advantages = generalized_advantage_estimation(
            values=values,
            rewards=rewards,
            lastvalue=lastvalue,
            gamma=gamma,
            lam=lam
        )
        advantages = torch.tensor(advantages).to(device)
        print(f"Explored {minibatchsize*nminibatches} steps")

        # Optimizer epochs
        for optstep in range(optimizersteps):
            losseshistory = []
            # Random shuffle of experiences
            idx = np.random.permutation(range(len(samples)))
            # One step of SGD for each minibatch
            for i in range(nminibatches):
                batchidx = idx[i*minibatchsize:(i+1)*minibatchsize]
                losses = ppostep(
                    policy=policy,
                    optimizer=optimizer,
                    states=[states[i] for i in batchidx],
                    actions=[actions[i] for i in batchidx],
                    baseprobs=[probs[i] for i in batchidx],
                    advantages=advantages[batchidx],
                    values=values[batchidx],
                    epscut=epscut,
                    gradclip=gradclip,
                    valuecoef=valuecoef,
                    entcoef=entcoef
                )
                losseshistory.append(losses)

            loss, pgloss, valueloss, entropyloss = map(torch.stack, zip(*losseshistory))
            print(f"Optimizer iteration {optstep+1}: loss {torch.mean(loss):.3f} (pg {torch.mean(pgloss):.3f} "
                  f"value {torch.mean(valueloss):.3f} entropy {torch.mean(entropyloss):.3f})")

        del states, rewards, probs, actions, terminals

        # Save policy network from time to time
        networkupdates += 1
        if not networkupdates % 10:
            torch.save(policy, checkpoint)

    # Save final model
    torch.save(policy, checkpoint)


def test(game, state=None, render=False, checkpoint='policygradient.pt', saveanimations=False, episodesteps=10000):
    """Tests a previously trained network"""
    env = make_env(game=game, state=state)
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
    parser.add_argument('--optimizersteps', type=int, default=4, help='Number of optimizer steps in each PPO update')
    parser.add_argument('--episodesteps', type=int, default=10000, help='Max number of steps to run in each episode')
    parser.add_argument('--maxsteps', type=int, default=50000000, help='Max number of training steps')
    parser.add_argument('--rewardscaling', type=float, default=0.01, help='Scaling of rewards in training')
    #TODO: maybe we are using too small batches? Check https://github.com/ray-project/ray/blob/master/examples/carla/train_ppo.py#L47

    args = parser.parse_args()
    if args.test:
        test(args.game, args.state, render=args.render, saveanimations=args.saveanimations,
             checkpoint=args.checkpoint, episodesteps=args.episodesteps)
    else:
        train(args.game, args.state, render=args.render, checkpoint=args.checkpoint, restart=args.restart,
              optimizersteps=args.optimizersteps, episodesteps=args.episodesteps, maxsteps=args.maxsteps,
              rewardscaling=args.rewardscaling)
