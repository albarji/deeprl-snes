# Agent that learns how to play a SNES game by using RLLib algorithms

import retro
import argparse
import ray
from ray.rllib.agents import ppo, impala
from ray.rllib.agents.agent import get_agent_class
from ray.tune import register_env
from ray.tune.logger import pretty_print
import envs
import time
import skimage
from skimage import color
import numpy as np
from functools import partial


def make_env(game, state, rewardscaling=1, skipframes=4, pad_action=None, keepcolor=False):
    """Creates the SNES environment"""
    env = retro.make(game=game, state=state)
    env = envs.RewardScaler(env, rewardscaling)
    env = envs.discretize_actions(env, game)
    env = envs.SkipFrames(env, skip=skipframes, pad_action=pad_action)
    env = envs.WarpFrame(env, togray=not keepcolor)
    env = envs.FrameStack(env)
    return env


def register_retro(game, state, **kwargs):
    """Registers a given retro game as a ray environment

    The environment is registered with name 'retro-v0'
    """
    env_creator = lambda env_config: make_env(game=game, state=state, **kwargs)
    register_env("retro-v0", env_creator)
    return partial(env_creator, {})


"""Algorithm configuration parameters."""
ALGORITHMS = {
    # Parameters from https://github.com/ray-project/ray/blob/master/python/ray/rllib/tuned_examples/atari-ppo.yaml
    "PPO": {
        "default_conf": ppo.DEFAULT_CONFIG,
        "conf": {
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_param": 0.1,
            "entropy_coeff": 0.01,
            "sample_batch_size": 500,
            "num_sgd_iter": 10,
            "num_envs_per_worker": 1,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "vf_share_layers": True,
            "num_gpus": 1,
            "lr_schedule": [
                [0, 0.0005],
                [20000000, 0.000000000001],
            ]
        }
    },
    # Parameters from https://github.com/ray-project/ray/blob/master/python/ray/rllib/tuned_examples/atari-impala.yaml
    "IMPALA": {
        "default_conf": impala.DEFAULT_CONFIG,
        "conf": {
            'sample_batch_size': 50,
            'train_batch_size': 500,
            'num_envs_per_worker': 1,
            'lr_schedule': [
                [0, 0.0005],
                [200000000, 0.000000000001],
            ],
            'sample_async': True
        }
    },
    # Random agent for testing purposes
    "random": {
        "default_conf": {},
        "conf": {}
    }
}


def create_config(alg="PPO", workers=4):
    """Returns a learning algorithm configuration"""
    if alg not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {alg}, must be one of {list(ALGORITHMS.keys())}")
    return {**ALGORITHMS[alg]["default_conf"], **ALGORITHMS[alg]["conf"], **{"num_workers": workers}}


def train(checkpoint=None, alg="PPO", workers=4):
    """Trains a policy network"""
    ray.init()
    config = create_config(alg, workers)
    print(f"Config for {alg}:", config)
    agent = get_agent_class(alg)(config=config, env="retro-v0")
    if checkpoint is not None:
        try:
            agent.restore(checkpoint)
            print(f"Resumed checkpoint {checkpoint}")
        except:
            print("Checkpoint not found: restarted policy network from scratch")
    print("Started policy network from scratch")

    for i in range(1000000):
        # Perform one iteration of training the policy with PPO
        result = agent.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)


def test(checkpoint=None, testdelay=0, alg="PPO", render=False, makemovie=False, envcreator=None, keepcolor=False):
    """Tests and renders a previously trained model"""
    ray.init()
    config = create_config(alg, workers=1)
    if alg == "random":
        env = envcreator()
    else:
        agent = get_agent_class(alg)(config=config, env="retro-v0")
        if checkpoint is None:
            raise ValueError(f"A previously trained checkpoint must be provided for algorithm {alg}")
        agent.restore(checkpoint)
        env = agent.local_evaluator.env

    while True:
        rawframes = []
        states = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done:
            if alg == "random":
                action = np.random.choice(range(env.action_space.n))
            else:
                action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            time.sleep(testdelay)
            reward_total += reward
            if render:
                env.render()
            if makemovie:
                rawframes.append(env.render(mode="rgb_array"))
                states.append(next_state)
            state = next_state
        if makemovie:
            envs.saveanimation(rawframes, f"{alg}_reward{reward_total}.mp4")
            processedframes = [
                st[:, :, -3:] if keepcolor else color.gray2rgb(st[:, :, -1])
                for st in states
            ]
            envs.saveanimation([skimage.img_as_ubyte(f) for f in processedframes],
                               f"{alg}_reward{reward_total}_processed.mp4")
        print("Episode reward", reward_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that learns how to play a retro game by using RLLib.')
    parser.add_argument('game', type=str, help='Game to play. Must be a valid Gym Retro game')
    parser.add_argument('state', type=str, help='State (level) of the game to play')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file from which to load learning progress')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no policy updates, render environment)')
    parser.add_argument('--skipframes', type=int, default=4, help='Run the environment in groups of N frames')
    parser.add_argument('--padaction', type=int, default=None, help='Index of action used to pad skipped frames')
    parser.add_argument('--keepcolor', action='store_true', help='Keep colors in image processing')
    parser.add_argument('--testdelay', type=float, default=0,
                        help='Introduced delay between test frames. Useful for debugging')
    parser.add_argument('--render', action='store_true', help='Render test episodes')
    parser.add_argument('--makemovie', action='store_true', help='Save videos of test episodes')
    parser.add_argument('--algorithm', type=str, default="IMPALA",
                        help=f'Algorithm to use for training: {list(ALGORITHMS.keys())}')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers to use during training')

    args = parser.parse_args()

    envcreator = register_retro(args.game, args.state, skipframes=args.skipframes,
                                pad_action=args.padaction, keepcolor=args.keepcolor)
    if args.test:
        test(checkpoint=args.checkpoint, testdelay=args.testdelay, alg=args.algorithm, render=args.render,
             makemovie=args.makemovie, envcreator=envcreator, keepcolor=args.keepcolor)
    else:
        train(checkpoint=args.checkpoint, alg=args.algorithm, workers=args.workers)