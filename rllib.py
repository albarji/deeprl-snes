"""Agent that learns how to play a SNES game by using RLLib algorithms"""

import retro
import argparse
import ray
from ray.rllib.agents import ppo, impala, dqn
from ray.rllib.agents import agent as rllibagent
from ray.tune import register_env
from ray.tune.logger import pretty_print
import envs
import models
import time
import numpy as np
from functools import partial
import rnd
import json


def make_env(game, state, rewardscaling=1, skipframes=4, pad_action=None, keepcolor=False,
             timepenalty=0, makemovie=None, makeprocessedmovie=None, cliprewards=False):
    """Creates the SNES environment"""
    env = retro.make(game=game, state=state)
    env = envs.RewardScaler(env, rewardscaling)
    if cliprewards:
        env = envs.RewardClipper(env)
    env = envs.ButtonsRemapper(env, game)
    env = envs.SkipFrames(env, skip=skipframes, pad_action=pad_action)
    if makemovie is not None:
        env = envs.MovieRecorder(env, fileprefix="raw", mode=makemovie)
    env = envs.WarpFrame(env, togray=not keepcolor)
    if makeprocessedmovie is not None:
        env = envs.ProcessedMovieRecorder(env, fileprefix="processed", mode=makeprocessedmovie)
    env = envs.FrameStack(env)
    env = envs.RewardTimeDump(env, timepenalty)
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
    # Parameters from https://github.com/ray-project/ray/blob/master/python/ray/rllib/tuned_examples/pong-rainbow.yaml
    "DQN": {  # DQN Rainbow
        "default_conf": dqn.DEFAULT_CONFIG,
        "conf": {
            "num_atoms": 51,
            "noisy": True,
            "lr": 1e-4,
            "learning_starts": 10000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0,
            "schedule_max_timesteps": 2000000,
            "prioritized_replay_alpha": 0.5,
            "beta_annealing_fraction": 0.2,
            "final_prioritized_replay_beta": 1.0,
            "n_step": 3,
            "gpu": True
        }
    },
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
    # Parameters from https://github.com/ray-project/ray/blob/master/python/ray/rllib/tuned_examples/atari-ppo.yaml
    # TODO: testing
    "PPORND": {
        "default_conf": rnd.DEFAULT_CONFIG,
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
    #  and IMPALA paper https://arxiv.org/abs/1802.01561 Appendix G
    "IMPALA": {
        "default_conf": impala.DEFAULT_CONFIG,
        "conf": {
            'sample_batch_size': 20,  # Unroll length
            'train_batch_size': 32,
            'num_envs_per_worker': 1,
            'lr_schedule': [
                [0, 0.0005],
                [200000000, 0.000000000001],
            ],
            "grad_clip": 40.0,
            "opt_type": "rmsprop",
            "momentum": 0.0,
            "epsilon": 0.01,
        }
    },
    # Random agent for testing purposes
    "random": {
        "default_conf": {},
        "conf": {}
    }
}


def get_agent_class(alg):
    """Returns the class of a known agent given its name."""
    if alg == "PPORND":
        # TODO: testing
        return rnd.PPORNDAgent
    else:
        return rllibagent.get_agent_class(alg)


def create_config(alg="PPO", workers=4, model=None):
    """Returns a learning algorithm configuration"""
    if alg not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {alg}, must be one of {list(ALGORITHMS.keys())}")
    config = {**ALGORITHMS[alg]["default_conf"], **ALGORITHMS[alg]["conf"], **{"num_workers": workers}}
    if model is not None:
        config['model'] = {
            "custom_model": model
        }
    return config


def train(checkpoint=None, alg="IMPALA", workers=4, entropycoeff=None, model=None):
    """Trains a policy network"""
    ray.init()
    config = create_config(alg, workers, model)
    if entropycoeff is not None:
        config["entropy_coeff"] = np.sign(config["entropy_coeff"]) * entropycoeff  # Each alg uses different sign
    print(f"Config for {alg}: {json.dumps(config, indent=4, sort_keys=True)}")
    agent = get_agent_class(alg)(config=config, env="retro-v0")
    if checkpoint is not None:
        try:
            agent.restore(checkpoint)
            print(f"Resumed checkpoint {checkpoint}")
        except:
            print("Checkpoint not found: restarted policy network from scratch")
    print("Started policy network from scratch")

    for i in range(1000000):
        # Perform one iteration of training the policy with the algorithm
        result = agent.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)


def test(checkpoint=None, testdelay=0, alg="IMPALA", render=False, envcreator=None,
         maxepisodelen=10000000, model=None):
    """Tests and renders a previously trained model"""
    ray.init()
    config = create_config(alg, workers=1, model=model)
    if alg == "random":
        env = envcreator()
    else:
        agent = get_agent_class(alg)(config=config, env="retro-v0")
        if checkpoint is None:
            raise ValueError(f"A previously trained checkpoint must be provided for algorithm {alg}")
        agent.restore(checkpoint)
        env = agent.local_evaluator.env

    while True:
        state = env.reset()
        done = False
        reward_total = 0.0
        step = 0
        while not done and step < maxepisodelen:
            if alg == "random":
                action = np.random.choice(range(env.action_space.n))
            else:
                action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            time.sleep(testdelay)
            reward_total += reward
            if render:
                env.render()
            state = next_state
            step = step + 1
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
    parser.add_argument('--makemovie', type=str, default=None,
                        help='Save videos of test episodes. '
                             'Valid modes: "all" to record all episodes, '
                             '"best" to record best episodes')
    parser.add_argument('--makeprocessedmovie', type=str, default=None,
                        help='Save videos of test episodes in form of processed frames. '
                             'Modes similar to those of --makemovie')
    parser.add_argument('--maxepisodelen', type=int, default=1000000, help='Maximum length of test episodes')
    parser.add_argument('--algorithm', type=str, default="IMPALA",
                        help=f'Algorithm to use for training: {list(ALGORITHMS.keys())}')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Deep network model to use for training: {[None] + list(models.MODELS.keys())}')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers to use during training')
    parser.add_argument('--timepenalty', type=float, default=0, help='Reward penalty to apply to each timestep')
    parser.add_argument('--entropycoeff', type=float, default=None, help='Entropy bonus to apply to diverse actions')
    parser.add_argument('--cliprewards', action="store_true", help='Clip rewards to {-1, 0, +1}')

    args = parser.parse_args()

    envcreator = register_retro(args.game, args.state, skipframes=args.skipframes,
                                pad_action=args.padaction, keepcolor=args.keepcolor,
                                timepenalty=args.timepenalty, makemovie=args.makemovie,
                                makeprocessedmovie=args.makeprocessedmovie, cliprewards=args.cliprewards)
    if args.test:
        test(checkpoint=args.checkpoint, testdelay=args.testdelay, alg=args.algorithm,
             render=args.render, envcreator=envcreator, maxepisodelen=args.maxepisodelen, model=args.model)
    else:
        train(checkpoint=args.checkpoint, alg=args.algorithm, workers=args.workers, entropycoeff=args.entropycoeff,
              model=args.model)
