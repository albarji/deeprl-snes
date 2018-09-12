# Agent that learns how to play a SNES game by using RLLib implementation of PPO

import retro
import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env
from ray.tune.logger import pretty_print
import envs
import time
from gym import wrappers


def make_env(game, state, rewardscaling=1, pad_action=None, keepcolor=False, videodir=None):
    """Creates the SNES environment"""
    env = retro.make(game=game, state=state)
    if videodir is not None:
        env = wrappers.Monitor(env, videodir, force=True, video_callable=lambda episode_id: True)
    env = envs.RewardScaler(env, rewardscaling)
    env = envs.discretize_actions(env, game)
    env = envs.SkipFrames(env, pad_action=pad_action)
    env = envs.WarpFrame(env, togray=not keepcolor)
    env = envs.FrameStack(env)
    return env


def register_snes(game, state, **kwargs):
    """Registers a given SNES game as a ray environment

    The environment is registered with name 'snes_env'
    """
    register_env("snes_env", lambda env_config: make_env(game=game, state=state, **kwargs))


def add_ppo_params(cfg):
    """Returns a modified configuration with PPO parameters"""
    config = cfg.copy()
    config["lambda"] = 0.95
    config["kl_coeff"] = 0.5
    config["clip_param"] = 0.1
    config["entropy_coeff"] = 0.01
    config["sample_batch_size"] = 500
    config["num_sgd_iter"] = 10
    config["num_workers"] = 4
    config["num_envs_per_worker"] = 1
    config["batch_mode"] = "truncate_episodes"
    config["observation_filter"] = "NoFilter"
    config["vf_share_layers"] = True
    config["num_gpus"] = 1
    config["lr_schedule"] = [
        [0, 0.0007],
        [20000000, 0.000000000001],
    ]
    return config


def train(checkpoint=None):
    """Trains a policy network"""
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    print("Default PPO config:", config)
    # Parameters from https://github.com/ray-project/ray/blob/master/python/ray/rllib/tuned_examples/atari-ppo.yaml
    config = add_ppo_params(config)
    print("PPO config:", config)
    agent = ppo.PPOAgent(config=config, env="snes_env")
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


def test(checkpoint, num_steps=10000, testdelay=0):
    """Tests and renders a previously trained model"""
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config = add_ppo_params(config)
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
            time.sleep(testdelay)
            reward_total += reward
            env.render()
            steps += 1
            state = next_state
        print("Episode reward", reward_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent that learns how to play a SNES game by using RLLib.')
    parser.add_argument('game', type=str, help='Game to play. Must be a valid Gym Retro game')
    parser.add_argument('state', type=str, help='State (level) of the game to play')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file from which to load learning progress')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no policy updates, render environment)')
    parser.add_argument('--padaction', type=int, default=None, help='Index of action used to pad skipped frames')
    parser.add_argument('--keepcolor', action='store_true', help='Keep colors in image processing')
    parser.add_argument('--testdelay', type=float, default=0,
                        help='Introduced delay between test frames. Useful for debugging')
    parser.add_argument('--videodir', type=str, default=None, help='Directory in which to save playthrough videos')

    args = parser.parse_args()

    register_snes(args.game, args.state, pad_action=args.padaction, keepcolor=args.keepcolor, videodir=args.videodir)
    if args.test:
        test(checkpoint=args.checkpoint, testdelay=args.testdelay)
    else:
        train(checkpoint=args.checkpoint)
