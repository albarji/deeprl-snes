# Agent that learns how to play a SNES game by using RLLib implementation of PPO

import retro
import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env
from ray.tune.logger import pretty_print
import envs


def make_env(game, state, rewardscaling=1, pad_action=None):
    """Creates the SNES environment"""
    env = retro.make(game=game, state=state)
    env = envs.RewardScaler(env, rewardscaling)
    env = envs.discretize_actions(env, game)
    env = envs.SkipFrames(env, pad_action=pad_action)
    env = envs.WarpFrame(env)
    env = envs.FrameStack(env)
    return env


def register_snes(game, state, pad_action):
    """Registers a given SNES game as a ray environment

    The environment is registered with name 'snes_env'
    """
    register_env("snes_env", lambda env_config: make_env(game=game, state=state, pad_action=pad_action))


def train(checkpoint=None):
    """Trains a policy network"""
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
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

        if i % 100 == 0:
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
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file from which to load learning progress')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no policy updates, render environment)')
    parser.add_argument('--padaction', type=int, default=None, help='Index of action used to pad skipped frames')

    args = parser.parse_args()

    register_snes(args.game, args.state, pad_action=args.padaction)
    if args.test:
        test(checkpoint=args.checkpoint)
    else:
        train(checkpoint=args.checkpoint)
