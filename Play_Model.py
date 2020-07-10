import gym
import copy
import random
import argparse
import numpy as np

from TD3 import TD3

def eval_policy(policy, env_name, seed, render=False):
    eval_env = gym.make(env_name)
    eval_env.seed(seed)
    reward_epi = 0.
    state, done = eval_env.reset(), False
    while not done:
        if render: env.render()
        action = policy.select_action(np.array(state))
        state, reward, done, _ = eval_env.step(action)
        reward_epi += reward

    return reward_epi

parser = argparse.ArgumentParser()

# Env
parser.add_argument('-env_name', default="HalfCheetah-v2", type=str)
parser.add_argument('-seed', default=1, type=int)
parser.add_argument('-render', help='Render gym episodes', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    ########################## Make env and save inform. ###########################
    env = gym.make(args.env_name)
    args.action_dim = env.action_space.shape[0]
    args.max_action = int(env.action_space.high[0])
    args.state_dim = env.observation_space.shape[0]

    rl_agent = TD3(args.state_dim, args.action_dim, args.max_action, args)
    policy = copy.deepcopy(rl_agent.actor)

    random.seed(args.seed)
    np.random.seed(args.seed)

    param = np.load("learned_models/"+args.env_name+".npy")
    policy.set_params(param)
    score = eval_policy(policy, args.env_name, args.seed, args.render)

    print(f"Score: {score}")




