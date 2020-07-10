import os
import gym
import copy
import torch
import pprint
import random
import argparse
import numpy as np
import pandas as pd
import time

from ES import CEM
from TD3 import TD3
from utils import get_output_folder
from Replay_Buffer import ReplayBuffer
from Parameters import Set_parameter
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
Device = torch.device("cuda" if USE_CUDA else "cpu")

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    reward_all = []
    for _ in range(eval_episodes):
        reward_epi = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            reward_epi += reward
        reward_all.append(reward_epi)
    return reward_all

def rollout(args, policy, env, max_step, replay_buffer=False, is_random_action=False):
    # initialize parameter for game
    state, done = env.reset(), False
    fitness, step = 0, 0
    while step < max_step and not done:
        # Determine action
        if is_random_action:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))
            if args.action_noise > 0:
                action += np.random.normal(0, args.max_action * args.action_noise, size=args.action_dim)
            action = action.clip(-args.max_action, args.max_action)

        # Take action, and get inform.
        next_state, reward, done, _ = env.step(action)

        # Save the transition to replay buffer
        if replay_buffer:
            not_done = 1 - float(done) if step < env._max_episode_steps else 1
            replay_buffer.add(state, action, next_state, reward, not_done)

        # Update state
        state = next_state
        fitness += reward
        step += 1

    return fitness, step



parser = argparse.ArgumentParser()

# Env
parser.add_argument('-env_name', default="HalfCheetah-v2", type=str)
parser.add_argument('-seed', default=1, type=int)
parser.add_argument('-warmup_steps', default=10000, type=int)
parser.add_argument('-max_steps', default=1000000, type=int)
parser.add_argument('-epi_steps', default=1000, type=int)
parser.add_argument('-ada_steps', dest="ada_steps", action='store_true')
parser.add_argument('-ada_inter', default=100000, type=int)
parser.add_argument('-ada_init', default=400, type=int)

# TD3
parser.add_argument('-nonlinearity_actor', default="elu", type=str)  # help: tanh, relu, and elu
parser.add_argument('-nonlinearity_critic', default="elu", type=str)  # help: leaky_relu, relu, and elu
parser.add_argument('-h1_actor', default=400, type=int)
parser.add_argument('-h2_actor', default=300, type=int)
parser.add_argument('-h1_critic', default=400, type=int)
parser.add_argument('-h2_critic', default=300, type=int)
parser.add_argument('-critic_lr', default=1e-3, type=float)
parser.add_argument('-actor_lr', default=2e-3, type=float)
parser.add_argument('-l2_rate', default=1e-5, type=float)
parser.add_argument('-discount', default=0.99, type=float)
parser.add_argument('-tau', default=0.005, type=float)
parser.add_argument('-btC_ratio', default=1, type=float)
parser.add_argument('-btA_ratio', default=1, type=float)
parser.add_argument('-dv_num', default=5, type=float)

parser.add_argument('-policy_noise', default=0.2, type=float)
parser.add_argument('-noise_clip', default=0.5, type=float)
parser.add_argument('-policy_freq', default=2, type=int)

# For exploration
parser.add_argument('--action_noise', default=0.00, type=float)

# Guided-learning beta
parser.add_argument('-guided', dest="guided", action='store_true')
parser.add_argument('-guided_beta', default=1, type=float)
parser.add_argument('-gl_target', default=0.05, type=float)

# CEM
parser.add_argument('-pop_size', default=10, type=int)
parser.add_argument('-cov_init', default=7.5e-3, type=float)
parser.add_argument('-cov_limit', default=1e-5, type=float)
parser.add_argument('-cov_alpha', default=1e-2, type=float)
parser.add_argument('-elitism', default=True, type=bool)

# The period of RL to CEM
parser.add_argument('-RtoE', default=1, type=int)

# Q-surrogate model
parser.add_argument('-q_surr', dest="q_surr", action='store_true')
parser.add_argument('-surr_ratio', default=50, type=int)
parser.add_argument('-surr_start', default=150000, type=int)
parser.add_argument('-surr_explor', default=0.5, type=float)

# ReplayBuffer
parser.add_argument('-buffer_size', default=1e6, type=float)
parser.add_argument('-batch_size', default=256, type=float)

# The number of episodes for policy test
parser.add_argument('-n_eval', default=10, type=int)

# Save information
parser.add_argument('-output', default='results/', type=str)
parser.add_argument('-algo_name', default='PGPS', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    Start_time = time.time()
    Total_time = 0
    Eval_time = 0
    Test_time = 0
    CEM_time = 0
    TD3_time = 0
    Gener_time = 0    
    
    torch.set_printoptions(precision=5)
    np.set_printoptions(precision=5)

    args = Set_parameter(args=args)
    folder2 = f"{args.env_name}-{args.seed}-{args.nonlinearity_actor}-{args.h1_actor}-{args.pop_size}-{args.RtoE}-{args.cov_alpha}"

    ############ Set save-folder, and print and save experiment setting ############
    folder = get_output_folder(args.output, folder2)
    with open(folder + "/experiment_setting.txt", 'w') as file:
        for key, value in vars(args).items(): file.write("{} = {}\n".format(key, value))
    print(f"\nThe results are saved in this routes:{folder}")
    print(f"Policy:{args.algo_name}, Env: {args.env_name}, Seed: {args.seed}\n")
    print(pprint.pformat(vars(args), indent=4))

    ########################## Make env and save inform. ###########################
    env = gym.make(args.env_name)
    args.action_dim = env.action_space.shape[0]
    args.max_action = int(env.action_space.high[0])
    args.state_dim = env.observation_space.shape[0]

    ################################### Set seed ###################################
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ############## Set replay-buffer, rl-agent, es-agents, and actor ###############
    replay_buffer = ReplayBuffer(args, args.state_dim, args.action_dim, max_size=args.buffer_size)
    rl_agent = TD3(args.state_dim, args.action_dim, args.max_action, args)
    es_agent = CEM(args, num_params=rl_agent.actor.get_size(), mu_init=rl_agent.actor.get_params())
    actor = copy.deepcopy(rl_agent.actor)

    ##################### Set data-frame for saving results ########################
    df_log = pd.DataFrame(columns=["Step", "AvgES", "BestES", "RL"])
    df_steps = pd.DataFrame(columns=["Step"] + [f"Ind{i}" for i in range(1, args.pop_size + 1)])
    df_fitness = pd.DataFrame(columns=["Step"] + [f"Ind{i}" for i in range(1, args.pop_size + 1)])
    df_mu = pd.DataFrame(columns=["Step", "Mean", "Std"] + [f"Reward{i}" for i in range(1, args.n_eval + 1)])
    df_best = pd.DataFrame(columns=["Step", "Mean", "Std"] + [f"Reward{i}" for i in range(1, args.n_eval + 1)])
    df_rl = pd.DataFrame(columns=["Step", "Mean", "Std"] + [f"Reward{i}" for i in range(1, args.n_eval + 1)])    

    ######################### Evaluate initial policy ##############################
    Test_start = time.time()
    f_mu = eval_policy(actor, args.env_name, args.seed, eval_episodes=args.n_eval)
    Test_time += time.time() - Test_start
    res = {"Step": 0, "AvgES": 0, "BestES": 0, "RL": np.mean(f_mu).round(0)}
    df_log = df_log.append(res, ignore_index=True)
    print(f"\nInitial policy performance\n{res}\n")

    ################################## Set params ##################################
    rl_policy = []
    tot_steps, eval_steps, generation = 0, 0, 0
    guided_learning = False
    num_high = int(args.pop_size * (1 - args.surr_explor))
    num_low = args.pop_size - num_high    

    ################################ Warmup games ##################################
    Eval_start = time.time()
    remaining_step = args.warmup_steps
    while tot_steps < args.warmup_steps:
        _, step = rollout(args, actor, env, remaining_step, replay_buffer, is_random_action=True)
        remaining_step -= step
        tot_steps += step
    Eval_time += time.time() - Eval_start
    ################################# Main games ###################################
    params = es_agent.sampling(args.pop_size)
    q_values = []
    
    while tot_steps < args.max_steps:
        generation += 1

        ################# Run games using each actor-individual ####################
        scores, steps = [], []
        if args.ada_steps:
            args.epi_steps = min(100 * (tot_steps // args.ada_inter) + args.ada_init, 1000)
        else:
            args.epi_steps = 1000
        
        Eval_start = time.time()
        for i in range(args.pop_size):
            actor.set_params(params[i])
            score, step = rollout(args, actor, env, args.epi_steps, replay_buffer)
            scores.append(score)
            steps.append(step)
        Eval_time += time.time() - Eval_start
        ########################### Update total step ##############################
        tot_steps += sum(steps)
        eval_steps += sum(steps)

        ###################### Print an experiment results #########################
        res = {"Step": tot_steps, "AvgES": np.mean(scores[:-2]).round(0), "BestES": np.max(scores[:-2]).round(0),
               "RL": np.round(scores[-1], 0) if (generation % args.RtoE) == 0 else 0}        
        print(f"\nGeneration: {generation},  Participate Rl-agent as last individual"
              if (generation % args.RtoE) == 0 else f"\nGeneration: {generation},")
        print(f"{steps}\n{np.round(scores, 0)}  STD:{np.std(scores).round() / 2}\n"
              f"Avg(Q-filter):{np.mean(scores[1:num_high]).round()}, Avg(random):{np.mean(scores[-num_low:-1]).round()}")
        print( f"{q_values}\n{res}\n" if len(q_values) > 0 else f"{res}\n")

        ############################### ES Update #################################
        CEM_start = time.time()
        es_agent.update(params, scores)
        CEM_time += time.time() - CEM_start
        ##### Check Q-vale and distance between best and rl before the update #####
        stateOut, _, _, _, _ = replay_buffer.sample(512)
        actor.set_params(es_agent.elite_param)

        predistOut = copy.deepcopy(((rl_agent.actor(stateOut) - actor(stateOut)) ** 2).mean().item())
        preQ = copy.deepcopy(rl_agent.critic.Q1(stateOut, rl_agent.actor(stateOut)).mean().item())
        print(f"Before the learning, Distance:{predistOut}, Q-value:{preQ}, and G_beta:{rl_agent.guided_beta}")

        ######################### Check Guided-learning ###########################
        guided_learning = False
        if (generation % args.RtoE) == 0:
            if args.guided and (scores[-1] < (np.mean(scores) - np.std(score))):
                print(f"[{scores[-1]} < {(np.mean(scores) - np.std(score)).round()}]"
                      f" and distance:{predistOut} > 0.1: Yes Guided_learning")
                guided_learning = True
            else:
                print(f"[{scores[-1]} > {(np.mean(scores) - np.std(score)).round()}]"
                      f" and distance:{predistOut}:No Guided_learning")

        ########################### Set RL Update steps ###########################
        TD3_start = time.time()
        rl_agent.actor_target.set_params(es_agent.elite_param)
        step_critic = int((sum(steps) * args.btC_ratio))
        step_actor = int((sum(steps) * args.btA_ratio))

        if guided_learning:
            ########################### RL-guided Update ##########################
            for k in range(args.dv_num):
                stateIn, _, _, _, _ = replay_buffer.sample(512)
                actor.set_params(es_agent.elite_param)
                predistIn = copy.deepcopy(((rl_agent.actor(stateIn) - actor(stateIn)) ** 2).mean().item())

                for i in range(step_critic // args.dv_num):
                    rl_agent.train_critic(replay_buffer)

                for i in range(step_actor // args.dv_num):
                    rl_agent.train_actor_guided(replay_buffer, es_agent.elite_param)

                postdistIn = ((rl_agent.actor(stateIn) - actor(stateIn)) ** 2).mean().item()
                print(f"Guided-learning, Predist:{predistIn}, PostDist:{postdistIn}, G_beta:{rl_agent.guided_beta}")

                if postdistIn > args.gl_target * 1.5:
                    rl_agent.guided_beta *= 2
                    rl_agent.guided_beta = 1024 if rl_agent.guided_beta > 1024 else rl_agent.guided_beta
                elif postdistIn < args.gl_target / 1.5:
                    rl_agent.guided_beta /= 2
        else:
            ########################### Basic RL Update ###########################
            for k in range(args.dv_num):
                for i in range(step_critic // args.dv_num):
                    rl_agent.train_critic(replay_buffer)

                for i in range(step_actor // args.dv_num):
                    rl_agent.train_actor(replay_buffer)

        rl_policy = rl_agent.actor.get_params()
        TD3_time += time.time() - TD3_start
        ### Check Q-vale and dist between ES-best and RL-agent after the update ###
        postdistOut = ((rl_agent.actor(stateOut) - actor(stateOut)) ** 2).mean().item()
        postQ = rl_agent.critic.Q1(stateOut, rl_agent.actor(stateOut)).mean().item()
        print(f"After the learning, Distance:{postdistOut}, Q-value:{postQ}, and G_beta:{rl_agent.guided_beta}")
        print(f"Learning Time-Tot:{np.round(time.time()-Start_time,2)}, Eval:{np.round(Eval_time,2)}, "
              f"CEM:{np.round(CEM_time,2)}, TD3:{np.round(TD3_time,2)}, "
              f"Gener:{np.round(Gener_time,2)}, Test:{np.round(Test_time,2)}")
        ######################## Generate next EA-individual#######################
        Gener_start = time.time()
        if not args.q_surr:
            ############### Generate next individual using only EA ################
            params = es_agent.sampling(args.pop_size)

        elif args.q_surr and tot_steps > args.surr_start:
            ########### Generate next indiv using Q-surrogate model ##############
            params = es_agent.sampling(args.pop_size * args.surr_ratio)
            params = params[1:]
            q_values = []

            ############## Estimate Q-value of sampled individual ################
            states, _, _, _, _ = replay_buffer.sample(32)
            for param in params:
                actor.set_params(param)
                q_val = copy.deepcopy(rl_agent.critic.Q1(states, actor(states)).mean().item())
                q_values.append(q_val)
            q_values = np.array(q_values)
            idx_sorted = np.argsort(-q_values)

            ############ Determine next pop using q-value estimation #############
            ################## Almost indivs have high q-value ###################
            ############ Some indivs have low q-value for exploration ############
            params[1:num_high] = params[idx_sorted[:num_high - 1]]            
            q_values[1:num_high] = q_values[idx_sorted[:num_high - 1]]                                
                                
            expl_params = es_agent.sampling(num_low+1)
            expl_params = expl_params[1:]
            expl_q = []
            for param in expl_params:
                actor.set_params(param)
                q_val = copy.deepcopy(rl_agent.critic.Q1(states, actor(states)).mean().item())
                expl_q.append(q_val)
                                
            params[num_high:args.pop_size] = expl_params
            q_values[num_high:args.pop_size] = expl_q

            ######## Set params[0] as best policy of previous generation #########
            params[0] = es_agent.elite_param
            actor.set_params(es_agent.elite_param)
            q_values[0] = copy.deepcopy(rl_agent.critic.Q1(states, actor(states)).mean().item())

            params = params[:args.pop_size]
            q_values = np.round(q_values[:args.pop_size], 2)
        Gener_time += time.time() - Gener_start
        ############## Inject RL-agent to EA population periodically #############
        if (generation + 1) % args.RtoE == 0:
            params[-1] = rl_policy
            if args.q_surr and tot_steps > args.surr_start:
                q_values[-1] = copy.deepcopy(rl_agent.critic.Q1(states, rl_agent.actor(states)).mean().item())

        ######################## Save current results ############################
        df_log = df_log.append(res, ignore_index=True)
        df_steps.loc[len(df_steps)] = [tot_steps] + steps
        df_fitness.loc[len(df_fitness)] = [tot_steps] + np.round(scores, 2).tolist()

        #################### Test and Write current policies ####################
        if eval_steps >= 5000:

            ######################### Evaluate policies #########################
            Test_start = time.time()
            ev_policy = [rl_policy, es_agent.mu, es_agent.elite_param]
            ev_res = []
            for i in range(3):
                actor.set_params(ev_policy[i])
                ev_res.append(eval_policy(actor, args.env_name, args.seed, eval_episodes=args.n_eval))
            [f_erl, f_mu, f_best] = ev_res
            Test_time += time.time() - Test_start
            #################### Save the evaluation resutls ####################
            re = {"Mu": np.mean(f_mu).round(0), "Best": np.mean(f_best).round(0), "RL": np.mean(f_erl).round(0)}
            df_rl.loc[len(df_rl)] = [tot_steps, re["RL"], np.std(f_erl).round(0)] + np.round(f_erl, 2).tolist()
            df_best.loc[len(df_best)] = [tot_steps, re["Best"], np.std(f_best).round(0)] + np.round(f_best, 2).tolist()
            df_mu.loc[len(df_mu)] = [tot_steps, re["Mu"], np.std(f_mu).round(0)] + np.round(f_mu, 2).tolist()

            ################## Write the experiment results ####################
            df_log.to_csv(folder + "/log.csv")
            df_steps.to_csv(folder + "/steps.csv")
            df_fitness.to_csv(folder + "/fitness.csv")
            df_mu.to_csv(folder + "/Evaluation_mu.csv")
            df_rl.to_csv(folder + "/Evaluation_erl.csv")
            df_best.to_csv(folder + "/Evaluation_best.csv")

            np.save(f"{folder}/model_mu.npy", es_agent.mu)
            print(f"\n=====================================================\n[Eval]"
                  f"{re}\n=====================================================\n")
            eval_steps = 0

    print()
    print()
    print("End the learning")
    print("End the learning")
    print(f"Learning Time-Tot:{np.round(time.time()-Start_time,2)}, Eval:{np.round(Eval_time,2)}, "
              f"CEM:{np.round(CEM_time,2)}, TD3:{np.round(TD3_time,2)}, "
              f"Gener:{np.round(Gener_time,2)}, Test:{np.round(Test_time,2)}")






