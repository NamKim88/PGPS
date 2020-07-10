import numpy as np
import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as FF

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
Device = torch.device("cuda" if USE_CUDA else "cpu")

class RLNN(nn.Module):
    def __init__(self, args):
        super(RLNN, self).__init__()
        self.args = args
        self.nonlinearity_actor = args.nonlinearity_actor
        self.nonlinearity_critic = args.nonlinearity_critic

    def set_params(self, params):
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())
            param.data.copy_(torch.from_numpy(params[cpt:cpt + tmp]).view(param.size()).to(Device))
            cpt += tmp

    def get_params(self):
        return copy.deepcopy(np.hstack([v.cpu().data.numpy().flatten() for v in self.parameters()]))

    def get_grads(self):
        return copy.deepcopy(np.hstack([v.grad.cpu().data.numpy().flatten() for v in self.parameters()]))

    def get_size(self):
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        if filename is None: return
        params = np.load('{}/{}.npy'.format(filename, net_name))
        self.set_params(params)

    def save_model(self, output, net_name):
        params = self.get_params()
        np.save('{}/{}.npy'.format(output, net_name), params)


class Actor(RLNN):
    def __init__(self, args, state_dim, action_dim, max_action, hidden1_node=400, hidden2_node=300):
        super(Actor, self).__init__(args)

        self.l1 = nn.Linear(state_dim, hidden1_node)
        self.l2 = nn.Linear(hidden1_node, hidden2_node)
        self.l3 = nn.Linear(hidden2_node, action_dim)

        self.max_action = max_action
        self.to(Device)

    def forward(self, state):
        # Relu was used in original TD3
        if self.nonlinearity_actor == "relu":
            a = FF.relu(self.l1(state))
            a = FF.relu(self.l2(a))
            a = torch.tanh(self.l3(a))
        # Elu was used in CERL
        elif self.nonlinearity_actor == "elu":
            a = FF.elu(self.l1(state))
            a = FF.elu(self.l2(a))
            a = torch.tanh(self.l3(a))
        # Tanh was used in ERL, CEM-RL, and PDERL, this is basic setting
        else:
            a = torch.tanh(self.l1(state))
            a = torch.tanh(self.l2(a))
            a = torch.tanh(self.l3(a))

        return self.max_action * a

    def select_action(self, state):
        # Input state is np.array(), therefore, convert np.array() to tensor
        state = FloatTensor(state).unsqueeze(0)

        # Get action from current policy
        action = self.forward(state)

        # Must be env.step(np.array* or lis*), therefore, convert tensor to np.array()
        return action.cpu().data.numpy().flatten()


class Critic(RLNN):
    def __init__(self, args, state_dim, action_dim, hidden1_node=400, hidden2_node=300):
        super(Critic, self).__init__(args)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden1_node)
        self.l2 = nn.Linear(hidden1_node, hidden2_node)
        self.l3 = nn.Linear(hidden2_node, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden1_node)
        self.l5 = nn.Linear(hidden1_node, hidden2_node)
        self.l6 = nn.Linear(hidden2_node, 1)

        self.to(Device)

    def forward(self, state, action):
        # The input of critic-Q is [state, action]
        sa = torch.cat([state, action], 1)

        # Relu was used in original TD3
        if self.nonlinearity_critic == "relu":
            q1 = FF.relu(self.l1(sa))
            q1 = FF.relu(self.l2(q1))
            q2 = FF.relu(self.l4(sa))
            q2 = FF.relu(self.l5(q2))
        # Elu was used in ERL, CERL, and PDERL
        elif self.nonlinearity_critic == "elu":
            q1 = FF.elu(self.l1(sa))
            q1 = FF.elu(self.l2(q1))
            q2 = FF.elu(self.l4(sa))
            q2 = FF.elu(self.l5(q2))
        # Leaky_relu was used in CEM-RL, this is basic setting
        else:
            q1 = FF.leaky_relu(self.l1(sa))
            q1 = FF.leaky_relu(self.l2(q1))
            q2 = FF.leaky_relu(self.l4(sa))
            q2 = FF.leaky_relu(self.l5(q2))
        q1 = self.l3(q1)
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        if self.nonlinearity_critic == "relu":
            q1 = FF.relu(self.l1(sa))
            q1 = FF.relu(self.l2(q1))
        elif self.nonlinearity_critic == "elu":
            q1 = FF.elu(self.l1(sa))
            q1 = FF.elu(self.l2(q1))
        else:
            q1 = FF.leaky_relu(self.l1(sa))
            q1 = FF.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        # Parameters about the neural net structure of critic and actor
        self.args = args
        self.max_action = max_action

        # Training batch size
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau

        # Action noise is added in the action of target Q
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

        # Parameters for Asynchronous update frequency
        self.total_iterC = 0
        self.total_iterA = 0
        self.policy_freq = args.policy_freq

        # Guided Beta
        self.guided_beta = args.guided_beta

        # Define critics and actors
        self.critic = Critic(args, state_dim, action_dim, self.args.h1_critic, self.args.h2_critic)
        self.actor = Actor(args, state_dim, action_dim, max_action, self.args.h1_actor, self.args.h2_actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        # Define optimizer in which Adam is used
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.l2_rate)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.l2_rate)

    def select_action(self, state):
        # Call the select_action function of actor
        return self.actor.select_action(state)

    def train(self, replay_buffer):
        self.total_iterC += 1

        # Sample mini-batch from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

        # Define target_Q used to estimate critic loss (=TD error)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Calculate the target_Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current_Q value
        current_Q1, current_Q2 = self.critic(state, action)

        # Calculate critic loss (=difference between target_Q and current_Q)
        critic_loss = FF.mse_loss(current_Q1, target_Q) + FF.mse_loss(current_Q2, target_Q)

        # Optimize the critic parameters
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

        if self.total_iterC % self.policy_freq == 0:
            self.total_iterA += 1

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor parameters
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_guided(self, replay_buffer, guided_param):
        self.total_iterC += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = FF.mse_loss(current_Q1, target_Q) + FF.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

        if self.total_iterC % self.policy_freq == 0:
            self.total_iterA += 1

            with torch.no_grad():
                guided_actor = copy.deepcopy(self.actor)
                guided_actor.set_params(guided_param)

            distance = ((self.actor(state) - guided_actor(state)) ** 2).mean()
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() + self.guided_beta * distance

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_critic(self, replay_buffer):
        self.total_iterC += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = FF.mse_loss(current_Q1, target_Q) + FF.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_actor(self, replay_buffer):
        self.total_iterA += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_actor_guided(self, replay_buffer, guided_param):
        self.total_iterA += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            guided_actor = copy.deepcopy(self.actor)
            guided_actor.set_params(guided_param)

        distance = ((self.actor(state) - guided_actor(state)) ** 2).mean()
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() + self.guided_beta * distance

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        np.save(filename + "_critic.npy", self.critic.state_dict().data.cpu().numpy())
        np.save(filename + "_actor.npy", self.actor.state_dict().data.cpu().numpy())

    def load(self, filename):
        params_critic = np.laod(filename + "_critic.npy")
        self.critic.set_params(params_critic)
        self.critic_optimizer = copy.deepcopy(self.critic)
        params_actor = np.laod(filename + "_actor.npy")
        self.critic.set_params(params_actor)
        self.actor_target = copy.deepcopy(self.actor)





