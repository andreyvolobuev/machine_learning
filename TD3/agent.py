import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork


class Agent:
    def __init__(self, input_dims, n_actions, env,
                 fc1_dims, fc2_dims, alpha, beta,
                 gamma, tau, noise1, noise2, clamp,
                 delay, max_size, batch_size, warmup):

        self.gamma = gamma
        self.tau = tau
        self.noise1 = noise1
        self.noise2 = noise2
        self.clamp = clamp
        self.delay = delay
        self.batch_size = batch_size
        self.warmup = warmup
        self.learn_cntr = 0
        self.env = env
        self.n_actions = n_actions

        self.actor = ActorNetwork(
                     input_shape=input_dims,
                     n_actions=n_actions,
                     fc1_dims=fc1_dims,
                     fc2_dims=fc2_dims,
                     alpha=alpha,
                     name='Actor_TD3PG.cpt',
                     checkpoint_dir='tmp/models')

        self.critic_1 = CriticNetwork(
                        input_shape=input_dims,
                        n_actions=n_actions,
                        fc1_dims=fc1_dims,
                        fc2_dims=fc2_dims,
                        beta=beta,
                        name='Critic_1_TD3PG.cpt',
                        checkpoint_dir='tmp/models')

        self.critic_2 = CriticNetwork(
                        input_shape=input_dims,
                        n_actions=n_actions,
                        fc1_dims=fc1_dims,
                        fc2_dims=fc2_dims,
                        beta=beta,
                        name='Critic_2_TD3PG.cpt',
                        checkpoint_dir='tmp/models')

        self.target_actor = ActorNetwork(
                            input_shape=input_dims,
                            n_actions=n_actions,
                            fc1_dims=fc1_dims,
                            fc2_dims=fc2_dims,
                            alpha=alpha,
                            name='Target_Actor_TD3PG.cpt',
                            checkpoint_dir='tmp/models')

        self.target_critic_1 = CriticNetwork(
                               input_shape=input_dims,
                               n_actions=n_actions,
                               fc1_dims=fc1_dims,
                               fc2_dims=fc2_dims,
                               beta=beta,
                               name='Target_Critic_1_TD3PG.cpt',
                               checkpoint_dir='tmp/models')

        self.target_critic_2 = CriticNetwork(
                               input_shape=input_dims, 
                               n_actions=n_actions, 
                               fc1_dims=fc1_dims,
                               fc2_dims=fc2_dims, 
                               beta=beta, 
                               name='Target_Critic_2_TD3PG.cpt',
                               checkpoint_dir='tmp/models')

        self.memory = ReplayBuffer(
                      max_size=max_size, 
                      input_shape=input_dims, 
                      n_actions=n_actions)

        self.update_target_networks()

    def update_target_networks(self):
        tau = self.tau

        actor = dict(self.actor.named_parameters())
        critic_1 = dict(self.critic_1.named_parameters())
        critic_2 = dict(self.critic_2.named_parameters())
        target_actor = dict(self.target_actor.named_parameters())
        target_critic_1 = dict(self.target_critic_1.named_parameters())
        target_critic_2 = dict(self.target_critic_2.named_parameters())
        
        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()
        
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()
        
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()
        
        self.target_actor.load_state_dict(actor)
        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
    
    def choose_action(self, observation):
        if self.learn_cntr < self.warmup:
            mu = np.random.normal(scale=self.noise1, 
                                  size=self.n_actions)
            mu = T.tensor(mu).to(self.actor.device)
        else:
            state = T.tensor(observation,
                             dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state)
        noise = T.tensor(np.random.normal(scale=self.noise1,
                                          size=self.n_actions), 
                         dtype=T.float).to(self.actor.device)
        mu_ = T.clamp(T.add(mu, noise), min=self.env.action_space.low[0],
                                        max=self.env.action_space.high[0])
        self.learn_cntr += 1
        return mu_.cpu().detach().numpy()
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        
    def sample(self):
        states, actions, rewards, states_, done = \
                                self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done, dtype=T.int).to(self.critic_1.device)
        
        return states, actions, rewards, states_, done
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_, done = self.sample()

        Vs1 = self.critic_1.forward(states, actions)
        Vs2 = self.critic_2.forward(states, actions)

        actions_ = self.target_actor.forward(states_)

        noise = T.tensor(np.random.normal(scale=self.noise1,
                                          size=self.n_actions), 
                         dtype=T.float).to(self.actor.device)
        noise = T.clamp(noise, min=-self.clamp, max=self.clamp)
        
        actions_ = T.add(actions_, noise)
        actions_ = T.clamp(actions_, min=self.env.action_space.low[0], 
                                     max=self.env.action_space.high[0])

        critic_1_Vs_ = self.target_critic_1.forward(states_, actions_)
        critic_2_Vs_ = self.target_critic_2.forward(states_, actions_)
        min_Vs_ = T.min(critic_1_Vs_, critic_2_Vs_)

        target = rewards + self.gamma*min_Vs_*(1-done)

        self.critic_1.optim.zero_grad()
        self.critic_2.optim.zero_grad()
        critic_1_loss = F.mse_loss(Vs1, target)
        critic_2_loss = F.mse_loss(Vs2, target)
        critic_loss = T.add(critic_1_loss, critic_2_loss)
        critic_loss.backward()
        self.critic_1.optim.step()
        self.critic_2.optim.step()

        if self.learn_cntr % self.delay == 0:
            self.actor.optim.zero_grad()
            actor_loss = self.critic_1.forward(states_, self.actor.forward(states_))
            actor_loss = -T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optim.step()
            
            self.update_target_networks()
