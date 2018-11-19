#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from A2C_nstep import ActorCritic

# Simple A2C--Code
# 
# This is a simple implementation of an Actor-Advantage-Critic (A2C) model. For an intuitive guide to the mechanics of the model itself please check out the comic in this repository. 
# 
# To keep things clear, we're using an easy challenge--Cartpole--and have pruned the A2C to only the necessary bits, sacrificing a bit of performance. We're building an n-step A2C with a single agent that takes in a simple Cartpole state as 4 float values, but notebooks for a Monte Carlo, multiple parallel agents, and raw pixels versions are in this directory as well. For a more industrial-strength A2C, check out our PyTorch implementation of the OpenAI Baselines A2C.

# In[12]:


# Discount factor. Model is not very sensitive to this value.
GAMMA = .95

# LR of 3e-2 explodes the gradients, LR of 3e-4 trains slower
LR = 3e-3
N_GAMES = 2000

# OpenAI baselines uses nstep of 5.
N_STEPS = 20

env = gym.make('CartPole-v0')
N_ACTIONS = 2 # get from env
N_INPUTS = 4 # get from env

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=LR)    


# The training loop:
# 
# There are two parts to the training loop. First, we gather a minibatch of training data by moving around the environment. Once we've filled out a minibatch, we stop and reflect. 

# In[13]:


state = env.reset()
finished_games = 0

while finished_games < N_GAMES:
    states, actions, rewards, dones = [], [], [], []

    # Gather training data
    for i in range(N_STEPS):
        s = Variable(torch.from_numpy(state).float().unsqueeze(0))

        action_probs = model.get_action_probs(s)
        action = action_probs.multinomial().data[0][0]
        next_state, reward, done, _ = env.step(action)

        states.append(state); actions.append(action); rewards.append(reward); dones.append(done)

        if done: state = env.reset(); finished_games += 1
        else: state = next_state

    # Reflect on training data
    reflect(states, actions, rewards, dones)


# The cell above contains everything. Now we'll go through and look at the individual parts of it.

# 1) Gather training data
# The agent moves around the environment gathering training data. Notice that we’re using the only Policy head of the model to help us choose actions but we’re not reflecting on anything yet. The model is being used purely for inference.
# 
# What does the training data look like? In the MC case the size of the minibatch fed into the model would be equal to the length of the episode. In the case of N-Step, however, we’ll gather the same amount of training data for each minibatch. A 5-step model always deals in minibatches of 5 observations

# In[ ]:


#Pic of normal 5 steps with no failure. Well-trained.


# If the episode ends in the middle of a set we simply start a new game and keep playing. 

# In[ ]:


#Pic of minibatch with failure in the middle.


# 2) Reflect on training data
# 
# Now that our model has gathered its training data, it’s time to reflect
# 
# Calculate true values
# 
# First, we need to calculate what the returns from each state ACTUALLY turned out to be. These will be the labels we use to train the critic. 
# 
# Let's check out what these would look like in the MC case:

# In[ ]:


#pic of MC returns


# What do the returns look like in the NSTEP case? Instead of backing up from a terminal state where we know the value is zero (like in a MC model), we’ll have to back up from an estimate of the value of the last state in our set. 

# In[ ]:


#Pic of df backing up from bootstrapped v(s). Note that last value is an estimate.


# This model is already well-trained. Notice how the predicted v(s) nicely tracks the actual v(s)s. We can test the accuracy of our bootstrapped v(s) by checking it against the actual returns from the first row of data in the next minibatch. In an untrained model, predicted v(s) would be way off.
# 
# How do we calculate these true state values? We start from the end and work our way backwards.

# In[7]:


def calc_actual_state_values(rewards, dones):
    R = []
    rewards.reverse()

    # If we happen to end the set on a terminal state, set next return to zero
    if dones[-1] == True: next_return = 0
        
    # If not terminal state, bootstrap v(s) using our critic
    # TODO: don't need to estimate again, just take from last value of v(s) estimates
    else: 
        s = torch.from_numpy(states[-1]).float().unsqueeze(0)
        next_return = model.get_state_value(Variable(s)).data[0][0] 
    
    # Backup from last state to calculate "true" returns for each state in the set
    R.append(next_return)
    dones.reverse()
    for r in range(1, len(rewards)):
        if not dones[r]: this_return = rewards[r] + next_return * GAMMA
        else: this_return = 0
        R.append(this_return)
        next_return = this_return

    R.reverse()
    state_values_true = Variable(torch.FloatTensor(R)).unsqueeze(1)
    
    return state_values_true


# Just a classic regression problem
# 
# Once we have labels for our minibatch of training data, we treat it like we would any other supervised learning problem. We calculate the loss and backpropagate it through the model.
# 
# Our first step is to send our states as input into the NN. In return we get a list of state value predictions and a list of action recommendations. We use these, along with our lists of bootstrapped target state values and actual actions taken to compute the advantage / TD error. 

# In[4]:


def reflect(states, actions, rewards, dones):
    
    # Calculating the ground truth "labels" as described above
    state_values_true = calc_actual_state_values(rewards, dones)

    s = Variable(torch.FloatTensor(states))
    action_probs, state_values_est = model.evaluate_actions(s)
    action_log_probs = action_probs.log() 
    
    a = Variable(torch.LongTensor(actions).view(-1,1))
    chosen_action_log_probs = action_log_probs.gather(1, a)

    # This is also the TD error
    advantages = state_values_true - state_values_est

    entropy = (action_probs * action_log_probs).sum(1).mean()
    action_gain = (chosen_action_log_probs * advantages).mean()
    value_loss = advantages.pow(2).mean()
    total_loss = value_loss - action_gain - 0.0001*entropy

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), 0.5)
    optimizer.step()


# The Model
# 
# This is a modular unit that can be swapped for any other input-output mapping machine--another ML model, a lookup table, etc. When we upgrade the N-step model to accept pixels in a later notebook, all we have to do is swap out this fully-connected NN for a CNN.

# In[2]:


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)
    
    # In a PyTorch model, you only have to define the forward pass. PyTorch computes the backwards pass for you!
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x) 
        return x
    
    # Only the Actor head
    def get_action_probs(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        return action_probs
    
    # Only the Critic head
    def get_state_value(self, x):
        x = self(x)
        state_value = self.critic(x)
        return state_value
    
    # Both heads
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        state_values = self.critic(x)
        return action_probs, state_values  


# How does the model actually train? Let's watch its score over 1000 games. This takes about 30 seconds on a consumer laptop.

# In[3]:


def test_model(model):
    score = 0
    done = False
    env = gym.make('CartPole-v0')
    state = env.reset()
    global action_probs
    while not done:
        score += 1
        s = torch.from_numpy(state).float().unsqueeze(0)
        
        action_probs = model.get_action_probs(Variable(s))
        
        _, action_index = action_probs.max(1)
        action = action_index.data[0] 
        next_state, reward, done, thing = env.step(action)
        state = next_state
    return score


# In[17]:


test_model(model)


# There are a number of improvements that we can make to this model:
# 
# --You'll notice that after reaching a perfect score of 200, the model's performance fluctuates wildly. We can fix this by only training on episodes where a failure occured--perfect games have no training signal bc returns for all states are identical. 
# 
# -- Further limiting our training to only those frames directly before a failure seems to speed training as well (more work required here), perhaps because we're downsampling "uninteresting" observations that have no variation in returns. 
# 
# -- We're not recording our scores or losses throughout training. Other versions in this repo chart progress.
# 
# -- We haven't experimented with other step sizes, which significantly affect training. Other a2cs in this repo show experiments along these lines
# 
# -- We haven't added in multiple actors, whichs help by decorrelating training data
