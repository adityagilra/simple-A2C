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


# In[2]:


SEED = 2
N_ACTIONS = 2
N_INPUTS = 4

env = gym.make('CartPole-v0')
env.observation_space.high


# In[ ]:





# In[23]:


GAMMA = .95
LR = 3e-3
N_GAMES = 1000
N_STEPS = 20

run_set(N_GAMES, N_STEPS, LR, GAMMA)


# In[ ]:





# In[13]:


lrs = [8e-3, 3e-3]
eps = [.95]
n_steps = [5, 10, 20]
#EXP_NAME = "Cartpole_nstep_gridsearch_LR_EPS_NSTEPS_121617_b"
#os.rmdir("experiments/"+EXP_NAME)
#os.mkdir("experiments/"+EXP_NAME)
for lr in lrs:
    for e in eps:
        for ns in n_steps:  
            try: run_set(2000, ns, lr, e)
            except: print("Failed to run set: NS "+str(ns)+" lr "+str(lr)+" eps "+str(e))
            


# In[22]:


def run_set(N_GAMES, N_STEPS, LR, EPS):
    env = gym.make('CartPole-v0')

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    state = env.reset()

    finished_games = 0
    num_games = []
    scores = []
    value_losses = []
    action_gains = []
    
    already_logged = False
    
    states = []
    actions = []
    rewards = []
    dones = []
    cum_scores = []
    
    game_current_score = 0

    while finished_games < N_GAMES:

        del states[:]
        del actions[:]
        del rewards[:]
        del dones[:]
        del cum_scores[:]

        # act phase
        for i in range(N_STEPS):
            s = torch.from_numpy(state).float().unsqueeze(0)

            action_probs = model.get_action_probs(Variable(s))
            action = action_probs.multinomial().data[0][0]
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            game_current_score += 1
            cum_scores.append(game_current_score)

            if done: 
                game_current_score = 0
                state = env.reset()
                finished_games += 1
                already_logged = False

            else: state = next_state

        # only taking windows in which failure occurs
        if True in dones and not 200 in cum_scores:
            # Reflect phase

            R = []
            rr = rewards
            rr.reverse()

            if dones[-1] == True:
                next_return = -30
            else:
                s = torch.from_numpy(states[-1]).float().unsqueeze(0)
                next_return = model.get_state_value(Variable(s)).data[0][0]

            R.append(next_return)
            dones.reverse()
            for r in range(1, len(rr)):
                if not dones[r]:
                    this_return = rr[r] + next_return * EPS
                else:
                    this_return = -30

                R.append(this_return)
                next_return = this_return

            R.reverse()
            dones.reverse()
            rewards = R

            s = Variable(torch.FloatTensor(states))

            action_probs, state_values = model.evaluate_actions(s)

            action_log_probs = action_probs.log() 

            advantages = Variable(torch.FloatTensor(rewards)).unsqueeze(1) - state_values

            entropy = (action_probs * action_log_probs).sum(1).mean()

            a = Variable(torch.LongTensor(actions).view(-1,1))

            chosen_action_log_probs = action_log_probs.gather(1, a)

            action_gain = (chosen_action_log_probs * advantages).mean()

            value_loss = advantages.pow(2).mean()

            total_loss = value_loss - action_gain - 0.0001*entropy

            optimizer.zero_grad()

            total_loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()

        if finished_games % 50 == 0 and not already_logged:
            try:
                s = test_model(model)
                scores.append(s)
                num_games.append(finished_games)
                action_gains.append(action_gain.data.numpy()[0])
                value_losses.append(value_loss.data.numpy()[0])
                already_logged = True
            except:
                continue

    EXP = "Cartpole_nstep_"+"LR_"+str(LR)+"_N_STEPS_"+str(N_STEPS)+"_EPS_"+str(EPS)+".png"

    plt.plot(num_games, scores)
    plt.xlabel("N_GAMES")
    plt.ylabel("Score")
    plt.title(EXP)
    plt.show()
    
    plt.plot(num_games, value_losses)
    plt.xlabel("N_GAMES")
    plt.ylabel("Value loss")
    plt.show()
    
    plt.plot(num_games, action_gains)
    plt.xlabel("N_GAMES")
    plt.ylabel("action gains")
    plt.show()


# In[3]:


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        
        x = self.linear2(x)
        x = F.relu(x)
        
        x = self.linear3(x)
        x = F.relu(x)
        
        return x
    
    def get_action_probs(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        return action_probs
    
    def get_state_value(self, x):
        x = self(x)
        state_value = self.critic(x)
        return state_value
    
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        state_values = self.critic(x)
        return action_probs, state_values
        
        


# In[4]:


def test_model(model):
    score = 0
    done = False
    env = gym.make('CartPole-v0')
    env.seed(SEED)
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


# 12.7.17 experiment. set up model from scratch. was having problems initially because
# wasn't passing discounted returns into the past. There was no signal for model to grab 
# on to, seeing as how it was just all ones. Model improved immediately after discounting 
# future returns. consistently beat game during training after 7-12k games
# 
# 12.9.17 Added in n-steps. Had similiar problem. Wasn't estimating final state and working back from there, 
# instead was plugging in zero, thus forcing to the model to think that every last step was worth nothing,
# which of course was confusing the hell out of it. Frames from a perfectly safe space were being labelled
# as end-states. using v(s) instead of zero as base for discounting returns works much better. 
# ACTUALLY NO. A small n_steps(5) leads to model blowing up. 
# 
# Large n_steps (100) works. Score of 150 after half a minute. 130 after another minute.
# 
# 50 works, but only score 100 or so after a minute of training.
# 
# 20 log probs get janky after half a minute, errors out.
# 
# 200 gets perfect scores of 200 after two min of training.
# 
# Is it because we're using very small bits of extremely correlated data? Let's try with multiple envs and see...
# Nope, multiple envs didn't solve it. 
# 
# 12.13. Gradient clipping fixed it! Trains now. Must have been exploding or vanishing gradients. I noticed the learning rate was pretty high (3e-2). Lowering learning rate probably would have fixed it as well. Handles steps of 5 with no problem. perfect scores after 10k periods, probably sooner. 
# 
# 12.15 OH NO!! Actually still broken, i thought it was fixed but i think the model was just cached!!! 
# 
# No worries, reducing LR by order of magnitude to 3e-3 solved the problem. Interesting: Goes through waves of scores. Hits 200 a bunch right away then dips right down to the teens again. WTF? 
# 
# 12.16
# Trying to get reproducability. Not quite there. Model seems to learn similar parameters, at least for first row of inputs, but getting very different series of test results.
# 
# Experiment:
# Failed to run set: NS 2 lr 0.003 eps 0.95
# Failed to run set: NS 2 lr 0.006 eps 0.95
# Failed to run set: NS 2 lr 0.006 eps 0.9
# Failed to run set: NS 2 lr 0.006 eps 0.9
# 
# Step size of 2 combined with larger LR  and higher epsilon seems to blow things up. Larger n_steps, ie 50, seem to work best. Best combo was n_steps 50, lr .003, eps .9 or above. Low eps + low LR + low stepn = doesn't catch very well. Higher LR makes catch earlier. More variable? Why does higher N_steps work better? Is it just bc we have bigger batch sizes or is there something qualitatively different? how would LR annealing perform? 
# 
# Step size 2 never worked well. Eps of .5 worked less well. 
# 
# 
# 
# 12.17
# Value loss is all over the place compared with no-nstep version. Model is not learning to accurately predict value of states. Predicting all states to be highly valued with little variation btwn them.
# 
# 12.18 moving experiment from vanilla version.
# 
# baseline: v(s) pretty much always around 10. Even after reduce fail reward to -30.
# 
# Hypothesis: we can increase accuracy of v(s) estimates by deepening model. 
# Approach: deepen model to 64-128-64 instead of single 128.
# 
# Result: v(s) estimates are MUCH better. Consequently, returns are better as well (bc they're bootstrapped from better estimate). Disappointingly, scores do not seem much improved. 
# 
# Let's try only training on windows where a failure occured. 
# 
# Result: Looks MUCH better. small problem! Realized we were assigning -30 even to episodes that finished on 200. This would tell the model that good states are actually bad. Let's fix that and try again.
# 
# Scores shoot up to 200 by around game 400. They seem to stay there indefinitely. It makes sense to only update if a window includes a failure. Why change your approach if everything is going well?
# 
# Let's try a grid search on nsteps, lr, gamma
# 
# 10 and 20 steps work. LR of 3e-2 too high. Let's go with 10 or 20 steps, gamma of .95. LR 3e-3 with the multiple agents model
# 
# Test: bringing failure reward back to 0 from -30 to see if it affects. No obvious difference btwn 0 and -30. -500 blows it up. -200 doesn't blow up, but doesn't train either.
# 
# 
