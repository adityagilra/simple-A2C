#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Adding another head to A2C: a next-state predictor. 
# If we can train the model to accurately predict its own next state and reward,
#  we could use it to generate additional training data.
#  Inspired by how humans do "mental practice" by imagining scenarios in their head.
#  Like that study with basketball players taking free throws:
#  Those who practiced mentally performed better, even with same amount of "live" data.
#  This sort of sample efficiency isn't really necessary when we have access to an env simulator, eg Gym,
#   but could be very helpful for robotics.

# In[7]:

#N_STEPS = 5
ENVSEED = 1
N_GAMES = 1000
envName = "Pendulum-v0"
DIM_ACTIONS = 1                 # dim of actions
R_ACTIONS = 2.                  # action can range between +- R_ACTIONS
N_INPUTS = 3                    # dim of env state

states = []
actions = []
rewards = []

env = gym.make(envName)
env.seed(ENVSEED)


# In[8]:


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, DIM_ACTIONS)
        self.critic = nn.Linear(64, 1)
        self.predictor = nn.Linear(64, N_INPUTS)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        
        x = self.linear2(x)
        x = F.relu(x)
        
        x = self.linear3(x)
        x = F.relu(x)
        
        return x
    
    def get_action(self, x):
        # self(x) calls nn.Module's __call__()
        x = self(x)
        #action = F.softmax(self.actor(x))
        action = self.actor(x) + np.random.normal(0.,R_ACTIONS/5.,size=(1))
        action = torch.clamp(action,-R_ACTIONS,R_ACTIONS)
        return action
    
    def evaluate_actions(self, x):
        # self(x) calls nn.Module's __call__()
        x = self(x)
        #action_probs = F.softmax(self.actor(x))
        action = self.actor(x) + np.random.normal(0.,R_ACTIONS/5.,size=(1))
        action = torch.clamp(action,-R_ACTIONS,R_ACTIONS)
        state_values = self.critic(x)
        next_state = self.predictor(x)
        
        return action, state_values, next_state



# In[9]:


def test_model(model):
    score = 0
    done = False
    env = gym.make(envName)
    state = env.reset()
    global action_probs
    while not done:
        score += 1
        s = torch.from_numpy(state).float().unsqueeze(0)
        
        action = model.get_action(Variable(s))
        action = action.data[0] 
        next_state, reward, done, thing = env.step(action.numpy())
        state = next_state
        
    return score
    


# In[10]:


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-3)


# In[12]:


scores = []
num_games = []
value_losses = []
action_gains = []
state_pred_losses = []

for i in range(N_GAMES):
    
    del states[:]
    del actions[:]
    del rewards[:]
    
    state = env.reset() 
    done = False
    
    # act phase
    while not done:
        s = torch.from_numpy(state).float().unsqueeze(0)
        
        action = model.get_action(Variable(s))
        next_state, reward, done, _ = env.step(action.numpy())
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state

    if True: #len(rewards) < 200: # only reflecting/training on episodes where a failure occured. No training
        # signal in perfect games. 
        # Reflect phase
        print("Training. Score was ", len(rewards))

        R = []
        rr = rewards
        rr.reverse()

        next_return = -30 #if len(rewards) < 200 else 1 # unnecessary now, should just be 0
        # punish failure hard

        for r in range(len(rr)):
            this_return = rr[r] + next_return * .9
            R.append(this_return)
            next_return = this_return
        R.reverse()

        rewards = R
        
        
        # taking only the last 20 states before failure. wow this really improves training
        """rewards = rewards[-20:]
        states = states[-20:]
        actions = actions[-20:]"""
        
        global ss
        ss = Variable(torch.FloatTensor(states))
        
        global next_states
        action_probs, state_values, next_states = model.evaluate_actions(ss)
        
        next_state_pred_loss = (ss[1:] - next_states[:-1]).pow(2).mean()

        #action_log_probs = action_probs.log() 

        advantages = Variable(torch.FloatTensor(rewards)).unsqueeze(1) - state_values

        #entropy = (action_probs * action_log_probs).sum(1).mean()

        a = Variable(torch.LongTensor(actions).view(-1,1))

        chosen_action_log_probs = action_log_probs.gather(1, a)

        action_gain = (chosen_action_log_probs * advantages).mean()

        value_loss = advantages.pow(2).mean()
        
        total_loss = value_loss/50.0 - action_gain - 0.0001*entropy + next_state_pred_loss
        #total_loss = next_state_pred_loss

        optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), 0.5)

        optimizer.step()
        
        #print("\nRewards", rewards, "\nState values",  state_values, )
        
    else: print("Not training, score of ", len(rewards))

    if i % 20 == 0:
        s = test_model(model)
        scores.append(s)
        num_games.append(i)

        action_gains.append(action_gain.data[0].numpy())
        value_losses.append(value_loss.data[0].numpy())
        state_pred_losses.append(next_state_pred_loss.data[0].numpy())

env.close()

fig = plt.figure()
fig.add_subplot(2,2,1)
plt.plot(num_games, scores)
plt.xlabel("N_GAMES")
plt.ylabel("Score")
#plt.title(EXP)

fig.add_subplot(2,2,2)
plt.plot(num_games, value_losses)
plt.xlabel("N_GAMES")
plt.ylabel("Value loss")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)

fig.add_subplot(2,2,3)
plt.plot(num_games, action_gains)
plt.xlabel("N_GAMES")
plt.ylabel("action gains")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)

fig.add_subplot(2,2,4)
plt.plot(num_games, state_pred_losses)
plt.xlabel("N_GAMES")
plt.ylabel("next state pred losses")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)

fig.tight_layout()

plt.show()



# In[14]:


ss[1:]


# In[15]:


next_states[:-1]


# 1.6.18 
# 
# Only taking last 20 frames really helps training. Otherwise spiky up and down, presumably bc interesting data is overpowered by no-signal data.
# 
# Adding in next frame prediction.
# 
# Simply adding next frame prediction loss to total loss doesn't seem to impede training (haven't tested scientifically)
# 
# value loss is of much higher magnitude than other losses. Fiddling with the weights to see if it speeds training.
# 
# Divide value loss by 50. Result: Doesn't seem to affect training? haven't tested scientifically. Investigate further.
# 
# Predictor is training, though not perfectly. Actor Critic learns too fast to allow proper tuning (bc we're not training on batches w no failure). Let's include only prediction loss and see how good a predictor we can get.
# 
# Only updating based on predictor loss. This seems to improve score? How would that be possible? Restart kernal and try again. OK, verified that it does NOT improve scores, whew.
# 
# Training on only policy loss blows up gradients. Try again with lower LR.
# 
# Training on only value loss score does NOT improve scores. Thought it did but restarted kernal and it didn't. wtf maybe it does... OK it does. Seems to even off at around 100, though extremely variable. First goes up to 100 consistently, then alternates btwn 100 and 10 ish. Investigate further
# 
# Training on predictor only does NOT improve score. It does of course improve predictor substantially. To be truly valuable, though, it needs to predict more states than just those directly before failure. 
# 
# Predictor loss decreases to .01 after 2000 games. 
# 
# TODO: 
# 1) try training weights for total loss composition
# 2) add head to model to also predict rewards
# 3) Use predictor to generate imagined data. Train on this data and see what it does to scores.
# 
# 
# 
# 
# 
