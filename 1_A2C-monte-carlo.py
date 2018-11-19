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


# In[2]:


#N_STEPS = 5
SEED = 1
N_GAMES = 1000
N_ACTIONS = 2
N_INPUTS = 4

states = []
actions = []
rewards = []

env = gym.make('CartPole-v0')
env.seed(SEED)


# In[6]:


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-3)


# In[9]:


for m in model.parameters():
    print(m)
    
test_model(model)


# In[9]:





# In[ ]:


scores = []
num_games = []
value_losses = []
action_gains = []

for i in range(N_GAMES):
    
    del states[:]
    del actions[:]
    del rewards[:]
    
    state = env.reset() 
    done = False
    
    # act phase
    while not done:
        s = torch.from_numpy(state).float().unsqueeze(0)
        
        action_probs = model.get_action_probs(Variable(s))
        action = action_probs.multinomial().data[0][0]
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state

    if len(rewards) < 200: # only reflecting/training on episodes where a failure occured. No training
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

        global rewards
        rewards = R
        
        # taking only the last 20 states before failure
        rewards = rewards[-20:]
        states = states[-20:]
        actions = actions[-20:]
        
        s = Variable(torch.FloatTensor(states))

        global state_values
        action_probs, state_values = model.evaluate_actions(s)

        action_log_probs = action_probs.log() 

        advantages = Variable(torch.FloatTensor(rewards)).unsqueeze(1) - state_values

        entropy = (action_probs * action_log_probs).sum(1).mean()

        a = Variable(torch.LongTensor(actions).view(-1,1))

        chosen_action_log_probs = action_log_probs.gather(1, a)

        action_gain = (chosen_action_log_probs * advantages).mean()

        value_loss = advantages.pow(2).mean()

        total_loss = value_loss - action_gain - 0.0001*entropy

        #total_loss /= len(rewards) # wow this allowed to reach high score faster. Wait. this shouldn't matter
        # bc we're using mean values. undoing now.

        optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), 0.5)

        optimizer.step()
        
        print("\nRewards", rewards, "\nState values",  state_values)
        
    else: print("Not training, score of ", len(rewards))

    if i % 20 == 0:
        s = test_model(model)
        scores.append(s)
        num_games.append(i)

        action_gains.append(action_gain.data.numpy()[0])
        value_losses.append(value_loss.data.numpy()[0])

        
plt.plot(num_games, scores)
plt.xlabel("N_GAMES")
plt.ylabel("Score")
#plt.title(EXP)
plt.show()

plt.plot(num_games, value_losses)
plt.xlabel("N_GAMES")
plt.ylabel("Value loss")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)
plt.show()

plt.plot(num_games, action_gains)
plt.xlabel("N_GAMES")
plt.ylabel("action gains")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)
plt.show()
    
    
env.close()



# In[ ]:





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
# 12.15.17 WTF now it errors out with nans in the parameters?? looks like exploding/vanishing gradients. Trying gradient clipping. Gradient clipping solves it. Weird scores shoot up fast then back down. Hit 200 after 600 games. Scores really all over the place.
# 
# 12.17 Dividing loss be length of rewards speeds training. Dividing inputs by their max value breaks it, probably bc max values are too big.
# 
# Rewards vs state values is relatively accurate. Much better than nstep version. At least they're in the ballpark and trend in the correct direction. state_values pretty much just converge on rewards values (which evens out due to discounting). So in essence model just learns rewards values and assigns that value to all states, regardless of state. Even states in which the game loses are marked as same value. Safest bet from model's perspective is to just predict the long term reward value every time. 
# 
# Why does nstep version blow up state_value predictions? They just keep getting bigger and bigger
# 
# epsilon makes sense when thinking how far back we want responsability to go. It cartpole, probably not very far so a low epsilon is probably better. All stable states look relatively the same--it would be hard to differentiate btwn a stable state that will end in 20 steps or a stable state that will end in 200 steps. Both stable states should be valued the same.
# 
# by epsilon, I mean gamma. bc i'm an amateur. 
# 
# Note: We were assigning value of 0 to last state indisciminantly, even if termination was due to max score of 200. Fixed now.
# 
# 
# 12.18
# Problem: V(s) is not learning well. It's just predicting the steady state (based on gamma) for all states--even those right before failure. This is ok in non-nstep version bc we're not using v(s) to bootstrap, but it's a big problem with nstep models. Hypothesis: It's bc our steady-state samples, where no real signal is being providing, are overwhelming the frames directly before failure--frames with strong signal. ALso, those steady state frames have no variation, no signal. Approach: Oversample important frames. Take only the 15 frames prior to failure for training.
# 
# Result: wow after score of 200 remained 200 thereafter. makes sense bc we're not training it anymore. v(s) looks better, descending from 8 to 5, ideal would be 8 to 1. not enough spread in the v(s) estimates. Trying with window of 20 before failure, decreasing reward for ending state to -30. Success in terms of not training on no-signal data, failure in terms of capturing variability of v(s). 
# 
# Result: first run didn't catch at all. second run caught early and right to 200. v(s) now flattened out as before. 
# 
# Why can't we capture variability of v(s)? 
# 
# hypothesis: neural network isn't deep enough. 
# 
# Approach: change architecture to 64-128-64 instead of single 128.
# 
# Result: Didn't catch on first run of 500, but wow v(s) estimates look much better. Was that the case in other failed runs? Often blowing up gradients though.
# 
# Try reducing LR to 3e-3 from 3e-4. Result: This allows it to train. We're capturing range of about 0 to -12 or -20, which is good although ideal range would be 5 to -26.
# 
# Let's go back and try shallower net to see if also captures this range. Changing ONLY the model architecture back to 128. Keeping lr 3e-3 and failure reward of -30. Result: very little variation (about 3)! Probably reason: deeper network better captures variation. Although v(s) estimate less good with simpler model, simpler model appears to perform better (?)
# 
# Question: why does deeper model require smaller LR to not blow up gradients? Is it just bc there are more parameters so more opportunity for blowup?
# 
# Hypothesis: if deeper models better capture variation, an even deeper model will do even better. Instead of a three layer 64-128-64, let's try a 64-128-256-128-64. Again, changing ONLY model architecture.
# 
# Result: Hmm, sometimes capturing lots of variation. sometimes too much even. Takes much longer to reach 200. v(s) sometimes very good, sometimes very bad. 
# 
# I wonder how a two hidden layer model would do? Not capturing as much variation. about 8. Let's go back to 64-128-64 model and try with n-steps. First, verify model again: Verified. Notes: early on in training we're seeing some really nice progressions of v(s). Later on not so good, is that just selection bias bc we're only seeing failed episodes, and the episodes model fails on get harder and harder? 
