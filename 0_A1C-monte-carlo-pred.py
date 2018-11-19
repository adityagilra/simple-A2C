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
SEEDS = [1,2,3,4,5,6,7,8,9,10]
N_SEEDS = len(SEEDS)
N_GAMES = 1000
N_ACTIONS = 2
N_INPUTS = 4
GAMMA = 0.9

env = gym.make('CartPole-v0')

cuda = False

# In[8]:


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, N_ACTIONS)
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
    
    def get_action_probs(self, x):
        # self(x) calls nn.Module's __call__()
        x = self(x)
        action_probs = F.softmax(self.actor(x),dim=1)
        return action_probs
    
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x),dim=1)
        state_values = self.critic(x)
        next_state = self.predictor(x)
        
        return action_probs, state_values, next_state
          


# In[9]:


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
        next_state, reward, done, thing = env.step(action.numpy())
        state = next_state
        
    return score
    


# In[10]:


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-3)

if cuda:
    model.cuda()

# In[12]:

# taken from https://discuss.pytorch.org/t/reset-model-weights/19180/2
def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight.data)

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

scores = np.zeros((N_SEEDS,N_GAMES//20))
num_games = np.arange(0,N_GAMES,20)
value_losses = np.zeros((N_SEEDS,N_GAMES//20))
action_gains = np.zeros((N_SEEDS,N_GAMES//20))
state_pred_losses = np.zeros((N_SEEDS,N_GAMES//20))

for SEEDi,SEED in enumerate(SEEDS):
    env.seed(ENVSEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # be sure to reset the weights of the model!
    model.apply(weights_init)

    for i in range(N_GAMES):
        
        states = []
        actions = []
        rewards = []
        dones = []
        
        state = env.reset() 
        done = False
        
        # act phase
        while not done:
            s = torch.from_numpy(state).float().unsqueeze(0)
            
            action_probs = model.get_action_probs(Variable(s))
            action = action_probs.multinomial(num_samples=1)[0][0]
            next_state, reward, done, _ = env.step(action.numpy())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state

        if True: #len(rewards) < 200: # only reflecting/training on episodes where a failure occured. No training
            # signal in perfect games. 
            # Reflect phase
            #print("Training. Score was ", len(rewards))

            rewards_orig = torch.tensor(rewards)
            ###### Old way of calculating value, obsolete, see calc_actual_state_values() function
            #R = []
            #rr = rewards
            #rr.reverse()
            #next_return = -30 #if len(rewards) < 200 else 1 # unnecessary now, should just be 0
            ## punish failure hard
            #for r in range(len(rr)):
            #    this_return = rr[r] + next_return * .9
            #    R.append(this_return)
            #    next_return = this_return
            #R.reverse()
            #rewards = R

            # choose one of these -- old vs new method of calculating rewards and values
            #state_values_true = Variable(torch.FloatTensor(rewards)).unsqueeze(1)
            state_values_true = calc_actual_state_values(rewards, dones)
            
            # taking only the last 20 states before failure. wow this really improves training
            """rewards = rewards[-20:]
            states = states[-20:]
            actions = actions[-20:]"""
            
            global ss
            ss = Variable(torch.FloatTensor(states))
            
            global next_states
            action_probs, state_values_est, next_states = model.evaluate_actions(ss)
            
            next_state_pred_loss = (ss[1:] - next_states[:-1]).pow(2).mean()

            action_log_probs = action_probs.log() 

            # same as TD error at each time (n-steps from t to the end of the trial)
            advantages = state_values_true - state_values_est

            entropy = (action_probs * action_log_probs).sum(1).mean()

            a = Variable(torch.LongTensor(actions).view(-1,1))

            chosen_action_log_probs = action_log_probs.gather(1, a)

            action_gain = (chosen_action_log_probs * advantages).mean()
            
            td_errors = (rewards_orig[:-1] + GAMMA*state_values_true[1:] - state_values_est[:-1])
            td_loss = td_errors.pow(2).mean()

            value_loss = advantages.pow(2).mean()
            
            #total_loss = td_loss/50.0
            total_loss = value_loss/50.0 - action_gain - 0.0001*entropy + next_state_pred_loss
            #total_loss = next_state_pred_loss
            #total_loss = - action_gain

            optimizer.zero_grad()

            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
                        
        else: print("Not training, score of ", len(rewards))

        if i % 20 == 0:
            s = test_model(model)
            print "Test score at run",i,"of seed",SEED,"is",s
            scores[SEEDi,i//20] = s
            action_gains[SEEDi,i//20] = action_gain.item()
            value_losses[SEEDi,i//20] = value_loss.item()
            state_pred_losses[SEEDi,i//20] = next_state_pred_loss.item()

env.close()

fig = plt.figure()
fig.add_subplot(2,2,1)
plt.errorbar(num_games, np.mean(scores,axis=0),np.std(scores,axis=0))
plt.xlabel("N_GAMES")
plt.ylabel("Score")
#plt.title(EXP)

fig.add_subplot(2,2,2)
plt.errorbar(num_games, np.mean(value_losses,axis=0),np.std(value_losses,axis=0))
plt.xlabel("N_GAMES")
plt.ylabel("Value loss")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)

fig.add_subplot(2,2,3)
plt.errorbar(num_games, np.mean(action_gains,axis=0),np.std(action_gains,axis=0))
plt.xlabel("N_GAMES")
plt.ylabel("action gains")
#plt.savefig("experiments/"+EXP_NAME+'/'+EXP)

fig.add_subplot(2,2,4)
plt.errorbar(num_games, np.mean(state_pred_losses,axis=0),np.std(state_pred_losses,axis=0))
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
