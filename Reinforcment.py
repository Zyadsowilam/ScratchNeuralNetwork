import numpy as np
import gym
from collections import deque
import random

EPISODES = 2000  
MAX_STEPS = 100 
LEARNING_RATE = 0.8  
GAMMA = 0.95  
RENDER = False  
epsilon = 1.0  
min_epsilon = 0.01
epsilon_decay = 0.995  
batch_size = 32  


positive_reward = 10
negative_reward = -10 


env = gym.make('FrozenLake-v1', render_mode='human')

state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))
experience_replay = deque(maxlen=10000)  #


rewards = []


for episode in range(EPISODES):
    state, info = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()

        
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(Q[state, :])  

       
        next_state, reward, done, truncated, info = env.step(action)

       
        if done:
            if reward == 1:  
                reward = positive_reward
            else:  
                reward = negative_reward

        
        experience_replay.append((state, action, reward, next_state, done))

       
        if len(experience_replay) > batch_size:
            batch = random.sample(experience_replay, batch_size)
            for state, action, reward, next_state, done in batch:
                target = reward
                if not done:
                    target = reward + GAMMA * np.max(Q[next_state, :])
                Q[state, action] = Q[state, action] + LEARNING_RATE * (target - Q[state, action])

        state = next_state
        total_reward += reward

        if done or truncated:
            break  

   
    rewards.append(total_reward)

   
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    
    if episode % 100 == 0:
        print(f"Episode {episode}/{EPISODES}, Average Reward: {np.mean(rewards[-100:]):.2f}, Epsilon: {epsilon:.2f}")

env.close()


print(Q)
print(f"Average reward: {np.mean(rewards):.2f}")
