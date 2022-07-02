
import numpy as np
import gym
import random
from time import sleep
import matplotlib.pyplot as plt

class q_learn:
    def __init__(self,alpha,gamma,epsilon,env_name = "Taxi-v3"):
        self.env = gym.make(env_name)
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def epsilon_greedy(self):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[self.state])
        return action
    
    def boltzman(self):
        w = [np.exp(i*self.epsilon) for i in self.q_table[self.state]]
        p = random.uniform(0, sum(w))
        for i in range(6):
            if p< w[i]:
                return i
            p -= w[i]

    def train(self,episodes, boltzmann = False):
        for i in range(1, episodes):
            self.state = self.env.reset()
            done = False
            while not done:
                action = (self.boltzman() if boltzmann else self.epsilon_greedy())
                next_state, reward, done, _ = self.env.step(action) 
                next_max = np.max(self.q_table[next_state])
                self.q_table[self.state, action] *= (1 - self.alpha)
                self.q_table[self.state, action] += self.alpha * (reward + self.gamma * next_max)
                self.state = next_state
                
            if i % int(episodes/100) == 0:
                print(f"Episode: {i}")

    def test(self, episodes, render = False):
        timesteps = 0
        for ep in range(episodes):
            self.state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_table[self.state])
                self.state, reward, done, _ = self.env.step(action)
                if (render):
                    self.env.render()
                    sleep(.1)
                timesteps +=1
        print(f"Average {timesteps / episodes} timesteps in {episodes} episodes:")
        return timesteps / episodes
def meta_test(episodes):
    eps_t = {}
    for alpha in [0.1 , 0.5, 0.9]:
        for gamma in [0.1 , 0.5, 0.9]:
            for epsilon in [0.1 , 0.5, 0.9]:
                q = q_learn(alpha,gamma,epsilon)
                q.train(episodes)
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                eps_t[f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}"] = q.test(1000)
    print(eps_t,"\n")
def plot_test_eps(optimum):
    q = q_learn(0.5,0.9,0.1)
    q.train(500)
    average = q.test(1000)
    x1 = [500]
    y1= [average]
    while average > optimum:
        q.train(100)
        average = q.test(1000)
        x1.append(x1[-1]+100)
        y1.append(average)
    q = q_learn(0.9,0.9,0.1)
    q.train(500)
    average = q.test(1000)
    x2 = [500]
    y2= [average]
    while average > optimum:
        q.train(100)
        average = q.test(1000)
        x2.append(x2[-1]+100)
        y2.append(average)
    plt.plot(x1, y1,label='alfa 0.5')
    plt.plot(x2, y2,label= 'alfa 0.9')
    plt.legend()
    plt.show()
def plot_test_boltzmann(optimum):
    q = q_learn(0.9,0.9,0.1)
    q.train(200)
    average = q.test(1000)
    x1 = [200]
    y1= [average]
    while average > optimum:
        q.train(100, True)
        average = q.test(1000)
        x1.append(x1[-1]+100)
        y1.append(average)
    q = q_learn(0.9,0.9,0.5)
    q.train(200)
    average = q.test(1000)
    x2 = [200]
    y2= [average]
    while average > optimum:
        q.train(100, True)
        average = q.test(1000)
        x2.append(x2[-1]+100)
        y2.append(average)
    q = q_learn(0.9,0.9,0.9)
    q.train(200)
    average = q.test(1000)
    x3 = [200]
    y3= [average]
    while average > optimum:
        q.train(100, True)
        average = q.test(1000)
        x3.append(x3[-1]+100)
        y3.append(average)
    plt.plot(x1, y1,label='temperatura 0.1')
    plt.plot(x2, y2,label= 'temperatura 0.5')
    plt.plot(x3, y3,label= 'temperatura 0.9')
    plt.legend()
    plt.show()

#q = q_learn(0.9,0.9,0.1) # (alfa,gamma,epsilon)

# q.train(1500) #epsilon greedy
q.train(500,True) #Boltzman

# q.test(1000)
q.test(1000,True) # render on

# meta_test(1000)
# plot_test_eps(14)
# plot_test_boltzmann(14)
