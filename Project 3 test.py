#import numpy as np
#import matplotlib.pyplot as plt
#from IPython import display as ipythondisplay

import gymnasium as gym
import pygame


env = gym.make('Blackjack-v1', render_mode='human')

for i_episode in range(100):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break

for i_episodes in range(400):
    state = env.reset()
    episode = []
    while True:
        action = 0 if state[0] > 18 else 1
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break


#env.reset()
#pygame.init()
#prev_screen = env.render(mode='rgb_array')
#plt.imshow(prev_screen)
#num_episodes = 10000
#dis=0.99
#rList=[]

for i in range(num_episodes):
    state = env.reset()
    rALL = 0
    done = False

for i_episodes in range(400):
    action = 0 if state[0] > 18 else 1
    next_state, reward, done, info = env.step(action)
    episode.append((state, action, reward))
    state = next_state

    #screen = env.render(mode='rgb_array')

    #plt.imshow(screen)
    #ipythondisplay.clear_output(wait=True)
    #ipythondisplay.display(plt.gcf())

    #if done:
        #break

    #ipythondisplay.clear_output(wait=True)
    #env.close()

    #ipythondisplay.clear_output(wait=True)
    #env.close()
