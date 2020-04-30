from gridworld_envt import Gridworld
from sarsa_agent import SarsaAgent

import matplotlib.pyplot as plt
import math


num_episodes = 150
episode_scores = []

gridworld = Gridworld()
actions = gridworld.action_space

agent = SarsaAgent(actions)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=0.2)
        # print(f'State: {state}, action: {action}')
        new_state, reward, done = gridworld.step(action)

        episode_score += reward

        new_state_action = agent.act(new_state)
        agent.learn(state, action, reward, new_state, new_state_action)

        state = new_state
        episode_actions.append(action)
        if done:
            break
        
    episode_scores.append(episode_score)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions

    print(f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 100): {sum(episode_scores[:-100])/len(episode_scores)}', end='')

print(f'\nAfter {num_episodes}, average score: {sum(episode_scores)/len(episode_scores)}, Average(last 100): {sum(episode_scores[:-100])/len(episode_scores)}')
print(f'Best score: {best_score}, Sequence of actions: {best_path_actions}')

plt.plot(range(len(episode_scores)), episode_scores)
plt.xlabel('Episodes ->')
plt.ylabel('Score ->')
plt.title('Training progress')
plt.show()
