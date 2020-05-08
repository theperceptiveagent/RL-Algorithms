from gridworld_envt import Gridworld
from expected_sarsa_agent import ExpectedSarsaAgent

import matplotlib.pyplot as plt
import math

num_episodes = 1000
episode_scores = []

# Epsilon greedy action selection
epsilon = 1 # Start at all actions random
eps_decay_factor = 0.999 # After every episode, eps is 0.9 times the previous one
eps_min = 0.05 # 5% exploration is compulsory till the end

gridworld = Gridworld()
actions = gridworld.action_space

agent = ExpectedSarsaAgent(actions)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=epsilon)
        # print(f'State: {state}, action: {action}')
        new_state, reward, done = gridworld.step(action)

        episode_score += reward

        agent.learn(state, action, reward, new_state, epsilon)

        state = new_state
        episode_actions.append(action)
        if done:
            break

    episode_scores.append(episode_score)
    # Decay epsilon
    epsilon = max(epsilon * eps_decay_factor, eps_min)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions

    print(f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 100): {sum(episode_scores[:-100])/len(episode_scores)}, epsilon: {epsilon}', end='')

print(f'\nAfter {num_episodes}, average score: {sum(episode_scores)/len(episode_scores)}, Average(last 100): {sum(episode_scores[:-100])/len(episode_scores)}')
print(f'Best score: {best_score}, Sequence of actions: {[gridworld.num2action[action] for action in best_path_actions]}')

plt.plot(range(len(episode_scores)), episode_scores)
plt.xlabel('Episodes ->')
plt.ylabel('Score ->')
plt.title('Training progress')
plt.show()
