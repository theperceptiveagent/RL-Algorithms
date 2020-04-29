import random

class SarsaAgent:

    def __init__(self, actions, alpha = 0.2, gamma=0.9):
        """
        The Q-values will be stored in a dictionary. Each key will be of the format: ((x, y), a). 
        params:
            actions (list): A list of all the possible action values.
            alpha (float): step size
            gamma (float): discount factor
        """
        self.Q = {}
        
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma

    def get_Q_value(self, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return self.Q.get((state, action), 0.0) # Return 0.0 if state-action pair does not exist

    def act(self, state, epsilon):
        # Choose a random action
        if random.random() < epsilon:
            action = random.choice(self.actions)
        # Choose the greedy action
        else:
            # Get all the Q-values for all possible actions for the state
            q_values = [self.get_Q_value(state, action) for action in self.actions]
            maxQ = max(q_values)
            # There might be cases where there are multiple actions with the same high q_value. Choose randomly then
            count_maxQ = q_values.count(maxQ)
            if count_maxQ > 1:
                # Get all the actions with the maxQ
                best_action_indexes = [i for i in range(len(self.actions)) if q_values[i] == maxQ]
                action_index = random.choice(best_action_indexes)
            else:
                action_index = q_values.index(maxQ)
            
            action = self.actions[action_index]
        
        return action

    def learn(self, state, action, reward, next_state, next_state_action):
        
        # Note to self: The next_state and next_state_action will be none if the episode is terminated.
        
        # Learning section
        if next_state == None or next_state_action == None:
            self.Q[(state, action)] += reward
        else:
            q_current = self.Q.get((state, action), None) # If this is the first time the state action pair is encountered
            if q_current == None:
                self.Q[(state, action)] = reward
            else:
                q_next = self.get_Q_value(next_state, next_state_action)
                self.Q[(state, action)] += self.alpha * (reward + self.gamma * q_next - q_current)


if __name__ == "__main__":
    agent = SarsaAgent([0, 1, 2, 3])

    action = agent.act((2, 3), 0.001)
    agent.learn((2,3), 0, 5, (4, 5), 1)
    agent.learn((4,5), 3, 1, (8, 18), 2)
    print(agent.Q)
    # print(action)
