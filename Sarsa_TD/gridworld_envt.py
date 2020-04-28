import numpy as np

class Gridworld:

    def __init__(self, start_pos=(3, 0), goal_pos=(3, 7)):
        self.grid = np.zeros((7, 10))

        self.start_x, self.start_y = start_pos[0], start_pos[1]
        self.curr_x, self.curr_y = None, None
        self.goal_x, self.goal_y = goal_pos[0], goal_pos[1]

        self.wind_offsets = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.num2action = {
                0: 'left',
                1: 'right', 
                2: 'up', 
                3: 'down'
            }
        self.done = False # A flag to check if goal is reached.

    def reset(self):
        self.curr_x, self.curr_y = self.start_x, self.start_y

        self.done = False

        return (self.curr_x, self.curr_y)

    def step(self, action):

        assert self.curr_x is not None and self.curr_y is not None, "No current state defined. Did you reset the environment? "\
                                                                    "Call env.reset()"
        
        assert action in [0, 1, 2, 3], 'action must be a number from 0-3'

        # Add wind's effect
        self.wind_effect()

        if action == 0:
            self.curr_y -= 1 if self.curr_y != 0 else 0
        elif action == 1:
            self.curr_y += 1 if self.curr_y != 9 else 0
        elif action == 2:
            self.curr_x -= 1 if self.curr_x != 0 else 0
        elif action == 3:
            self.curr_x += 1 if self.curr_x != 6 else 0

        reward_for_action = self.calculate_reward()

        return ((self.curr_x, self.curr_y), reward_for_action, self.done)
    
    def wind_effect(self):
        displacement = self.wind_offsets[self.curr_y]
        self.curr_x -= displacement if self.curr_x != 0 else 0

    def calculate_reward(self):
        if self.curr_x == self.goal_x and self.curr_y == self.curr_y:
            self.done = True
            return 10
        else:
            return -1

    def render(self):
        assert self.curr_x is not None and self.curr_y is not None, "No current state defined. Did you reset the environment? "\
                                                                    "Call env.reset()"
        # print('P -> Current Position, G -> Goal Position.\n')
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if i == self.curr_x and j == self.curr_y:
                    print(' P ', end='' if i<10 else '\n')
                elif i == self.goal_x and j == self.goal_y:
                    print(' G ', end='' if i<10 else '\n')
                else:
                    print(' - ', end='')
            print()
        
        print()
        if self.done:
            print('Goal reached!')


gridworld = Gridworld()
# state = gridworld.reset()
# gridworld.render()
for i in range(9):
    state, reward, done = gridworld.step(1)
for i in range(4):
    state, reward, done = gridworld.step(3)
for i in range(2):
    state, reward, done = gridworld.step(0)
gridworld.render()
print(state)

