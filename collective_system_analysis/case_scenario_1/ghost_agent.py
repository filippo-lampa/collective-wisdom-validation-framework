import math
from enum import Enum

import mesa
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from wall_agent import WallAgent
from pacman_agent import PacmanAgent


class DataInclusionLogic(Enum):
    INCLUDE_HIGHER_VALUES = 1
    INCLUDE_LOWER_VALUES = 2
    INCLUDE_NOT_PRESENT_VALUES = 3


class GhostAgent(mesa.Agent):
    def __init__(self, unique_id, model, initial_learning_rate, discount_factor, exploration_rate, should_use_protocol,
                 vicinity_radius, bandwidth, data_inclusion_logic, learning_rate_decay):
        super().__init__(unique_id, model)
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.n_states = self.model.grid.width * self.model.grid.height
        self.n_actions = 4
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.steps = 0
        self.should_use_protocol = should_use_protocol
        self.vicinity_radius = vicinity_radius
        self.bandwidth = bandwidth
        self.data_inclusion_logic = data_inclusion_logic
        self.current_state = (6, 6)
        self.current_state = self.current_state[0] * self.model.grid.width + self.current_state[1]
        self.goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        self.goal_state = self.goal_state[0] * self.model.grid.width + self.goal_state[1]
        self.learning_rate_decay = learning_rate_decay

    def init_agent(self):
        self.learning_rate = self.initial_learning_rate
        self.model.grid.move_agent(self, (6, 6))
        self.current_state = (6, 6)
        self.current_state = self.current_state[0] * self.model.grid.width + self.current_state[1]
        self.steps = 0
        return

    def step(self):

        #learning rate decay
        #if self.steps > 500:
        #    self.learning_rate = self.learning_rate * math.exp(-self.learning_rate_decay)

        self.goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        self.goal_state = self.goal_state[0] * self.model.grid.width + self.goal_state[1]

        if self.should_use_protocol:
            nearby_agents = self.model.grid.get_neighbors(self.pos, moore=True, radius=self.vicinity_radius)
            for agent in nearby_agents:
                if isinstance(agent, GhostAgent):
                    self.exchange_data(agent)

        if self.current_state != self.goal_state:

            print("Ghost ", self.unique_id, " performing step number ", self.steps)

            action = self.choose_action(self.current_state)
            next_state = self.move(action)
            terrain_penalty_factor = self.model.get_terrain_penalty(self.current_state)
            reward = self.get_reward(next_state, terrain_penalty_factor)
            self.update(self.current_state, action, reward, next_state, terrain_penalty_factor)
            self.current_state = next_state
            self.steps += 1

        else:

            print("Ghost ", self.unique_id, " reached the goal state")

            self.model.end_episode()

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def move(self, action):
        x, y = self.model.get_agents_of_type(GhostAgent).select(lambda agent: agent.unique_id == self.unique_id)[0].pos
        walls_positions = []
        for cell_content, (x_axis, y_axis) in self.model.grid.coord_iter():
            if len(cell_content) > 0 and isinstance(cell_content[0], WallAgent):
                walls_positions.append((x_axis, y_axis))

        if action == 0:
            if x > 0:
                x -= 1
        elif action == 1:
            if x < self.model.grid.width - 1:
                x += 1
        elif action == 2:
            if y > 0:
                y -= 1
        elif action == 3:
            if y < self.model.grid.height - 1:
                y += 1
        if (x, y) in walls_positions:
            move = self.model.get_agents_of_type(GhostAgent).select(lambda agent: agent.unique_id == self.unique_id)[0].pos
            return move[0] * self.model.grid.width + move[1]

        move = x * self.model.grid.width + y

        self.model.grid.move_agent(self, (x, y))

        print("Ghost ", self.unique_id, " moved to ", x, y)

        return move

    def get_reward(self, state, terrain_penalty_factor):
        goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        goal_state = goal_state[0] * self.model.grid.width + goal_state[1]
        if state == goal_state:
            return 1
        else:
            return 1 / (abs(state - goal_state) + 1) * (1 - terrain_penalty_factor)

    def update(self, state, action, reward, next_state, terrain_penalty):
        q_value = self.q_table[state, action]
        next_q_value = np.max(self.q_table[next_state])
        self.q_table[state, action] = (q_value + self.learning_rate *
                                       (reward + self.discount_factor * next_q_value - q_value))# * (1 - terrain_penalty))

    def exchange_data(self, agent):
        data = []
        temp_q_table = self.q_table.copy()
        for i in range(self.bandwidth):
            state, action = np.unravel_index(np.argmax(temp_q_table, axis=None), temp_q_table.shape)
            reward = temp_q_table[state, action]
            temp_q_table[state, action] = 0
            data.append({'state': state, 'action': action, 'reward': reward})
        del temp_q_table
        agent.get_data_from_exchange(data)
        print("Ghost {} sent data to ghost {}".format(self.unique_id, agent.unique_id))

    def get_data_from_exchange(self, data):

        def include_higher_values(data):
            #substitute the entries in the q_table where the value in the q_table is lower than the value in
            # [data['state'], data['action']] for the corresponding state and action
            for row in data:
                if self.q_table[row['state'], row['action']] < row['reward']:
                    self.q_table[row['state'], row['action']] = row['reward']

        def include_lower_values(data):
            #substitute the entries in the q_table where the value in the q_table is higher than the value in
            # [data['state'], data['action']] for the corresponding state and action
            for row in data:
                if self.q_table[row['state'], row['action']] > row['reward']:
                    self.q_table[row['state'], row['action']] = row['reward']

        def include_not_present_values(data):
            #include in the q_table the entries in data that are not present in the q_table
            for row in data:
                if self.q_table[row['state'], row['action']] == 0.0:
                    self.q_table[row['state'], row['action']] = row['reward']

        for logic in self.data_inclusion_logic:
            if logic == DataInclusionLogic.INCLUDE_LOWER_VALUES:
                include_lower_values(data)
            elif logic == DataInclusionLogic.INCLUDE_NOT_PRESENT_VALUES:
                include_not_present_values(data)
            elif logic == DataInclusionLogic.INCLUDE_HIGHER_VALUES:
                include_higher_values(data)

        print("Ghost {} received data".format(self.unique_id))

        return
