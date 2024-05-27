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
    def __init__(self, unique_id, model, learning_rate, discount_factor, exploration_rate, should_use_protocol,
                 vicinity_radius, bandwidth, data_inclusion_logic):
        super().__init__(unique_id, model)
        self.learning_rate = learning_rate
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

    def step(self):

        if self.should_use_protocol:
            nearby_agents = self.model.grid.get_neighbors(self[0].pos, moore=True, radius=self.vicinity_radius)
            for agent in nearby_agents:
                if isinstance(agent, GhostAgent):
                    self.exchange_data(agent)

        current_state = self.model.get_agents_of_type(GhostAgent).select(lambda agent: agent.unique_id == self.unique_id)[0].pos
        current_state = current_state[0] * self.model.grid.width + current_state[1]
        goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        print("Goal: ", goal_state)
        goal_state = goal_state[0] * self.model.grid.width + goal_state[1]

        while current_state != goal_state:

            if self.steps % 100000 == 0:
                agent_counts = np.zeros((self.model.grid.width, self.model.grid.height))
                for cell_content, (x, y) in self.model.grid.coord_iter():
                    agent_count = len(cell_content)
                    agent_counts[x][y] = agent_count

                g = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
                g.figure.set_size_inches(4, 4)
                g.set(title="Number of agents on each cell of the grid")
                plt.show()

            print("Ghost agent ", self.unique_id, ", episode number ", self.model.schedule.steps)
            print("Ghost ", self.unique_id, " performing step number ", self.steps, " of the episode")

            action = self.choose_action(current_state)
            next_state = self.move(action)
            reward = self.get_reward(next_state)
            terrain_penalty_factor = self.model.get_terrain_penalty(current_state)
            self.update(current_state, action, reward, next_state, terrain_penalty_factor)
            current_state = next_state
            self.steps += 1

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.q_table[state])
        print(action)
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

    def get_reward(self, state):
        goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        goal_state = goal_state[0] * self.model.grid.width + goal_state[1]
        if state == goal_state:
            return 1
        else:
            return 1 / (abs(state - goal_state) + 1)

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
            for index, row in data.iterrows():
                if self.q_table[row['state'], row['action']] < row['reward']:
                    self.q_table[row['state'], row['action']] = row['reward']

        def include_lower_values(data):
            #substitute the entries in the q_table where the value in the q_table is higher than the value in
            # [data['state'], data['action']] for the corresponding state and action
            for index, row in data.iterrows():
                if self.q_table[row['state'], row['action']] > row['reward']:
                    self.q_table[row['state'], row['action']] = row['reward']

        def include_not_present_values(data):
            #include in the q_table the entries in data that are not present in the q_table
            for index, row in data.iterrows():
                if self.q_table[row['state'], row['action']] == 0:
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
