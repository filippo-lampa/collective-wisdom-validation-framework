import math
from enum import Enum

import mesa
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import a_star as astar
from wall_agent import WallAgent
from pacman_agent import PacmanAgent


class DataInclusionLogic(Enum):
    # Substitute the entries in the q_table where the value in the q_table is lower than the value received from the
    # exchange for the corresponding state and action.
    INCLUDE_HIGHER_VALUES = 1
    # Substitute the entries in the q_table where the value in the q_table is higher than the value received from the
    # exchange for the corresponding state and action.
    INCLUDE_LOWER_VALUES = 2
    # Include in the q_table the entries in data that are not present in the q_table.
    INCLUDE_NOT_PRESENT_VALUES = 3


class DataSelectionLogic(Enum):
    HIGHEST_REWARD = 1
    LOWEST_REWARD = 2
    RANDOM = 3


class GhostAgent(mesa.Agent):
    def __init__(self, unique_id, model, initial_learning_rate, discount_factor, exploration_rate, should_use_protocol,
                 vicinity_radius, bandwidth, data_inclusion_logic, learning_rate_decay, data_selection_logic,
                 should_penalize_terrain, get_closer_reward, get_farther_reward, catch_pacman_reward, hit_wall_reward,
                 max_steps, early_stopping):
        super().__init__(unique_id, model)
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.n_states = self.model.grid.width * self.model.grid.height
        self.n_actions = 4
        self.q_table = np.zeros((self.n_states, self.n_actions))
        #self.q_table = np.full((self.n_states, self.n_actions), 10.0) #understand why they get stuckd for some iterations
        self.steps = 0
        self.should_use_protocol = should_use_protocol
        self.vicinity_radius = vicinity_radius
        self.bandwidth = bandwidth
        self.target_bandwidth = bandwidth
        assert self.bandwidth <= self.n_states * self.n_actions
        self.data_inclusion_logic = data_inclusion_logic
        self.current_state = (7, 7)
        self.current_state = self.current_state[0] * self.model.grid.width + self.current_state[1]
        self.goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        self.goal_state = self.goal_state[0] * self.model.grid.width + self.goal_state[1]
        self.learning_rate_decay = learning_rate_decay
        self.data_selection_logic = data_selection_logic
        self.should_penalize_terrain = should_penalize_terrain
        self.movement_countdown = self.model.get_terrain_penalty(self.current_state)
        self.get_closer_reward = get_closer_reward
        self.get_farther_reward = get_farther_reward
        self.catch_pacman_reward = catch_pacman_reward
        self.hit_wall_reward = hit_wall_reward
        self.max_steps = max_steps
        self.early_stopping = early_stopping

    def init_agent(self):
        self.learning_rate = self.initial_learning_rate
        self.model.grid.move_agent(self, (7, 7))
        self.current_state = (7, 7)
        self.current_state = self.current_state[0] * self.model.grid.width + self.current_state[1]
        self.steps = 0
        self.movement_countdown = self.model.get_terrain_penalty(self.current_state)
        return

    def step(self):

        if self.early_stopping and self.steps == self.max_steps:
            self.model.end_episode(True, self.steps)
            return

        self.model.plot_env_status(self.steps)

        #learning rate decay
        # if self.steps > 50:
        #     self.learning_rate = self.learning_rate * math.exp(-self.learning_rate_decay)

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

            if not self.should_penalize_terrain:
                next_state = self.move(action)
            else:
                if self.movement_countdown == 0:
                    next_state = self.move(action)
                    self.movement_countdown = self.model.get_terrain_penalty(next_state)
                else:
                    next_state = self.current_state
                    self.movement_countdown -= 1
            terrain_penalty_factor = self.model.get_terrain_penalty(self.current_state)
            reward = self.get_reward(next_state, terrain_penalty_factor)
            self.update(self.current_state, action, reward, next_state)
            # if the position of the ghost is the same of the next state, it means that the ghost chose a valid action
            # and we can update the current state, otherwise the ghost hit a wall and we keep the current state
            if self.pos == (next_state // self.model.grid.width, next_state % self.model.grid.width):
                self.current_state = next_state
            self.steps += 1

        else:

            print("Ghost ", self.unique_id, " reached the goal state")

            self.model.end_episode(False, self.steps)

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

        if action == 0: # move left
            if x > 0:
                x -= 1
        elif action == 1: # move right
            if x < self.model.grid.width - 1:
                x += 1
        elif action == 2: # move up
            if y > 0:
                y -= 1
        elif action == 3: # move down
            if y < self.model.grid.height - 1:
                y += 1
        if (x, y) in walls_positions:
            print("Ghost ", self.unique_id, " hit a wall")
            return x * self.model.grid.width + y

        move = x * self.model.grid.width + y

        self.model.grid.move_agent(self, (x, y))

        print("Ghost ", self.unique_id, " moved to ", x, y)

        return move

    def get_reward(self, state, terrain_penalty_factor):
        goal_state = self.model.get_agents_of_type(PacmanAgent)[0].pos
        goal_state = goal_state[0] * self.model.grid.width + goal_state[1]
        if state == goal_state:
            return self.catch_pacman_reward
        else:

            if self.early_stopping and self.steps == self.max_steps:
                return -50

            path_from_current_state = astar.astar([[0] * self.model.grid.width for _ in range(self.model.grid.height)],
                                                  astar.Node(self.current_state // self.model.grid.width, self.current_state % self.model.grid.width),
                        astar.Node(goal_state // self.model.grid.width, goal_state % self.model.grid.width), self.model.get_wall_positions(),
                                                  self.model, self.should_penalize_terrain)

            path_from_next_state = astar.astar([[0] * self.model.grid.width for _ in range(self.model.grid.height)],
                                                    astar.Node(state // self.model.grid.width, state % self.model.grid.width),
                                                    astar.Node(goal_state // self.model.grid.width, goal_state % self.model.grid.width), self.model.get_wall_positions(),
                                                    self.model, self.should_penalize_terrain)

            proximity_reward = len(path_from_current_state) - len(path_from_next_state)

            agents_in_next_state = self.model.grid.get_cell_list_contents((state // self.model.grid.width, state % self.model.grid.width))
            if len(agents_in_next_state) > 0 and isinstance(agents_in_next_state[0], WallAgent):
                wall_penalty = self.hit_wall_reward
            else:
                wall_penalty = 0

            reward = proximity_reward + wall_penalty
            return reward

    def update(self, state, action, reward, next_state):
        q_value = self.q_table[state, action]
        next_q_value = np.max(self.q_table[next_state])
        self.q_table[state, action] = (q_value + self.learning_rate *
                                       (reward + self.discount_factor * next_q_value - q_value))

    def exchange_data(self, agent):
        selected_data = self.select_data_to_send()
        agent.get_data_from_exchange(selected_data)
        print("Ghost {} sent data to ghost {}".format(self.unique_id, agent.unique_id))

    def select_data_to_send(self):

        def select_highest_reward_entries():
            entries_to_select = []
            q_table_copy = np.copy(self.q_table)
            for i in range(self.bandwidth):
                state, action = np.unravel_index(np.argmax(q_table_copy), self.q_table.shape)
                entries_to_select.append({'state': state, 'action': action, 'reward': self.q_table[state, action]})
                q_table_copy[state, action] = -np.inf
            del q_table_copy
            return entries_to_select

        def select_lowest_reward_entries():
            entries_to_select = []
            q_table_copy = np.copy(self.q_table)
            for i in range(self.bandwidth):
                state, action = np.unravel_index(np.argmin(q_table_copy), self.q_table.shape)
                entries_to_select.append({'state': state, 'action': action, 'reward': self.q_table[state, action]})
                q_table_copy[state, action] = np.inf
            del q_table_copy
            return entries_to_select

        def select_random_entries():
            entries_to_select = []
            for i in range(self.bandwidth):
                state = np.random.randint(0, self.n_states)
                action = np.random.randint(0, self.n_actions)
                entries_to_select.append({'state': state, 'action': action, 'reward': self.q_table[state, action]})
            return entries_to_select

        if self.data_selection_logic == DataSelectionLogic.HIGHEST_REWARD:
            return select_highest_reward_entries()
        elif self.data_selection_logic == DataSelectionLogic.LOWEST_REWARD:
            return select_lowest_reward_entries()
        elif self.data_selection_logic == DataSelectionLogic.RANDOM:
            return select_random_entries()

    def get_data_from_exchange(self, data):

        def include_higher_values(data):
            for row in data:
                if self.q_table[row['state'], row['action']] < row['reward']:
                    self.q_table[row['state'], row['action']] = row['reward']

        def include_lower_values(data):
            for row in data:
                if row['reward'] == 0.0:
                    continue
                if self.q_table[row['state'], row['action']] > row['reward']:
                    self.q_table[row['state'], row['action']] = row['reward']

        def include_not_present_values(data):
            for row in data:
                if self.q_table[row['state'], row['action']] == 0.0:
                    self.q_table[row['state'], row['action']] = row['reward']

        for logic in self.data_inclusion_logic:
            if logic == DataInclusionLogic.INCLUDE_NOT_PRESENT_VALUES:
                include_not_present_values(data)
            elif logic == DataInclusionLogic.INCLUDE_LOWER_VALUES:
                include_lower_values(data)
            elif logic == DataInclusionLogic.INCLUDE_HIGHER_VALUES:
                include_higher_values(data)

        print("Ghost {} received data".format(self.unique_id))

        return

    def update_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        print("Ghost ", self.unique_id, " increased bandwidth to ", self.bandwidth)
        return
