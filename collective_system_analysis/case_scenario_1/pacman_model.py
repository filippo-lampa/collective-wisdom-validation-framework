'''
Multi-agent system using the mesa library to simulate the following case scenario:
Environment similar to Pacman with different sub-environments (q-lerning task): Each ghost in the game
represents a mobile agent. Each ghost explores a different sub-environment and acquires specific knowledge about it.
When agents meet, they exchange subsets of their data, allowing each agent to acquire information about environments it
has not directly explored. The overall goal is to ensure that each ghost has complete knowledge of all sub-environments
to maximize the chances of catching Pacman.
'''
import mesa
import numpy as np
import pandas as pd
import random

import seaborn as sns
import matplotlib.pyplot as plt

from collective_system_analysis.case_scenario_1.pacman_agent import PacmanAgent
from collective_system_analysis.case_scenario_1.ghost_agent import GhostAgent, DataInclusionLogic
from collective_system_analysis.case_scenario_1.wall_agent import WallAgent

from mesa import Agent, Model


class PacmanModel(Model):
    def __init__(self, num_ghosts):
        self.num_ghosts = num_ghosts
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.running = True
        self.grid = mesa.space.MultiGrid(13, 13, True)
        #define a dictionary with the mapping between cells and their corresponding terrain

        def map_terrains():
            # Define the ranges for each terrain
            grass_range = range(0, 48)
            mud_range = range(48, 96)
            standard_range = range(96, 145)

            # Define the slowing factors for each terrain type
            slowing_factors = {
                "standard": 0,
                "grass": 0.2,
                "mud": 0.4
            }

            # Create the dictionary to map grid locations to terrain types and slowing factors
            terrain_map = {}

            # Populate the dictionary with grass terrain
            for i in grass_range:
                terrain_map[i] = ("grass", slowing_factors["grass"])

            # Populate the dictionary with mud terrain
            for i in mud_range:
                terrain_map[i] = ("mud", slowing_factors["mud"])

            # Populate the dictionary with standard terrain
            for i in standard_range:
                terrain_map[i] = ("standard", slowing_factors["standard"])

            return terrain_map

        self.terrain_map = map_terrains()

        walls_positions = [(6, 0),
                           (1, 1), (3, 1), (4, 1), (6, 1), (8, 1), (9, 1), (11, 1),
                           (1, 3), (3, 3), (5, 3), (6, 3), (7, 3), (9, 3), (11, 3),
                           (0, 5), (1, 5), (4, 5), (5, 5), (7, 5), (8, 5), (11, 5), (12, 5),
                           (4, 6), (8, 6),
                           (0, 7), (1, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (11, 7), (12, 7),
                           (1, 9), (3, 9), (5, 9), (6, 9), (7, 9), (9, 9), (11, 9),
                           (1, 11), (3, 11), (4, 11), (6, 11), (8, 11), (9, 11), (11, 11),
                           (6, 12)]

        pacman = PacmanAgent(0, self)
        self.schedule.add(pacman)
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        if (x, y) not in walls_positions:
            self.grid.place_agent(pacman, (x, y))
        else:
            while (x, y) in walls_positions:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(pacman, (x, y))

        starting_ghosts_position = (6, 6)

        learning_rate = 0.1
        discount_factor = 0.9
        exploration_rate = 0.1

        for i in range(1, self.num_ghosts + 1):

            ghost = GhostAgent(i, self, 0.1, 0.9, 0.1, False,
                               1, 100, [DataInclusionLogic.INCLUDE_HIGHER_VALUES,
                                        DataInclusionLogic.INCLUDE_NOT_PRESENT_VALUES])

            self.schedule.add(ghost)

            x = starting_ghosts_position[0]
            y = starting_ghosts_position[1]

            self.grid.place_agent(ghost, (x, y))

        for i in range(self.num_ghosts + 1, self.num_ghosts + 1 + len(walls_positions)):
            wall = WallAgent(i, self)
            self.schedule.add(wall)
            x = walls_positions[i][0]
            y = walls_positions[i][1]
            self.grid.place_agent(wall, (x, y))

        #self.datacollector = mesa.DataCollector(
        #    agent_reporters={}
        #)

    def get_terrain_penalty(self, x, y):
        cell_id = x * self.grid.width + y
        terrain_type, slowing_factor = self.terrain_map[cell_id]
        return slowing_factor

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':

    model = PacmanModel(3)

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell_content, (x, y) in model.grid.coord_iter():
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    g = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
    g.figure.set_size_inches(4, 4)
    g.set(title="Number of agents on each cell of the grid")
    plt.show()

    for i in range(10000000):
        #every step corresponds to an episode in the q-learning task
        model.step()
        for agent in model.schedule.agents:
            if isinstance(agent, GhostAgent):
                model.grid.move_agent(agent, (6, 6))
                agent.steps = 0
            elif isinstance(agent, PacmanAgent):
                model.grid.move_agent(agent, (model.random.randrange(model.grid.width), model.random.randrange(model.grid.height)))

        '''
        if i % 5000 == 0 and i != 0:
            model.datacollector.collect(model)

            # get data for all the agents and plot them in a single graph

            # get data for all the agents and plot them in a single graph where data are not nan
            agents_data = model.datacollector.get_agent_vars_dataframe().dropna()

            g = sns.lineplot(data=agents_data, x="Step", y="MAE", hue="AgentID")
            g.set(title="MAE over time - Time step " + str(i), ylabel="MAE")
            plt.show()

            g = sns.lineplot(data=agents_data, x="Step", y="RMSE", hue="AgentID")
            g.set(title="RMSE data over time - Time step " + str(i), ylabel="RMSE")
            plt.show()

            g = sns.lineplot(data=agents_data, x="Step", y="Accuracy", hue="AgentID")
            g.set(title="Accuracy data over time - Time step " + str(i), ylabel="Accuracy")
            plt.show()

    # model_data = model.datacollector.get_model_vars_dataframe()

    # g = sns.lineplot(data=model_data)
    # g.set(title="Model data over Time", ylabel="Model Data")

    agents_data = model.datacollector.get_agent_vars_dataframe()

    g = sns.lineplot(data=agents_data, x="Step", y="MAE", hue="AgentID")
    g.set(title="MAE over time - Last step ", ylabel="MAE")
    plt.show()

    g = sns.lineplot(data=agents_data, x="Step", y="RMSE", hue="AgentID")
    g.set(title="RMSE data over time - Last step ", ylabel="RMSE")
    plt.show()

    g = sns.lineplot(data=agents_data, x="Step", y="Accuracy", hue="AgentID")
    g.set(title="Accuracy data over time - Last step ", ylabel="Accuracy")
    plt.show()
    '''