'''
Multi-agent system using the mesa library to simulate the following case scenario:
Environment similar to Pacman with different sub-environments (q-lerning task): Each ghost in the game
represents a mobile agent. Each ghost explores a different sub-environment and acquires specific knowledge about it.
When agents meet, they exchange subsets of their data, allowing each agent to acquire information about environments it
has not directly explored. The overall goal is to ensure that each ghost has complete knowledge of all sub-environments
to maximize the chances of catching Pacman.
'''
import os

import mesa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pacman_agent import PacmanAgent
from ghost_agent import GhostAgent, DataInclusionLogic
from wall_agent import WallAgent

from mesa import Model


class PacmanModel(Model):
    def __init__(self, num_ghosts):
        self.num_ghosts = num_ghosts
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.running = True
        self.grid = mesa.space.MultiGrid(13, 13, True)
        self.episode_ended = False
        #define a dictionary with the mapping between cells and their corresponding terrain

        def map_terrains():
            # Define the ranges for each terrain
            grass_range = range(0, 55)
            mud_range = range(55, 111)
            standard_range = range(111, 169)

            # Define the slowing factors for each terrain type
            slowing_factors = {
                "standard": 0,
                "grass": 0.8,
                "mud": 0.9
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

        self.walls_positions = [(6, 0),
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
        if (x, y) not in self.walls_positions:
            self.grid.place_agent(pacman, (x, y))
        else:
            while (x, y) in self.walls_positions:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(pacman, (x, y))

        starting_ghosts_position = (6, 6)

        for i in range(1, self.num_ghosts + 1):

            ghost = GhostAgent(i, self, 0.8, 0.95, 0.25, True,
                               1, 100, [DataInclusionLogic.INCLUDE_LOWER_VALUES,
                                        DataInclusionLogic.INCLUDE_NOT_PRESENT_VALUES], 0)

            self.schedule.add(ghost)

            x = starting_ghosts_position[0]
            y = starting_ghosts_position[1]

            self.grid.place_agent(ghost, (x, y))

        for i in range(self.num_ghosts + 1, self.num_ghosts + 1 + len(self.walls_positions)):
            wall = WallAgent(i, self)
            self.schedule.add(wall)
            x = self.walls_positions[i - self.num_ghosts - 1][0]
            y = self.walls_positions[i - self.num_ghosts - 1][1]
            self.grid.place_agent(wall, (x, y))

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Steps": "steps"}
        )

    def get_terrain_penalty(self, cell_id):
        #cell_id = x * self.grid.width + y
        terrain_type, slowing_factor = self.terrain_map[cell_id]
        return slowing_factor

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def end_episode(self):

        self.episode_ended = True

        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)

        if (x, y) not in model.walls_positions:
            self.grid.move_agent(self.get_agents_of_type(PacmanAgent)[0], (x, y))
        else:
            while (x, y) in model.walls_positions:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.move_agent(self.get_agents_of_type(PacmanAgent)[0], (x, y))

        for agent in self.schedule.agents:
            if isinstance(agent, GhostAgent):
                agent.init_agent()

        '''
        agent_counts = np.zeros((model.grid.width, model.grid.height))
        for cell_content, (x, y) in model.grid.coord_iter():
            agent_count = len(cell_content)
            agent_counts[x][y] = agent_count

        g = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
        g.figure.set_size_inches(4, 4)
        g.set(title="Number of agents on each cell of the grid")
        plt.show()
        '''

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

    num_episodes = 100000000

    steps_needed_per_episode = []

    if not os.path.exists('plots'):
        os.makedirs('plots')

    for i in range(num_episodes):

        model.episode_ended = False

        print("Episode number: ", i)

        count = 0

        while not model.episode_ended:

            print("Step number: ", count, " of episode number: ", i)

            model.step()
            '''
            if count % 100000 == 0 and count != 0:

                agent_counts = np.zeros((model.grid.width, model.grid.height))
                for cell_content, (x, y) in model.grid.coord_iter():
                    agent_count = len(cell_content)
                    agent_counts[x][y] = agent_count

                g = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
                g.figure.set_size_inches(4, 4)
                g.set(title="Number of agents on each cell of the grid")
                plt.show()
            '''
            count += 1

        steps_needed_per_episode.append(count)

        print("Average number of steps needed per episode: ", np.mean(steps_needed_per_episode))

        if i % 100 == 0 and i != 0:

            g = sns.lineplot(data=steps_needed_per_episode)
            #g.set(title="Number of steps needed per episode with protocol", ylabel="Number of steps", xlabel="Episode number")
            g.set(title="Number of steps needed per episode without protocol", ylabel="Number of steps",
                  xlabel="Episode number")
            plt.savefig(os.path.join(os.getcwd(), 'plots', 'steps_needed_per_episode.png'))
            plt.show()

    g = sns.lineplot(data=steps_needed_per_episode, x=range(len(steps_needed_per_episode)), y=steps_needed_per_episode)
    g.set(title="Number of steps needed per episode", ylabel="Number of steps", xlabel="Episode number")
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'steps_needed_per_episode.png'))
    plt.show()


