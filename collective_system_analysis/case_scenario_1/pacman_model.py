'''
Multi-agent system using the mesa library to simulate the following case scenario:
Environment similar to Pacman with different sub-environments (q-lerning task): Each ghost in the game
represents a mobile agent. Each ghost explores a different sub-environment and acquires specific knowledge about it.
When agents meet, they exchange subsets of their data, allowing each agent to acquire information about environments it
has not directly explored. The overall goal is to ensure that each ghost has complete knowledge of all sub-environments
to maximize the chances of catching Pacman.
'''
import os
import argparse

import mesa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

from pacman_agent import PacmanAgent
from ghost_agent import GhostAgent, DataInclusionLogic, DataSelectionLogic
from wall_agent import WallAgent

from mesa import Model


class PacmanModel(Model):

    def __init__(self, num_ghosts, args: argparse.Namespace):
        super().__init__()
        self.num_ghosts = num_ghosts
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.running = True
        self.grid = mesa.space.MultiGrid(15, 15, True)
        self.episode_ended = False
        self.args = args
        self.last_step_plotted = -1
        self.ax = None
        self.fig = None
        self.current_episode = 0
        self.interactive_plot = args.interactive_plot
        self.steps_needed_per_episode = []

        args_inclusion_logic_mapping = {
            'higher': DataInclusionLogic.INCLUDE_HIGHER_VALUES,
            'lower': DataInclusionLogic.INCLUDE_LOWER_VALUES,
            'not_present': DataInclusionLogic.INCLUDE_NOT_PRESENT_VALUES
        }

        args_data_selection_logic_mapping = {
            'highest_reward': DataSelectionLogic.HIGHEST_REWARD,
            'lowest_reward': DataSelectionLogic.LOWEST_REWARD,
            'random': DataSelectionLogic.RANDOM
        }

        args_data_inclusion_logics = args.data_inclusion_logics.split(',')

        def map_terrains():
            # Define the ranges for each terrain. Upper left side is grass, upper right side is mud and the rest is standard

            grass_range = [i for i in range(16, 22)] + [i for i in range(31, 37)] + [i for i in range(46, 52)] + [i for i in range(61, 67)]
            mud_range = [i for i in range(158, 164)] + [i for i in range(173, 179)] + [i for i in range(188, 194)] + [i for i in range(203, 209)]
            all_indices = set(range(self.grid.width * self.grid.height))
            grass_range = set(grass_range)
            mud_range = set(mud_range)
            standard_range = all_indices - (grass_range | mud_range)



            # Define the slowing factors for each terrain type
            slowing_factors = {
                "standard": args.standard_terrain_slowing_factor,
                "grass": args.grass_terrain_slowing_factor,
                "mud": args.mud_terrain_slowing_factor
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

        # self.walls_positions = [(6, 0),
        #                    (1, 1), (3, 1), (4, 1), (6, 1), (8, 1), (9, 1), (11, 1),
        #                    (1, 3), (3, 3), (5, 3), (6, 3), (7, 3), (9, 3), (11, 3),
        #                    (0, 5), (1, 5), (4, 5), (5, 5), (7, 5), (8, 5), (11, 5), (12, 5),
        #                    (4, 6), (8, 6),
        #                    (0, 7), (1, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (11, 7), (12, 7),
        #                    (1, 9), (3, 9), (5, 9), (6, 9), (7, 9), (9, 9), (11, 9),
        #                    (1, 11), (3, 11), (4, 11), (6, 11), (8, 11), (9, 11), (11, 11),
        #                    (6, 12)]

        self.walls_positions = [(7, 1),
                                (2, 2), (4, 2), (5, 2), (7, 2), (9, 2), (10, 2), (12, 2),
                                (2, 4), (4, 4), (6, 4), (7, 4), (8, 4), (10, 4), (12, 4),
                                (1, 6), (2, 6), (5, 6), (6, 6), (8, 6), (9, 6), (12, 6), (13, 6),
                                (5, 7), (9, 7),
                                (1, 8), (2, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (12, 8), (13, 8),
                                (2, 10), (4, 10), (6, 10), (7, 10), (8, 10), (10, 10), (12, 10),
                                (2, 12), (4, 12), (5, 12), (7, 12), (9, 12), (10, 12), (12, 12),
                                (7, 13)]

        boundary_walls = [(i, 0) for i in range(self.grid.width)] + [(i, self.grid.width - 1) for i in range(self.grid.width)] + \
                         [(0, i) for i in range(self.grid.height)] + [(self.grid.height - 1, i) for i in range(self.grid.height)]

        all_walls_positions = set(self.walls_positions + boundary_walls)

        self.walls_positions = list(all_walls_positions)
        '''
        pacman = PacmanAgent(0, self, self.args.should_penalize_terrain)
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

        starting_ghosts_position = (7, 7)

        for i in range(1, self.num_ghosts + 1):

            ghost = GhostAgent(i, self, self.args.initial_learning_rate, self.args.discount_factor,
                               self.args.exploration_rate, self.args.use_protocol,
                               self.args.vicinity_radius, self.args.bandwidth,
                               [args_inclusion_logic_mapping[logic] for logic in args_data_inclusion_logics],
                               self.args.learning_rate_decay, args_data_selection_logic_mapping[args.data_selection_logic],
                               self.args.should_penalize_terrain, self.args.get_closer_reward, self.args.get_farther_reward,
                               self.args.catch_pacman_reward, self.args.hit_wall_reward, self.args.max_steps,
                               self.args.early_stopping)

            self.schedule.add(ghost)

            x = starting_ghosts_position[0]
            y = starting_ghosts_position[1]

            self.grid.place_agent(ghost, (x, y))
        '''
        for i in range(self.num_ghosts + 1, self.num_ghosts + 1 + len(self.walls_positions)):
            wall = WallAgent(i, self)
            self.schedule.add(wall)
            x = self.walls_positions[i - self.num_ghosts - 1][0]
            y = self.walls_positions[i - self.num_ghosts - 1][1]
            self.grid.place_agent(wall, (x, y))

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Steps": "steps"}
        )

    def get_wall_positions(self):
        return self.walls_positions

    def get_terrain_penalty(self, cell_id):
        #cell_id = x * self.grid.width + y
        terrain_type, slowing_factor = self.terrain_map[cell_id]
        return slowing_factor

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def end_episode(self, stopped_early, step):

        if not stopped_early:
            self.steps_needed_per_episode.append(step)

        if self.interactive_plot:
            self.last_step_plotted = -1

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

    def plot_map_terrains(self):
        #plot grass terrain in green, mud terrain in brown and standard terrain in yellow.
        # The walls are plotted in black

        color_map = {
            0: "green",
            1: "olive",
            2: "yellow",
            3: "black"
        }

        grid_size = (15, 15)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set the limits of the plot
        ax.set_xlim(0, grid_size[0])
        ax.set_ylim(0, grid_size[1])

        # Draw the grid
        for x in range(grid_size[0] + 1):
            ax.axhline(x, color='gray', linewidth=0.5)
            ax.axvline(x, color='gray', linewidth=0.5)

        walls_positions = [(y, x) for x, y in self.walls_positions]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_id = i * grid_size[0] + j
                terrain_type, slowing_factor = self.terrain_map[cell_id]
                color = color_map[3] if (i, j) in walls_positions else color_map[0] if terrain_type == "grass" \
                    else color_map[1] if terrain_type == "mud" else color_map[2]
                rect = patches.Rectangle((j, grid_size[0] - i - 1), 1, 1, linewidth=1, edgecolor='none',
                                         facecolor=color)
                ax.add_patch(rect)

        # Set aspect of the plot to be equal
        ax.set_aspect('equal')

        # Hide the axis
        ax.axis('off')

        # Show the plot
        plt.show()

    def plot_env_status(self, step):
        if not self.interactive_plot:
            return
        print("Plotting environment status")
        if step > self.last_step_plotted:
            self.last_step_plotted = step
            agent_counts = np.zeros((self.grid.width, self.grid.height))
            for cell_content, (x, y) in self.grid.coord_iter():
                agent_count = len(cell_content)
                agent_counts[x][y] = agent_count

            def color_agent(agent):
                color = None
                if isinstance(agent, PacmanAgent):
                    color = 'yellow'
                elif isinstance(agent, GhostAgent):
                    if agent.unique_id == 1:
                        color = 'cyan'
                    elif agent.unique_id == 2:
                        color = 'orange'
                    else:
                        color = 'pink'
                elif isinstance(agent, WallAgent):
                    color = 'blue'

                rect = patches.Rectangle((agent.pos[0], agent.pos[1]), 1, 1, linewidth=1, edgecolor='black',
                                         facecolor=color)
                self.ax.add_patch(rect)

            if self.fig is None:
                plt.ion()
                plt.style.use('dark_background')
                self.fig, self.ax = plt.subplots()
                self.heatmap = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
                sns.set_style(rc={'figure.facecolor': 'red'})
                self.heatmap.figure.set_size_inches(4, 4)
                self.heatmap.set_title("Environment status at step " + str(step) + " of episode " + str(self.current_episode))
                pacman_agent = self.get_agents_of_type(PacmanAgent)[0]
                color_agent(pacman_agent)
                ghost_agents = self.get_agents_of_type(GhostAgent)
                for ghost in ghost_agents:
                    color_agent(ghost)
                '''
                wall_agents = self.get_agents_of_type(WallAgent)
                for wall in wall_agents:
                    color_agent(wall)
                '''
                plt.show()
            else:
                # Update existing plot
                self.ax.clear()  # Clear the axis before plotting new data
                self.heatmap = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True,
                                           ax=self.ax)
                self.heatmap.set_title("Environment status at step " + str(step) + " of episode " + str(self.current_episode))

                pacman_agent = self.get_agents_of_type(PacmanAgent)[0]
                color_agent(pacman_agent)
                ghost_agents = self.get_agents_of_type(GhostAgent)
                for ghost in ghost_agents:
                    color_agent(ghost)
                # wall_agents = self.get_agents_of_type(WallAgent)
                # for wall in wall_agents:
                #     color_agent(wall)

                plt.style.use("dark_background")
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()




if __name__ == '__main__':

    # model args
    parser = argparse.ArgumentParser(description='Run the multi-agent reinforcement learning system')
    parser.add_argument('--num_ghosts', type=int, default=3, help='Number of ghosts in the game')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of episodes to run')
    parser.add_argument('--plot_episodes', type=int, default=500, help='Number of steps between performance plots')
    parser.add_argument('--standard_terrain_slowing_factor', type=float, default=0, help='Slowing factor for standard terrain')
    parser.add_argument('--grass_terrain_slowing_factor', type=float, default=2, help='Slowing factor for grass terrain')
    parser.add_argument('--mud_terrain_slowing_factor', type=float, default=4, help='Slowing factor for mud terrain')
    parser.add_argument('--plots_name', type=str, default='', help='Name shown in the plots')
    parser.add_argument('--should_penalize_terrain', action='store_true', help='Enable terrain penalties for the ghosts')

    # ghosts args
    parser.add_argument('--bandwidth_increase_rate', type=float, default=0.1, help='Rate at which the bandwidth increases for the ghosts')
    parser.add_argument('--should_increase_bandwidth', action='store_true', help='Enable increasing bandwidth for the ghosts through time')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stop for the ghosts')
    parser.add_argument('--max_steps', type=int, default=400, help='Maximum number of steps per episode')
    parser.add_argument('--hit_wall_reward', type=float, default=-20, help='Reward for hitting a wall')
    parser.add_argument('--get_closer_reward', type=float, default=40, help='Reward for getting closer to Pacman')
    parser.add_argument('--get_farther_reward', type=float, default=-20, help='Reward for getting farther from Pacman')
    parser.add_argument('--catch_pacman_reward', type=float, default=100, help='Reward for catching Pacman')
    parser.add_argument('--initial_learning_rate', type=float, default=0.95, help='Initial learning rate for the ghosts')
    parser.add_argument('--discount_factor', type=float, default=0.95, help='Discount factor for the ghosts')
    parser.add_argument('--exploration_rate', type=float, default=0.25, help='Exploration rate for the ghosts')
    parser.add_argument('--use_protocol', action='store_true', help='Use protocol for data exchange between ghosts')
    parser.add_argument('--vicinity_radius', type=int, default=1, help='Vicinity radius for data exchange between ghosts')
    parser.add_argument('--bandwidth', type=int, default=100, help='Bandwidth for data exchange between ghosts')
    parser.add_argument('--data_inclusion_logics', type=str, default='lower,not_present', help='Logics for including exchange data between ghosts in q-tables (can choose more than one): higher, lower, not_present')
    parser.add_argument('--data_selection_logic', type=str, default='random', help='Logic for selecting data to exchange between ghosts in q-tables: highest_reward, lowest_reward, random')
    parser.add_argument('--learning_rate_decay', type=float, default=0, help='Learning rate decay for the ghosts')
    parser.add_argument('--interactive_plot', action='store_true', help='Enable interactive plot')

    args = parser.parse_args()

    if not os.path.exists('logs'):
        os.makedirs('logs')

    with open(os.path.join(os.getcwd(), 'logs', 'args_' + args.plots_name + '.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, value))

    model = PacmanModel(args.num_ghosts, args)

    model.plot_map_terrains()

    plots_name = args.plots_name

    num_episodes = args.num_episodes
    plot_episodes = args.plot_episodes

    mean_steps_needed_per_episode = []
    early_ratio = []

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell_content, (x, y) in model.grid.coord_iter():
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    g = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
    g.figure.set_size_inches(4, 4)
    g.set(title="Number of agents on each cell of the grid")
    plt.show()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    for i in range(num_episodes):

        if args.should_increase_bandwidth and i > 10:
            model.get_agents_of_type(GhostAgent)[0].update_bandwidth(int(
                min(args.bandwidth, args.bandwidth * args.bandwidth_increase_rate * (i - 10))))

        model.current_episode = i

        model.episode_ended = False

        print("Episode number: ", i)

        count = 0

        while not model.episode_ended:

            print("Step number: ", count, " of episode number: ", i, "for simulation: ", plots_name)

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

        mean_steps_needed = np.mean(model.steps_needed_per_episode)

        print("Average number of steps needed per episode: ", mean_steps_needed)

        mean_steps_needed_per_episode.append(mean_steps_needed)

        if args.early_stopping and i > 0:
            early_stops_ratio = len(model.steps_needed_per_episode) / i
            early_ratio.append(early_stops_ratio)

        if i % plot_episodes == 0 and i != 0:

            g = sns.lineplot(data=model.steps_needed_per_episode)
            #g.set(title="Number of steps needed per episode with protocol", ylabel="Number of steps", xlabel="Episode number")
            g.set(title=plots_name, ylabel="Number of steps",
                  xlabel="Episode number")
            plt.show()

            g = sns.lineplot(data=mean_steps_needed_per_episode)
            #g.set(title="Number of steps needed per episode with protocol", ylabel="Number of steps", xlabel="Episode number")
            g.set(title=plots_name, ylabel="Mean Number of Steps per Episode",
                  xlabel="Episode number")
            plt.savefig(os.path.join(os.getcwd(), 'plots_countdown_version', 'mean_steps_needed_per_episode_' + plots_name + '.png'))
            plt.show()

            #plot the ratio of the number of early stops to the total number of episodes to see how it evolves over time. The number of early
            # stops is the length of the steps_needed_per_episode list. The ratio is the number of early stops divided by the total number of episodes.
            if args.early_stopping:
                g = sns.lineplot(data=early_ratio)
                g.set(title="Ratio of non-early ended episodes to total number of episodes", ylabel="Ratio", xlabel="Episode number")
                plt.show()

    g = sns.lineplot(data=model.steps_needed_per_episode)
    g.set(title=plots_name, ylabel="Number of steps", xlabel="Episode number")
    plt.show()

    g = sns.lineplot(data=mean_steps_needed_per_episode)
    g.set(title=plots_name, ylabel="Mean Number of Steps per Episode", xlabel="Episode number")
    plt.savefig(os.path.join(os.getcwd(), 'plots_countdown_version', 'mean_steps_needed_per_episode_' + plots_name + '.png'))
    plt.show()



