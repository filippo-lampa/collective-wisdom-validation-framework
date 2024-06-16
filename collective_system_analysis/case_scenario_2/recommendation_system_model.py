'''
Multi-agent system for recommendation system model using the mesa library.

The model is a multi-agent system that simulates a recommendation system for a hotel. The agents are the hotels, and they
either collect new data after a guest checks in (after a random number of time steps) or when they exchange data with
nearby hotels (closer than a certain distance). The data consists of the guest's ID, the movie profile they chose, and the
rating they gave to the movie profile. The agents use this data to make personalized recommendations to future guests.
'''
import argparse
import os

import mesa
import torch

from river import optim, preprocessing

from torch import nn

from river import reco

from deep_river.regression import Regressor

import seaborn as sns

import numpy as np
import pandas as pd

from collective_system_analysis.case_scenario_2.model_architectures import RecommendationNN, NCF
from hotel_agent import HotelAgent
from customer_agent import CustomerAgent

from models.ml.online_regression_model import OnlineRegressionModel
from models.dl.online_regression_nn import OnlineRegressionNN

from collective_system_analysis.case_scenario_2.hotel_agent import DataSelectionLogic, MemoryManagementLogic

import matplotlib.pyplot as plt

import pickle


class RecommendationSystemModel(mesa.Model):

    def __init__(self, width, height, n_hotels, n_customers, load_existing_data, stratified_users, eval_steps, should_resume,
                 approach, args: argparse.Namespace):
        super().__init__()
        self.num_hotels = n_hotels
        self.num_customers = n_customers
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.load_existing_data = load_existing_data
        self.stratified_users = stratified_users
        self.eval_steps = eval_steps
        self.should_resume = should_resume
        self.approach = approach
        self.args = args

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(torch.device(device))

        agent_data_mapping, testing_data = self.prepare_starting_data()

        # Regression model. The protocol is supposed to work independently of the architecture.

        agent_positions_mapping = {
            0: (3, 1),
            1: (5, 3),
            2: (3, 5),
            3: (0, 9),
            4: (6, 5),
        }

        biased_mf_params = {
            'n_factors': 5,
            'bias_optimizer': optim.SGD(0.0025),
            'latent_optimizer': optim.SGD(0.005),
            'weight_initializer': optim.initializers.Zeros(),
            'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
            'l2_bias': 0.,
            'l2_latent': 0.
        }

        # Speed up embedding lookup by mapping user ids to a smaller range

        dataset = pd.concat([agent_data_mapping[i] for i in range(len(agent_data_mapping))])
        complete_dataset = pd.concat([dataset, testing_data])
        unique_user_ids = complete_dataset['user'].unique()
        del complete_dataset
        user_id_mapping = {i: idx for idx, i in enumerate(unique_user_ids)}

        for i in range(self.num_hotels):

            if self.approach == 'nn':
                '''
                if i % 2 == 0:
                    regression_model = Regressor(
                        module=NCF,
                        loss_fn=nn.L1Loss(),
                        optimizer_fn='adam',
                        lr=1e-4,
                        user_id_mapping=user_id_mapping,
                        device=device
                    )
                else:
                    regression_model = Regressor(
                            module=RecommendationNN,
                            loss_fn=nn.L1Loss(),
                            optimizer_fn='adam',
                            lr=1e-4,
                            user_id_mapping=user_id_mapping,
                            device=device
                    )
                '''
                regression_model = Regressor(
                    module=RecommendationNN,
                    loss_fn=nn.L1Loss(),
                    optimizer_fn='adam',
                    lr=1e-4,
                    user_id_mapping=user_id_mapping,
                    device=device
                )
                agent_online_regression_model = OnlineRegressionNN(regression_model, 'agent_{}_model'.format(i))

            elif self.approach == 'ml':

                regression_model = preprocessing.PredClipper(
                    regressor=reco.BiasedMF(**biased_mf_params),
                    y_min=1,
                    y_max=5
                )

                agent_online_regression_model = OnlineRegressionModel(regression_model, 'agent_{}_model'.format(i))

            agent_data = agent_data_mapping[i]

            agent_dataset = None

            if self.should_resume:

                assert os.path.exists('running_agents_models/agent_{}_model.pkl'.format(i)), "Model file not found"
                assert os.path.exists('running_agents_datasets/hotel_agent_dataset_{}.pkl'.format(i)), "Dataset file not found"

                if self.approach == 'ml':
                    with open('running_agents_models/agent_{}_model.pkl'.format(i), 'rb') as f:
                        agent_online_regression_model = pickle.load(f)
                else:
                    agent_online_regression_model = OnlineRegressionNN(regression_model, 'agent_{}_model'.format(i),
                                                                       True, i)

                with open('running_agents_datasets/hotel_agent_dataset_{}.pkl'.format(i), 'rb') as f:
                    agent_dataset = pickle.load(f)

            args_memory_logic_mapping = {
                'fifo': MemoryManagementLogic.FIFO,
                'entropy': MemoryManagementLogic.ENTROPY,
                'nearest_neighbors': MemoryManagementLogic.NEAREST_NEIGHBORS,
                'most_recently_sent': MemoryManagementLogic.MOST_RECENTLY_SENT,
                'most_uncertain': MemoryManagementLogic.MOST_UNCERTAIN
            }

            args_data_logic_mapping = {
                'ordering_untouched': DataSelectionLogic.ORDERING_UNTOUCHED,
                'ordering_loss': DataSelectionLogic.ORDERING_LOSS,
                'ordering_date': DataSelectionLogic.ORDERING_DATE,
                'random': DataSelectionLogic.RANDOM,
                'clustering': DataSelectionLogic.CLUSTERING,
                'loss_threshold': DataSelectionLogic.LOSS_THRESHOLD
            }

            # Initialize the online regression model with the starting data (a subset of the whole dataset), and the
            # hotel agent with all the data that will be fed to the model from the environment as the agent interacts
            # with it to simulate the recommendation system.

            # Note: the flow of data into and out of the memory should be faster than the ratio of data exchange with
            # the other agents to avoid overly redundant data

            a = HotelAgent(i, self, agent_online_regression_model, agent_data, testing_data,
                           args.nearby_agents_data_exchange_steps, args.vicinity_radius, args.memory_size,
                           self.eval_steps, args.bandwidth, args.maximum_time_steps_for_exchange,
                           args.testing_data_amount, args.starting_dataset_size, args.should_use_protocol,
                           args.should_keep_data_ordered, args_data_logic_mapping[args.data_selection_logic],
                           args.should_train_all_data, args.should_be_fixed_subset_exchange_percentage,
                           args.subset_exchange_percentage, args.loss_threshold,
                           args_memory_logic_mapping[args.memory_management_logic], agent_dataset,
                           args.should_send_first_entries)

            self.schedule.add(a)

            #x = self.random.randrange(self.grid.width)
            #y = self.random.randrange(self.grid.height)

            x, y = agent_positions_mapping[i]

            self.grid.place_agent(a, (x, y))

        for i in range(self.num_customers):
            c = CustomerAgent(i, self)
            self.schedule.add(c)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(c, (x, y))

        self.datacollector = mesa.DataCollector(
            agent_reporters={"MAE": "mae", "RMSE": "rmse"},
        )

    def prepare_starting_data(self):

        def prepare_dataset():
            df1 = pd.read_csv('../../models/data/case_scenario_2/netflix_prize_dataset/combined_data_1.txt'
                              , header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
            df2 = pd.read_csv('../../models/data/case_scenario_2/netflix_prize_dataset/combined_data_2.txt',
                              header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
            df3 = pd.read_csv('../../models/data/case_scenario_2/netflix_prize_dataset/combined_data_3.txt',
                              header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
            df4 = pd.read_csv('../../models/data/case_scenario_2/netflix_prize_dataset/combined_data_4.txt',
                              header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

            df1['Rating'] = df1['Rating'].astype(float)
            df2['Rating'] = df2['Rating'].astype(float)
            df3['Rating'] = df3['Rating'].astype(float)
            df4['Rating'] = df4['Rating'].astype(float)

            df = pd.concat([df1, df2, df3, df4])

            df.index = np.arange(0, len(df))

            df_nan = pd.DataFrame(pd.isnull(df.Rating))
            df_nan = df_nan[df_nan['Rating'] == True]
            df_nan = df_nan.reset_index()

            movie_np = []
            movie_id = 1

            for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
                # numpy approach
                temp = np.full((1, i - j - 1), movie_id)
                movie_np = np.append(movie_np, temp)
                movie_id += 1

            # Account for last record and corresponding length
            # numpy approach
            last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
            movie_np = np.append(movie_np, last_record)

            print('Movie numpy: {}'.format(movie_np))
            print('Length: {}'.format(len(movie_np)))

            # remove those Movie ID rows
            df = df[pd.notnull(df['Rating'])]

            df['Movie_Id'] = movie_np.astype(int)
            df['Cust_Id'] = df['Cust_Id'].astype(int)

            #shuffle the dataframe
            df = df.sample(frac=1).reset_index(drop=True)

            df = df.rename(columns={'Cust_Id': 'user', 'Movie_Id': 'item'})

            df.to_csv('dataset.csv', index=False)
            return df

        print("Loading dataset")

        if not os.path.exists('dataset.csv') or not self.load_existing_data:
            dataset = prepare_dataset()
        else:
            dataset = pd.read_csv('dataset.csv')

        print("Dataset loaded, with shape: ", dataset.shape)

        import pickle

        if (self.stratified_users and (not os.path.exists('agent_data_mapping_stratified.pkl') or not os.path.exists('testing_data_stratified.pkl'))) \
            or (not self.stratified_users and (not os.path.exists('agent_data_mapping.pkl') or not os.path.exists('testing_data.pkl'))) \
                or not self.load_existing_data:

            training_data = pd.DataFrame()
            testing_data = pd.DataFrame()

            if self.stratified_users:

                dataset = dataset.sort_values(by='user')
                unique_users = dataset['user'].unique()

                for index, user in enumerate(unique_users):
                    print("Concating data for user: ", index, " out of ", len(unique_users))
                    user_data = dataset[dataset['user'] == user]
                    training_data = pd.concat([training_data, user_data.iloc[:int(0.8 * len(user_data))]])
                    testing_data = pd.concat([testing_data, user_data.iloc[int(0.8 * len(user_data)):]])

            else:
                training_data = dataset.sample(frac=0.8, random_state=200)
                testing_data = dataset.drop(training_data.index)

            amount_of_data_per_agent = [len(training_data) // self.num_hotels for _ in range(self.num_hotels)]
            agent_data_mapping = {}
            for i in range(self.num_hotels):
                data = training_data.sample(amount_of_data_per_agent[i], random_state=200)
                agent_data_mapping[i] = data
                training_data = training_data.drop(data.index)

            if self.stratified_users:
                with open('agent_data_mapping_stratified.pkl', 'wb') as f:
                        pickle.dump(agent_data_mapping, f)
                with open('testing_data_stratified.pkl', 'wb') as f:
                    pickle.dump(testing_data, f)
            else:
                with open('agent_data_mapping.pkl', 'wb') as f:
                    pickle.dump(agent_data_mapping, f)
                with open('testing_data.pkl', 'wb') as f:
                    pickle.dump(testing_data, f)
        else:
            if self.stratified_users:
                with open('agent_data_mapping_stratified.pkl', 'rb') as f:
                    agent_data_mapping = pickle.load(f)
                with open('testing_data_stratified.pkl', 'rb') as f:
                    testing_data = pickle.load(f)
            else:
                with open('agent_data_mapping.pkl', 'rb') as f:
                    agent_data_mapping = pickle.load(f)
                with open('testing_data.pkl', 'rb') as f:
                    testing_data = pickle.load(f)

        print("Data prepared")

        return agent_data_mapping, testing_data

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def save_model(self, agents_stats):
        if not os.path.exists('running_agents_stats'):
            os.makedirs('running_agents_stats')

        with open('running_agents_stats/agents_stats.pkl', 'wb') as f:
            pickle.dump(agents_stats, f)

        if not os.path.exists('running_agents_models'):
            os.makedirs('running_agents_models')

        if not os.path.exists('running_agents_datasets'):
            os.makedirs('running_agents_datasets')

        for agent in self.schedule.agents:
            if isinstance(agent, HotelAgent):
                agent.save_model()


if __name__ == '__main__':

    # model parameters from args
    parser = argparse.ArgumentParser(description='Run the recommendation system model')
    parser.add_argument('--sim_steps', type=int, default=440001, help='Number of simulation steps')
    parser.add_argument('--eval_steps', type=int, default=10000, help='Number of simulation steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=20000, help='Number of simulation steps between saving the model')
    parser.add_argument('--plot_steps', type=int, default=20000, help='Number of simulation steps between plotting the model')
    parser.add_argument('--resume', type=bool, default=False, help='Whether to resume the simulation')
    parser.add_argument('--plots_name', type=str, default='', help='Name shown in the plots')
    parser.add_argument('--approach', type=str, default='ml', help='The type of model used for the recommendation system')
    parser.add_argument('--num_hotels', type=int, default=5, help='Number of hotels in the simulation')
    parser.add_argument('--num_customers', type=int, default=10, help='Number of customers in the simulation')
    parser.add_argument('--load_existing_data', type=bool, default=False, help='Whether to load existing preprocessed dataset')
    parser.add_argument('--stratified_users', type=bool, default=False, help='Whether to stratify the users in the dataset to ensure each one is both in training and testing set')
    parser.add_argument('--should_save_checkpoints', type=bool, default=False, help='Whether to save checkpoints of the model during computation')

    # hotel agent parameters from args
    parser.add_argument('--nearby_agents_data_exchange_steps', type=int, default=1, help='Number of steps between data exchange with nearby agents')
    parser.add_argument('--vicinity_radius', type=int, default=2, help='Radius of the vicinity for data exchange')
    parser.add_argument('--memory_size', type=int, default=1000, help='Size of the memory of the agent')
    parser.add_argument('--bandwidth', type=int, default=1000, help='Bandwidth of the agent')
    parser.add_argument('--maximum_time_steps_for_exchange', type=int, default=1000, help='Maximum number of time steps for data exchange')
    parser.add_argument('--testing_data_amount', type=int, default=1000, help='Amount of testing data')
    parser.add_argument('--starting_dataset_size', type=int, default=0, help='Size of the starting dataset')
    parser.add_argument('--should_use_protocol', type=bool, default=True, help='Whether to use the protocol for data exchange')
    parser.add_argument('--should_keep_data_ordered', type=bool, default=False, help='Whether to keep the data ordered')
    parser.add_argument('--data_selection_logic', type=str, default='ordering_untouched', help='Logic for selecting data: ordering_untouched, ordering_loss, ordering_date, random, clustering, loss_threshold')
    parser.add_argument('--should_train_all_data', type=bool, default=False, help='Whether to train all the data')
    parser.add_argument('--should_be_fixed_subset_exchange_percentage', type=bool, default=False, help='Whether the subset exchange percentage should be fixed')
    parser.add_argument('--subset_exchange_percentage', type=float, default=0.2, help='Percentage of the subset to exchange')
    parser.add_argument('--loss_threshold', type=int, default=3, help='Threshold for the loss')
    parser.add_argument('--memory_management_logic', type=str, default='fifo', help='Logics for managing the memory: fifo, entropy, nearest_neighbors, most_recently_sent, most_uncertain')
    parser.add_argument('--should_send_first_entries', type=bool, default=False, help='Whether to send the first entries')

    args = parser.parse_args()

    simulation_steps = args.sim_steps
    simulation_eval_steps = args.eval_steps
    simulation_save_steps = args.save_steps
    simulation_plot_steps = args.plot_steps
    should_resume = args.resume
    plots_name = args.plots_name
    approach = args.approach
    should_save_checkpoints = args.should_save_checkpoints

    model = RecommendationSystemModel(10, 10, 5, 10, True, False,
                                      simulation_eval_steps, should_resume, approach, args)

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

    resumed_number_of_steps = 0

    if should_resume:
        print("Resuming simulation")
        resumed_agents_data = pickle.load(open('running_agents_stats/agents_stats.pkl', 'rb'))
        resumed_number_of_steps = resumed_agents_data['MAE'].size
        #drop rows where mae is 0.0
        resumed_agents_data = resumed_agents_data[resumed_agents_data['MAE'] != 0.0]

    number_of_steps = simulation_steps if not should_resume else simulation_steps - resumed_number_of_steps

    if should_resume:
        model.schedule.steps = resumed_number_of_steps

        agents_data = model.datacollector.get_agent_vars_dataframe().dropna()
        g = sns.lineplot(data=resumed_agents_data, x="Step", y="MAE", hue="AgentID")
        g.set(title="{}: MAE over time - Time step ".format(plots_name) + str(resumed_number_of_steps), ylabel="MAE")
        plt.savefig(os.path.join(os.getcwd(), 'plots', 'MAE.png'))
        plt.show()
        g = sns.lineplot(data=resumed_agents_data, x="Step", y="RMSE", hue="AgentID")
        g.set(title="{}: RMSE data over time - Time step ".format(plots_name) + str(resumed_number_of_steps), ylabel="RMSE")
        plt.savefig(os.path.join(os.getcwd(), 'plots', 'RMSE.png'))
        plt.show()

    for i in range(number_of_steps):
        model.step()

        if should_save_checkpoints and i % simulation_save_steps == 0 and i != 0:
            model.save_model(model.datacollector.get_agent_vars_dataframe().dropna())

        if i % simulation_plot_steps == 0 and i != 0:

            model.datacollector.collect(model)

            #get data for all the agents and plot them in a single graph

            #get data for all the agents and plot them in a single graph where data are not nan
            agents_data = model.datacollector.get_agent_vars_dataframe().dropna()

            if should_resume:
                complete_agents_data = pd.concat([agents_data, resumed_agents_data])
                g = sns.lineplot(data=complete_agents_data, x="Step", y="MAE", hue="AgentID")
            else:
                g = sns.lineplot(data=agents_data, x="Step", y="MAE", hue="AgentID")
            g.set(title="{}: MAE over time - Time step ".format(plots_name) + str(i), ylabel="MAE")
            plt.savefig(os.path.join(os.getcwd(), 'plots', 'MAE.png'))
            plt.show()

            if should_resume:
                g = sns.lineplot(data=complete_agents_data, x="Step", y="RMSE", hue="AgentID")
            else:
                g = sns.lineplot(data=agents_data, x="Step", y="RMSE", hue="AgentID")
            g.set(title="{}: RMSE data over time - Time step ".format(plots_name) + str(i), ylabel="RMSE")
            plt.savefig(os.path.join(os.getcwd(), 'plots', 'RMSE.png'))
            plt.show()

    agents_data = model.datacollector.get_agent_vars_dataframe()

    if should_resume:
        complete_agents_data = pd.concat([agents_data, resumed_agents_data])
        g = sns.lineplot(data=complete_agents_data, x="Step", y="MAE", hue="AgentID")
    else:
        g = sns.lineplot(data=agents_data, x="Step", y="MAE", hue="AgentID")
    g.set(title="{}: MAE over time - Last step ".format(plots_name), ylabel="MAE")
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'MAE.png'))
    plt.show()

    if should_resume:
        g = sns.lineplot(data=complete_agents_data, x="Step", y="RMSE", hue="AgentID")
    else:
        g = sns.lineplot(data=agents_data, x="Step", y="RMSE", hue="AgentID")
    g.set(title="{}: RMSE data over time - Last step ".format(plots_name), ylabel="RMSE")
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'RMSE.png'))
    plt.show()





