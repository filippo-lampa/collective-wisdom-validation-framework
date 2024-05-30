'''
Multi-agent system for recommendation system model using the mesa library.

The model is a multi-agent system that simulates a recommendation system for a hotel. The agents are the hotels, and they
either collect new data after a guest checks in (after a random number of time steps) or when they exchange data with
nearby hotels (closer than a certain distance). The data consists of the guest's ID, the movie profile they chose, and the
rating they gave to the movie profile. The agents use this data to make personalized recommendations to future guests.
'''
import os

import mesa
import torch

from river import optim

from torch import nn

from river import reco

from deep_river.regression import Regressor
import seaborn as sns

import numpy as np
import pandas as pd

from hotel_agent import HotelAgent
from customer_agent import CustomerAgent

from models.ml.online_regression_model import OnlineRegressionModel
from models.dl.online_regression_nn import OnlineRegressionNN

import matplotlib.pyplot as plt

from collective_system_analysis.case_scenario_2.hotel_agent import DataSelectionLogic, MemoryManagementLogic


class RecommendationSystemModel(mesa.Model):

    def __init__(self, width, height, n_hotels, n_customers, load_existing_data=False):
        super().__init__()
        self.num_hotels = n_hotels
        self.num_customers = n_customers
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.load_existing_data = load_existing_data

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(torch.device(device))

        agent_data_mapping, testing_data = self.prepare_starting_data()

        # Regression model. The protocol is supposed to work independently of the architecture.

        biased_mf_params = {
            'n_factors': 5,
            'bias_optimizer': optim.SGD(0.025),
            'latent_optimizer': optim.SGD(0.05),
            'weight_initializer': optim.initializers.Zeros(),
            'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
            'l2_bias': 0.,
            'l2_latent': 0.
        }

        agent_positions_mapping = {
            0: (3, 1),
            1: (5, 3),
            2: (3, 5),
            3: (0, 9),
            4: (6, 5),
        }

        # Speed up embedding lookup by mapping user ids to a smaller range

        dataset = pd.concat([agent_data_mapping[i] for i in range(len(agent_data_mapping))])
        complete_dataset = pd.concat([dataset, testing_data])
        unique_user_ids = complete_dataset['user'].unique()
        del complete_dataset
        user_id_mapping = {i: idx for idx, i in enumerate(unique_user_ids)}

        for i in range(self.num_hotels):
            class MyModule(nn.Module):
                '''
                Recommendation neural network model based on collaborative filtering.
                '''

                def intermediate_mapping(self, user_ids):
                    '''
                    users ids range from 1 to 2649429, with gaps. However we have only 480189 users.
                    To avoid having larger embeddings than necessary, we will map each user id to a smaller range.
                    '''
                    #user_ids could be one or more user ids
                    if isinstance(user_ids, int):
                        return torch.tensor(user_id_mapping[user_ids])
                    return torch.tensor([user_id_mapping[i.item()] for i in user_ids])


                def __init__(self, n_features):
                    super().__init__()
                    self.user_embedding = nn.Embedding(480190, 50)
                    self.item_embedding = nn.Embedding(17771, 50)
                    self.fc1 = nn.Linear(50, 1)

                def forward(self, X, **kwargs):

                    if len(X) > 1:
                        users = X[:, 0].to(torch.int64)
                        items = X[:, 1].to(torch.int64)
                        user_embedding = self.user_embedding(self.intermediate_mapping(users))
                        item_embedding = self.item_embedding(items)
                        X = torch.mul(user_embedding, item_embedding)
                        X = self.fc1(X)
                        return X

                    user_embedding = self.user_embedding(self.intermediate_mapping(int(X[0][0].item())))
                    item_embedding = self.item_embedding(X[0][1].to(torch.int64))
                    X = torch.mul(user_embedding, item_embedding)
                    X = self.fc1(X.unsqueeze(0))
                    return X

            regression_model = Regressor(
                module=MyModule,
                loss_fn=nn.L1Loss(),
                optimizer_fn='adam',
                lr=1e-3,
                device=device
            )

            '''
            regression_model = preprocessing.PredClipper(
                regressor=reco.BiasedMF(**biased_mf_params),
                y_min=1,
                y_max=5
            )
            '''

            agent_data = agent_data_mapping[i]

            # Initialize the online regression model with the starting data (a subset of the whole dataset), and the
            # hotel agent with all the data that will be fed to the model from the environment as the agent interacts
            # with it to simulate the recommendation system.

            agent_online_regression_model = OnlineRegressionNN(regression_model, 'agent_{}_model'.format(i))

            a = HotelAgent(i, self, agent_online_regression_model, agent_data, testing_data,
                           10, 2, 1000, 20000,
                           20, 100000, 10000, 0,
                           True, False, DataSelectionLogic.ORDERING_LOSS,
                           True, False, 0.2,
                           3, MemoryManagementLogic.FIFO)

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

        '''
        self.datacollector = mesa.DataCollector(
            model_reporters={}, agent_reporters={}
        )
        '''
        self.datacollector = mesa.DataCollector(
            agent_reporters={"MAE": "mae", "RMSE": "rmse", "Accuracy": "accuracy"},
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

        if not os.path.exists('dataset.csv') or not self.load_existing_data:
            dataset = prepare_dataset()
        else:
            dataset = pd.read_csv('dataset.csv')

        print("Dataset loaded, with shape: ", dataset.shape)

        import pickle

        if not os.path.exists('agent_data_mapping.pkl') or not os.path.exists('testing_data.pkl') or not self.load_existing_data:
            dataset = dataset.sort_values(by='user')
            unique_users = dataset['user'].unique()
            training_data = pd.DataFrame()
            testing_data = pd.DataFrame()
            for index, user in enumerate(unique_users):
                print("Concating data for user: ", index, " out of ", len(unique_users))
                user_data = dataset[dataset['user'] == user]
                training_data = pd.concat([training_data, user_data.iloc[:int(0.8 * len(user_data))]])
                testing_data = pd.concat([testing_data, user_data.iloc[int(0.8 * len(user_data)):]])
            #training_data = dataset.sample(frac=0.8, random_state=200)
            #testing_data = dataset.drop(training_data.index)
            #amount_of_data_per_agent = [np.random.randint(0, len(training_data)) for i in range(self.num_agents)]
            amount_of_data_per_agent = [len(training_data) // self.num_hotels for _ in range(self.num_hotels)]
            agent_data_mapping = {}
            for i in range(self.num_hotels):
                data = training_data.sample(amount_of_data_per_agent[i], random_state=200)
                agent_data_mapping[i] = data
                training_data = training_data.drop(data.index)

            with open('agent_data_mapping.pkl', 'wb') as f:
                    pickle.dump(agent_data_mapping, f)
            with open('testing_data.pkl', 'wb') as f:
                pickle.dump(testing_data, f)
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


if __name__ == '__main__':

    model = RecommendationSystemModel(10, 10, 5, 2, True)

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell_content, (x, y) in model.grid.coord_iter():
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    g = sns.heatmap(agent_counts.T, cmap="viridis", annot=True, cbar=False, square=True)
    g.figure.set_size_inches(4, 4)
    g.set(title="Number of agents on each cell of the grid")
    plt.show()

    for i in range(10000000):
        model.step()

        if i % 20000 == 0 and i != 0:

            model.datacollector.collect(model)

            #get data for all the agents and plot them in a single graph

            #get data for all the agents and plot them in a single graph where data are not nan
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

    #model_data = model.datacollector.get_model_vars_dataframe()

    #g = sns.lineplot(data=model_data)
    #g.set(title="Model data over Time", ylabel="Model Data")

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

