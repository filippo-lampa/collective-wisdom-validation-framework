'''
Multi-agent system for recommendation system model using the mesa library.

The model is a multi-agent system that simulates a recommendation system for a hotel. The agents are the hotels, and they
either collect new data after a guest checks in (after a random number of time steps) or when they exchange data with
nearby hotels (closer than a certain distance). The data consists of the guest's ID, the movie profile they chose, and the
rating they gave to the movie profile. The agents use this data to make personalized recommendations to future guests.
'''
import os

import mesa

from river import preprocessing
from river import optim
from river import reco

import seaborn as sns

import numpy as np
import pandas as pd

from hotel_agent import HotelAgent

from models.ml.online_regression_model import OnlineRegressionModel
from models.dl.online_regression_nn import OnlineRegressionNN

class RecommendationSystemModel(mesa.Model):

    def __init__(self, width, height, n_agents):
        super().__init__()
        self.num_agents = n_agents
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.data = pd.DataFrame(columns=['guest_id', 'movie_profile', 'rating'])
        self.data = self.data.astype({'guest_id': 'int32', 'movie_profile': 'int32', 'rating': 'int32'})

        agent_data_mapping, testing_data = self.prepare_starting_data()

        # Create a regression model. The protocol is supposed to work independently of the architecture.

        '''
        nn online regression model example 

            model = (
                    pp.StandardScaler() |
                    nn.MLPRegressor(
                        hidden_dims=(5,),
                        activations=(
                            nn.activations.ReLU,
                            nn.activations.ReLU,
                            nn.activations.Identity
                        ),
                        optimizer=optim.SGD(1e-3),
                        seed=42
                    )
            )

        standard online regression model example

            funk_mf_params = {
                'n_factors': 10,
                'optimizer': optim.SGD(0.05),
                'l2': 0.1,
                'initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73)
            }

            model = preprocessing.PredClipper(
                regressor=reco.FunkMF(**funk_mf_params),
                y_min=1,
                y_max=5
            )  
        '''

        funk_mf_params = {
            'n_factors': 10,
            'optimizer': optim.SGD(0.05),
            'l2': 0.1,
            'initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73)
        }

        regression_model = preprocessing.PredClipper(
            regressor=reco.FunkMF(**funk_mf_params),
            y_min=1,
            y_max=5
        )

        for i in range(self.num_agents):
            agent_data = agent_data_mapping[i]
            # Initialize the online regression model with the starting data (a subset of the whole dataset), and the
            # hotel agent with all the data that will be fed to the model from the environment as the agent interacts
            # with it to simulate the recommendation system.

            agent_online_regression_model = OnlineRegressionModel(regression_model, 'agent_{}_model'.format(i))
            a = HotelAgent(i, self, agent_online_regression_model, agent_data, testing_data, 0.5,
                           0.5, 10, 1000, 300, 100)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = mesa.DataCollector(
            model_reporters={}, agent_reporters={}
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

        if not os.path.exists('dataset.csv'):
            dataset = prepare_dataset()
        else:
            dataset = pd.read_csv('dataset.csv')

        # split the data into training and testing
        training_data = dataset.sample(frac=0.8)
        testing_data = dataset.drop(training_data.index)
        amount_of_data_per_agent = [np.random.randint(10000, len(training_data)) for _ in range(self.num_agents)]
        agent_data_mapping = {}
        for i in range(self.num_agents):
            data = training_data.sample(amount_of_data_per_agent[i])
            agent_data_mapping[i] = data
            training_data = training_data.drop(data.index)

        return agent_data_mapping, testing_data

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':
    model = RecommendationSystemModel(10, 10, 10)
    model.prepare_starting_data()
    for i in range(100):
        model.step()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell_content, (x, y) in model.grid.coord_iter():
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    # Plot using seaborn, with a size of 5x5
    g = sns.heatmap(agent_counts, cmap="viridis", annot=True, cbar=False, square=True)
    g.figure.set_size_inches(4, 4)
    g.set(title="Number of agents on each cell of the grid")

    model_data = model.datacollector.get_model_vars_dataframe()

    g = sns.lineplot(data=model_data)
    g.set(title="Model data over Time", ylabel="Model Data")

    agent_data = model.datacollector.get_agent_vars_dataframe()

    g = sns.lineplot(data=agent_data, x="Step", y="Data")
    g.set(title="Agent data over Time", ylabel="Agent Data")






