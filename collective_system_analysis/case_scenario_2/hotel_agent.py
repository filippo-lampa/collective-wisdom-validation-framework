from enum import Enum

import mesa
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import torch

from scipy.stats import entropy

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import pickle

from models.dl.online_regression_nn import OnlineRegressionNN
from models.ml.online_regression_model import OnlineRegressionModel


class DataSelectionLogic(Enum):
    #EXPECTED RANKING:

    #ORDERING_UNTOUCHED_LAST-FIFO done
    #ORDERING_LOSS-MOST_UNCERTAIN done
    #ORDERING_LOSS-ENTROPY done / ORDERING_LOSS-NEAREST_NEIGHBORS done
    #ORDERING_LOSS-FIFO done / ORDERING_LOSS-MOST_RECENTLY_SENT (not necessary)
    #ORDERING_UNTOUCHED_LAST-ENTROPY done/ ORDERING_UNTOUCHED_LAST-NEAREST_NEIGHBORS done
    #ORDERING_UNTOUCHED_FIRST-ENTROPY / ORDERING_UNTOUCHED_FIRST-NEAREST_NEIGHBORS
    #ORDERING_RANDOM-ENTROPY / ORDERING_RANDOM-NEAREST_NEIGHBORS
    #ORDERING_RANDOM-FIFO done
    #ORDERING_RANDOM-MOST_RECENTLY_SENT
    #ORDERING_CLUSTERING-MOST_RECENTLY_SENT
    #ORDERING_CLUSTERING-FIFO
    #ORDERING_CLUSTERING-NEAREST_NEIGHBORS
    #ORDERING_CLUSTERING-ENTROPY
    #ORDERING_UNTOUCHED_LAST-MOST_RECENTLY_SENT
    #ORDERING_UNTOUCHED_FIRST-MOST_RECENTLY_SENT
    #ORDERING_UNTOUCHED_FIRST-FIFO to do


    # If we choose the first entries it is FIFO, the first data points to be collected are the first to be sent.
    # This means that as long as we use a fifo
    # memory management logic, the data points that are sent are the ones that are collected first, and if the number of
    # time steps for sharing data are lower than the amount of time steps after which we override data, we are sure that
    # all the data points are sent at least once, and are therefore spread across the network. This means that data are
    # surely spread until we fill the memory. After that, the data will only be sent by an agent if it first sends them,
    # and then receives the new ones from the other agent/s. However, since also the other agents will have a full memory,
    # one of the two agents will have to remove some data to make space for the new ones, and therefore some data will be
    # lost at each exchange without checking the amount of information that is lost.
    # On the other side, if we use the other memory management logics, we still don't have the guarantee that all the data
    # points are sent at least once, but we can have a better control on the amount of information that is lost at each
    # exchange, therefore the performance of the model comes down to the best memory management logic that we can use and
    # to the trade-off between the amount of information that is lost at each exchange and the amount of information that
    # is shared across the network. Fifo and most recently sent, in this case, should act the same way and be interchangable.
    # Entropy and nearest neighbors are independent from this data selection logic. Most uncertain is not compatible with
    # this data selection logic.
    # The size of the bandwidth at that point really comes down to the weight we want to give
    # to our own experience and to the experience of the other agents.
    # Naive approach.
    # If we select the last entries, we are sending the newest data points to the other agents, leaving outdated data
    # points that are probably outdated or already sent to the other agents. With a fifo memory management logic, this
    # ensures that newest data are sent and oldest data are removed, potentially maximizing information sharing and
    # minimizing the amount of information that is lost if we are sure that new data are more important than old data.
    # With most recently sent, we are removing the newest data points. If manage to send data before receving new data,
    # it is optimal since we shared ours and overrode them with the new ones. In the other case, i'm removing the
    # newest data points and sending the oldest ones which could already be in the system and outdated.
    ORDERING_UNTOUCHED = 0,
    # The data points are sent in order of increasing loss. This means that the data points that are sent are the ones
    # that our model is more confident about, and therefore, since we are building a collective knowledge, may be more
    # beneficial for the other agents. This approach holds the potential to provide better performances than the FIFO since
    # it aims to maximize the amount of information that is shared across the network, but it also could mean introducing
    # more noise in the other agents' models, since especially in the early stages the model could be less accurate,
    # gradually drifting towards the wrong data points. Using this approach with a FIFO memory management logic means
    # ensuring self-sabotage, since we are removing the data points that could be more beneficial for the other agents.
    # Same for most recently sent, since the data points that are sent are the ones with the lowest loss, and we are
    # them.
    # With this approach, we can use the most_uncertain memory management logic to remove the data points that are more
    # uncertain, and therefore more likely to be less beneficial for the other agents. Nearest neighbors is independent
    # from this data selection logic, same goes for entropy.
    ORDERING_LOSS = 1,
    # The data points are sent in order of increasing date. Equivalent to FIFO, but potentially more precise and useful
    # to restore the order of the data points in the other agents' datasets. Same considerations as FIFO. May be useful
    # to use when we are precisely interested in the order of the data points in the dataset, which is not the case in this
    # scenario.
    ORDERING_DATE = 2,
    # The data points are sent in random order. This means that the data points that are sent are randomly selected from
    # the dataset, trying to get a sample of the distribution of the data points. All the memory logics are independent
    # from this data selection logic, and its performance comes down to it.
    RANDOM = 3
    # The centroids of the clusters are sent to the other agents. With semantically meaningful data, this could be useful,
    # but in this case, since we are dealing with user ids, item ids, and ratings, it is not. System performances with this
    # selection logic independent from the memory management logic, except for the nearest neighbors memory management
    # since removing similar data points could reduce the quality of the centroids and therefore the quality of the data
    # points that are sent to the other agents. There is also a correlation with the entropy memory management logic, since
    # we are removing the data that are the most similar to the others, and therefore the most likely to be the centroids.
    CLUSTERING = 4
    # The data points with loss values above a certain threshold are sent. Same considerations as the loss data selection
    # logic, just gives slightly more control on the amount of information that is shared across the network, for cases
    # where we know the application field.
    LOSS_THRESHOLD = 5


class MemoryManagementLogic(Enum):
    FIFO = 1
    ENTROPY = 2
    NEAREST_NEIGHBORS = 3
    MOST_RECENTLY_SENT = 4
    MOST_UNCERTAIN = 5


class HotelAgent(mesa.Agent):
    def __init__(self, unique_id, model, regression_model, data, testing_data, nearby_agents_data_exchange_steps,
                 vicinity_radius, memory_size, model_eval_steps, bandwidth, maximum_time_steps_for_exchange,
                 testing_data_amount, starting_dataset_size, should_use_protocol, should_keep_data_ordered,
                 data_selection_logic, should_train_all_data, should_be_fixed_subset_exchange_percentage,
                 subset_exchange_percentage, loss_threshold, memory_management_logic, resumed_dataset,
                 should_send_first_entries):

        super().__init__(unique_id, model)

        self.environment_dataset = data
        self.data_selection_logic = data_selection_logic

        self.regression_model = regression_model

        if resumed_dataset is not None:
            self.dataset = resumed_dataset
        else:
            self.dataset = pd.DataFrame(columns=data.columns, dtype=np.float64)

            if data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
                self.dataset.insert(len(self.dataset.columns), 'loss', 0, True)
            elif data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
                self.dataset.insert(len(self.dataset.columns), 'date', 0, True)


            if starting_dataset_size > 0:
                self.dataset = pd.concat([self.dataset, self.environment_dataset.iloc[:starting_dataset_size]], ignore_index=True)
                self.regression_model.train_many(self.environment_dataset.iloc[:starting_dataset_size])

            if memory_management_logic.name == MemoryManagementLogic.MOST_RECENTLY_SENT.name:
                self.dataset.insert(len(self.dataset.columns), 'date_sent', 0, True)

        self.should_train_all_data = should_train_all_data
        self.testing_data_amount = testing_data_amount
        #self.testing_data = testing_data
        self.testing_data = testing_data.iloc[:testing_data_amount]
        self.nearby_agents_data_exchange_steps = nearby_agents_data_exchange_steps
        self.vicinity_radius = vicinity_radius
        self.memory_size = memory_size
        self.model_eval_steps = model_eval_steps
        self.bandwidth = bandwidth # number of datapoints an agent can send to another agent every time step
        self.maximum_time_steps_for_exchange = maximum_time_steps_for_exchange
        self.should_use_protocol = should_use_protocol
        self.should_keep_data_ordered = should_keep_data_ordered
        self.should_be_fixed_subset_exchange_percentage = should_be_fixed_subset_exchange_percentage
        self.subset_exchange_percentage = subset_exchange_percentage
        self.loss_threshold = loss_threshold
        self.memory_management_logic = memory_management_logic
        self.should_send_first_entries = should_send_first_entries
        self.regression_model_type = 'nn' if isinstance(self.regression_model, OnlineRegressionNN) else 'ml'
        self.mae = None
        self.rmse = None

    def train_whole_network(self):
        '''
        Train the network with all the data available.
        Intended to be used for online learning test purposes.
        '''
        self.environment_dataset.reset_index(drop=True, inplace=True)
        mae_list = []
        rmse_list = []

        for index, row in self.environment_dataset.iterrows():
            print("Training row number: ", index, " out of ", len(self.environment_dataset))
            self.regression_model.train(row)
            if index % 200000 == 0 and index != 0:
                #mae, rmse = self.regression_model.evaluate(
                #    self.testing_data.sample(self.testing_data_amount, random_state=42))
                mae, rmse = self.regression_model.evaluate(self.testing_data, self.regression_model_type)
                mae_list.append(mae)
                rmse_list.append(rmse)
                self.mae = mae
                self.rmse = rmse
                print(self.mae, self.rmse)

                g = sns.lineplot(data=mae_list)
                g.set(title="MAE over time - Time step {}".format(index), ylabel="MAE")
                plt.show()

                g = sns.lineplot(data=rmse_list)
                g.set(title="RMSE data over time - Time step {}".format(index), ylabel="RMSE")
                plt.show()

        #self.mae, self.rmse = self.regression_model.evaluate(
        #self.testing_data.sample(self.testing_data_amount, random_state=42))
        self.mae, self.rmse = self.regression_model.evaluate(self.testing_data, self.regression_model_type)
        print(self.mae, self.rmse)

        mae_list.append(self.mae)
        rmse_list.append(self.rmse)

        g = sns.lineplot(data=mae_list)
        g.set(title="Final MAE over time", ylabel="MAE")
        plt.show()

        g = sns.lineplot(data=rmse_list)
        g.set(title="Final RMSE over time", ylabel="RMSE")
        plt.show()

        return

    def step(self):

        if self.should_train_all_data:
            self.train_whole_network()
        else:
            print("Hotel {} step number {}".format(self.unique_id, self.model.schedule.steps), "Dataset size: ", len(self.dataset))
            if self.model.schedule.steps % self.model_eval_steps == 0 and self.model.schedule.steps != 0:
                print("Hotel {} evaluation:".format(self.unique_id))
                #mae, rmse = self.regression_model.evaluate(self.testing_data.sample(self.testing_data_amount, random_state=42))
                mae, rmse = self.regression_model.evaluate(self.testing_data, self.regression_model_type)
                print("Hotel {} MAE: {}".format(self.unique_id, mae))
                print("Hotel {} RMSE: {}".format(self.unique_id, rmse))
                self.mae = mae
                self.rmse = rmse
            if self.should_use_protocol:
                nearby_agents = self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.vicinity_radius, moore=True)
                print("Hotel {} nearby hotels: {}".format(self.unique_id, [agent.unique_id for agent in nearby_agents
                                                                           if isinstance(agent, HotelAgent)]))
                if self.model.schedule.steps % self.nearby_agents_data_exchange_steps == 0:
                    for agent in nearby_agents:
                        if isinstance(agent, HotelAgent):
                            self.exchange_data(agent)

    def collect_data(self):

        #get data from the environment (next data from the environment dataset)
        data = self.environment_dataset.iloc[0]
        if len(self.dataset) + 1 > self.memory_size:
            self.manage_memory(data)
        loss = self.regression_model.train(data)
        if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
            new_data = data.to_frame().T
            new_data['loss'] = loss
            self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)
            self.dataset.sort_values(by='loss', inplace=True)
        elif self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
            new_data = data.to_frame().T
            new_data['date'] = self.model.schedule.steps
            self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)
            self.dataset.sort_values(by='date', inplace=True)
        else:
            self.dataset = pd.concat([self.dataset, data.to_frame().T], ignore_index=True)
        self.environment_dataset = self.environment_dataset.iloc[1:]
        print("Hotel {} collected data".format(self.unique_id))

    def exchange_data(self, agent):
        if len(self.dataset) <= 1:
            return
        data_to_send = self.select_data_to_send()
        if len(data_to_send) == 0:
            return
        if self.memory_management_logic.name == MemoryManagementLogic.MOST_RECENTLY_SENT.name:
            data_to_send['date_sent'] = self.model.schedule.steps
        print("Hotel {} sent data to hotel {}".format(self.unique_id, agent.unique_id))
        agent.get_data_from_exchange(data_to_send)

    def select_data_to_send(self):
        '''
        Define here the logic for the agent to select data to send to another agent.
        '''
        def select_first_entries():
            if self.should_be_fixed_subset_exchange_percentage:
                return self.dataset.iloc[:round(len(self.dataset) * self.subset_exchange_percentage)]
            return self.dataset.iloc[:self.bandwidth if len(self.dataset) >= self.bandwidth else len(self.dataset)]

        def select_last_entries():
            if self.should_be_fixed_subset_exchange_percentage:
                return self.dataset.iloc[-round(len(self.dataset) * self.subset_exchange_percentage):]
            return self.dataset.iloc[-self.bandwidth if len(self.dataset) >= self.bandwidth else -len(self.dataset):]

        def select_random_entries():
            if self.should_be_fixed_subset_exchange_percentage:
                return self.dataset.sample(round(len(self.dataset) * self.subset_exchange_percentage))
            return self.dataset.sample(self.bandwidth) if len(self.dataset) >= self.bandwidth else self.dataset

        def select_clustered_entries():
            '''
            Applying clustering to group similar data points together and prioritize sending representatives from each
            cluster (e.g. centroids). This method is of no use in this context, as clustering user ids, item ids, and
            ratings is not meaningful.
            '''
            pass

        def select_loss_threshold_entries():
            '''
            Select data points with loss values above a certain threshold. Loss values range from 0 to 4. The amount
            of data point to be sent should not exceed the bandwidth.
            '''
            return self.dataset[self.dataset['loss'] > self.loss_threshold].sort_values(by='loss') \
                       .iloc[:self.bandwidth if len(self.dataset) >= self.bandwidth else len(self.dataset)]

        if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
            return select_first_entries() if self.should_send_first_entries else select_last_entries()
        if self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
            return select_last_entries() if self.should_send_first_entries else select_first_entries()
        if self.data_selection_logic.name == DataSelectionLogic.RANDOM.name:
            return select_random_entries()
        if self.data_selection_logic.name == DataSelectionLogic.CLUSTERING.name:
            return select_clustered_entries()
        if self.data_selection_logic.name == DataSelectionLogic.LOSS_THRESHOLD.name:
            return select_loss_threshold_entries()
        if self.data_selection_logic.name == DataSelectionLogic.ORDERING_UNTOUCHED.name:
            return select_first_entries() if self.should_send_first_entries else select_last_entries()

    def get_data_from_exchange(self, data):
        '''
        Define here the logic for the agent to receive data from another agent.
        '''

        #remove entries from data if they are already present in the dataset

        '''
        indexes_to_remove = []
        for index, row in data.iterrows():
            for index_dataset, row_dataset in self.dataset.iterrows():
                if row['user'] == row_dataset['user'] and row['item'] == row_dataset['item']:
                    indexes_to_remove.append(index)
                    break
        '''
        '''
        # Create a set of (user, item) tuples for fast lookup
        dataset_tuples = set(zip(self.dataset['user'], self.dataset['item']))

        # Find indexes to remove by checking if (user, item) tuple exists in the set
        indexes_to_remove = [index for index, row in data.iterrows() if (row['user'], row['item']) in dataset_tuples]

        data = data.drop(indexes_to_remove)
        '''

        data.reset_index(drop=True, inplace=True)
        merged = pd.merge(data, self.dataset, on=['user', 'item'], how='left', indicator='Exist')
        #merged = merged.drop('Rating', inplace=True, axis=1)
        merged['Exist'] = np.where(merged.Exist == 'both', True, False)

        data = merged[merged['Exist'] == False].drop(columns=['Exist']).rename(columns={'Rating_x': 'Rating'})

        if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
            data = data.filter(items=['user', 'item', 'Rating', 'loss'])
        elif self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
            data = data.filter(items=['user', 'item', 'Rating', 'date'])
        elif self.memory_management_logic.name == MemoryManagementLogic.MOST_RECENTLY_SENT.name:
            data = data.filter(items=['user', 'item', 'Rating', 'date_sent'])
        else:
            data = data.filter(items=['user', 'item', 'Rating'])

        if len(data) == 0:
            return

        if len(self.dataset) + len(data) > self.memory_size:
            self.manage_memory(data)

        # if the data is a DataFrame, then the agent received multiple data points
        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
                losses = self.regression_model.train_many(data)
                data['loss'] = losses
                self.dataset = pd.concat([self.dataset, data], ignore_index=True)
                self.dataset.sort_values(by='loss', inplace=True)
                print("Hotel {} received data".format(self.unique_id))
                return
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
                data['date'] = self.model.schedule.steps
                self.dataset = pd.concat([self.dataset, data], ignore_index=True)
                self.dataset.sort_values(by='date', inplace=True)
                self.regression_model.train_many(data)
                print("Hotel {} received data".format(self.unique_id))
                return
            self.dataset = pd.concat([self.dataset, data], ignore_index=True)
            self.regression_model.train_many(data)
            print("Hotel {} received data".format(self.unique_id))
            return
        # if the data is a Series, then the agent received a single data point
        else:
            data = data.to_frame().T
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
                data['loss'] = self.regression_model.train(data)
                self.dataset = pd.concat([self.dataset, data], ignore_index=True)
                self.dataset.sort_values(by='loss', inplace=True)
                print("Hotel {} received data".format(self.unique_id))
                return
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
                data['date'] = self.model.schedule.steps
                self.dataset = pd.concat([self.dataset, data], ignore_index=True)
                self.dataset.sort_values(by='date', inplace=True)
                self.regression_model.train(data)
                print("Hotel {} received data".format(self.unique_id))
                return
            self.dataset = pd.concat([self.dataset, data], ignore_index=True)
            self.regression_model.train(data)

        print("Hotel {} received data".format(self.unique_id))

    def manage_memory(self, new_data):
        '''
        Define here the logic for the agent to manage its memory when new data is collected.
        '''

        space_to_free = len(self.dataset) + len(new_data) - self.memory_size

        if self.memory_management_logic.name == MemoryManagementLogic.FIFO.name:
            self.dataset = self.dataset.iloc[space_to_free:]
            self.dataset.reset_index(drop=True, inplace=True)
            return

        if self.memory_management_logic.name == MemoryManagementLogic.ENTROPY.name:
            '''
            Remove the data points with the lowest entropy.
            '''
            def calculate_entropy(ratings):
                value, counts = np.unique(ratings, return_counts=True)
                return entropy(counts, base=2)

            self.dataset.insert(len(self.dataset.columns), 'entropy', 0, True)
            user_entropies = self.dataset.groupby('user')['Rating'].apply(calculate_entropy).reset_index()

            self.dataset = pd.merge(self.dataset, user_entropies, on='user')
            self.dataset.sort_values(by='entropy', inplace=True)
            self.dataset = self.dataset.rename(columns={'Rating_x': 'Rating'})

            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
                if 'date_sent' in self.dataset.columns:
                    self.dataset = self.dataset.filter(items=['user', 'item', 'Rating', 'loss', 'date_sent'])
                else:
                    self.dataset = self.dataset.filter(items=['user', 'item', 'Rating', 'loss'])
            elif self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
                if 'date_sent' in self.dataset.columns:
                    self.dataset = self.dataset.filter(items=['user', 'item', 'Rating', 'date', 'date_sent'])
                else:
                    self.dataset = self.dataset.filter(items=['user', 'item', 'Rating', 'date'])
            else:
                if 'date_sent' in self.dataset.columns:
                    self.dataset = self.dataset.filter(items=['user', 'item', 'Rating', 'date_sent'])
                else:
                    self.dataset = self.dataset.filter(items=['user', 'item', 'Rating'])

            self.dataset = self.dataset.iloc[space_to_free:]

            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
                self.dataset.sort_values(by='loss', inplace=True)
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
                self.dataset.sort_values(by='date', inplace=True)

            self.dataset.reset_index(drop=True, inplace=True)
            return

        if self.memory_management_logic.name == MemoryManagementLogic.NEAREST_NEIGHBORS.name:
            '''
            Remove the data points that are closer to others to ensure having a high diversity of data points.
            '''

            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(self.dataset[['user', 'item']])

            distances, indices = nbrs.kneighbors(self.dataset[['user', 'item']])
            distances = pd.DataFrame(distances, columns=['distance1', 'distance2'])
            indices = pd.DataFrame(indices, columns=['index1', 'index2'])

            distances = pd.concat([distances, indices], axis=1)

            distances.sort_values(by='distance2', inplace=True)

            self.dataset = self.dataset.drop(distances.head(space_to_free)['index2'])
            self.dataset.reset_index(drop=True, inplace=True)

            return

        if self.memory_management_logic.name == MemoryManagementLogic.MOST_RECENTLY_SENT.name:
            '''
            Remove the data points that have been sent recently to other agents as the model is trained on them and
            the information is already shared.
            '''
            self.dataset.sort_values(by='date_sent', inplace=True, ascending=False)
            self.dataset = self.dataset.iloc[space_to_free:]
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name:
                self.dataset.sort_values(by='loss', inplace=True)
            if self.data_selection_logic.name == DataSelectionLogic.ORDERING_DATE.name:
                self.dataset.sort_values(by='date', inplace=True)
            self.dataset.reset_index(drop=True, inplace=True)
            return

        if self.memory_management_logic.name == MemoryManagementLogic.MOST_UNCERTAIN.name:
            '''
            Remove the data points with the highest loss.
            '''
            assert self.data_selection_logic.name == DataSelectionLogic.ORDERING_LOSS.name, \
                "Memory management logic MOST_UNCERTAIN is only compatible with data selection logic ORDERING_LOSS"
            #remove the last space_to_free data points
            self.dataset = self.dataset.iloc[:-space_to_free]
            self.dataset.reset_index(drop=True, inplace=True)
            return

    def save_model(self):
        if isinstance(self.regression_model, OnlineRegressionModel):
            with open('running_agents_models/agent_{}_model.pkl'.format(self.unique_id), 'wb') as f:
                pickle.dump(self.regression_model.model, f)
        else:
            checkpoint = {'model': self.regression_model.model.module.state_dict(),
                          'optimizer': self.regression_model.model.optimizer.state_dict()}
            torch.save(checkpoint, 'running_agents_models/agent_{}_model.pt'.format(self.unique_id))

        with open('running_agents_datasets/hotel_agent_dataset_{}.pkl'.format(self.unique_id), 'wb') as f:
            pickle.dump(self.dataset, f)

        print("Hotel {} saved model".format(self.unique_id))




