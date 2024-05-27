from enum import Enum

import mesa
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

class DataSelectionLogic(Enum):
    ORDERING_LOSS = 1,
    ORDERING_DATE = 2,
    ORDERING_RANDOM = 3
    CLUSTERING = 4


class HotelAgent(mesa.Agent):
    def __init__(self, unique_id, model, regression_model, data, testing_data, nearby_agents_data_exchange_steps,
                 vicinity_radius, memory_size, model_eval_steps, bandwidth, maximum_time_steps_for_exchange,
                 testing_data_amount, starting_dataset_size, should_use_protocol, should_keep_data_ordered,
                 data_selection_logic, should_train_all_data):
        super().__init__(unique_id, model)

        self.environment_dataset = data
        self.data_selection_logic = data_selection_logic

        if data_selection_logic == DataSelectionLogic.ORDERING_LOSS:
            self.dataset = pd.DataFrame(columns=['user', 'Rating', 'item', 'loss'])
            for index, row in data.iterrows():
                row['loss'] = regression_model.train(row)
                self.dataset = pd.concat([self.dataset, row.to_frame().T], ignore_index=True)
        else:
            self.dataset = pd.DataFrame(columns=data.columns)
            self.dataset = pd.concat([self.dataset, self.environment_dataset.iloc[:starting_dataset_size]], ignore_index=True)

        self.should_train_all_data = should_train_all_data
        self.testing_data = testing_data
        self.regression_model = regression_model
        self.nearby_agents_data_exchange_steps = nearby_agents_data_exchange_steps
        self.vicinity_radius = vicinity_radius
        self.memory_size = memory_size
        self.model_eval_steps = model_eval_steps
        self.bandwidth = bandwidth # number of datapoints an agent can send to another agent every time step
        self.maximum_time_steps_for_exchange = maximum_time_steps_for_exchange
        self.testing_data_amount = testing_data_amount
        self.should_use_protocol = should_use_protocol
        self.should_keep_data_ordered = should_keep_data_ordered
        self.mae = 0
        self.rmse = 0
        self.accuracy = 0

    def train_whole_network(self):
        '''
        Train the network with all the data available.
        Intended to be used for online learning test purposes.
        '''
        self.environment_dataset.reset_index(drop=True, inplace=True)
        mae_list = []
        rmse_list = []
        accuracy_list = []

        for index, row in self.environment_dataset.iterrows():
            print("Training row number: ", index, " out of ", len(self.environment_dataset))
            self.regression_model.train(row)
            if index % 1000 == 0 and index != 0:
                mae, rmse, self.accuracy = self.regression_model.evaluate(
                    self.testing_data.sample(self.testing_data_amount, random_state=42))
                mae_list.append(mae.get())
                rmse_list.append(rmse.get())
                accuracy_list.append(self.accuracy)
                self.mae = mae.get()
                self.rmse = rmse.get()
                print(self.mae, self.rmse, self.accuracy)

                g = sns.lineplot(data=mae_list)
                g.set(title="MAE over time - Time step ", ylabel="MAE")
                plt.show()

                g = sns.lineplot(data=rmse_list)
                g.set(title="RMSE data over time - Time step ", ylabel="RMSE")
                plt.show()

                g = sns.lineplot(data=accuracy_list)
                g.set(title="Accuracy data over time - Time step ", ylabel="Accuracy")
                plt.show()

        self.mae, self.rmse, self.accuracy = self.regression_model.evaluate(
        self.testing_data.sample(self.testing_data_amount, random_state=42))
        print(self.mae, self.rmse, self.accuracy)

        mae_list.append(self.mae)
        rmse_list.append(self.rmse)
        accuracy_list.append(self.accuracy)

        g = sns.lineplot(data=mae_list)
        g.set(title="MAE over time - Time step ", ylabel="MAE")
        plt.show()

        g = sns.lineplot(data=rmse_list)
        g.set(title="RMSE data over time - Time step ", ylabel="RMSE")
        plt.show()

        return

    def step(self):

        if self.should_train_all_data:
            self.train_whole_network()
        else:
            print("Hotel {} step number {}".format(self.unique_id, self.model.schedule.steps), "Dataset size: ", len(self.dataset))
            if self.model.schedule.steps % self.model_eval_steps == 0 and self.model.schedule.steps != 0:
                print("Hotel {} evaluation:".format(self.unique_id))
                mae, rmse, accuracy = self.regression_model.evaluate(self.testing_data.sample(self.testing_data_amount, random_state=42))
                print("Hotel {} MAE: {}".format(self.unique_id, mae.get()))
                print("Hotel {} RMSE: {}".format(self.unique_id, rmse.get()))
                print("Hotel {} Accuracy: {}".format(self.unique_id, accuracy))
                self.mae = mae.get()
                self.rmse = rmse.get()
                self.accuracy = accuracy
            if self.should_use_protocol:
                nearby_agents = self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.vicinity_radius, moore=True)
                print("Hotel {} nearby agents: {}".format(self.unique_id, [agent.unique_id for agent in nearby_agents]))
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
        if self.data_selection_logic == DataSelectionLogic.ORDERING_LOSS:
            new_data = data.to_frame().T
            new_data['loss'] = loss
            self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)
            self.dataset.sort_values(by='loss', inplace=True)
        else:
            self.dataset = pd.concat([self.dataset, data.to_frame().T], ignore_index=True)
        self.environment_dataset = self.environment_dataset.iloc[1:]
        print("Hotel {} collected data".format(self.unique_id))

    def exchange_data(self, agent):
        if len(self.dataset) <= 1:
            return
        data_to_send = self.select_data_to_send()
        print("Hotel {} sent data to hotel {}".format(self.unique_id, agent.unique_id))
        agent.get_data_from_exchange(data_to_send)

    def select_data_to_send(self):
        '''
        Define here the logic for the agent to select data to send to another agent.
        '''
        def select_subset_size_fixed():
            return len(self.dataset) * 0.2

        def select_subset_size_maximize_bandwidth():
            return self.bandwidth if len(self.dataset) >= self.bandwidth else len(self.dataset)

        return self.dataset.iloc[:select_subset_size_maximize_bandwidth()]

    def get_data_from_exchange(self, data):
        '''
        Define here the logic for the agent to receive data from another agent.
        '''
        if len(self.dataset) + len(data) > self.memory_size:
            self.manage_memory(data)
        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)
            if self.data_selection_logic == DataSelectionLogic.ORDERING_LOSS:
                for index, row in data.iterrows():
                    row['loss'] = self.regression_model.train(row)
                self.dataset = pd.concat([self.dataset, data], ignore_index=True)
                self.dataset.sort_values(by='loss', inplace=True)
                return
        else:
            data = data.to_frame().T
            if self.data_selection_logic == DataSelectionLogic.ORDERING_LOSS:
                data['loss'] = self.regression_model.train(data)
                self.dataset = pd.concat([self.dataset, data], ignore_index=True)
                self.dataset.sort_values(by='loss', inplace=True)
                return
        self.dataset = pd.concat([self.dataset, data], ignore_index=True)
        for index, row in data.iterrows():
            self.regression_model.train(row)

        print("Hotel {} received data".format(self.unique_id))

    def manage_memory(self, new_data):
        '''
        Define here the logic for the agent to manage its memory when new data is collected.
        '''
        space_to_free = len(self.dataset) + len(new_data) - self.memory_size
        self.dataset = self.dataset.iloc[space_to_free:]
        self.dataset.reset_index(drop=True, inplace=True)
