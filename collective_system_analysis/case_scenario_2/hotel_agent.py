import mesa
import pandas as pd
import numpy as np


class HotelAgent(mesa.Agent):
    def __init__(self, unique_id, model, regression_model, data, testing_data, environment_data_collection_probability=0.5,
                  nearby_agents_data_exchange_steps=10, vicinity_radius=10, memory_size=1000, model_eval_steps=50, bandwidth=100,
                 maximum_time_steps_for_exchange=1000):
        super().__init__(unique_id, model)
        self.environment_dataset = data
        self.dataset = []
        self.testing_data = testing_data
        self.regression_model = regression_model
        self.environment_data_collection_probability = environment_data_collection_probability
        self.nearby_agents_data_exchange_steps = nearby_agents_data_exchange_steps
        self.vicinity_radius = vicinity_radius
        self.memory_size = memory_size
        self.model_eval_steps = model_eval_steps
        self.bandwidth = bandwidth # number of datapoints an agent can send to another agent every time step
        self.maximum_time_steps_for_exchange = maximum_time_steps_for_exchange


    def step(self):
        if self.model.schedule.steps % self.model_eval_steps == 0:
            print("Agent {} evaluation:".format(self.unique_id))
            self.regression_model.evaluate(self.testing_data)
        should_collect_data = np.random.rand() < self.environment_data_collection_probability
        if should_collect_data:
            self.collect_data()
        nearby_agents = self.model.space.get_neighbors(self.pos, include_center=False, radius=self.vicinity_radius)
        if self.model.schedule.steps % self.nearby_agents_data_exchange_steps == 0:
            for agent in nearby_agents:
                self.exchange_data(agent)

    def collect_data(self):
        #get data from the environment (next data from the environment dataset
        new_data = self.environment_dataset.iloc[0]
        if len(self.dataset) + 1 > self.memory_size:
            self.manage_memory(new_data)
        self.dataset = self.dataset.append(new_data, ignore_index=True)
        self.environment_dataset = self.environment_dataset.iloc[1:]
        self.regression_model.train(new_data)

    def exchange_data(self, agent):
        data_to_send = self.select_data_to_send()
        agent.get_data_from_exchange(data_to_send)

    def select_data_to_send(self):
        '''
        Define here the logic for the agent to select data to send to another agent.
        '''

        def select_subset_size():
            '''
            Define here the logic for the agent to select the size of the subset to send to another agent.
            '''
            return len(self.dataset) // 2
        return self.dataset[:select_subset_size()]

    def get_data_from_exchange(self, data):
        '''
        Define here the logic for the agent to receive data from another agent.
        '''
        if len(self.dataset) + len(data) > self.memory_size:
            self.manage_memory(data)
        self.dataset = self.dataset.append(data, ignore_index=True)
        self.regression_model.train(data)

    def manage_memory(self, new_data):
        '''
        Define here the logic for the agent to manage its memory when new data is collected.
        '''
        # For example, remove the oldest data to free enough space for the new data
        space_to_free = len(self.dataset) + len(new_data) - self.memory_size
        self.dataset = self.dataset.iloc[space_to_free:]
        self.dataset = self.dataset.append(new_data, ignore_index=True)
