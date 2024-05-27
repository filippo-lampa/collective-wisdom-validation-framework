'''
A customer agent randomly moves around the environment and hits hotels. When a customer agent hits a hotel, it calls
the hotel's collect_data method to collect data from the environment, passing its own id as parameter. The hotel agent
the retrieves one entry from the environment dataset relative to the customer agent's id and trains its regression model.
'''

import numpy as np
import random
import mesa
from hotel_agent import HotelAgent


class CustomerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        self.move()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        if self.model.grid.is_cell_empty(new_position):
            return
        else:
            for agent in self.model.grid.get_cell_list_contents([new_position]):
                if isinstance(agent, HotelAgent):
                    print("Customer agent {} hit hotel agent {}".format(self.unique_id, agent.unique_id))
                    agent.collect_data()
                    return
                else:
                    return


