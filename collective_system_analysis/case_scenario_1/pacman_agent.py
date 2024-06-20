import mesa

from wall_agent import WallAgent


class PacmanAgent(mesa.Agent):
    def __init__(self, unique_id, model, should_penalize_terrain):
        super().__init__(unique_id, model)
        self.should_penalize_terrain = should_penalize_terrain
        self.movement_countdown = None

    def step(self):

        print("Pacman location: ", self.pos)

        if self.movement_countdown is None:
            starting_position = self.model.get_agents_of_type(PacmanAgent)[0].pos
            position_index = starting_position[0] + starting_position[1] * self.model.grid.width
            self.movement_countdown = self.model.get_terrain_penalty(position_index)

        if self.movement_countdown == 0:
            next_position = self.move()
            position_index = next_position[0] + next_position[1] * self.model.grid.width
            self.movement_countdown = self.model.get_terrain_penalty(position_index)
        else:
            self.movement_countdown -= 1

    def move(self):

        from ghost_agent import GhostAgent

        # ghosts_positions = []
        # walls_positions = []
        # for cell_content, (x, y) in self.model.grid.coord_iter():
        #     if not len(cell_content) == 0 and isinstance(cell_content[0], GhostAgent):
        #         ghosts_positions.append((x, y))
        #     elif not len(cell_content) == 0 and isinstance(cell_content[0], WallAgent):
        #         walls_positions.append((x, y))
        ghost_agents = self.model.get_agents_of_type(GhostAgent)
        wall_agents = self.model.get_agents_of_type(WallAgent)

        ghosts_positions = [(agent.pos[0], agent.pos[1]) for agent in ghost_agents]
        walls_positions = [(agent.pos[0], agent.pos[1]) for agent in wall_agents]

        possible_steps = []
        optimal_steps = []

        x, y = self.model.get_agents_of_type(PacmanAgent)[0].pos

        if x > 0 and (x - 1, y) not in walls_positions:
            possible_steps.append((x - 1, y))
        if x < self.model.grid.width - 1 and (x + 1, y) not in walls_positions:
            possible_steps.append((x + 1, y))
        if y > 0 and (x, y - 1) not in walls_positions:
            possible_steps.append((x, y - 1))
        if y < self.model.grid.height - 1 and (x, y + 1) not in walls_positions:
            possible_steps.append((x, y + 1))

        closest_ghost = None

        for ghost_position in ghosts_positions:
            if closest_ghost is None:
                closest_ghost = ghost_position
            elif abs(x - ghost_position[0]) + abs(y - ghost_position[1]) < abs(x - closest_ghost[0]) + abs(y - closest_ghost[1]):
                closest_ghost = ghost_position

        # move in the opposite direction of the closest ghost

        dx = x - closest_ghost[0]
        dy = y - closest_ghost[1]

        if abs(dx) > abs(dy):
            if dx > 0 and (x + 1, y) in possible_steps:
                optimal_steps.append((x + 1, y))
            elif dx < 0 and (x - 1, y) in possible_steps:
                optimal_steps.append((x - 1, y))
        else:
            if dy > 0 and (x, y + 1) in possible_steps:
                optimal_steps.append((x, y + 1))
            elif dy < 0 and (x, y - 1) in possible_steps:
                optimal_steps.append((x, y - 1))

        if not possible_steps:
            return x, y

        move = self.random.choice(optimal_steps) if len(optimal_steps) > 0 else self.random.choice(possible_steps)

        self.model.grid.move_agent(self, move)

        print("Pacman moved to ", move)

        return move







