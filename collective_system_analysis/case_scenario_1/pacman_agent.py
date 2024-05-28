import mesa

from wall_agent import WallAgent


class PacmanAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        self.move()

    def move(self):

        from ghost_agent import GhostAgent

        ghosts_positions = []
        walls_positions = []
        for cell_content, (x, y) in self.model.grid.coord_iter():
            if isinstance(cell_content, GhostAgent):
                ghosts_positions.append((x, y))
            elif isinstance(cell_content, WallAgent):
                walls_positions.append((x, y))

        possible_steps = []
        x, y = self.model.get_agents_of_type(PacmanAgent)[0].pos

        closest_ghost = None

        for ghost_position in ghosts_positions:
            if closest_ghost is None:
                closest_ghost = ghost_position
            elif abs(x - ghost_position[0]) + abs(y - ghost_position[1]) < abs(x - closest_ghost[0]) + abs(y - closest_ghost[1]):
                closest_ghost = ghost_position

        if closest_ghost is not None:
            if x < closest_ghost[0]:
                possible_steps.append((x + 1, y))
            elif x > closest_ghost[0]:
                possible_steps.append((x - 1, y))
            if y < closest_ghost[1]:
                possible_steps.append((x, y + 1))
            elif y > closest_ghost[1]:
                possible_steps.append((x, y - 1))

        if len(possible_steps) == 0:
            if x > 0:
                possible_steps.append((x - 1, y))
            if x < self.model.grid.width - 1:
                possible_steps.append((x + 1, y))
            if y > 0:
                possible_steps.append((x, y - 1))
            if y < self.model.grid.height - 1:
                possible_steps.append((x, y + 1))

        possible_steps = [step for step in possible_steps if step not in walls_positions]

        move = self.random.choice(possible_steps)

        self.model.grid.move_agent(self, move)

        print("Pacman moved to ", move)

        return move







