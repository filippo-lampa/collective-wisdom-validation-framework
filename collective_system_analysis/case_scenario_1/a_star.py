import heapq

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')  # Distance from start node
        self.h = float('inf')  # Heuristic distance to end node
        self.f = float('inf')  # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


def heuristic(node, goal):
    # Manhattan distance heuristic
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def astar(grid, start, end, walls, environment, should_penalize_terrain):
    open_set = []
    heapq.heappush(open_set, (0, start))
    start.g = 0
    start.h = heuristic(start, end)
    start.f = start.g + start.h

    closed_set = set()
    came_from = {}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = []
            while current in came_from:
                path.append((current.x, current.y))
                current = came_from[current]
            path.append((start.x, start.y))
            path.reverse()
            return path

        closed_set.add(current)

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Movement in 4 directions (right, left, down, up)
        for dx, dy in neighbors:
            neighbor = Node(current.x + dx, current.y + dy)

            if neighbor.x < 0 or neighbor.x >= len(grid) or neighbor.y < 0 or neighbor.y >= len(grid[0]):
                continue  # Out of bounds

            if (neighbor.x, neighbor.y) in walls:
                continue  # This neighbor is a wall

            if neighbor in closed_set:
                continue  # Already evaluated

            tentative_g = current.g + environment.get_terrain_penalty(neighbor.x * len(grid[0]) + neighbor.y) + 1 \
                if should_penalize_terrain else current.g + 1

            if tentative_g < neighbor.g:
                came_from[neighbor] = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, end)
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_set, (neighbor.f, neighbor))

    return None  # No path found