import numpy as np
from queue import PriorityQueue

def Shannon_entropy(proba_dist: np.array, axis: int=None) -> float | np.ndarray:
    # Compute the Shannon Entropy 
    tab = proba_dist * np.log2(proba_dist)
    tab[np.isnan(tab)] = 0
    return - np.sum(tab, axis=axis, where=(proba_dist.any() != 0))

def Manhattan_dist(position, goal):
    # Calculate the Manhattan distance between two positions
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def get_neighbors(position, grid):
    # Get valid neighboring positions in the grid
    rows, cols = grid.shape
    i, j = position
    neighbors = []

    # left, right, move up, move dow
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)] 

    for action in actions:
        new_i = i + action[0]
        new_j = j + action[1]
        if 0 <= new_i < rows and 0 <= new_j < cols and grid[new_i, new_j] != 1:
            neighbors.append((new_i, new_j))

    return neighbors

def A_star_algorithm(start: tuple, goal: tuple, grid: np.ndarray) -> list | None:

    # Grid with 0 if empty cell 1 if object (i.e. obstacle)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    
    parent = {}
    g_scores = np.inf * np.ones((rows, cols))
    g_scores[start] = 0
    f_scores = np.inf * np.ones((rows, cols))
    f_scores[start] = Manhattan_dist(start, goal)

    queue = PriorityQueue()
    queue.put((f_scores[start], start))

    while not queue.empty():
        _, current = queue.get()

        if current == goal:
            # Reconstruct the sequence of actions from goal to start
            successive_pos = []
            while current in parent:
                previous = parent[current]
                successive_pos.append((previous, current))
                current = previous
            return successive_pos[::-1]

        visited[current] = True

        for neighbor in get_neighbors(current, grid):
            if visited[neighbor]:
                continue

            g_score = g_scores[current] + 1  # Cost of moving to the neighbor is 1
            if g_score < g_scores[neighbor]:
                parent[neighbor] = current
                g_scores[neighbor] = g_score
                f_scores[neighbor] = g_score + Manhattan_dist(neighbor, goal)
                queue.put((f_scores[neighbor], neighbor))

    # If the goal is not reachable
    return None