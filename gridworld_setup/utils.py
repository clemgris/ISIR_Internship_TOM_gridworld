import numpy as np
import matplotlib.pyplot as plt

from queue import PriorityQueue
import heapq

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

def Dijkstra(grid: np.ndarray, g_x: int, g_y: int) -> np.ndarray:
    rows, cols = grid.shape

    distance = np.full_like(grid, np.inf)
    distance[g_x, g_y] = 0

    heap = [(0, g_x, g_y)]

    while heap:
        dist, x, y = heapq.heappop(heap)

        if dist > distance[x, y]:
            continue

        # Check neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != 1:
                cost = dist + 1  # Assuming each cell has a cost of 1
                if cost < distance[nx, ny]:
                    distance[nx, ny] = cost
                    heapq.heappush(heap, (cost, nx, ny))

    return distance

##
# Visualization
##

def plot_grid(start, num, size, alpha=0.5):
    idx = np.linspace(start, size, num)
    for x in idx:
        plt.plot([x, x], [start, size], alpha=alpha, c='gray')
        plt.plot([start, size], [x, x], alpha=alpha, c='gray')

def plot_agent(pos: tuple, dir: int) -> None:
    if dir == 0:
        marker = ">"
    elif dir == 1:
        marker = "v"
    elif dir == 2:
        marker = "<"
    elif dir == 3:
        marker = "^"
    plt.scatter(pos[0], pos[1], marker=marker, c='r', s=120)

def plot_error_episode_length(colors: np.ndarray, rf_values: list, num_colors: int, dict: dict) -> None:
    labels = np.concatenate((np.array(rf_values)[:-1], np.array(['full obs'])))
    for rf_idx, receptive_field in reversed(list(enumerate(rf_values))):
        all_length = []
        all_accuracy = []
        for goal_color in range(num_colors):
            all_length += dict[receptive_field][goal_color]['length']
            all_accuracy += dict[receptive_field][goal_color]['accuracy']['rf']

        bins = np.arange(0, (np.max(all_length) // 20 + 1) * 20 + 1, 20)

        mean_accuracy = []
        std_accuracy = []
        n = []
        for i in range(len(bins) - 1):
            lower_bound = bins[i]
            upper_bound = bins[i + 1]
            filtered_accuracy = [acc for dist, acc in zip(all_length, all_accuracy) if lower_bound <= dist <= upper_bound]
            mean_accuracy.append(np.mean(filtered_accuracy))
            std_accuracy.append(np.std(filtered_accuracy))
            n.append(len(filtered_accuracy))

        
        plt.bar(range(len(bins) - 1), mean_accuracy, yerr=1.96 * np.array(std_accuracy) / np.array(n),
                color=colors[rf_idx], label=f'rf={labels[rf_idx]}')

        plt.xlabel('Length of the observed episode')
        plt.ylabel('Mean Accuracy (MAP)')
        plt.title('Mean accuracy (MAP) per episode length')

        plt.xticks(range(len(bins) - 1), [f'[{bins[i]},{bins[i + 1]}]' for i in range(len(bins) - 1)])

    plt.plot([-0.5, len(bins) - 1.5], [1, 1], label='Max', ls='--', c='k')
    plt.legend()
