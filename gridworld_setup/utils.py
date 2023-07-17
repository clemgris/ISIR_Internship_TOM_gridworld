import numpy as np
import matplotlib.pyplot as plt

from minigrid.core.constants import DIR_TO_VEC

from environment import MultiGoalsEnv, MultiRoomsGoalsEnv

from queue import PriorityQueue
import heapq

def draw(proba_dist: np.array) -> int:
    assert(np.isclose(proba_dist.sum(), 1.))
    rand_num = np.random.uniform(0, 1)
    cum_prob = np.cumsum(proba_dist)
    for idx, _ in enumerate(proba_dist):
        if cum_prob[idx] >= rand_num:
            selected_idx = idx
            break
    return selected_idx

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

def map_actions(learner_pos: tuple, pos_dest: tuple, learner_dir: int) -> list:
    # Mapping position transition --> actions
    dx = learner_pos[0] - pos_dest[0]
    dy = learner_pos[1] - pos_dest[1]
    actions = []
    if dx < 0:
        if learner_dir == 0:
            actions.append(2)
        elif learner_dir == 1:
            actions.append(0)
            actions.append(2)
        elif learner_dir == 2:
            actions.append(1)
            actions.append(1)
            actions.append(2)
        elif learner_dir == 3:
            actions.append(1)
            actions.append(2)

    if dx > 0:
        if learner_dir == 0:
            actions.append(1)
            actions.append(1)
            actions.append(2)
        elif learner_dir == 1:
            actions.append(1)
            actions.append(2)
        elif learner_dir == 2:
            actions.append(2)
        elif learner_dir == 3:
            actions.append(0)
            actions.append(2)

    if dy < 0:
        if learner_dir == 0:
            actions.append(1)
            actions.append(2)
        elif learner_dir == 1:
            actions.append(2)
        elif learner_dir == 2:
            actions.append(0)
            actions.append(2)
        elif learner_dir == 3:
            actions.append(1)
            actions.append(1)
            actions.append(2)

    if dy > 0:
        if learner_dir == 0:
            actions.append(0)
            actions.append(2)
        elif learner_dir == 1:
            actions.append(1)
            actions.append(1)
            actions.append(2)
        elif learner_dir == 2:
            actions.append(1)
            actions.append(2)
        elif learner_dir == 3:
            actions.append(2)

    return actions

def get_view(agent_pos: tuple, agent_dir: int, receptive_field: int) -> tuple:
    # Facing right
    if agent_dir == 0:
        topX = agent_pos[0]
        topY = agent_pos[1] - receptive_field // 2
    # Facing down
    elif agent_dir == 1:
        topX = agent_pos[0] - receptive_field // 2
        topY = agent_pos[1]
    # Facing left
    elif agent_dir == 2:
        topX = agent_pos[0] - receptive_field + 1
        topY = agent_pos[1] - receptive_field // 2
    # Facing up
    elif agent_dir == 3:
        topX = agent_pos[0] - receptive_field // 2
        topY = agent_pos[1] - receptive_field + 1
    else:
        assert False, "invalid agent direction"

    botX = topX + receptive_field
    botY = topY + receptive_field

    return (topX, topY, botX, botY)

def obj_in_view(agent_pos: tuple, agent_dir: int, receptive_field: int, obj_pos: tuple, env: MultiGoalsEnv | MultiRoomsGoalsEnv) -> bool:
    
    topX, topY, _, _ = get_view(agent_pos, agent_dir, receptive_field)
    
    grid = env.grid.slice(topX, topY, receptive_field, receptive_field)
    for _ in range(agent_dir + 1):
        grid = grid.rotate_left()

    if env.see_through_walls:
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
    else:
        vis_mask = grid.process_vis(agent_pos=(receptive_field // 2, receptive_field - 1))

    f_vec = DIR_TO_VEC[agent_dir]
    dir_vec = DIR_TO_VEC[agent_dir]
    dx, dy = dir_vec
    r_vec =  np.array((-dy, dx))
    top_left = (
        agent_pos
        + f_vec * (receptive_field - 1)
        - r_vec * (receptive_field // 2)
    )

    # For each cell in the visibility mask
    for vis_j in range(0, receptive_field):
        for vis_i in range(0, receptive_field):
            
            if not vis_mask[vis_i, vis_j]:
                continue

            # Compute the world coordinates of this cell
            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
            if (abs_i, abs_j) == obj_pos:
                return True

    return False

##
# Visualization
##

def plot_grid(start, num, size, alpha=0.5):
    idx = np.linspace(start, size, num)
    for x in idx:
        plt.plot([x, x], [start, size], alpha=alpha, c='gray')
        plt.plot([start, size], [x, x], alpha=alpha, c='gray')

def plot_agent_play(pos: tuple, dir: int) -> None:
    if dir == 0:
        marker = ">"
    elif dir == 1:
        marker = "v"
    elif dir == 2:
        marker = "<"
    elif dir == 3:
        marker = "^"
    plt.scatter(pos[0], pos[1], marker=marker, c='r', s=120)

def plot_agent_obs(pos: tuple, GRID_SIZE: int, img: np.ndarray, hide: bool=False) -> None:
    ratio = img.shape[0] / GRID_SIZE
    im_agent_pos =np.array([(pos[0] + 0.5) * ratio, (pos[1] + 0.5) * ratio]).astype('int')
    if hide:
        plt.scatter(im_agent_pos[0], im_agent_pos[1], color=rgb_to_hex((76, 76, 76)), marker='s', s=145)
    plt.scatter(im_agent_pos[0], im_agent_pos[1], c='w', marker='*', s=120)

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

        
        plt.bar(range(len(bins) - 1), mean_accuracy, yerr=1.96 * np.array(std_accuracy) / np.sqrt(np.array(n)),
                color=colors[rf_idx], label=f'rf={labels[rf_idx]}')

        plt.xlabel('Length of the observed episode')
        plt.ylabel('Mean Accuracy (MAP)')
        plt.title('Mean accuracy (MAP) per episode length')

        plt.xticks(range(len(bins) - 1), [f'[{bins[i]},{bins[i + 1]}]' for i in range(len(bins) - 1)])

    plt.plot([-0.5, len(bins) - 1.5], [1, 1], label='Max', ls='--', c='k')
    plt.legend()

def rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(channel))) for channel in rgb]
    # Convert RGB to hexadecimal color code (i.e. map to color type in python)
    hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    return hex_code
