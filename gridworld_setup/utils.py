import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.actions import Actions

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

def compute_learner_obs(pos: tuple, dir: int, receptive_field: int, env: MultiGoalsEnv | MultiRoomsGoalsEnv) -> np.ndarray:
    
    topX, topY, _, _ = get_view(pos, dir, receptive_field)
    
    grid = env.grid.slice(topX, topY, receptive_field, receptive_field)
    for _ in range(dir + 1):
        grid = grid.rotate_left()

    if env.see_through_walls:
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
    else:
        vis_mask = grid.process_vis(agent_pos=(receptive_field // 2, receptive_field - 1))

    obs = grid.encode(vis_mask)

    return obs, vis_mask


def generate_demo(env: MultiGoalsEnv | MultiRoomsGoalsEnv, rf: int, goal_color: int):
    
    # Save current config of the env
    prev_agent_view_size = env.agent_view_size
    
    gridsize = env.height
    assert(env.height == env.width)

    # Teacher has full observability
    env.agent_view_size = gridsize
    env.reset_grid()
    
    # Get image of the env
    env_image = np.ones((gridsize, gridsize))
    obs = env.grid.encode(np.ones((gridsize, gridsize)))
    
    for abs_j in range(0, gridsize):
        for abs_i in range(0, gridsize):
            color_idx = obs[abs_i, abs_j, 1]

            # Goal
            if obs[abs_i, abs_j, 0] == 4 and color_idx == goal_color + 1:
                value = 3
                goal_pos = (abs_i, abs_j)
            # Subgoal (key)
            elif obs[abs_i, abs_j, 0] == 5 and color_idx == goal_color + 1:
                value = 2
                subgoal_pos = (abs_i, abs_j)
            # Wall
            elif obs[abs_i, abs_j, 0] in [2, 5, 4]:
                value = 1
            # Nothing
            else:
                value= 0

            env_image[abs_i, abs_j] = value
    obstacle_grid = (env_image != 0)

    # Create demo
    env.agent_view_size = rf
    env.reset_grid()

    subgoal_dist = Manhattan_dist(env.agent_pos, subgoal_pos)
    goal_dist = Manhattan_dist(env.agent_pos, goal_pos)
    
    # Go first to the closest object
    if subgoal_dist < goal_dist:
        Goals = [subgoal_pos, goal_pos]
    else:
        Goals = [goal_pos, subgoal_pos]

    traj = []
    for g in Goals:
        # Remove object g from obstacles
        obstacle_grid[g[0], g[1]] = 0
        # Compute shortest path
        path = A_star_algorithm(env.agent_pos, g, obstacle_grid)
        # Put object g in obtacles
        obstacle_grid[g[0], g[1]] = 1

        ii = 0
        while not obj_in_view(env.agent_pos, env.agent_dir, rf, g, env):
            transition = path[ii]
            actions = map_actions(env.agent_pos, transition[1], env.agent_dir)
            for a in actions:
                env.step(Actions(a))
                traj.append(a)
                if obj_in_view(env.agent_pos, env.agent_dir, rf, g, env):
                    break
            ii += 1

    # Reset the position of the agent in the env and the env config
    env.agent_view_size = prev_agent_view_size
    env.reset_grid()
    
    return traj
    
def compute_opt_length(env: MultiGoalsEnv, goal_color: int):
    
    # Save current config of the env
    prev_agent_view_size = env.agent_view_size
    
    gridsize = env.height
    assert(env.height == env.width)

    # Teacher has full observability
    env.agent_view_size = gridsize
    env.reset_grid()

    env_image = np.ones((gridsize, gridsize))
    obs = env.grid.encode(np.ones((gridsize, gridsize)))
    
    # Get image of the env
    for abs_j in range(0, gridsize):
        for abs_i in range(0, gridsize):
            color_idx = obs[abs_i, abs_j, 1]

            # Goal
            if obs[abs_i, abs_j, 0] == 4 and color_idx == goal_color + 1:
                value = 3
                goal_pos = (abs_i, abs_j)
            # Subgoal (key)
            elif obs[abs_i, abs_j, 0] == 5 and color_idx == goal_color + 1:
                value = 2
                subgoal_pos = (abs_i, abs_j)
            # Wall
            elif obs[abs_i, abs_j, 0] in [2, 5, 4]:
                value = 1
            # Nothing
            else:
                value= 0

            env_image[abs_i, abs_j] = value

    obstacle_grid = (env_image == 1)
    obstacle_grid[goal_pos[0], goal_pos[1]] = 1
    
    # Compute length of the optimal path to finish the task
    length_opt_path = 0
    agent_dir = env.agent_start_dir

    path_subgoal = A_star_algorithm(env.agent_start_pos, subgoal_pos, obstacle_grid)
    for trans in path_subgoal:
        actions = map_actions(trans[0], trans[1], agent_dir)
        for a in actions:
            if a == 0:
                agent_dir = (agent_dir - 1) % 4
            elif a == 1:
                agent_dir = (agent_dir + 1) % 4
            length_opt_path += 1
    obstacle_grid[goal_pos[0], goal_pos[1]] = 0
    path_goal = A_star_algorithm(subgoal_pos, goal_pos, obstacle_grid)
    for trans in path_goal:
        actions = map_actions(trans[0], trans[1], agent_dir)
        for a in actions:
            if a == 0:
                agent_dir = (agent_dir - 1) % 4
            elif a == 1:
                agent_dir = (agent_dir + 1) % 4
            length_opt_path += 1

    # Reset the position of the agent in the env and the env config
    env.agent_view_size = prev_agent_view_size
    env.reset_grid()

    return length_opt_path