import numpy as np
from queue import SimpleQueue
from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_COLOR, DIR_TO_VEC
from PIL import Image
import pickle

from environment import MultiGoalsEnv
from utils import *

class BayesianLearner:
    def __init__(self,
                 goal: int,
                 receptive_field: int,
                 grid_size: int=20,
                 num_doors: int=4,
                 save_render: bool=False,
                 max_steps: int=100
                 ) -> None:
        
        self.goal = goal + 1
        self.receptive_field = receptive_field
        self.max_steps = max_steps

        # Init with uniform beliefs (no prior)
        self.beliefs = 1 / 4 * np.ones((grid_size, grid_size, 4))

        # Prior on the walls
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                self.beliefs[0, i, :] = np.array([0,1,0,0])
                self.beliefs[grid_size - 1, i, :] = np.array([0,1,0,0])
                self.beliefs[j, 0, :] = np.array([0,1,0,0])
                self.beliefs[j, grid_size - 1, :] = np.array([0,1,0,0])

        # Generate feasable environment
        has_path_key_door = False
        while not has_path_key_door:

            # Generate env
            self.init_env(grid_size=grid_size,
                        num_doors=num_doors)
            self.env.reset()

            # Check feasability
            grid = np.ones((self.env.height, self.env.width))
            for i in range(self.env.height):
                for j in range(self.env.width):
                    grid[i, j] = self.env.grid.get(i,j) is not None

            key_pos = self.env.obj_idx[2 * self.goal]
            grid[key_pos[0], key_pos[1]] = 0
            path_key = A_star_algorithm(start=self.env.agent_pos, goal=key_pos, grid=grid)

            door_pos = self.env.obj_idx[2 * self.goal - 1]
            grid[door_pos[0], door_pos[1]] = 0
            path_door = A_star_algorithm(start=self.env.agent_pos, goal=door_pos, grid=grid)

            has_path_key_door = path_door is not None and path_key is not None
        
        self.has_key = False

        self.actions = SimpleQueue()
        self.transitions = SimpleQueue()

        self.reward = 0
        self.going_to_key = False
        self.going_to_door = False

        # For rendering
        self.save_render = save_render
        self.render_frames = []
        self.render_beliefs = []

    def init_env(self, grid_size: int=20, num_doors: int=4) -> None:
        agent_start_pos = (grid_size//2, grid_size-2) # tuple(np.random.randint(1, grid_size - 1, size=2))
        agent_start_dir = 3 # np.random.randint(0, 4)

        self.env = MultiGoalsEnv(render_mode = "rgb_array",
                                 agent_goal=self.goal,
                                 size=grid_size,
                                 agent_start_pos=agent_start_pos,
                                 agent_start_dir=agent_start_dir,
                                 agent_view_size=self.receptive_field,
                                 num_doors=num_doors,
                                 max_steps=self.max_steps)
    
    def play(self, size: int=None) -> None:
        
        if size is None:
            size = self.max_steps

        for _ in range(size):
            a = self.policy()
            action = Actions(a)
            obs, reward, terminated, _, _ = self.env.step(action)
            self.update_beliefs(obs['image'])
            self.reward = reward

            # Save for rendering
            self.render_frames.append(self.env.render())
            beliefs_image = Shannon_entropy(self.beliefs, axis=2).T
            self.render_beliefs.append(np.repeat(np.repeat(beliefs_image, 20, axis=0), 20, axis=1))

            if terminated:
                break

        if self.save_render:

            # Save frames
            gif_file = f"./outputs_rendering/output.gif"
            pil_images = [Image.fromarray(image.astype('uint8'), 'RGB') for image in self.render_frames]
            pil_images[0].save(gif_file, save_all=True, append_images=pil_images[1:], duration=100, loop=0)

            # Save associeted beliefs entropy
            gif_file = f"./outputs_rendering/output_belief.gif"
            pil_images = [Image.fromarray(image / (Shannon_entropy( 1 / 4 * np.ones(4)) + 0.5) * 255) for image in self.render_beliefs]
            pil_images[0].save(gif_file, save_all=True, append_images=pil_images[1:], duration=100, loop=0)


    def update_beliefs(self, obs: np.ndarray) -> None:
        f_vec = self.env.dir_vec
        r_vec = self.env.right_vec
        top_left = (
            self.env.agent_pos
            + f_vec * (self.receptive_field - 1)
            - r_vec * (self.receptive_field // 2)
        )

        # Compute which cells are visible to the agent
        _, vis_mask = self.env.gen_obs_grid()

        # For each cell in the visibility mask
        for vis_j in range(0, self.receptive_field):
            for vis_i in range(0, self.receptive_field):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue

                # Door
                if (obs[vis_i, vis_j, :2] == np.array([4, self.goal])).all() and (self.env.agent_pos != (abs_i, abs_j)):
                    # Projection
                    self.beliefs[:, :, 2] = np.zeros_like(self.beliefs[:, :,2])
                    self.beliefs /= self.beliefs.sum(axis=2).reshape(self.env.height, self.env.width, -1)
                    # Only one door
                    self.beliefs[abs_i, abs_j, :] = np.array([0, 0, 1, 0])
                # Key
                elif (obs[vis_i, vis_j, :2] == np.array([5, self.goal])).all() and (self.env.agent_pos != (abs_i, abs_j)):
                    # Projection
                    self.beliefs[:, :, 3] = np.zeros_like(self.beliefs[:, :, 3])
                    self.beliefs /= self.beliefs.sum(axis=2).reshape(self.env.height, self.env.width, -1)
                    # Only one key
                    self.beliefs[abs_i, abs_j, :] = np.array([0, 0, 0, 1])
                # Obstacle
                elif obs[vis_i, vis_j, 0] in [2, 4, 5] and (self.env.agent_pos != (abs_i, abs_j)):
                    self.beliefs[abs_i, abs_j, :] = np.array([0, 1, 0, 0])
                # Nothing
                else:
                    if (obs[vis_i, vis_j, :2] == np.array([5, self.goal])).all() and (self.env.agent_pos == (abs_i, abs_j)):
                        self.has_key = True
                    self.beliefs[abs_i, abs_j, :] = np.array([1, 0, 0, 0])
    
    def compute_exploration_score(self, dir: int, pos: tuple) -> float:
        f_vec = DIR_TO_VEC[dir]
        dir_vec = DIR_TO_VEC[dir]
        dx, dy = dir_vec
        r_vec =  np.array((-dy, dx))

        top_left = (
            pos
            + f_vec * (self.receptive_field - 1)
            - r_vec * (self.receptive_field // 2)
        )

        exploration_score = 0

        # For each cell in the visibility mask
        for vis_j in range(0, self.receptive_field):
            for vis_i in range(0, self.receptive_field):

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue

                exploration_score += Shannon_entropy(self.beliefs[abs_i, abs_j, :])
            
        return exploration_score
    
    def best_exploration_action(self, forced: bool=False) -> int:

        # Action that maximizes the exploration
        scores = np.zeros(3)
        # Turn left
        scores[0] = self.compute_exploration_score(dir=(self.env.agent_dir - 1) % 4, pos=self.env.agent_pos)
        # Turn right
        scores[1] = self.compute_exploration_score(dir=(self.env.agent_dir + 1) % 4, pos=self.env.agent_pos)
        # Move forward
        next_pos = self.env.agent_pos + DIR_TO_VEC[self.env.agent_dir]
        if np.all(self.beliefs[next_pos[0], next_pos[1], :] == np.array([0, 1, 0, 0])) or \
            np.all(self.beliefs[next_pos[0], next_pos[1], :] == np.array([0, 0, 1, 0])): # Obstacle in front
            scores[2] = 0.
        else:
            scores[2] = self.compute_exploration_score(dir=self.env.agent_dir, pos=next_pos)

        argmax_set = np.where(np.isclose(scores, np.max(scores)))[0]

        # If actions better than the others
        if len(argmax_set) < 3 or forced:
            best_action =  np.random.choice(argmax_set)
            return best_action
        
        # If all the actions are equal
        else:

            # Unexplored locations
            unexplored_pos = np.where(Shannon_entropy(self.beliefs, axis=2) != 0)
            # At least one unexplored position
            if len(unexplored_pos[0]) > 0:
                # Manhattan distance to the unexplored locations
                dist = np.array([Manhattan_dist(self.env.agent_pos, pos) for pos in zip(unexplored_pos[0], unexplored_pos[1])])
                # Closest unexplored locations
                argmin_set = np.where(np.isclose(dist, np.min(dist)))[0]
                dest_idx = np.random.choice(argmin_set)
                dest_pos = (unexplored_pos[0][dest_idx], unexplored_pos[1][dest_idx])

                # Obstacle grid
                grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
                grid[dest_pos[0], dest_pos[1]] = 0
                
                # If the intermediate goal is not reacheable change goal
                while dist[dest_idx] < 10e5:
                    path = A_star_algorithm(self.env.agent_pos, dest_pos, grid)
                    if path is None:
                        dist[dest_idx] = 10e5
                        argmin_set = np.where(np.isclose(dist, np.min(dist)))[0]
                        dest_idx = np.random.choice(argmin_set)
                        dest_pos = (unexplored_pos[0][dest_idx], unexplored_pos[1][dest_idx])
                        grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
                        grid[dest_pos[0], dest_pos[1]] = 0
                    else:
                        # Add transitions to go to exploratory goal
                        for transition in path:
                            self.transitions.put(transition)

                        return self.policy()

            return self.best_exploration_action(forced=True)
            

    def add_actions(self, pos_dest: tuple) -> None:
        dx = self.env.agent_pos[0] - pos_dest[0]
        dy = self.env.agent_pos[1] - pos_dest[1]
        if dx < 0:
            if self.env.agent_dir == 0:
                self.actions.put(2)
            elif self.env.agent_dir == 1:
                self.actions.put(0)
                self.actions.put(2)
            elif self.env.agent_dir == 2:
                self.actions.put(1)
                self.actions.put(1)
                self.actions.put(2)
            elif self.env.agent_dir == 3:
                self.actions.put(1)
                self.actions.put(2)

        if dx > 0:
            if self.env.agent_dir == 0:
                self.actions.put(1)
                self.actions.put(1)
                self.actions.put(2)
            elif self.env.agent_dir == 1:
                self.actions.put(1)
                self.actions.put(2)
            elif self.env.agent_dir == 2:
                self.actions.put(2)
            elif self.env.agent_dir == 3:
                self.actions.put(0)
                self.actions.put(2)

        if dy < 0:
            if self.env.agent_dir == 0:
                self.actions.put(1)
                self.actions.put(2)
            elif self.env.agent_dir == 1:
                self.actions.put(2)
            elif self.env.agent_dir == 2:
                self.actions.put(0)
                self.actions.put(2)
            elif self.env.agent_dir == 3:
                self.actions.put(1)
                self.actions.put(1)
                self.actions.put(2)

        if dy > 0:
            if self.env.agent_dir == 0:
                self.actions.put(0)
                self.actions.put(2)
            elif self.env.agent_dir == 1:
                self.actions.put(1)
                self.actions.put(1)
                self.actions.put(2)
            elif self.env.agent_dir == 2:
                self.actions.put(1)
                self.actions.put(2)
            elif self.env.agent_dir == 3:
                self.actions.put(2)

    def obs_in_front(self, obj_idx: int) -> bool:

        dx, dy = 0, 0
        if self.env.agent_dir == 0:
            dx = 1
        elif self.env.agent_dir == 2:
            dx = -1
        elif self.env.agent_dir == 3:
            dy = -1
        elif self.env.agent_dir == 1:
            dy = 1

        agent_pos = self.env.agent_pos
        return self.beliefs[agent_pos[0] + dx, agent_pos[1] + dy, obj_idx] == 1
        
    def policy(self):
            
        if self.env.step_count == 0:
            return 4 # unused
    
        if self.obs_in_front(obj_idx=3):

            self.going_to_key = False
            # Key reached --> empty queues
            while not self.transitions.empty():
                _ = self.transitions.get()
            while not self.actions.empty():
                _ = self.actions.get()
            # Pickup the key
            return 3
        
        if self.obs_in_front(obj_idx=2) and self.has_key:

            self.going_to_door = False
            # Door reached --> empty queues
            while not self.transitions.empty():
                _ = self.transitions.get()
            while not self.actions.empty():
                _ = self.actions.get()
            # Open the door
            return 5

        # If know where is the key & not already going to the key --> go to the key
        if (self.beliefs[:, :, 3] == 1).any() and not self.going_to_key:

            key_pos = np.where(self.beliefs[:, :, 3] == 1)
            # Obstacle grid
            grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
            grid[key_pos[0], key_pos[1]] = 0
            
            path = A_star_algorithm(self.env.agent_pos, key_pos, grid)

            if path is not None:
                # Empty queues
                while not self.transitions.empty():
                    _ = self.transitions.get()
                while not self.actions.empty():
                    _ = self.actions.get()
                # Add transitions to go to key
                for transition in path:
                    self.transitions.put(transition)
                # Set variable
                self.going_to_key = True
                # Return action
                return self.policy()

        # If know where is the door & has key & not already going to the door --> go to the door
        elif (self.beliefs[:, :, 2] == 1).any() and self.has_key and not self.going_to_door:

            door_pos = np.where(self.beliefs[:, :, 2] == 1)
            # Obstacle grid
            grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
            grid[door_pos[0], door_pos[1]] = 0

            path = A_star_algorithm(self.env.agent_pos, door_pos, grid)

            if path is not None:
                # Empty queues
                while not self.transitions.empty():
                    _ = self.transitions.get()
                while not self.actions.empty():
                    _ = self.actions.get()
                # Add transitions to go to door
                for transition in path:
                    self.transitions.put(transition)
                # Set variable
                self.going_to_door = True
                # Return action
                return self.policy()
        
        # Action to be played
        if not self.actions.empty():
            action = self.actions.get()
            return action
        
        # Position to be reached
        if not self.transitions.empty():
            pos_init, pos_dest = self.transitions.get()
            
            # Sanity check
            assert(pos_init == self.env.agent_pos)

            # If not an obstacle --> add action to reach pos_dest
            if np.any(self.beliefs[pos_dest[0], pos_dest[1], :] != np.array([0, 1, 0, 0])):
                self.add_actions(pos_dest)

            return self.policy()
        
        # Nothing to do 
        else:
            # Action that maximizes the exploration
            return self.best_exploration_action()