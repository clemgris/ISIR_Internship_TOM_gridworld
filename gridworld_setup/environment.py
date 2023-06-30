from __future__ import annotations

from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_COLOR
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

import numpy as np

class MultiGoalsEnv(MiniGridEnv):
    def __init__(
        self,
        agent_goal: int,
        agent_view_size: int,
        size=20,
        agent_start_pos: tuple=(1, 1),
        agent_start_dir: int=0,
        num_colors: int=4,
        max_steps: int | None = None,
        **kwargs,
    ):  
        self.agent_goal = agent_goal
        self.num_doors = num_colors
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_view_size = agent_view_size

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Open the door with the right color"

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        self.obj_idx = [self.agent_start_pos]

        # Place walls around
        for i in range(0, height):
            for j in range(0, width):
                self.grid.set(0, i, Wall())
                self.grid.set(width - 1, i, Wall())
                self.grid.set(j, 0, Wall())
                self.grid.set(j, height - 1, Wall())

        # Place the doors and keys
        self.doors = []
        self.keys = []
        for ii in range(self.num_doors):

            # Create door and key at random position
            i_door, i_key = np.random.randint(1, self.width - 1, size=2)
            j_door, j_key = np.random.randint(1, self.height - 1, size=2)

            # Ensure no other object at the position
            while (i_door, j_door) in self.obj_idx:
                i_door = np.random.randint(1, self.width - 1)
                j_door = np.random.randint(1, self.height - 1)
            self.obj_idx.append((i_door, j_door))
            while (i_key, j_key) in self.obj_idx:
                i_key = np.random.randint(1, self.width - 1)
                j_key = np.random.randint(1, self.height - 1)
            self.obj_idx.append((i_key, j_key))

            door = Door(IDX_TO_COLOR[ii+1], is_locked=True)
            key = Key(IDX_TO_COLOR[ii+1])
            self.doors.append(door)
            self.keys.append(key)
            # Add door and key to the env
            self.grid.set(i_door, j_door, door)
            self.grid.set(i_key, j_key, key)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Open the door with the right color"


    def step(self, action: Actions):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.doors[self.agent_goal-1].is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info