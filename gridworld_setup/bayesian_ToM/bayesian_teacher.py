from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.actions import Actions

from environment import MultiGoalsEnv, MultiRoomsGoalsEnv
from utils import *

import numpy as np
from queue import SimpleQueue

from typing import Callable

@dataclass
class BayesianTeacher(ABC):
    """
    Bayesian teacher that knows rational learner
    """
    
    env: MultiGoalsEnv | MultiRoomsGoalsEnv
    num_colors: int = 4
    rf_values_basic: np.ndarray = np.array([3, 5, 7])
    Na: int = 6
    add_full_obs: bool = True
    LOG: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        
        self.num_rf = len(self.rf_values_basic) + 1 if self.add_full_obs else len(self.rf_values_basic)

        # Init beliefs on the type of learner
        self.beliefs = (
            1.0 / (self.num_colors * len(self.rf_values_basic)) * np.ones((self.num_colors, self.num_rf))
        )
        # Init env and learner beliefs about the env
        self.init_env(self.env)
                
    def init_env(
        self, env: MultiGoalsEnv | MultiRoomsGoalsEnv
    ) -> None:
        self.env = env
        self.gridsize = self.env.height

        if self.add_full_obs:
            self.rf_values = np.concatenate(
                (self.rf_values_basic, np.array([self.gridsize]))
            )
        else:
            self.rf_values = self.rf_values_basic

        self.learner_beliefs = (
            1.0
            / (2 + 2 * self.num_colors)
            * np.ones(
                (self.num_rf, self.gridsize, self.gridsize, 2 + 2 * self.num_colors)
            )
        )

        self.learner_going_to_subgoal = np.zeros(
            (self.num_colors, self.num_rf), dtype=bool
        )
        self.learner_going_to_goal = np.zeros(
            (self.num_colors, self.num_rf), dtype=bool
        )
        self.learner_reached_subgoal = np.zeros(
            (self.num_colors, self.num_rf), dtype=bool
        )
        self.learner_step_count = -1

    @abstractmethod
    def update_learner_belief(self, rf_idx: int) -> None:
        ...

    @abstractmethod
    def learner_exploration_policy(self, goal_color: int, rf_idx: int) -> np.ndarray:
        ...

    @abstractmethod
    def learner_greedy_policy(
        self, obj: str, rf_idx: int, goal_color: int
    ) -> np.ndarray:
        ...

    def obj_in_front(self, rf_idx: int, obj_idx: int) -> bool:
        dx, dy = 0, 0
        if self.learner_dir == 0:
            dx = 1
        elif self.learner_dir == 2:
            dx = -1
        elif self.learner_dir == 3:
            dy = -1
        elif self.learner_dir == 1:
            dy = 1

        return (
            self.learner_beliefs[
                rf_idx, self.learner_pos[0] + dx, self.learner_pos[1] + dy, obj_idx
            ]
            == 1
        )

    def compute_obstacle_grid(self, rf_idx: int) -> np.ndarray:
        one_hot = np.zeros(2 + self.num_colors * 2)
        one_hot[0] = 1
        return np.ones((self.gridsize, self.gridsize)) - np.all(
            self.learner_beliefs[rf_idx, ...] == one_hot.reshape(1, 1, -1), axis=2
        )

    @abstractmethod
    def learner_policy(self, goal_color: int, rf_idx: int) -> np.ndarray:
        ...

    def update_knowledge(
        self,
        learner_pos: tuple,
        learner_dir: int,
        learner_step_count: int,
        rf_idx: int | None = None,
    ) -> None:
        self.learner_pos = learner_pos
        self.learner_dir = learner_dir
        self.learner_step_count += 1
        assert self.learner_step_count == learner_step_count

        if rf_idx is None:
            for rf_idx in range(self.num_rf):
                # Update what the learner knows about the env
                self.update_learner_belief(rf_idx)
        else:
            self.update_learner_belief(rf_idx)

    def observe(self, action: int) -> None:
        self.LOG.append(f"step t={self.learner_step_count}")
        self.LOG.append(f"True action {action}")

        for rf_idx in range(self.num_rf):
            for goal_color in range(self.num_colors):
                # Predict policy of the learner
                predicted_policy = self.learner_policy(goal_color, rf_idx)
                rf = self.rf_values[rf_idx]
                if goal_color == 0:
                    self.LOG.append(
                        f"agent_pos {self.env.agent_pos} dir {self.env.agent_dir} rf {rf} goal_color {goal_color} policy {np.round(predicted_policy, 4)}"
                    )

                # Bayesian update
                self.beliefs[goal_color, rf_idx] *= predicted_policy[action]

        self.beliefs /= self.beliefs.sum()
        self.LOG.append(f"pred {list(np.around(self.beliefs, 4))}")

    def predicted_reward(self, demo: list, goal_color: int, rf_idx: int) -> float:
        current_receptve_field = self.env.agent_view_size

        # Reset env AND estimate beliefs of the learner
        self.env.agent_view_size = self.rf_values[rf_idx]
        self.env.reset_grid()
        self.init_env(self.env)

        self.learner_step_count = 0

        if len(demo) > 0:
            # Add first unused action to get the first observation
            demo = [4] + demo
        else:
            demo = demo

        # Simulate the learner observing the demo
        for a in demo:
            action = Actions(a)
            _, _, _, _, _ = self.env.step(action)
            self.update_knowledge(
                self.env.agent_pos, self.env.agent_dir, self.env.step_count, rf_idx
            )

        # Simulate the learner playing on the env AFTER seen the demo
        self.env.reset_grid()
        self.learner_step_count = 0
        terminated = False
        while (not terminated) and (self.env.step_count < self.env.max_steps):
            a = draw(self.learner_policy(goal_color, rf_idx))
            action = Actions(a)
            _, reward, terminated, _, _ = self.env.step(action)
            self.update_knowledge(
                self.env.agent_pos, self.env.agent_dir, self.env.step_count, rf_idx
            )

        # Reset env
        self.env.agent_view_size = current_receptve_field
        self.env.reset_grid()

        # Return the predicted reward
        return reward

    def select_demo(
        self,
        cost_function: Callable[[int, int], float] = lambda x, l: exp_cost(
            l - x, l, alpha=0.3, beta=5
        ),
    ) -> list:
        goal_color_belief = np.sum(self.beliefs, axis=1)
        argmax_set = np.where(np.isclose(goal_color_belief, np.max(goal_color_belief)))[
            0
        ]
        pred_goal_color = np.random.choice(argmax_set)
        demos = []
        for rf in self.rf_values:
            demo = generate_demo(self.env, rf, pred_goal_color)
            demos.append(demo)

        # Compute longest demo
        demo_all = generate_demo_all(self.env)
        l_max = len(demo_all)

        demos.append(demo_all)

        predicted_utility = []
        for demo_idx, demo in enumerate(demos):
            pred_u = 0
            for rf_idx_demo, _ in enumerate(self.rf_values):
                hat_r = self.predicted_reward(demo, pred_goal_color, rf_idx_demo)
                cost = cost_function(len(demo), l_max)
                pred_u += (hat_r - cost) * self.beliefs[pred_goal_color, rf_idx_demo]
            predicted_utility.append(pred_u)

        argmax_set = np.where(np.isclose(predicted_utility, np.max(predicted_utility)))[
            0
        ]
        demo_idx = np.random.choice(argmax_set)

        predicted_best_utility = np.max(predicted_utility)

        return demos[demo_idx], demo_idx, predicted_best_utility, demos
