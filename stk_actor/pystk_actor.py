from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import logging

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import *
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations

#: The base environment name (you can change that)
env_name = "supertuxkart/full-v0"

#: Player name (you must change that)
player_name = "TQCars"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        #lambda env: SkipCountdownWrapper(env,11),
        lambda env: NewRewardWrapper(env, offtrack_ratio=1.2),
        lambda env: ConstantSizedObservations(env),
        lambda env: PolarObservations(env),
        lambda env: Minimal56EssentialObsWrapper(env),
        lambda env: ActionToDictWrapper(env),
    ]


def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    """Creates a new actor (BBRL agent) that write into `action`

    :param state: The saved `stk_actor/pystk_actor.pth` (if it exists)
    :param observation_space: The environment observation space (with wrappers)
    :param action_space: The environment action space (with wrappers)
    :return: a BBRL agent
    """
    actor = TQCRacingAgent(
        obs_dim=56,
        continuous_dim=2,  # Steering
        discrete_dim=1,    # brake, nitro, rescue, drift, fire, acceleration
        hidden_dim=256,
        n_quantiles=25,
        n_critics=5
    )
    
    actor.load_state_dict(state['agent'])
    return Agents(actor)
