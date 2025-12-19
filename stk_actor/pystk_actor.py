from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import logging

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor, TQCRacingAgent, MinimalEssentialObsWrapper, ActionToDictWrapper, SkipCountdownWrapper
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
        lambda env: SkipCountdownWrapper(env, skip_steps=10),
        lambda env: ConstantSizedObservations(env),
        lambda env: MinimalEssentialObsWrapper(env),
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
    obs_dim = observation_space['continuous'].shape[0]
    actor = TQCRacingAgent(
        obs_dim=obs_dim,
        continuous_dim=1,  # Steering
        discrete_dim=5,    # brake, nitro, rescue, drift, fire, acceleration
        hidden_dim=256,
        n_quantiles=25,
        n_critics=5
    )

    # Returns a dummy actor
    if state is None:
        return SamplingActor(action_space)

    filename = "stk_actor/pystk_actor.pth"
    logging.info(f"Loading trained model from {filename}...")
    state_dict = torch.load(filename)
    actor.load_state_dict(state_dict['agent'])
    return Agents(actor)
