from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent#, make_env
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations

# Note the use of relative imports
from .actors import Actor, MinimalEssentialObsWrapper, TQCRacingAgent
from .pystk_actor import env_name, get_wrappers, player_name
import gymnasium as gym
import numpy as np
from numpy import array, float32, int64

def make_env(env_id, gamma):
	
	def thunk():
		env = gym.make(env_id, render_mode="human", agent=AgentSpec(use_ai=True, name=player_name))
		env = ConstantSizedObservations(env)
		env = PolarObservations(env)
		env = MinimalEssentialObsWrapper(env)

		return env

	return thunk

if __name__ == "__main__":
	# Setup the environment
	num_envs = 4
	env = make_env("supertuxkart/full-v0", gamma=0.99)()
	
	
	# (2) Learn
	actor = TQCRacingAgent(env.observation_space['continuous'].shape[0])
	# actor.load_state_dict(torch.load("trained_models/pystk_actor.pth"))
	observation, info = env.reset()
	done = False
	step = 0
	while not done:
		flat_action = actor.get_action(torch.tensor(observation['continuous'], dtype=torch.float32))[0].cpu().numpy()
		action = {
			'steer': flat_action[0:1],
			'acceleration': flat_action[1:2],
			'brake': flat_action[2:3],
			'drift': flat_action[3:4],
			'nitro': flat_action[4:5],
			'rescue': flat_action[5:6],
			'fire': 1
		}
		observation, reward, terminated, truncated, info = env.step(action)
		if step == 1250:
			truncated = True
		done = terminated or truncated
		step += 1
	print("Episode finished after {} steps".format(step))

	# (3) Save the actor state
	mod_path = Path(inspect.getfile(get_wrappers)).parent
	torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
