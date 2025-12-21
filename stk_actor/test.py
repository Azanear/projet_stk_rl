import gymnasium as gym
from pystk2_gymnasium import AgentSpec, ConstantSizedObservations, PolarObservations
from .actors import TQCRacingAgent, MinimalEssentialObsWrapper, ActionToDictWrapper, SkipCountdownWrapper
import numpy as np
import torch
import sys

# Créer l'environnement
env = gym.make(
    "supertuxkart/full-v0",
    agent=AgentSpec(use_ai=False),
    render_mode='human',  # Pas de render, juste les prints
    num_kart=4,
    difficulty=2,  # Difficulté max pour voir plus d'items/attachments
    max_episode_steps=1000
)
env = SkipCountdownWrapper(env,10)
env = ConstantSizedObservations(env)
env = PolarObservations(env)
env = MinimalEssentialObsWrapper(env)


obs, info = env.reset()

# Tracking pour voir les changements
last_attachment = None
seen_item_types = set()
seen_attachments = set()

actor = TQCRacingAgent(
    obs_dim=29,
    continuous_dim=2,  # Steering
    discrete_dim=2,    # brake, nitro, rescue, drift, fire, acceleration
    hidden_dim=256,
    n_quantiles=25,
    n_critics=5
)

dir_path = sys.path[0]

filename = f"{dir_path}/stk_actor/pystk_actor.pth"
state = torch.load(filename,map_location=torch.device('cpu'))
actor.load_state_dict(state['agent'])

for step in range(5000):

    #action = env.action_space.sample()

    obs = torch.tensor(obs['continuous'], dtype=torch.float32).unsqueeze(0)

    flat_action_tensor, _  = actor.get_action(obs, deterministic=True)
    if isinstance(flat_action_tensor, torch.Tensor):
        flat_action = flat_action_tensor.cpu().numpy()
    else:
        flat_action = flat_action_tensor
    if flat_action.ndim == 1:
        flat_action = flat_action[np.newaxis, :]  # Add batch dim if needed

    action = {
        'steer': flat_action[:,0:1],
        'acceleration': 0.5,
        'brake': flat_action[:,2:3],
        'drift': 0,
        'nitro': 1,
        'rescue': 0,
        'fire': 1
    }

    
    print("Il ya un objet = ",obs[:,10], " C'est un bon objet ?: ",obs[:,14])
    print("Il ya un objet = ",obs[:,15], " C'est un bon objet ?: ",obs[:,19])
    print("Il ya un objet = ",obs[:,20], " C'est un bon objet ?: ",obs[:,24])
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"\n🔄 Episode ended at step {step}, resetting...")
        obs, info = env.reset()
        last_attachment = None

env.close()
