import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn
import numpy as np

class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action
    
class MinimalEssentialObsWrapper(gym.Wrapper):
    """
    Extrait uniquement les observations essentielles pour le racing.
    Adapté au format exact de SuperTuxKart.
    
    Réduit de ~92 dims continuous → ~20 dims continuous
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict({
            'continuous': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(31,), #  PREVIOUS 21, 
                dtype=np.float32
            )
        })
        
        # Update observation space    
    def reset(self, **kwargs):
        obs,info = self.env.reset(**kwargs)
        
        processed_obs = self._process_obs(obs)
        info['distance_down_track'] = float(obs['distance_down_track'][0])
        return processed_obs,info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Process observation
        processed_obs = self._process_obs(obs)
        info['distance_down_track'] = float(obs['distance_down_track'][0])
        return processed_obs, reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        continuous_features = []
        
        # 1. Distance down track (1)
        #continuous_features.append(float(obs['distance_down_track'][0]))
        
        # 2. Velocity (3)
        velocity = obs['velocity'] / 40.0
        continuous_features.extend(velocity.tolist())
        
        # 3. Speed (1) - norme de velocity
        speed = np.linalg.norm(velocity)
        continuous_features.append(float(speed))
        
        # 4. Front vector (3) - orientation du kart
        front = obs['front']
        continuous_features.extend(front.tolist())
        
        # 5. Center path distance (1)
        center_dist = float(obs['center_path_distance'][0]) / 10.
        continuous_features.append(center_dist)

         # 5. Center path distance (1)
        skeed_factor = float(obs['skeed_factor'][0]) 
        continuous_features.append(skeed_factor)
        
        # 6. Center path direction x, z (2) - ignore y
        center_path = obs['center_path']
        continuous_features.append(float(center_path[0]))  # x
        continuous_features.append(float(center_path[2]))  # z
        

        indices = [0, 1,2,3,4] # Modifiable selon len(path_starts)
        
        VISION_SCALE = 80.0  # Basé sur ton 77.87m
        WIDTH_SCALE = 20.0   # Basé sur ton 19.34m
        
        path_starts = obs['paths_start']
        path_widths = obs['paths_width']
        available_len = len(path_starts)
        
        for i in indices:
            # Sécurité si on a moins de points que prévu
            idx = min(i, available_len - 1)
            
            p_start = path_starts[idx]
            
            # Normalisation des coordonnées X et Z
            continuous_features.append(float(p_start[0]) / VISION_SCALE)
            continuous_features.append(float(p_start[2]) / VISION_SCALE)
            
            # Normalisation de la largeur de la route à cet endroit
            raw_w = path_widths[idx]
            p_width = float(raw_w[0]) if hasattr(raw_w, '__len__') else float(raw_w)
            continuous_features.append(p_width / WIDTH_SCALE)


        """
        # 7. Current path start x, z (2)
        paths_start = obs['paths_start'][0]  # Premier path
        continuous_features.append(float(paths_start[0]))  # x
        continuous_features.append(float(paths_start[2]))  # z
        
        # 8. Current path end x, z (2)
        paths_end = obs['paths_end'][0]
        continuous_features.append(float(paths_end[0]))  # x
        continuous_features.append(float(paths_end[2]))  # z

        # ADDED +++++++++++++++++++++++++++++++++++++++++++
        next_paths_end = obs['paths_end'][1]
        continuous_features.append(float(next_paths_end[0]))  # x
        continuous_features.append(float(next_paths_end[2]))  # z
        
        # 9. Path width (1)
        path_width = float(obs['paths_width'][0][0])
        continuous_features.append(path_width)
        
        # ADDED +++++++++++++++++++++++++++++++++++++++++++
        next_path_width = float(obs['paths_width'][1][0])
        continuous_features.append(next_path_width)
        """

        # 10. Closest kart position x, z (2)
        kart_pos = obs['karts_position'][0]        
        continuous_features.append(float(kart_pos[0]) / VISION_SCALE )  # x
        continuous_features.append(float(kart_pos[2]) / VISION_SCALE)  # z
        
        # 11. Max steer angle (1)
        continuous_features.append(float(obs['max_steer_angle'][0]))
        
        # 12. Energy (1)
        continuous_features.append(float(obs['energy'][0]) / 10.0)
        
        # 13. Off-track indicator (1) - calculé

        curr_w_raw = path_widths[0][0]
        curr_w = float(curr_w_raw)
        
        # Dist brute
        dist_raw = float(obs['center_path_distance'][0])
        
        off_track = 1.0 if abs(dist_raw) > curr_w / 2 else 0.0
        continuous_features.append(off_track)

        #off_track = 1.0 if abs(center_dist) > path_width / 2 else 0.0
        #continuous_features.append(off_track)
        
        
        return {
            'continuous': np.array(continuous_features, dtype=np.float32)
        }


class TQCRacingAgent(nn.Module):
    """
    TQC (Truncated Quantile Critics) for racing with continuous + discrete actions.
    """
    
    def __init__(self, obs_dim, continuous_dim=1, discrete_dim=6, 
                 hidden_dim=64, n_quantiles=25, n_critics=5):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        
        # ===== ACTOR NETWORK =====
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Continuous actions (steering)
        self.continuous_mean = nn.Linear(hidden_dim, continuous_dim)
        self.continuous_log_std = nn.Linear(hidden_dim, continuous_dim)
        
        # Discrete actions (brake, nitro, rescue, drift, fire, acceleration)
        self.discrete_logits = nn.Linear(hidden_dim, discrete_dim)
        
        # ===== QUANTILE CRITICS =====
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + continuous_dim + discrete_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_quantiles)
            )
            for _ in range(n_critics)
        ])
        
    def prepare_to_send_actions(self,flat_action_tensor):
        # Convert to numpy if it's a tensor
        if isinstance(flat_action_tensor, torch.Tensor):
            flat_action = flat_action_tensor.cpu().numpy()
        else:
            flat_action = flat_action_tensor
        
       
        # Extract components
        # THE FLAT ACTION WILL CONTAIN {'acceleration','steer', 'brake', 'drift', 'nitro', 'rescue'}
        # FIRE WONT BE HERE THOUGHH
        #{'acceleration','steer', 'brake', 'drift', 'fire', 'nitro', 'rescue'}
        to_send = {
            'steer': flat_action[:,0:1],
            'acceleration': flat_action[:,1:2],
            'brake': flat_action[:,2:3],
            'drift': flat_action[:,3:4],
            'nitro': flat_action[:,4:5],
            'rescue': flat_action[:,5:6]
        }
        return to_send
        
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.
        
        Returns:
            continuous_action: (batch, continuous_dim)
            discrete_action: (batch, discrete_dim)
            log_prob: Log probability for training
        """
        
        with torch.no_grad():
            features = self.encoder(obs)
            
            # === CONTINUOUS ACTIONS ===
            mean = self.continuous_mean(features)
            log_std = self.continuous_log_std(features)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            
            if deterministic:
                continuous_action = torch.tanh(mean)
                log_prob_continuous = None
            else:
                # Reparameterization trick
                normal = torch.distributions.Normal(mean, std)
                x = normal.rsample()
                continuous_action = torch.tanh(x)
                
                # Log probability with tanh correction
                log_prob_continuous = normal.log_prob(x).sum(dim=-1, keepdim=True)
                log_prob_continuous -= torch.log(1 - continuous_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            # === DISCRETE ACTIONS ===
            discrete_logits = self.discrete_logits(features)
            
            if deterministic:
                discrete_action = (torch.sigmoid(discrete_logits) > 0.5).float()
                log_prob_discrete = None
            else:
                discrete_probs = torch.sigmoid(discrete_logits)
                discrete_action = torch.bernoulli(discrete_probs)
                
                # Log probability for Bernoulli
                log_prob_discrete = (
                    discrete_action * torch.log(discrete_probs + 1e-8) +
                    (1 - discrete_action) * torch.log(1 - discrete_probs + 1e-8)
                ).sum(dim=-1, keepdim=True)
            
            # Combine actions and log probs
            
            action = torch.cat([continuous_action, discrete_action], dim=-1)
            
            if deterministic:
                log_prob = None
            else:
                log_prob = log_prob_continuous + log_prob_discrete
            
            return action, log_prob
    
    def get_action_and_log_prob(self, obs):
        """
        Get action and log probability for training (with gradients).
        """
        features = self.encoder(obs)
        
        # === CONTINUOUS ACTIONS ===
        mean = self.continuous_mean(features)
        log_std = self.continuous_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        continuous_action = torch.tanh(x)
        
        # Log probability with tanh correction
        log_prob_continuous = normal.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob_continuous -= torch.log(1 - continuous_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        # === DISCRETE ACTIONS ===
        discrete_logits = self.discrete_logits(features)
        discrete_probs = torch.sigmoid(discrete_logits)
        discrete_action = torch.bernoulli(discrete_probs)
        
        # Log probability
        log_prob_discrete = (
            discrete_action * torch.log(discrete_probs + 1e-8) +
            (1 - discrete_action) * torch.log(1 - discrete_probs + 1e-8)
        ).sum(dim=-1, keepdim=True)
        
        # Combine
        action = torch.cat([continuous_action, discrete_action], dim=-1)
        log_prob = log_prob_continuous + log_prob_discrete
        
        return action, log_prob

    def get_action_deterministic(self, obs):
        """
        Get deterministic action for BC training (no sampling).
        Returns raw outputs before sampling.
        """
        features = self.encoder(obs)
        
        # === CONTINUOUS ACTIONS ===
        mean = self.continuous_mean(features)
        # Pour BC, on utilise directement la mean (pas de sampling)
        continuous_action = torch.tanh(mean)
        
        # === DISCRETE ACTIONS ===
        discrete_logits = self.discrete_logits(features)
        # Pour BC, on retourne les logits (pas les samples)
        # La loss BCEWithLogits s'occupera de la sigmoid
        
        # Combine
        # Pour les actions discrÃ¨tes, on garde les logits
        action = torch.cat([continuous_action, discrete_logits], dim=-1)
        
        return action 
        
    def get_quantile_values(self, obs, action):
        """
        Get quantile estimates from all critics.
        
        Returns:
            quantiles: (batch, n_critics, n_quantiles)
        """
        obs_action = torch.cat([obs, action], dim=-1)
        
        # Get quantiles from all critics
        quantiles = torch.stack([critic(obs_action) for critic in self.critics], dim=1)
        
        return quantiles

class Actor(Agent):
    """Computes probabilities over action"""
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.continuous_space = observation_space["continuous"]
        self.discrete_space = observation_space["discrete"]
        self.action_space = action_space
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.continuous_space.shape[0] + self.discrete_space.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_space.shape[0])
        )

    def forward(self, t: int):
        # Computes probabilities over actions
        obs = self.get(("env/obs", t))
        logits = self.model(obs)
        probas = torch.nn.functional.softmax(logits, dim=-1)
        self.set(("action", t), probas)


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        probas = self.get(("action", t))
        action = probas.argmax(-1)
        self.set(("action", t), action)


class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
