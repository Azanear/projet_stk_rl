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
    
# Dans actors.py
class ActionToDictWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    def action(self, action):
        # Sécurité : passage sur CPU et conversion Numpy si c'est un tenseur
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        
        # Gestion du batch dimension éventuelle (si action arrive en [1, 7])
        if len(action.shape) > 1:
            action = action[0]

        return {
            'steer': float(action[0]),
            'acceleration': float(action[1]),
            'brake': float(action[2]),
            'drift': float(action[3]),
            'nitro': 1,
            'rescue': 0,
            'fire': 1
        }
    
import numpy as np



import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium import ConstantSizedObservations,PolarObservations


import numpy as np


import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium import ConstantSizedObservations,PolarObservations


import numpy as np
 
def angle_to_sincos(angle_rad: float):
    """
    angle_rad: angle in radians
    returns: (sin(angle), cos(angle))
    """
    return float(np.sin(angle_rad)), float(np.cos(angle_rad))

def target_ahead_features(karts_position, max_dist=15.0, fov_deg=20.0, min_dist=0.5):
    
    fov = np.deg2rad(fov_deg)

    best_dist = None
    best_yaw = 0.0

    yaw, pitch, dist = karts_position

    yaw = float(yaw)
    dist = float(dist)

    if dist < min_dist or dist > max_dist:
        return 0.0, 0.0, 0.0, 0.0  # exists, dist, sin, cos

    if abs(yaw) > fov:
        return 0.0, 0.0, 0.0, 0.0

    if best_dist is None or dist < best_dist:
        best_dist = dist
        best_yaw = yaw

    if best_dist is None:
        return 0.0, 0.0, 0.0, 0.0

    exists = 1.0
    dist_norm = float(np.clip(best_dist / max_dist, 0.0, 1.0))
    sin_yaw, cos_yaw = angle_to_sincos(best_yaw)
    
    return exists, dist_norm, sin_yaw, cos_yaw

def item_value(item_type):
    """
    +1 = BON (nitro) → aller vers
    -1 = MAUVAIS (banana, bubble_gum, easter_egg) → éviter
     0 = neutre (bonus_box, none)
    """
    item_type = int(item_type)
    if item_type in [3, 4]:  # nitro_small, nitro_big
        return 1.0   # BON → priorité!
    elif item_type in [2, 5, 6]:  # banana, bubble_gum, easter_egg
        return -1.0  # MAUVAIS → éviter absolument
    else:  # 0=none, 1=bonus_box
        return 0.0   # Neutre
    
def nearest_items_features(items_position, items_type, max_dist=20.0, n_items=3):
    """
    items_position: liste de (horizontal_angle, vertical_angle, distance) en POLAR
    items_type: liste des types d'items
    """
    features = []
    
    items_with_dist = []
    for i in range(3):
        pos = items_position[i]
        itype = int(items_type[i])
        
        # POLAR: (horizontal_angle, vertical_angle, distance)
        h_angle = float(pos[0])  # Angle horizontal (yaw)
        v_angle = float(pos[1])  # Angle vertical (pitch) - on ignore
        dist = float(pos[2])     # Distance
        
        # Ignorer si trop loin ou type neutre
        if dist > max_dist or itype == 0:
            continue
        
        items_with_dist.append({
            'dist': dist,
            'yaw': h_angle,
            'type': itype,
            'value': item_value(itype)
        })
    
    # Trier: DANGERS d'abord (-1), puis par distance
    items_with_dist.sort(key=lambda x: (x['value'], x['dist']))
    
    # Prendre les N premiers
    for i in range(n_items):
        if i < len(items_with_dist):
            item = items_with_dist[i]
            exists = 1.0
            dist_norm = float(np.clip(item['dist'] / max_dist, 0.0, 1.0))
            sin_yaw = float(np.sin(item['yaw']))
            cos_yaw = float(np.cos(item['yaw']))
            value = item['value']
        else:
            # Pas d'item → valeurs neutres
            exists = 0.0
            dist_norm = 1.0
            sin_yaw = 0.0
            cos_yaw = 1.0
            value = 0.0
        
        features.extend([exists, dist_norm, sin_yaw, cos_yaw, value])
    
    return features



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
                shape=(28,),
                dtype=np.float32
            )
        })
    
    def reset(self, **kwargs):
        obs,info = self.env.reset(**kwargs)
        processed_obs = self._process_obs(obs)
       
        self.observation_space = gym.spaces.Dict({
            'continuous': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(28,),
                dtype=np.float32
            )
        })

        info['distance_down_track'] = obs['distance_down_track']
        info['off_track'] = processed_obs['continuous'][-2]
        return processed_obs,info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        processed_obs = self._process_obs(obs)
        info['distance_down_track'] = obs['distance_down_track']
        info['off_track'] = processed_obs['continuous'][-2]
        return processed_obs, reward, terminated, truncated, info
    
    def normalize_dist(self,dist, max_dist):
        """
        Normalize a distance into [0, 1] with clipping.
        """
        return np.clip(dist / max_dist, 0.0, 1.0)
    
    def _process_obs(self, obs):
        # IT IS POLAR OBSERVATIONS 
        continuous_features = []
         
        vel_yaw   = float(obs['velocity'][0])   # angle horizontal (rad)
        speed     = float(obs['velocity'][2]) / 40.  # ✅ SPEED (m/s)
        continuous_features.extend([vel_yaw,speed])
        
        cp_yaw,_,dist = obs["center_path"]
        cp_sin, cp_cos = angle_to_sincos(cp_yaw)
        
        dist_norm = self.normalize_dist(dist, obs['paths_width'][0][0])
        continuous_features.extend([cp_sin, cp_cos,dist_norm])
       
        exists,dist_normed,sin_yaw,cos_yaw = target_ahead_features(obs['karts_position'][0], max_dist=15.0, fov_deg=20.0, min_dist=0.5)
        continuous_features.extend([exists,dist_normed,sin_yaw,cos_yaw])

        itemsss = nearest_items_features(obs['items_position'], obs['items_type'], max_dist=50.0, n_items=3)
        continuous_features.extend(itemsss)
      
        continuous_features.extend([float(obs['max_steer_angle'][0]),float(obs['energy'][0]) / 10.0])
        
        curr_w_raw = obs['paths_width'][0][0]
        curr_w = float(curr_w_raw)
        off_track = 1.0 if abs(float(obs['center_path_distance'][0])) > (curr_w / 2) + 3.5 else 0.0


        continuous_features.append(off_track)

        start_horizontal_yaw,_,start_dist = obs['paths_start'][0]
        end_horizontal_yaw,_,end_dist = obs['paths_end'][0]

        continuous_features.append((end_dist - start_dist) / (obs['paths_distance'][0][-1] - obs['paths_distance'][0][0]) )
     
        
        return {
            'continuous': np.array(continuous_features, dtype=np.float32)
        }
    


class Minimal56EssentialObsWrapper(gym.Wrapper):
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
                shape=(28,),
                dtype=np.float32
            )
        })

        self.prev_ddt = 0.0
        self.prev_cp_yaw = 0.0
        self.prev_steer = 0.0

    def reset(self, **kwargs):
        obs,info = self.env.reset(**kwargs)

        self.prev_ddt = float(obs["distance_down_track"][0])
        self.prev_cp_yaw = float(obs["center_path"][0])
        self.prev_steer = 0.0

        processed_obs = self._process_obs(obs)
      

        processed_obs['continuous'] = np.concatenate([processed_obs['continuous'], 
                                                      np.array([self.prev_steer], dtype=np.float32)])
        
        processed_obs['continuous'] = np.concatenate([processed_obs['continuous'], np.array([info['nb_time_low_speed'] / 50], dtype=np.float32)])
        self.observation_space = gym.spaces.Dict({
            'continuous': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(28,),
                dtype=np.float32
            )
        })

        info['distance_down_track'] = obs['distance_down_track']
        info['off_track'] = processed_obs['continuous'][-3]
        return processed_obs,info
    
    def tile_progress_smooth(self, obs):
        pd = np.asarray(obs["paths_distance"], dtype=np.float32)
        ddt = float(obs["distance_down_track"][0])

        starts = pd[:, 0]
        ends   = pd[:, 1]

        idx = int(np.searchsorted(ends, ddt, side="right"))
        idx = np.clip(idx, 0, len(ends) - 1)

        tile_len = max(ends[idx] - starts[idx], 1e-6)
        frac_in_tile = (ddt - starts[idx]) / tile_len
        frac_in_tile = np.clip(frac_in_tile, 0.0, 1.0)

        return (idx + frac_in_tile) / max(len(ends), 1)

    def step(self, action):
        self.prev_steer = action['steer'] if isinstance(action, dict) else action[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        processed_obs = self._process_obs(obs)

        self.prev_steer = float(np.asarray(action["steer"]).reshape(-1)[0])
        processed_obs['continuous'] = np.concatenate([processed_obs['continuous'], np.array([self.prev_steer], dtype=np.float32)])
        processed_obs['continuous'] = np.concatenate([processed_obs['continuous'], np.array([info['nb_time_low_speed'] / 50], dtype=np.float32)])

        info['distance_down_track'] = obs['distance_down_track']
        info['off_track'] = processed_obs['continuous'][-3]

        return processed_obs, reward, terminated, truncated, info
    
    def normalize_dist(self,dist, max_dist):
        """
        Normalize a distance into [0, 1] with clipping.
        """
        return np.clip(dist / max_dist, 0.0, 1.0)
    
    def _process_obs(self, obs):
        # IT IS POLAR OBSERVATIONS 
        continuous_features = []

        # ---- progress local (delta) ----
        front = obs["front"]
        continuous_features.extend([float(front[0]), float(front[1]), float(front[2])])
        ddt = float(obs["distance_down_track"][0])
        progress = ddt - self.prev_ddt
        self.prev_ddt = ddt
        progress_norm = np.clip(progress / 1.0, -1.0, 1.0)

        track_progress = self.tile_progress_smooth(obs)

        continuous_features.extend([
            progress_norm,          # local speed along track
            track_progress,         # absolute position [0,1]
            1.0 - track_progress    # remaining track
        ])

        cp_yaw = float(obs["center_path"][0])
        cp_yaw_rate = cp_yaw - self.prev_cp_yaw
        # wrap to [-pi, pi]
        cp_yaw_rate = (cp_yaw_rate + np.pi) % (2*np.pi) - np.pi
        self.prev_cp_yaw = cp_yaw

        continuous_features.extend([progress_norm, cp_yaw_rate])

        width = float(obs["paths_width"][0][0])
        half = max(width * 0.5, 1e-6)
        cp_dist = float(obs["center_path_distance"][0])

        lat = np.clip(cp_dist / half, -1.0, 1.0)
        margin_norm = np.clip((half - abs(cp_dist)) / half, -1.0, 1.0)
        continuous_features.extend([lat, margin_norm])

        starts = np.asarray(obs["paths_start"], dtype=np.float32)  # (K, 3) maybe
        ends   = np.asarray(obs["paths_end"], dtype=np.float32)

        K = 3
        prev_yaw = None

        for i in range(K):
            yaw_i = float(starts[i][0])
            sin_i, cos_i = angle_to_sincos(yaw_i)

            rank = i / max(K - 1, 1)

            continuous_features.extend([sin_i, cos_i, rank])

            if prev_yaw is None:
                continuous_features.extend([0.0, 1.0])
            else:
                dy = yaw_i - prev_yaw
                dy = (dy + np.pi) % (2*np.pi) - np.pi
                sdy, cdy = angle_to_sincos(dy)
                continuous_features.extend([sdy, cdy])

            prev_yaw = yaw_i

       
        vel_yaw   = float(obs['velocity'][0])   # angle horizontal (rad)
        speed     = float(obs['velocity'][2]) / 40.  # ✅ SPEED (m/s)
        continuous_features.extend([vel_yaw,speed])
        
        cp_yaw,_,dist = obs["center_path"]
        cp_sin, cp_cos = angle_to_sincos(cp_yaw)
        
        dist_norm = self.normalize_dist(dist, obs['paths_width'][0][0])
        continuous_features.extend([cp_sin, cp_cos,dist_norm])
       
        exists,dist_normed,sin_yaw,cos_yaw = target_ahead_features(obs['karts_position'][0], max_dist=15.0, fov_deg=20.0, min_dist=0.5)
        continuous_features.extend([exists,dist_normed,sin_yaw,cos_yaw])

        itemsss = nearest_items_features(obs['items_position'], obs['items_type'], max_dist=50.0, n_items=3)
        continuous_features.extend(itemsss)
      
        continuous_features.extend([float(obs['max_steer_angle'][0])])
        
        
        curr_w_raw = obs['paths_width'][0][0]
        curr_w = float(curr_w_raw)
        off_track = 1.0 if abs(float(obs['center_path_distance'][0])) > (curr_w / 2) + 3.5 else 0.0

        continuous_features.append(off_track)

        start_horizontal_yaw,_,start_dist = obs['paths_start'][0]
        end_horizontal_yaw,_,end_dist = obs['paths_end'][0]

        continuous_features.append((end_dist - start_dist) / (obs['paths_distance'][0][-1] - obs['paths_distance'][0][0]) )
        continuous_features.extend([start_horizontal_yaw, end_horizontal_yaw])
        
        return {
            'continuous': np.array(continuous_features, dtype=np.float32)
        }
    



class VariableTileVisitRewardWrapper(gym.Wrapper):
    def __init__(self, env, offtrack_ratio=1.2):
        super().__init__(env)
        self.offtrack_ratio = float(offtrack_ratio)

        self.nb_tile = 1
        self.tile_starts = None
        self.tile_ends = None

        self.visited_idx = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        pd = np.asarray(obs["paths_distance"], dtype=np.float64)  # shape (N, 2)
        self.tile_starts = pd[:, 0].copy()
        self.tile_ends   = pd[:, 1].copy()
        self.nb_tile = int(pd.shape[0])

        self.visited_idx = set()
        return obs, info

    def _pos_ratio(self, obs) -> float:
        curr_w = float(obs["paths_width"][0][0])
        off_track = 1.0 if abs(float(obs["center_path_distance"][0])) > (curr_w / 2) + 1.0 else 0.0
        return off_track

    def _tile_index_from_s(self, s: float) -> int:
        # find closest start (robust to float noise)
        return int(np.argmin(np.abs(self.tile_starts - s)))

    def _tile_index_from_ddt(self, ddt: float) -> int:
        # tile_ends is increasing along the track
        idx = int(np.searchsorted(self.tile_ends, ddt, side="right"))
        return min(max(idx, 0), self.nb_tile - 1)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = -0.1

        # Option A: use s (your method)
        #s = float(obs["paths_distance"][0][0])
        #tile_idx = self._tile_index_from_s(s)

        # Option B: better: use distance_down_track (more stable)
        ddt = float(obs["distance_down_track"][0])
        tile_idx = self._tile_index_from_ddt(ddt)
        
        if tile_idx not in self.visited_idx:
            self.visited_idx.add(tile_idx)
            reward += 1000.0 / float(self.nb_tile)

        """
        off_track = self._pos_ratio(obs)
        if off_track:
            reward -= 100
            terminated = True
        """
        return obs, float(reward), terminated, truncated, info

   

class NewRewardWrapper(gym.Wrapper):
    def __init__(self, env, offtrack_ratio=1.2):
        super().__init__(env)
        self.offtrack_ratio = float(offtrack_ratio)

        self.nb_tile = 1
        self.tile_starts = None
        self.tile_ends = None

        self.visited_idx = set()
        self.nb_time_low_speed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        pd = np.asarray(obs["paths_distance"], dtype=np.float64)  # shape (N, 2)
        self.tile_starts = pd[:, 0].copy()
        self.tile_ends   = pd[:, 1].copy()
        self.nb_tile = int(pd.shape[0])
        self.nb_time_low_speed = 0

        self.visited_idx = set()

        info['nb_time_low_speed'] = self.nb_time_low_speed
        return obs, info

    def _pos_ratio(self, obs) -> float:
        curr_w = float(obs["paths_width"][0][0])
        off_track = 1.0 if abs(float(obs["center_path_distance"][0])) > (curr_w / 2) + 10. else 0.0
        return off_track

    def _tile_index_from_s(self, s: float) -> int:
        # find closest start (robust to float noise)
        return int(np.argmin(np.abs(self.tile_starts - s)))

    def _tile_index_from_ddt(self, ddt: float) -> int:
        # tile_ends is increasing along the track
        idx = int(np.searchsorted(self.tile_ends, ddt, side="right"))
        return min(max(idx, 0), self.nb_tile - 1)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        
        reward = -0.1

        # Option A: use s (your method)
        #s = float(obs["paths_distance"][0][0])
        #tile_idx = self._tile_index_from_s(s)
        if np.linalg.norm(obs['velocity']) < 1.0:
            self.nb_time_low_speed += 1
            
        else:
            self.nb_time_low_speed = 0

       
        # Option B: better: use distance_down_track (more stable)
        ddt = float(obs["distance_down_track"][0])
        tile_idx = self._tile_index_from_ddt(ddt)
        
        if tile_idx not in self.visited_idx and tile_idx > 0:
            self.visited_idx.add(tile_idx)
            reward += 1000.0 / float(self.nb_tile)

        if terminated:
            reward += 100 
        if self.nb_time_low_speed > 50:
            reward = -100
            terminated = True

        info['nb_time_low_speed'] = self.nb_time_low_speed
        return obs, float(reward), terminated, truncated, info





class SkipCountdownWrapper(gym.Wrapper):
    """Skip les N premiers steps où la course n'a pas commencé."""
    
    def __init__(self, env, skip_steps=10):
        super().__init__(env)
        self.skip_steps = skip_steps
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Skip les premiers steps avec action nulle
        for _ in range(self.skip_steps):
            # Action neutre pendant le countdown
            action = self.env.action_space.sample()
            to_send = {
                'acceleration': np.zeros_like(action['acceleration']),
                'steer': np.zeros_like(action['acceleration']),
                'brake': np.zeros_like(action['acceleration']),
                'drift': np.zeros_like(action['acceleration']),
                'fire':  np.zeros_like(action['acceleration']),
                'nitro': np.zeros_like(action['acceleration']),
                'rescue': np.zeros_like(action['acceleration'])
            }
           
            obs, _, terminated, truncated, info = self.env.step(to_send)
            
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
                break
        
        return obs, info




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
            'drift': np.zeros((len(flat_action),1)),
            'fire': np.ones((len(flat_action),1)),
            'nitro': np.ones((len(flat_action),1)),
            'rescue': np.zeros((len(flat_action),1))
        }
        return to_send
        
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            features = self.encoder(obs)

            mean = self.continuous_mean(features)          # (B,2)
            log_std = self.continuous_log_std(features)    # (B,2)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            # --- sample pre-squash ---
            if deterministic:
                u = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                u = dist.rsample()

            # split dims
            u_steer = u[:, :1]
            u_accel = u[:, 1:2]

            # squash per-dim
            steer = torch.tanh(u_steer)          # [-1,1]
            accel = torch.sigmoid(u_accel)       # [0,1]
            continuous_action = torch.cat([steer, accel], dim=1)

            # --- log prob for continuous ---
            if deterministic:
                log_prob_continuous = None
            else:
                # base log prob in u-space
                logp_u = dist.log_prob(u)  # (B,2)

                # tanh correction for steer
                # steer = tanh(u_steer) => d/du = 1 - steer^2
                corr_steer = torch.log(1 - steer.pow(2) + 1e-6)

                # sigmoid correction for accel
                # accel = sigmoid(u_accel) => d/du = accel*(1-accel)
                corr_accel = torch.log(accel * (1 - accel) + 1e-6)

                # total correction (sum over dims)
                log_prob_continuous = (logp_u[:, :1] - corr_steer) + (logp_u[:, 1:2] - corr_accel)
                # shape (B,1)

            # --- discrete part ---
            discrete_logits = self.discrete_logits(features)

            if deterministic:
                discrete_action = (torch.sigmoid(discrete_logits) > 0.5).float()
                log_prob_discrete = None
            else:
                probs = torch.sigmoid(discrete_logits)
                discrete_action = torch.bernoulli(probs)
                log_prob_discrete = (
                    discrete_action * torch.log(probs + 1e-8) +
                    (1 - discrete_action) * torch.log(1 - probs + 1e-8)
                ).sum(dim=-1, keepdim=True)

            action = torch.cat([continuous_action, discrete_action], dim=-1)

            if deterministic:
                log_prob = None
            else:
                log_prob = log_prob_continuous + log_prob_discrete

            return action, log_prob
    
    def get_action_and_log_prob(self, obs):
        """
        Get action and log probability for training (with gradients).
        steer: tanh -> [-1, 1]
        accel: sigmoid -> [0, 1]
        """
        features = self.encoder(obs)

        # === CONTINUOUS ACTIONS ===
        mean = self.continuous_mean(features)          # (B, 2)
        log_std = self.continuous_log_std(features)    # (B, 2)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()  # pre-squash (B, 2)

        u_steer = u[:, :1]
        u_accel = u[:, 1:2]

        steer = torch.tanh(u_steer)        # [-1,1]
        accel = torch.sigmoid(u_accel)     # [0,1]
        continuous_action = torch.cat([steer, accel], dim=1)

        # base log prob in u-space
        logp_u = normal.log_prob(u)  # (B,2)

        # Jacobian corrections
        corr_steer = torch.log(1.0 - steer.pow(2) + 1e-6)                 # tanh'
        corr_accel = torch.log(accel * (1.0 - accel) + 1e-6)              # sigmoid'

        # log prob in action space (sum dims, keepdim)
        log_prob_continuous = (logp_u[:, :1] - corr_steer) + (logp_u[:, 1:2] - corr_accel)  # (B,1)

        # === DISCRETE ACTIONS ===
        discrete_logits = self.discrete_logits(features)
        discrete_probs = torch.sigmoid(discrete_logits)
        discrete_action = torch.bernoulli(discrete_probs)

        log_prob_discrete = (
            discrete_action * torch.log(discrete_probs + 1e-8) +
            (1.0 - discrete_action) * torch.log(1.0 - discrete_probs + 1e-8)
        ).sum(dim=-1, keepdim=True)

        # Combine
        action = torch.cat([continuous_action, discrete_action], dim=-1)
        log_prob = log_prob_continuous + log_prob_discrete

        return action, log_prob

        
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

    def forward(self, t: int):
        observation = self.get(("env/env_obs/continuous", t))
       
        action, _ = self.get_action(observation)
        self.set(("action", t), action)
        