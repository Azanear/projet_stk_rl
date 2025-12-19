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
            'nitro': float(action[4]),
            'rescue': float(action[5]),
            'fire': 1
        }
    
import numpy as np

def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def angle_xz(v3) -> float:
    # v3: (x,y,z) in kart frame
    x = float(v3[0])
    z = float(v3[2])
    return np.arctan2(x, z)  # x left, z forward


class MinimalEssentialObsWrapper(gym.ObservationWrapper):
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
                shape=(34,),
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


        ps = obs["paths_start"]
        pe = obs["paths_end"]
        seg0 = np.array(pe[0], dtype=np.float32) - np.array(ps[0], dtype=np.float32)
        seg0[1] = 0.0
        track_angle0 = angle_xz(seg0)

        heading_error = wrap_to_pi(track_angle0)   # kart heading = 0 dans son repère
        continuous_features.append(heading_error / np.pi)  # normalisé [-1,1] # UN FEATURE EN + LA

        k = 2  # lookahead (1 ou 2 marche bien)
        idx = min(k, len(ps)-1)

        segk = np.array(pe[idx], dtype=np.float32) - np.array(ps[idx], dtype=np.float32)
        segk[1] = 0.0
        track_anglek = angle_xz(segk)

        curvature = wrap_to_pi(track_anglek - track_angle0)
        continuous_features.append(curvature / np.pi)  # [-1,1]

        seg0_len = float(np.linalg.norm(seg0[[0,2]]) + 1e-6)
        curv_per_m = curvature / seg0_len
        continuous_features.append(np.clip(curv_per_m, -1.0, 1.0))


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
   
        
        return {
            'continuous': np.array(continuous_features, dtype=np.float32)
        }
        
    def observation(self, observation):
        return self._process_obs(observation)


class SkipCountdownWrapper(gym.ObservationWrapper):
    """Skip les N premiers steps où la course n'a pas commencé."""
    
    def __init__(self, env, skip_steps=11):
        super().__init__(env)
        self.skip_steps = skip_steps
    
    def observation(self, obs):
        # Retourne l'obs tel quel, sans modification
        return obs
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Skip les premiers steps avec action nulle
        for _ in range(self.skip_steps):
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

class NormalizedProgressRewardWrapper(gym.Wrapper):
    def __init__(self, env, progress_scale=200.0):
        super().__init__(env)
        self.prev_distance = 0.0
        self.total_track_length = 1.0
        self.progress_scale = progress_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # ... calcul track length ...
        self.prev_distance = float(obs["distance_down_track"][0])
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        current_distance = float(obs["distance_down_track"][0])
        progress_meters = current_distance - self.prev_distance

        # === POSITION SUR LA PISTE ===
        center_dist = abs(float(obs['center_path_distance'][0]))
        pw = obs['paths_width'][0]
        path_width = float(pw[0]) if hasattr(pw, '__len__') else float(pw)
        half_w = path_width / 2
        
        pos_ratio = center_dist / half_w  # 0 = centre, 1 = bord, >1 = hors piste

        # === REWARD LOGIC ===
        if pos_ratio > 1.2:  # HORS PISTE
            reward = 0.0  # Pas de progress reward
            # PAS de grosse pénalité, juste 0
        else:
            # Progress normal
            reward = (progress_meters / self.total_track_length) * self.progress_scale
            
            # PETIT bonus si bien centré (pos_ratio < 0.3)
            if pos_ratio < 0.3:
                reward += 0.02  # Mini bonus centre
        
        # Pénalité immobilité
        speed = np.linalg.norm(obs['velocity'])
        if speed < 1.0:
            reward -= 0.02

        # Bonus finish
        if terminated and not truncated:
            print("I SWEAR WE FINISHED THE FUCKIN")
            reward += 20

        self.prev_distance = current_distance
        reward = np.clip(reward, -1.0, 5.0)

        return obs, reward, terminated, truncated, info


class SmartNitroBonusWrapper(gym.Wrapper):

    """
    DONNE UN BONUS si l'agent utilise sa Nitro INTELLIGEMMENT :
    1. Il a de l'énergie (> seuil)
    2. Il est bien placé sur la piste (safe zone)
    3. Il appuie sur Nitro
    
    C'est la méthode "Carotte" : on récompense l'audace.
    """
    def __init__(self, env, energy_threshold=0.4, safe_zone_ratio=0.2, bonus=0.1):
        super().__init__(env)
        self.energy_threshold = energy_threshold
        self.safe_zone_ratio = safe_zone_ratio
        self.bonus = bonus # C'est un bonus positif (+)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- Récupération des données ---
        energies = obs['energy']
        nitro_actions = action['nitro']
        center_dists = obs['center_path_distance']
        widths = obs['paths_width']

        
        # Normalisation en listes
        
        energies = [energies]
        nitro_actions = [nitro_actions]
        center_dists = [center_dists]
        widths = [widths]
        reward = reward

        for i in range(1):
            e = float(energies[i])
            n = float(nitro_actions[i])
            dist = abs(float(center_dists[i]))
            
            # Gestion largeur
            raw_w = widths[i][0]
            w = float(raw_w)
            
            # Ratio de sécurité (0 = centre, 1 = bord)
            pos_ratio = dist / max(w, 0.1)
            
            # LA LOGIQUE BONUS (+)
            # Si (J'ai du jus) ET (Je suis au centre) ET (J'APPUIE sur Nitro)
            if e > self.energy_threshold and pos_ratio < self.safe_zone_ratio and n > 0.5:
                # BRAVO ! Tu as compris le jeu !
                reward += self.bonus

        return obs, reward, terminated, truncated, info


class RacingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames_stuck = 0
        self.min_speed_threshold = 1.0
        self.max_stuck_frames = 100 # Si bloqué > 100 frames -> Mort

    def reset(self, **kwargs):
        self.frames_stuck = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Récupération des infos brutes
        stucked = False
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- 1. CALCUL DE LA PHYSIQUE ---
        # On s'assure d'avoir la vitesse brute (pas normalisée)
        # Si 'velocity' est un vecteur 3D
        velocity = obs['velocity']
        speed = np.linalg.norm(velocity) 
        
        # --- 2. LA NOUVELLE REWARD (SPEED & TIME) ---
        # A. La Taxe de Temps (Time Penalty)
        # Force l'IA à trouver le chemin le plus court/rapide
        reward = -0.05 
        
        # B. Le Bonus de Vitesse (Speed Bonus)
        # Encourage à garder le pied au plancher
        # Hypothèse: Speed max ~25.0 -> 25 * 0.02 = +0.5 pts/frame
        reward += speed * 0.02
        
        # C. Bonus Drift (Si dispo dans les obs brutes)
        if 'skeed_factor' in obs:
             skeed = float(obs['skeed_factor'])
             # Si on glisse ET qu'on va vite = Style & Nitro
             if skeed > 0.5 and speed > 15.0:
                 
                 reward += 0.1

        # --- 3. LE KILL SWITCH (Vital pour le training) ---
        # Si le kart n'avance pas, on arrête tout de suite pour ne pas polluer le buffer
        if speed < self.min_speed_threshold:
            self.frames_stuck += 1
        else:
            self.frames_stuck = 0
            
        if self.frames_stuck > self.max_stuck_frames:
            # print("💀 STUCK! Killing episode.")
            reward -= 2.0 # Punition pour l'échec
            terminated = True
            stucked = True
            
        # --- 4. BONUS FINISH LINE ---
        # On garde ton super bonus si elle finit la course
        if terminated and not truncated and not stucked: # Si fini naturellement (pas par le temps)
             # On vérifie que c'est pas le kill switch qui a trigger le terminated
            print("🏁 I SWEAR WE FINISHED THE FUCKIN RACE!")
            reward += 20.0

        return obs, reward, terminated, truncated, info


class TQCRacingAgent(Agent):
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
        if flat_action.ndim == 1:
            flat_action = flat_action[np.newaxis, :]  # Add batch dim if needed
        to_send = {
            'steer': flat_action[:,0:1],
            'acceleration': flat_action[:,1:2],
            'brake': flat_action[:,2:3],
            'drift': flat_action[:,3:4],
            'nitro': flat_action[:,4:5],
            'rescue': flat_action[:,5:6],
            'fire': 1
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
    
    def forward(self, t: int):
        
        observation = self.get(("env/env_obs/continuous", t))
        action, _ = self.get_action(observation,True)
        #print('ACTIONS = ',action)
        #action[:, [0, 1]] = action[:, [1, 0]]
        
        if observation[0][-1] == 1.0:
            action[0][5] = 1.0
        
       
        self.set(("action", t), action)
        

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