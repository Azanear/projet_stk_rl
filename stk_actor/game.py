"""
STK Manual Debug + Recorder

Controls:
- LEFT / RIGHT : steer (INVERTED)  -> LEFT = - , RIGHT = +
- UP           : accel
- DOWN         : brake
- LSHIFT       : drift
- P            : pause/resume
- R            : reset episode
- T            : start/stop recording (obs/actions/...)
- Q or ESC     : quit

Output:
- saves a .npz dataset with obs/actions/rewards/dones/infos
"""

import sys
import time
import gymnasium as gym
import numpy as np
import pygame

from pystk2_gymnasium import AgentSpec, ConstantSizedObservations, PolarObservations
from actors import MinimalEssentialObsWrapper, SkipCountdownWrapper, VariableTileVisitRewardWrapper
from actors import *

# =============================
# Config
# =============================
FPS = 10
DATASET_PATH = "YOOOOOO.npz"

# Manual smoothing
STEER_STEP = 0.12
STEER_RETURN = 0.12

IDLE_ACCEL = 0.15
rec_on = False
records = []      # buffer du chunk courant
chunk_id = 0      # compteur de fichiers (1,2,3...)


ferrari = TQCRacingAgent(56,2,1,256)
dir_path = sys.path[0]
filename = f"{dir_path}/pystk_actor.pth"
ferrari.load_state_dict(torch.load(filename, map_location=torch.device('cpu'))['agent'])


from pathlib import Path
import itertools

def chunk_path(base_path: str, chunk_id: int) -> str:
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{chunk_id:04d}{p.suffix}"))


def to_float(x):
    """Robust float conversion for scalars/arrays/tensors."""
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return 0.0


def action_to_flat(action_dict):
    """Store only the useful action fields as a flat vector."""
    steer = to_float(action_dict["steer"])
    accel = to_float(action_dict["acceleration"])
    brake = to_float(action_dict["brake"])
    drift = to_float(action_dict["drift"])
    nitro = to_float(action_dict.get("nitro", 0))
    rescue = to_float(action_dict.get("rescue", 0))
    fire = to_float(action_dict.get("fire", 0))
    return np.array([steer, accel, brake, drift, nitro, rescue, fire], dtype=np.float32)


def make_manual_action(steer, accel, brake, drift):
    """Build action dict expected by env (keep it consistent with your original)."""
    return {
        "steer": np.array([[steer]], dtype=np.float32),
        "acceleration": np.array([[accel]], dtype=np.float32),
        "brake": np.array([[brake]], dtype=np.float32),
        "drift": 0,
        "nitro": 1,
        "rescue": 0,
        "fire": 1,
    }


def save_dataset(path, records):
    """Save list of transitions to NPZ."""
    if len(records) == 0:
        print("⚠️ Nothing to save (0 recorded steps).")
        return

    obs_arr = np.stack([r["obs"] for r in records], axis=0)          # (N, obs_dim)
    act_arr = np.stack([r["action"] for r in records], axis=0)       # (N, act_dim)
    rew_arr = np.asarray([r["reward"] for r in records], dtype=np.float32)
    done_arr = np.asarray([r["done"] for r in records], dtype=np.bool_)

    # infos can be different shapes/types → store as object array
    info_arr = np.asarray([r["info"] for r in records], dtype=object)

    np.savez_compressed(
        path,
        obs=obs_arr,
        action=act_arr,
        reward=rew_arr,
        done=done_arr,
        info=info_arr,
        saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"✅ Saved dataset: {path}")
    print(f"   Steps: {len(records)} | obs shape: {obs_arr.shape} | action shape: {act_arr.shape}")


def main():
    # =============================
    # Pygame minimal HUD
    # =============================
    pygame.init()
    screen = pygame.display.set_mode((520, 180))
    pygame.display.set_caption("STK Manual + Recorder (T=REC)")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    def draw(hud_lines, rec_on, paused):
        screen.fill((15, 15, 15))
        y = 10
        for line in hud_lines:
            surf = font.render(line, True, (230, 230, 230))
            screen.blit(surf, (10, y))
            y += 26

        status = f"{'PAUSED' if paused else 'RUNNING'} | {'REC ON' if rec_on else 'REC OFF'}"
        color = (255, 80, 80) if rec_on else (160, 160, 160)
        surf = font.render(status, True, color)
        screen.blit(surf, (10, 140))
        pygame.display.flip()

    # =============================
    # Env
    # =============================
    env = gym.make(
        "supertuxkart/full-v0",
        agent=AgentSpec(use_ai=False),
        render_mode="human",
        num_kart=4,
        difficulty=2,
        max_episode_steps=1500,
    )
    env = SkipCountdownWrapper(env, skip_steps=30)
    env = NewRewardWrapper(env)
    env = ConstantSizedObservations(env)
    env = PolarObservations(env)
    env = Minimal56EssentialObsWrapper(env)

    obs, info = env.reset()
    print("Obs shape - ", obs["continuous"].shape)
    # =============================
    # State
    # =============================
    running = True
    paused = False

    # manual controls
    steer = 0.0
    accel = IDLE_ACCEL
    brake = 0.0
    drift = 0.0

    step = 0
    episode = 0
    ep_reward = 0.0

    # recording
    rec_on = False
    records = []
    chunk_id = 0

    print("\n=== STK MANUAL + RECORDER ===")
    print("LEFT/RIGHT: steer (inverted) | UP: accel | DOWN: brake | SHIFT: drift")
    print("T: toggle REC | P: pause | R: reset | Q/ESC: quit")
    print(f"Dataset output: {DATASET_PATH}\n")

    while running:
        # -------- events --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_p:
                    paused = not paused
                    print("⏸️ PAUSED" if paused else "▶️ RESUMED")

                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    step = 0
                    ep_reward = 0.0
                    episode += 1
                    print(f"🔄 Reset episode -> {episode}")

                elif event.key == pygame.K_t:
                    if rec_on:
                        # STOP -> save current chunk into a NEW file
                        out_path = chunk_path(DATASET_PATH, chunk_id)
                        print(f"⏹️ REC OFF -> saving {out_path}", flush=True)
                        save_dataset(out_path, records)
                        rec_on = False
                    else:
                        # START -> create new chunk buffer
                        chunk_id += 1
                        records = []
                        rec_on = True
                        out_path = chunk_path(DATASET_PATH, chunk_id)
                        print(f"⏺️ REC ON  -> recording to {out_path}", flush=True)
                        # -------- manual keys --------
        keys = pygame.key.get_pressed()

        # Steering (INVERTED)
        # LEFT => steer negative, RIGHT => steer positive
        if keys[pygame.K_LEFT]:
            steer = max(steer - STEER_STEP, -1.0)
        elif keys[pygame.K_RIGHT]:
            steer = min(steer + STEER_STEP, 1.0)
        else:
            steer *= STEER_RETURN

        # Accel/Brake
        if keys[pygame.K_UP]:
            accel = 1.0
            brake = 0.0
        elif keys[pygame.K_DOWN]:
            accel = 0.0
            brake = 1.0
        else:
            accel = IDLE_ACCEL
            brake = 0.0

        drift = 1.0 if keys[pygame.K_d] else 0.0

        # -------- step env -----f---
        if not paused:
            action_to_send = make_manual_action(steer, accel, brake, drift)
            action = env.action_space.sample() 


            # keep obs before step if you want (s,a,r,s') dataset style:
            obs_before = np.asarray(obs["continuous"], dtype=np.float32).copy()
       
            ff_actions,_ = ferrari.get_action(torch.tensor(obs_before).unsqueeze(0))

            action_ferrari = ferrari.prepare_to_send_actions(ff_actions)

            #action_to_send['acceleration'] = 0.2
            obs, reward, terminated, truncated, info = env.step(action_ferrari)

            done = bool(terminated or truncated)

            step += 1
            ep_reward += float(reward)

            if rec_on:
                records.append(
                    {
                        "obs": obs_before,
                        "action": action_to_flat(action_to_send),
                        "reward": float(reward),
                        "done": done,
                        "info": dict(info) if isinstance(info, dict) else info,
                    }
                )

            if done:
                print(f"🏁 Episode {episode} finished | steps={step} | reward={ep_reward:.2f}")
                obs, info = env.reset()
                step = 0
                ep_reward = 0.0
                episode += 1

        # -------- HUD --------
        hud = [
            f"Episode: {episode}  Step: {step}  EpReward: {ep_reward:.2f}",
            f"Steer: {steer:+.2f}  Accel: {accel:.2f}  Brake: {brake:.2f}  Drift: {drift:.0f}",
            f"Recorded steps: {len(records)}",
            "T=REC  P=Pause  R=Reset  Q/ESC=Quit",
        ]
        draw(hud, rec_on, paused)
        clock.tick(FPS)

    # -------- exit: save once more --------
    print("\nExiting...")
    save_dataset(DATASET_PATH, records)
    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
