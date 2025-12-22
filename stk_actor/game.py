import sys
import time
import gymnasium as gym
import numpy as np
import pygame
import torch  # ✅ manquait

from pystk2_gymnasium import AgentSpec, ConstantSizedObservations, PolarObservations
from actors import MinimalEssentialObsWrapper, SkipCountdownWrapper, VariableTileVisitRewardWrapper
from actors import *

# =============================
# Config
# =============================
FPS = 10
DATASET_PATH = "HOLDON.npz"

# Manual smoothing
STEER_STEP = 0.12
STEER_RETURN = 0.12
IDLE_ACCEL = 0.15

from pathlib import Path

def chunk_path(base_path: str, chunk_id: int) -> str:
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{chunk_id:04d}{p.suffix}"))

def to_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return 0.0

def action_to_flat(action_dict):
    steer = to_float(action_dict["steer"])
    accel = to_float(action_dict["acceleration"])
    brake = to_float(action_dict["brake"])
    drift = to_float(action_dict["drift"])
    nitro = to_float(action_dict.get("nitro", 0))
    rescue = to_float(action_dict.get("rescue", 0))
    fire = to_float(action_dict.get("fire", 0))
    return np.array([steer, accel, brake, drift, nitro, rescue, fire], dtype=np.float32)

def make_manual_action(steer, accel, brake, drift):
    return {
        "steer": np.array([[steer]], dtype=np.float32),
        "acceleration": np.array([[accel]], dtype=np.float32),
        "brake": np.array([[brake]], dtype=np.float32),
        "drift": np.array([[drift]], dtype=np.float32),
        "nitro": np.array([[1.0]], dtype=np.float32),
        "rescue": np.array([[0.0]], dtype=np.float32),
        "fire": np.array([[1.0]], dtype=np.float32),
    }

def save_dataset(path, records):
    if len(records) == 0:
        print("⚠️ Nothing to save (0 recorded steps).")
        return

    obs_arr = np.stack([r["obs"] for r in records], axis=0)
    act_arr = np.stack([r["action"] for r in records], axis=0)
    rew_arr = np.asarray([r["reward"] for r in records], dtype=np.float32)
    done_arr = np.asarray([r["done"] for r in records], dtype=np.bool_)
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

# =============================
# Ferrari (AI)
# =============================
ferrari = TQCRacingAgent(56, 2, 1, 256)
filename = "/Users/abdulhamid/Documents/projet_stk_rl/stk_actor/tqc_LAST_DAGGER_300000.pt"
ckpt = torch.load(filename, map_location=torch.device("cpu"))
ferrari.load_state_dict(ckpt["agent"])
ferrari.eval()

def get_ai_action(obs_continuous: np.ndarray):
    """
    Retourne une action dict STK depuis l'IA.
    Adapte si ton agent expose get_action/prepare_to_send_actions.
    """
    with torch.no_grad():
        x = torch.tensor(obs_continuous, dtype=torch.float32).unsqueeze(0)

        # --- OPTION A: tu as get_action() + prepare_to_send_actions() ---
        if hasattr(ferrari, "get_action") and hasattr(ferrari, "prepare_to_send_actions"):
            ff_actions, _ = ferrari.get_action(x)
            return ferrari.prepare_to_send_actions(ff_actions)

        # --- OPTION B: tu as juste forward qui renvoie déjà les actions ---
        if callable(ferrari):
            out = ferrari(x)
            # si out est déjà un dict action → retourne
            if isinstance(out, dict):
                return out

    # fallback (au cas où)
    return make_manual_action(0.0, IDLE_ACCEL, 0.0, 0.0)

def main():
    pygame.init()
    screen = pygame.display.set_mode((520, 180))
    pygame.display.set_caption("STK AI default + Manual REC toggle (T)")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    def draw(hud_lines, rec_on, paused, manual_override):
        screen.fill((15, 15, 15))
        y = 10
        for line in hud_lines:
            surf = font.render(line, True, (230, 230, 230))
            screen.blit(surf, (10, y))
            y += 26

        mode = "MANUAL" if manual_override else "AI"
        status = f"{'PAUSED' if paused else 'RUNNING'} | MODE={mode} | {'REC ON' if rec_on else 'REC OFF'}"
        color = (255, 80, 80) if rec_on else (160, 160, 160)
        surf = font.render(status, True, color)
        screen.blit(surf, (10, 140))
        pygame.display.flip()

    env = gym.make(
        "supertuxkart/full-v0",
        agent=AgentSpec(use_ai=False),
        render_mode="human",
        num_kart=4,
        difficulty=2,
        max_episode_steps=1500,
    )
    env = SkipCountdownWrapper(env, skip_steps=10)
    env = NewRewardWrapper(env)
    env = ConstantSizedObservations(env)
    env = PolarObservations(env)
    env = Minimal56EssentialObsWrapper(env)

    obs, info = env.reset()
    print("Obs shape - ", obs["continuous"].shape)

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

    # recording + control switching
    rec_on = False
    manual_override = False   # ✅ IA par défaut
    records = []
    chunk_id = 0

    print("\n=== STK AI DEFAULT + MANUAL REC ===")
    print("Par défaut: IA (Ferrari) pilote.")
    print("T: REC ON -> tu pilotes + record | T encore: save + retour IA")
    print("P: pause | R: reset | Q/ESC: quit\n")

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
                        # STOP REC + SAVE + RETURN AI
                        out_path = chunk_path(DATASET_PATH, chunk_id)
                        print(f"⏹️ REC OFF -> saving {out_path}", flush=True)
                        save_dataset(out_path, records)
                        rec_on = False
                        manual_override = False  # ✅ retour IA
                    else:
                        # START REC + TAKE MANUAL
                        chunk_id += 1
                        records = []
                        rec_on = True
                        manual_override = True   # ✅ toi tu pilotes
                        out_path = chunk_path(DATASET_PATH, chunk_id)
                        print(f"⏺️ REC ON  -> MANUAL CONTROL -> recording to {out_path}", flush=True)

        keys = pygame.key.get_pressed()

        # -------- manual keys (only used if manual_override) --------
        if manual_override:
            # Steering (INVERTED)
            if keys[pygame.K_LEFT]:
                steer = max(steer - STEER_STEP, -1.0)
            elif keys[pygame.K_RIGHT]:
                steer = min(steer + STEER_STEP, 1.0)
            else:
                steer *= STEER_RETURN

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

        # -------- step env --------
        if not paused:
            obs_before = np.asarray(obs["continuous"], dtype=np.float32).copy()

            if manual_override:
                action_to_send = make_manual_action(steer, accel, brake, drift)
            else:
                action_to_send = get_ai_action(obs_before)
                action_to_send['acceleration'] = 1.0

            obs, reward, terminated, truncated, info = env.step(action_to_send)
            done = bool(terminated or truncated)

            step += 1
            ep_reward += float(reward)

            # record only when REC ON (which implies manual_override)
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
            "T=REC toggle manual  P=Pause  R=Reset  Q/ESC=Quit",
        ]
        draw(hud, rec_on, paused, manual_override)
        clock.tick(FPS)

    print("\nExiting...")
    # option: si tu veux sauver le dernier chunk uniquement si rec_on
    if rec_on:
        out_path = chunk_path(DATASET_PATH, chunk_id)
        save_dataset(out_path, records)

    env.close()
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
