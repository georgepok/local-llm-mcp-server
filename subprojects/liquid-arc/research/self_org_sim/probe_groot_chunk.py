"""Compare v10's predicted chunk vs GR00T's chunk for the SAME observation.
Reveals scale/format mismatches."""
import pickle
import zmq
import numpy as np
from pathlib import Path

SUITE_DIR = Path("/home/pokazge/datasets/libero-object-expert-v1")
GROOT_PORT = 5557
LIQUID_PORT = 7777

# Load one observation from libero_object expert ep0
idx = np.load(SUITE_DIR / "index.npz")
starts = idx["episode_starts"]
n_total = int(idx["n_total"])
img_size = int(idx["img_size"])

imgs = np.memmap(SUITE_DIR / "imgs.dat", dtype=np.uint8, mode="r",
                 shape=(n_total, img_size, img_size, 3))
wrists = np.memmap(SUITE_DIR / "wrists.dat", dtype=np.uint8, mode="r",
                   shape=(n_total, img_size, img_size, 3))
states = np.memmap(SUITE_DIR / "states.dat", dtype=np.float32, mode="r",
                   shape=(n_total, 8))

# Use first step of episode 0
global_idx = int(starts[0])
img_raw = np.array(imgs[global_idx])
wrist_raw = np.array(wrists[global_idx])
state8 = np.array(states[global_idx])
task_lang = "pick up the alphabet soup and place it in the basket"

# Load teacher chunk for same sample
labels = np.load(SUITE_DIR / "labels_index.npz")
sample_idx = labels["sample_idx"]
n_samples = int(labels["n_samples"])
chunks_mm = np.memmap(SUITE_DIR / "teacher_chunks.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, 16, 7))
# Find sample for ep=0, t=0
mask = (sample_idx[:, 0] == 0) & (sample_idx[:, 1] == 0)
s_local = int(np.where(mask)[0][0])
teacher_chunk = np.array(chunks_mm[s_local])

# === Query GR00T server ===
def build_groot_obs(img_256, wrist_256, st8, task):
    state_slots = {"x": (0, 1), "y": (1, 2), "z": (2, 3),
                   "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
                   "gripper": (6, 8)}
    obs = {"video": {}, "state": {}, "language": {}}
    for k, arr in [("image", img_256), ("wrist_image", wrist_256)]:
        obs["video"][k] = arr[None, None, ...]
    for k, (lo, hi) in state_slots.items():
        obs["state"][k] = st8[lo:hi].astype(np.float32)[None, None, :]
    obs["language"]["annotation.human.action.task_description"] = [[task]]
    return obs

ctx = zmq.Context.instance()
gs = ctx.socket(zmq.REQ); gs.setsockopt(zmq.RCVTIMEO, 30000)
gs.connect(f"tcp://localhost:{GROOT_PORT}")
gs.send(pickle.dumps({"op": "get_action_with_state",
                       "obs": build_groot_obs(img_raw, wrist_raw, state8, task_lang)}))
resp = pickle.loads(gs.recv())
groot_chunk = resp["chunk"]

# === Query Liquid (v10) server ===
ls = ctx.socket(zmq.REQ); ls.setsockopt(zmq.RCVTIMEO, 30000)
ls.connect(f"tcp://localhost:{LIQUID_PORT}")
ls.send(pickle.dumps({"cmd": "init"}))
print("init:", pickle.loads(ls.recv()))
ls.send(pickle.dumps({"cmd": "episode_reset"}))
print("reset:", pickle.loads(ls.recv()))
ls.send(pickle.dumps({"cmd": "predict_chunk",
                       "img_raw": img_raw, "wrist_raw": wrist_raw, "state8": state8,
                       "n_steps": 10}))
resp = pickle.loads(ls.recv())
v10_chunk = resp.get("chunk")

print(f"\n=== CHUNK COMPARISON (obs: ep0, t=0, task: '{task_lang[:60]}...') ===")
print(f"\nteacher_chunk (saved offline by gen_groot_labels):")
print(f"  shape: {teacher_chunk.shape}, dtype: {teacher_chunk.dtype}")
print(f"  position 0: {teacher_chunk[0]}")
print(f"  position 7: {teacher_chunk[7]}")
print(f"  position 15: {teacher_chunk[15]}")
print(f"  per-dim min/max across chunk: min={teacher_chunk.min(axis=0)} max={teacher_chunk.max(axis=0)}")

print(f"\ngroot_chunk (live from groot_server):")
print(f"  shape: {groot_chunk.shape}, dtype: {groot_chunk.dtype}")
print(f"  position 0: {groot_chunk[0]}")
print(f"  position 7: {groot_chunk[7]}")
print(f"  position 15: {groot_chunk[15]}")
print(f"  per-dim min/max across chunk: min={groot_chunk.min(axis=0)} max={groot_chunk.max(axis=0)}")

if v10_chunk is not None:
    v10_chunk = np.asarray(v10_chunk)
    print(f"\nv10_chunk (live from liquid_server):")
    print(f"  shape: {v10_chunk.shape}, dtype: {v10_chunk.dtype}")
    print(f"  position 0: {v10_chunk[0]}")
    print(f"  position 7: {v10_chunk[7]}")
    print(f"  position 15: {v10_chunk[15]}")
    print(f"  per-dim min/max across chunk: min={v10_chunk.min(axis=0)} max={v10_chunk.max(axis=0)}")

print(f"\n=== DIFF (groot vs teacher at this same expert state) ===")
diff = groot_chunk - teacher_chunk
print(f"per-dim mean abs diff: {np.abs(diff).mean(axis=0)}")
print(f"per-dim max abs diff: {np.abs(diff).max(axis=0)}")
