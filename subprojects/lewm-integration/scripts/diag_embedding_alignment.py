"""Diagnostic: do predicted embeddings align with encoder embeddings?

If cosine similarity is low, CEM's cost function is broken regardless
of prediction MSE — the predictor outputs live in a different subspace
than the encoder's goal embeddings.
"""
import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "le-wm"))

import stable_pretraining as spt
import stable_worldmodel as swm
from patch_vit_stem import replace_vit_patch_embeddings
from jepa import JEPA
from module import Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor
from omegaconf import OmegaConf

from liquid_arc.config import LiquidARCConfig
from liquid_arc_lewm import LiquidARCPredictor


def _load(model, path, skip_prefix=""):
    st = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    own = dict(model.state_dict())
    sd = {}
    for k, v in st.items():
        k2 = k[6:] if k.startswith("model.") else k
        if skip_prefix and k2.startswith(skip_prefix):
            continue
        if k2.endswith("projection.weight") and k2 not in own:
            k2 = k2.replace("projection.weight", "projection.proj.weight")
        if k2.endswith("projection.bias") and k2 not in own:
            k2 = k2.replace("projection.bias", "projection.proj.bias")
        if k2 in own and own[k2].shape != v.shape and v.ndim == 4:
            v = v.reshape(v.shape[0], -1)
        sd[k2] = v
    model.load_state_dict(sd, strict=False)


enc = spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224,
                                 pretrained=False, use_mask_token=False)
replace_vit_patch_embeddings(enc)
ode_cfg = LiquidARCConfig(
    d_model=192, d_metric=48, d_metric_bottleneck=96, metric_rank=32,
    d_ffn=512, n_ode_steps=16, ode_steps_min=16, ode_steps_max=16,
    integration_time=2.0, tau_min=0.5, tau_max=1.0, use_torch_compile=False)
pred = LiquidARCPredictor(input_dim=192, action_emb_dim=192,
                          ode_config=ode_cfg, output_dim=192, dropout=0.1)
ae = Embedder(input_dim=10, emb_dim=192)
proj = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
           norm_fn=torch.nn.LayerNorm)
pp = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
         norm_fn=torch.nn.LayerNorm)
model = JEPA(encoder=enc, predictor=pred, action_encoder=ae,
             projector=proj, pred_proj=pp)

_load(model, "/workspace/models/stable-wm/checkpoints/lewm-pusht/weights.pt",
      skip_prefix="predictor.")
_load(model, "/workspace/models/stable-wm/liquid_crit_weights.ckpt")
model = model.cuda().eval()

dcfg = OmegaConf.create({
    "dataset": {"num_steps": 6, "frameskip": 5, "name": "pusht_expert_train",
                "keys_to_load": ["pixels", "action", "proprio", "state"],
                "keys_to_cache": ["action", "proprio", "state"]}
})
ds = swm.data.HDF5Dataset(**dcfg.dataset, transform=None)
tr = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
for c in dcfg.dataset.keys_to_load:
    if c.startswith("pixels"):
        continue
    tr.append(get_column_normalizer(ds, c, c))
ds.transform = spt.data.transforms.Compose(*tr)
gen = torch.Generator().manual_seed(0)
_, val = spt.data.random_split(ds, [0.9, 0.1], generator=gen)
loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False,
                                     num_workers=4, drop_last=True)

cos_sims, mse_vals = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 15:
            break
        moved = {k: v.cuda() if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model.encode(moved)
        emb = out["emb"]
        act = out["act_emb"]
        pred_emb = model.predict(emb[:, :3], act[:, :3])
        predicted = pred_emb[:, -1]
        target = emb[:, 3]
        cs = F.cosine_similarity(predicted, target, dim=-1)
        ms = (predicted - target).pow(2).sum(dim=-1)
        cos_sims.append(cs)
        mse_vals.append(ms)
        if i == 0:
            print(f"pred norm: {predicted.norm(dim=-1).mean():.4f}  "
                  f"target norm: {target.norm(dim=-1).mean():.4f}")
            print(f"pred mean: {predicted.mean():.4f}  "
                  f"target mean: {target.mean():.4f}")
            print(f"pred std:  {predicted.std():.4f}  "
                  f"target std:  {target.std():.4f}")

cos_all = torch.cat(cos_sims)
mse_all = torch.cat(mse_vals)
print(f"\n=== EMBEDDING ALIGNMENT DIAGNOSTIC ===")
print(f"cosine similarity: mean={cos_all.mean():.4f}  "
      f"std={cos_all.std():.4f}  min={cos_all.min():.4f}")
print(f"MSE:               mean={mse_all.mean():.6f}  "
      f"std={mse_all.std():.6f}")
print(f"samples: {len(cos_all)}")
if cos_all.mean() > 0.95:
    print("ALIGNED — cost function should work for MPC")
elif cos_all.mean() > 0.8:
    print("PARTIALLY ALIGNED — MPC may work with tuning")
else:
    print("MISALIGNED — cost function broken, fix projector first")
