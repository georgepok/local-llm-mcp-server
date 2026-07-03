"""Debug encoder: load pretrained, run on actual PushT frame, check for NaN."""
import sys
import torch
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(True)

import stable_pretraining as spt
import stable_worldmodel as swm
from patch_vit_stem import replace_vit_patch_embeddings
from utils import get_column_normalizer, get_img_preprocessor
from omegaconf import OmegaConf

data_cfg = OmegaConf.create({
    "dataset": {
        "num_steps": 10, "frameskip": 5, "name": "pusht_expert_train",
        "keys_to_load": ["pixels", "action", "proprio", "state"],
        "keys_to_cache": ["action", "proprio", "state"],
    }
})
ds = swm.data.HDF5Dataset(**data_cfg.dataset, transform=None)
tr = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
for col in data_cfg.dataset.keys_to_load:
    if col.startswith("pixels"):
        continue
    tr.append(get_column_normalizer(ds, col, col))
ds.transform = spt.data.transforms.Compose(*tr)

sample = ds[0]
pix = sample["pixels"].unsqueeze(0).cuda()
print("pixels", pix.shape, pix.dtype, "range", pix.min().item(), pix.max().item(),
      "nan?", torch.isnan(pix).any().item(), "inf?", torch.isinf(pix).any().item())

enc = spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224,
                                 pretrained=False, use_mask_token=False)
replace_vit_patch_embeddings(enc)
enc = enc.cuda().eval()

pt = torch.load("/workspace/models/stable-wm/checkpoints/lewm-pusht/weights.pt",
                map_location="cpu", weights_only=False)
if isinstance(pt, dict) and "state_dict" in pt:
    pt = pt["state_dict"]
own = dict(enc.state_dict())
sd = {}
for k, v in pt.items():
    k2 = k[6:] if k.startswith("model.") else k
    if not k2.startswith("encoder."):
        continue
    k2 = k2[len("encoder."):]
    if k2.endswith("patch_embeddings.projection.weight") and k2 not in own:
        k2 = k2.replace("projection.weight", "projection.proj.weight")
    if k2.endswith("patch_embeddings.projection.bias") and k2 not in own:
        k2 = k2.replace("projection.bias", "projection.proj.bias")
    if k2 in own and own[k2].shape != v.shape and v.ndim == 4:
        v = v.reshape(v.shape[0], -1)
    sd[k2] = v
res = enc.load_state_dict(sd, strict=False)
print("load missing=", len(res.missing_keys), "unexp=", len(res.unexpected_keys))
if res.missing_keys:
    print("missing:", res.missing_keys[:10])

img = pix[:, 0] if pix.ndim == 5 else pix
print("img", img.shape, img.dtype, "range", img.min().item(), img.max().item())

with torch.no_grad():
    stem = enc.embeddings.patch_embeddings.projection(img.float())
    print("stem", stem.shape, "nan?", torch.isnan(stem).any().item(),
          "mean", stem.mean().item(), "std", stem.std().item())
    y = enc(img.float(), interpolate_pos_encoding=True).last_hidden_state
    print("full", y.shape, "nan?", torch.isnan(y).any().item(),
          "mean", y.mean().item(), "std", y.std().item())
