#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diverse environment: wider range of rooms, space sizes, connectivity
CW_DIVERSE='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 150.0, "connect_radius": 40.0, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'
CW_STANDARD='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1}'
STEPS=3000

echo "============================================"
echo "  Seeds Batch — Creative Iteration"
echo "============================================"

# ---- Seed A: d_proj=2, MLP, diverse env ----
echo ""
echo ">>> Seed A: d_proj=2, MLP, lambda=1.0, diverse env"
cat > /tmp/seed_a.yaml <<EOF
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 512
d_ffn_fluid: 512
d_metric: 64
n_scales: 3
vocab_size: 50304
max_seq_len: 1024
model_type: fgn
architecture_version: "fluid"
geo_metric_type: learned
curvature_lambda: 0.0
curvature_reward_mu: 0.0
structural_energy_lambda: 1.0
structural_energy_max_pairs: 2048
structural_energy_d_proj: 2
structural_energy_proj_mlp: true
dropout: 0.1
use_torch_compile: true
EOF

python scripts/train_resonant.py \
    --config /tmp/seed_a.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 500 --grad_clip 1.0 --log_every 100 \
    --save_every 1000 --max_steps $STEPS \
    --lambda_struct 1.0 \
    --task_kwargs "$CW_DIVERSE" \
    --output_dir output_seed_a_2d_mlp_diverse

echo ""
echo "--- Seed A Fidelity ---"
python scripts/diagnose_geometry_fidelity.py \
    --config /tmp/seed_a.yaml \
    --checkpoint output_seed_a_2d_mlp_diverse/checkpoints/final.pt \
    --n_episodes 50

# ---- Seed B: d_proj=2, linear, diverse env ----
echo ""
echo ">>> Seed B: d_proj=2, linear, lambda=1.0, diverse env"
cat > /tmp/seed_b.yaml <<EOF
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 512
d_ffn_fluid: 512
d_metric: 64
n_scales: 3
vocab_size: 50304
max_seq_len: 1024
model_type: fgn
architecture_version: "fluid"
geo_metric_type: learned
curvature_lambda: 0.0
curvature_reward_mu: 0.0
structural_energy_lambda: 1.0
structural_energy_max_pairs: 2048
structural_energy_d_proj: 2
structural_energy_proj_mlp: false
dropout: 0.1
use_torch_compile: true
EOF

python scripts/train_resonant.py \
    --config /tmp/seed_b.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 500 --grad_clip 1.0 --log_every 100 \
    --save_every 1000 --max_steps $STEPS \
    --lambda_struct 1.0 \
    --task_kwargs "$CW_DIVERSE" \
    --output_dir output_seed_b_2d_lin_diverse

echo ""
echo "--- Seed B Fidelity ---"
python scripts/diagnose_geometry_fidelity.py \
    --config /tmp/seed_b.yaml \
    --checkpoint output_seed_b_2d_lin_diverse/checkpoints/final.pt \
    --n_episodes 50

# ---- Seed C: d_proj=8, MLP, diverse env, lambda=5.0 (really force it) ----
echo ""
echo ">>> Seed C: d_proj=8, MLP, lambda=5.0, diverse env"
cat > /tmp/seed_c.yaml <<EOF
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 512
d_ffn_fluid: 512
d_metric: 64
n_scales: 3
vocab_size: 50304
max_seq_len: 1024
model_type: fgn
architecture_version: "fluid"
geo_metric_type: learned
curvature_lambda: 0.0
curvature_reward_mu: 0.0
structural_energy_lambda: 5.0
structural_energy_max_pairs: 2048
structural_energy_d_proj: 8
structural_energy_proj_mlp: true
dropout: 0.1
use_torch_compile: true
EOF

python scripts/train_resonant.py \
    --config /tmp/seed_c.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 500 --grad_clip 1.0 --log_every 100 \
    --save_every 1000 --max_steps $STEPS \
    --lambda_struct 5.0 \
    --task_kwargs "$CW_DIVERSE" \
    --output_dir output_seed_c_8d_mlp_lambda5

echo ""
echo "--- Seed C Fidelity ---"
python scripts/diagnose_geometry_fidelity.py \
    --config /tmp/seed_c.yaml \
    --checkpoint output_seed_c_8d_mlp_lambda5/checkpoints/final.pt \
    --n_episodes 50

echo ""
echo "============================================"
echo "  All Seeds Complete"
echo "============================================"
