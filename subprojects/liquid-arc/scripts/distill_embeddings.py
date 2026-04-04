"""Distill sentence-transformer similarity structure into MindTokenizer embeddings.

Transfers semantic neighborhoods from sentence-transformers (384-dim) to the
Mind's ODE embedding pipeline (768-dim) via pairwise cosine similarity matching.

No dimension alignment needed — we match the *similarity structure*, not the
vectors themselves. Each encoder uses its own tokenizer.

Usage (on DGX Spark, in fgn-train or liquid-mind container):
    python scripts/distill_embeddings.py \
        --checkpoint output_30m/checkpoints/step_10000.pt \
        --output distilled_embeddings.pt \
        --n_texts 500 --n_steps 300 --batch_size 32 --lr 3e-3

The output is a state_dict for ConversationEmbedding that can be loaded
by the Mind on startup via --distilled_embeddings flag.
"""

import argparse
import random
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import ContinuousDynamics, ContextPool
from liquid_arc.conversation_embedding import ConversationEmbedding


# ── Corpus ──────────────────────────────────────────────────────────────────

CORPUS_TEXTS = [
    # Topology / geometry
    "A Klein bottle is a non-orientable surface that cannot be embedded in three dimensions without self-intersection.",
    "The fundamental group of a torus is Z cross Z, while the sphere is simply connected.",
    "Geodesics on a sphere are great circles, the shortest paths between any two points.",
    "A Mobius strip has only one side and one boundary curve.",
    "Euler characteristic relates vertices, edges, and faces of a polyhedron.",
    "Riemannian curvature measures how parallel transport around a loop rotates a vector.",
    "The Gauss-Bonnet theorem connects total curvature to topology.",
    "Hyperbolic space has constant negative curvature and exponential growth of circles.",
    "Diffeomorphisms preserve the smooth structure of a manifold.",
    "A fiber bundle locally looks like a product space but may be globally twisted.",

    # Music
    "A fugue subject enters in different voices at different pitch levels creating overlapping temporal patterns.",
    "Polyrhythm of three against four creates interference patterns resolving at the least common multiple.",
    "A deceptive cadence substitutes the expected tonic with the submediant chord.",
    "Counterpoint follows rules of consonance and dissonance between independent melodic lines.",
    "The harmonic series determines which intervals sound consonant to the human ear.",
    "Syncopation displaces expected rhythmic accents creating tension against the meter.",
    "A twelve-tone row uses all chromatic pitches exactly once before repeating.",
    "The circle of fifths arranges all twelve major keys by ascending perfect fifths.",
    "Sonata form has exposition, development, and recapitulation sections.",
    "Microtonal music divides the octave into more than twelve equal parts.",

    # Biology
    "Turing patterns emerge from reaction-diffusion of activator and inhibitor chemicals.",
    "Neural crest cells migrate using contact inhibition and chemotaxis during development.",
    "The genetic code maps 64 codons to 20 amino acids with systematic redundancy.",
    "Protein folding converts a linear amino acid chain into a three-dimensional structure.",
    "Mitochondria were once free-living bacteria that formed an endosymbiotic relationship.",
    "CRISPR-Cas9 uses guide RNA to find and cut specific DNA sequences.",
    "Epigenetic modifications alter gene expression without changing the DNA sequence.",
    "Homeotic genes control body plan organization along the anterior-posterior axis.",
    "Action potentials propagate along axons through voltage-gated sodium and potassium channels.",
    "The immune system distinguishes self from non-self using MHC molecules.",

    # Physics
    "Spontaneous symmetry breaking occurs when the ground state has less symmetry than the Hamiltonian.",
    "Renormalization group flow connects microscopic and macroscopic descriptions of physical systems.",
    "A soap bubble minimizes surface area subject to a volume constraint via mean curvature.",
    "Noether's theorem connects continuous symmetries to conservation laws.",
    "Entanglement means quantum states of particles cannot be described independently.",
    "The Higgs mechanism gives mass to gauge bosons through spontaneous symmetry breaking.",
    "Black holes have an event horizon beyond which nothing can escape.",
    "Superconductivity emerges when electrons form Cooper pairs below a critical temperature.",
    "Hawking radiation allows black holes to slowly evaporate through quantum effects.",
    "General relativity describes gravity as curvature of spacetime caused by mass and energy.",

    # Philosophy
    "Whitehead's actual occasions of experience prehend previous occasions creating a web of mutual influence.",
    "Merleau-Ponty describes the body schema as the origin of spatial experience rather than an object in space.",
    "The ship of Theseus asks whether identity persists through complete material replacement.",
    "Heidegger's ready-to-hand describes how tools become transparent extensions of the body in use.",
    "Wittgenstein argued that the limits of language are the limits of one's world.",
    "Phenomenology studies the structures of conscious experience from the first-person perspective.",
    "Process philosophy treats becoming as more fundamental than being.",
    "The hard problem of consciousness asks why physical processes give rise to subjective experience.",
    "Qualia are the subjective experiential properties of mental states.",
    "Emergence describes how complex systems exhibit properties not reducible to their components.",

    # Mathematics
    "A strange attractor in a chaotic system has fractional dimension between a surface and a volume.",
    "Fisher information metric turns probability distributions into a Riemannian manifold.",
    "A functor between categories maps objects to objects and morphisms to morphisms preserving composition.",
    "Godel's incompleteness theorem shows that consistent formal systems cannot prove all true statements.",
    "The Riemann zeta function connects prime number distribution to complex analysis.",
    "Group theory studies symmetry through sets with an associative binary operation and inverses.",
    "Topology studies properties preserved under continuous deformation like stretching and bending.",
    "Eigenvalues of a matrix determine its behavior under repeated application.",
    "The central limit theorem explains why many natural distributions are approximately Gaussian.",
    "Bayesian inference updates probability distributions as new evidence is observed.",

    # Poetry / language
    "Enjambment creates tension between syntactic structure and prosodic line breaks.",
    "A villanelle's two repeating refrains create a spiral accumulating new meaning each pass.",
    "Haiku's 5-7-5 structure creates a breath, expansion, and compression of attention.",
    "Metaphor maps structure from a source domain to a target domain.",
    "Iambic pentameter has five pairs of unstressed and stressed syllables per line.",
    "Alliteration repeats initial consonant sounds to create rhythmic emphasis.",
    "A sonnet's volta marks a turn in argument or perspective.",
    "Free verse abandons regular meter in favor of natural speech rhythms.",
    "Paradox holds contradictory ideas in tension to reveal deeper truth.",
    "Imagery uses sensory language to create vivid mental pictures.",

    # Ecology
    "Ecological succession transforms bare landscape into forest through species that modify the environment.",
    "Competitive exclusion creates a geometric packing problem in ecological niche space.",
    "A keystone species structures the entire ecosystem and its removal collapses the community.",
    "Trophic cascades propagate effects through food webs from predators to plants.",
    "Island biogeography predicts species richness from island size and distance to mainland.",
    "Mycorrhizal networks connect trees underground enabling nutrient and signal transfer.",
    "Coral reefs are built by tiny polyps depositing calcium carbonate over centuries.",
    "Nitrogen fixation by bacteria converts atmospheric N2 into biologically usable forms.",
    "Invasive species disrupt native ecosystems by outcompeting established organisms.",
    "Biodiversity hotspots contain high concentrations of endemic species under threat.",

    # Computing / information
    "Neural networks learn hierarchical representations through backpropagation of error gradients.",
    "Information entropy measures the average surprise in a probability distribution.",
    "A hash function maps arbitrary data to fixed-size values with collision resistance.",
    "Turing machines define computability through a simple tape-reading abstract model.",
    "PageRank measures web page importance through the link structure of the graph.",
    "Gradient descent finds local minima by stepping in the direction of steepest decrease.",
    "Attention mechanisms in transformers compute weighted sums based on query-key similarity.",
    "Compression algorithms exploit statistical redundancy to reduce data size.",
    "Convolutional networks detect local patterns through learned sliding filters.",
    "Reinforcement learning agents maximize cumulative reward through trial and error.",

    # Simple / everyday
    "The cat sat on the warm windowsill watching birds in the garden.",
    "Rain fell steadily all afternoon turning the streets into small rivers.",
    "She opened the book to chapter three and began reading aloud.",
    "The coffee was too hot to drink so he waited and watched the steam rise.",
    "Children played in the park while their parents talked on nearby benches.",
    "The old bridge creaked under the weight of the passing truck.",
    "Stars appeared one by one as the sky darkened after sunset.",
    "He sorted the mail into two piles: bills and everything else.",
    "The dog chased its tail in circles until it got dizzy and fell over.",
    "Fresh bread from the bakery filled the kitchen with a warm scent.",

    # Emotional / experiential
    "The grief came in waves, each one a little smaller than the last.",
    "Standing at the edge of the canyon, she felt the immensity of geological time.",
    "The moment before sleep is a dissolution of boundaries between self and world.",
    "Nostalgia is a bittersweet recognition that the past exists only in memory.",
    "Awe expands attention outward and diminishes the sense of individual self.",
    "Flow state dissolves the boundary between action and awareness.",
    "Loneliness is not the absence of others but the absence of connection.",
    "The sublime combines terror and beauty in a single overwhelming experience.",
    "Empathy requires modeling another mind's interior from external behavioral cues.",
    "Joy arrives unannounced and leaves before you can name it.",

    # Technical / mechanical
    "A differential gear allows wheels on the same axle to rotate at different speeds.",
    "Heat exchangers transfer thermal energy between fluids without mixing them.",
    "Feedback control systems use error signals to adjust output toward a setpoint.",
    "Hydraulic systems transmit force through incompressible fluid in enclosed pipes.",
    "Ball bearings reduce friction by replacing sliding contact with rolling contact.",
    "A transistor amplifies or switches electronic signals using semiconductor junctions.",
    "Fiber optic cables transmit data as pulses of light through glass strands.",
    "GPS triangulates position using precise timing signals from multiple satellites.",
    "Capacitors store electrical energy in an electric field between two conductors.",
    "Laser light is coherent, meaning all photons have the same phase and frequency.",

    # Abstract / structural
    "A bridge connects two separate regions and the shortest path must cross it.",
    "Patterns repeat at different scales in fractal structures.",
    "Symmetry means a transformation leaves the essential structure unchanged.",
    "Hierarchy organizes elements into levels where each level contains the ones below.",
    "A cycle returns to its starting point after traversing a sequence of states.",
    "Boundaries separate inside from outside and define what belongs to a region.",
    "Resonance occurs when a system is driven at its natural frequency.",
    "Equilibrium is a state where opposing forces or processes exactly balance.",
    "Threshold effects mean small changes near a critical point cause large responses.",
    "Networks consist of nodes connected by edges with emergent global properties.",
]


def build_corpus(n_texts: int) -> list:
    """Select n_texts from the built-in corpus, cycling if needed."""
    if n_texts <= len(CORPUS_TEXTS):
        return random.sample(CORPUS_TEXTS, n_texts)
    # Cycle through corpus with slight variations
    texts = list(CORPUS_TEXTS)
    while len(texts) < n_texts:
        base = random.choice(CORPUS_TEXTS)
        texts.append(base)
    return texts[:n_texts]


# ── Teacher encoding ────────────────────────────────────────────────────────

def encode_teacher(texts: list, batch_size: int = 64) -> torch.Tensor:
    """Encode all texts with sentence-transformers. Returns [N, 384]."""
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Encoding {len(texts)} texts with teacher...")
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True, convert_to_tensor=True)
    # Normalize for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


# ── Student encoding ────────────────────────────────────────────────────────

def encode_student_batch(
    texts: list,
    embedding: ConversationEmbedding,
    dynamics: ContinuousDynamics,
    context_pool: ContextPool,
    n_steps: int = 16,
    T: float = 2.0,
    device: str = 'cuda',
) -> torch.Tensor:
    """Encode texts through the ODE pipeline. Returns [N, d_model] normalized."""
    d_model = dynamics.norm_geo.normalized_shape[0]
    all_reprs = []

    for text in texts:
        token_h, token_mask = embedding.embed_tokens(text, device)
        T_actual = token_mask.sum().item()
        if T_actual == 0:
            all_reprs.append(torch.zeros(d_model, device=device))
            continue

        context = context_pool(token_h, token_mask)
        dynamics.set_context(context, mask=None)
        dynamics.set_n_steps(n_steps)

        # Euler ODE integration WITH gradients
        dt = T / n_steps
        t = 0.0
        h = token_h
        for step_i in range(n_steps):
            if hasattr(dynamics, 'set_step_index'):
                dynamics.set_step_index(step_i, n_steps)
            dy = dynamics(t, h)
            h = h + dt * dy
            t += dt

        # Mean-pool non-padding positions
        mask_exp = token_mask.unsqueeze(-1).float()
        h_pooled = (h * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        all_reprs.append(h_pooled.squeeze(0))

    result = torch.stack(all_reprs)
    return F.normalize(result, p=2, dim=1)


# ── Distillation loop ──────────────────────────────────────────────────────

def distill(
    texts: list,
    teacher_sims: torch.Tensor,
    embedding: ConversationEmbedding,
    dynamics: ContinuousDynamics,
    context_pool: ContextPool,
    n_steps_ode: int = 16,
    T: float = 2.0,
    lr: float = 3e-3,
    n_steps: int = 300,
    batch_size: int = 32,
    device: str = 'cuda',
):
    """Train embeddings to match teacher similarity structure."""
    # Only train tokenizer embeddings — freeze everything else
    for p in dynamics.parameters():
        p.requires_grad_(False)
    for p in context_pool.parameters():
        p.requires_grad_(False)
    for p in embedding.event_proj.parameters():
        p.requires_grad_(False)
    for p in embedding.content_proj.parameters():
        p.requires_grad_(False)
    for p in embedding.metadata_proj.parameters():
        p.requires_grad_(False)
    for p in embedding.type_embed.parameters():
        p.requires_grad_(False)
    for p in embedding.pos_embed.parameters():
        p.requires_grad_(False)
    for p in embedding.norm.parameters():
        p.requires_grad_(False)

    # Tokenizer embeddings are trainable
    for p in embedding.tokenizer.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in embedding.tokenizer.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} (tokenizer embeddings)")

    optimizer = torch.optim.Adam(embedding.tokenizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    N = len(texts)
    best_loss = float('inf')
    best_state = None

    for step in range(n_steps):
        # Sample a random batch of text indices
        indices = random.sample(range(N), min(batch_size, N))
        batch_texts = [texts[i] for i in indices]

        # Extract teacher similarity sub-matrix
        idx_t = torch.tensor(indices, device=teacher_sims.device)
        teacher_sub = teacher_sims[idx_t][:, idx_t]  # [B, B]

        # Encode student batch
        student_reprs = encode_student_batch(
            batch_texts, embedding, dynamics, context_pool,
            n_steps=n_steps_ode, T=T, device=device,
        )

        # Student similarity matrix
        student_sims = student_reprs @ student_reprs.T  # [B, B]

        # MSE on similarity structure
        loss = F.mse_loss(student_sims, teacher_sub.to(device))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embedding.tokenizer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in embedding.tokenizer.state_dict().items()}

        if step % 10 == 0 or step == n_steps - 1:
            # Diagnostic: average off-diagonal similarity spread
            with torch.no_grad():
                mask = ~torch.eye(len(indices), dtype=torch.bool, device=device)
                student_spread = student_sims[mask].std().item()
                teacher_spread = teacher_sub.to(device)[mask].std().item()
            print(f"  step {step:4d} | loss={loss.item():.5f} | "
                  f"student_spread={student_spread:.3f} | "
                  f"teacher_spread={teacher_spread:.3f} | "
                  f"lr={scheduler.get_last_lr()[0]:.1e}")

    return best_state, best_loss


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distill sentence-transformer → ODE embeddings")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to LiquidARC model checkpoint (e.g. step_10000.pt)')
    parser.add_argument('--output', type=str, default='distilled_embeddings.pt',
                        help='Output path for distilled embedding state_dict')
    parser.add_argument('--n_texts', type=int, default=150,
                        help='Number of corpus texts to use (max ~150 built-in)')
    parser.add_argument('--n_steps', type=int, default=300,
                        help='Number of distillation training steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Texts per distillation step')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate for tokenizer embeddings')
    parser.add_argument('--n_ode_steps', type=int, default=16,
                        help='ODE integration steps for student encoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (overrides checkpoint config)')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.config:
        import yaml
        with open(args.config) as f:
            config = LiquidARCConfig(**yaml.safe_load(f))
    elif 'config' in ckpt:
        cfg = ckpt['config']
        if isinstance(cfg, LiquidARCConfig):
            config = cfg
        elif isinstance(cfg, dict):
            config = LiquidARCConfig(**cfg)
        else:
            config = LiquidARCConfig()
    else:
        config = LiquidARCConfig()

    # Build model components
    dynamics = ContinuousDynamics(config).to(device)
    context_pool = ContextPool(config).to(device)
    embedding = ConversationEmbedding(
        d_model=config.d_model,
        max_tokens=64,
    ).to(device)

    # Load weights (handle _orig_mod. prefix from torch.compile)
    model_state = ckpt.get('model_state_dict', ckpt.get('model', {}))
    cleaned = {}
    for k, v in model_state.items():
        k_clean = k.replace('_orig_mod.', '')
        cleaned[k_clean] = v

    # Load dynamics weights
    dyn_prefix = 'dynamics.'
    dyn_state = {k[len(dyn_prefix):]: v for k, v in cleaned.items()
                 if k.startswith(dyn_prefix)}
    if dyn_state:
        dynamics.load_state_dict(dyn_state, strict=False)
        print(f"  Loaded dynamics: {len(dyn_state)} keys")

    # Load context_pool weights
    cp_prefix = 'context_pool.'
    cp_state = {k[len(cp_prefix):]: v for k, v in cleaned.items()
                if k.startswith(cp_prefix)}
    if cp_state:
        context_pool.load_state_dict(cp_state, strict=False)
        print(f"  Loaded context_pool: {len(cp_state)} keys")

    # Eagerly load tokenizer (needed for encoding)
    embedding.tokenizer._load_tokenizer()

    # ── Build corpus ────────────────────────────────────────────────────
    texts = build_corpus(args.n_texts)
    print(f"Corpus: {len(texts)} texts")

    # ── Teacher encoding ────────────────────────────────────────────────
    teacher_embs = encode_teacher(texts)
    teacher_sims = teacher_embs @ teacher_embs.T  # [N, N] cosine similarity
    print(f"Teacher similarity matrix: {teacher_sims.shape}")

    # Diagnostic: teacher similarity statistics
    mask = ~torch.eye(len(texts), dtype=torch.bool)
    off_diag = teacher_sims[mask]
    print(f"  Teacher off-diagonal sim: mean={off_diag.mean():.3f}, "
          f"std={off_diag.std():.3f}, min={off_diag.min():.3f}, max={off_diag.max():.3f}")

    # ── Pre-distillation baseline ───────────────────────────────────────
    print("\nPre-distillation baseline:")
    with torch.no_grad():
        student_reprs = encode_student_batch(
            texts[:min(32, len(texts))], embedding, dynamics, context_pool,
            n_steps=args.n_ode_steps, device=device,
        )
        student_sims = student_reprs @ student_reprs.T
        s_mask = ~torch.eye(student_sims.size(0), dtype=torch.bool, device=device)
        s_off = student_sims[s_mask]
        print(f"  Student off-diagonal sim: mean={s_off.mean():.3f}, "
              f"std={s_off.std():.3f}")
        t_sub = teacher_sims[:student_sims.size(0), :student_sims.size(0)].to(device)
        baseline_loss = F.mse_loss(student_sims, t_sub).item()
        print(f"  Baseline similarity MSE: {baseline_loss:.5f}")

    # ── Distill ─────────────────────────────────────────────────────────
    print(f"\nDistilling ({args.n_steps} steps, batch_size={args.batch_size}, "
          f"lr={args.lr})...")
    t0 = time.time()

    best_state, best_loss = distill(
        texts=texts,
        teacher_sims=teacher_sims,
        embedding=embedding,
        dynamics=dynamics,
        context_pool=context_pool,
        n_steps_ode=args.n_ode_steps,
        T=2.0,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"\nDistillation complete in {elapsed:.1f}s")
    print(f"  Best loss: {best_loss:.5f} (baseline: {baseline_loss:.5f})")
    print(f"  Improvement: {baseline_loss / max(best_loss, 1e-8):.1f}x")

    # ── Save ────────────────────────────────────────────────────────────
    # Save just the tokenizer state (what the Mind needs)
    output = {
        'tokenizer_state_dict': best_state,
        'distill_loss': best_loss,
        'baseline_loss': baseline_loss,
        'n_texts': len(texts),
        'n_steps': args.n_steps,
        'source_checkpoint': args.checkpoint,
    }
    torch.save(output, args.output)
    print(f"Saved distilled embeddings to: {args.output}")

    # Also save full embedding state for direct Mind loading
    embedding.tokenizer.load_state_dict(best_state)
    full_output = args.output.replace('.pt', '_full.pt')
    torch.save({
        'embedding_state_dict': embedding.state_dict(),
        'distill_loss': best_loss,
        'baseline_loss': baseline_loss,
        'n_texts': len(texts),
        'n_steps': args.n_steps,
        'source_checkpoint': args.checkpoint,
    }, full_output)
    print(f"Saved full embedding state to: {full_output}")


if __name__ == '__main__':
    main()
