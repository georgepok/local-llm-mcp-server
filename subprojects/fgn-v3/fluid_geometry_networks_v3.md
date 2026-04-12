fluid_geometry_networks_v3.md
74.57 KB •1,303 lines
•
Formatting may be inconsistent from source

# Fluid Geometry Networks: Intrinsic Riemannian Architectures for Adaptive Neural Computation

**A Research Foundation Document**

---

**Authors**: [Research Collaboration]

**Date**: February 2026

**Document Type**: Theoretical Foundation / Position Paper (Revised)

**Keywords**: Neural Architecture, Riemannian Geometry, Heat Kernels, Parallel Transport, Transformers, State Space Models, Geometric Deep Learning, Variable Associativity, Adaptive Resolution, Cognitive Architecture

---

## Abstract

Current neural architectures face a fundamental tension: transformer architectures excel at parallel pattern recognition through temporally-agnostic processing in flat embedding spaces, while recurrent and state-space architectures excel at sequential temporal reasoning through inherently path-dependent processing. Previous work, including the initial formulation of Fluid Geometry Networks (FGN v2), proposed resolving this through learnable associativity parameters (Î±) that interpolate between processing modes. While productive, this approach suffered from the "leaky masking" problem and treated geometry as an external parameter rather than an intrinsic property.

We present a fundamental revision grounded in **Intrinsic Riemannian Geometry**. We show that a single learned object â€” the **metric tensor** g on the sequence-embedding manifold â€” unifies three previously separate concepts: (1) adaptive resolution emerges from the **heat kernel** of the Laplace-Beltrami operator determined by g, (2) the associativity parameter Î± is **derived from scalar curvature** rather than independently learned, and (3) the generative "meets and becomes" operator (~>) is formalized as **parallel transport** along the Levi-Civita connection uniquely determined by g. This consolidation dissolves the leaky masking problem (geometry is structural, not interpolated), provides a rigorous mathematical foundation connecting to established differential geometry, and reveals that standard transformer attention is a **flat-space approximation** that systematically discards curvature information â€” precisely the information encoding emergence, causality, and novelty generation.

We retain and extend the **Causal-Symbolic Gap (CSG)** benchmark, update architectural specifications for the metric-network paradigm, and outline a revised research roadmap. The framework has implications for machine cognition, embodied AI, generative systems, and the mathematical foundations of neural computation.

---

## 1. Introduction

### 1.1 The Grounding Problem

Large language models demonstrate remarkable capabilities in pattern recognition, reasoning, and generation within the domain of human language and formal systems. However, their relationship to physical reality remains mediated entirely through human-generated representations. Language, while extraordinarily rich, constitutes a compressed encoding of human temporal experience â€” it is map, not territory.

The challenge of grounding AI systems in physical reality â€” enabling them to interface with temporal, causal, embodied domains â€” represents one of the central problems in artificial intelligence. Current approaches typically involve:

1. **Architectural modification**: Adding recurrent connections, memory mechanisms, or temporal processing layers to transformer backbones
2. **Hybrid systems**: Combining separate transformer and recurrent components with learned routing
3. **Enhanced encoding**: Improving positional and temporal encodings within existing architectures

While functionally effective for specific tasks, these approaches sacrifice a potentially crucial capability: the transformer's unique capacity for simultaneous, parallel attention across entire contexts â€” what we term "flat seeing." This capability may be essential for certain forms of pattern recognition, abstraction, and reasoning that sequential processing cannot achieve.

### 1.2 From Variable Associativity to Intrinsic Geometry

The initial FGN formulation (v2) proposed that the flat/sequential dichotomy reflects a fundamental geometric structure â€” the relationship between associative (Euclidean-like) and non-associative (Riemannian-like) computational geometries â€” and introduced a learnable associativity parameter Î± âˆˆ [0,1] to interpolate between them.

This formulation, while productive, had a critical limitation: it treated Î± as an **extrinsic parameter** governing the blending of two separate processing pathways. This led to the leaky masking problem (soft interpolation allows information leakage) and missed a deeper mathematical structure.

The present revision proposes that the correct framework is not parametric interpolation but **intrinsic Riemannian geometry**. The key insight:

**A single learned metric tensor g determines all aspects of fluid geometry.**

Specifically:
- **Resolution** (how broadly or narrowly the model attends) emerges from the heat kernel on the Riemannian manifold (M, g)
- **Associativity** (whether processing is order-dependent) is derived from the scalar curvature Îº of g, not independently parameterized
- **Emergence** (the generative ~> operator) is formalized as parallel transport along the Levi-Civita connection âˆ‡ uniquely determined by g

This is not merely a change of mathematical notation. It represents a shift from **geometry as parameter** to **geometry as structure** â€” from selecting between pre-existing modes to learning the intrinsic shape of computational space.

### 1.3 Core Thesis (Revised)

**Central Claims**:

1. **Transformers operate in flat computational geometry**: Standard self-attention implements a flat-space (zero curvature) approximation where all positions share a common coordinate frame. The dot-product attention kernel approximates the heat kernel on a Euclidean manifold.

2. **Recurrent/State-Space architectures operate in curved computational geometry**: Sequential processing follows geodesics on a manifold with non-zero curvature, where path-dependence arises from the geometry itself rather than from architectural constraint.

3. **The Î± parameter is derived, not fundamental**: What we previously treated as an independent associativity parameter is the scalar curvature Îº of the learned metric, mapped through Î±(x) â‰ˆ exp(âˆ’Îº(x)/Îºâ‚€). High curvature â†’ low Î± (path-dependent); zero curvature â†’ Î± = 1 (order-independent).

4. **Emergence requires curvature**: The ~> operator ("meets and becomes") is parallel transport â€” the movement of representations across regions of non-zero curvature. On a flat manifold, parallel transport is trivial (no genuine novelty). Emergence is proportional to curvature.

5. **Standard attention systematically discards curvature information**: By treating all positions as living in the same vector space, transformers implicitly flatten the computational manifold. The information lost in this flattening is precisely the curvature information that encodes causal structure, temporal dependence, and generative potential.

### 1.4 Contributions

This paper makes the following contributions:

- **Unified Geometric Framework**: We show that resolution, associativity, and emergence are three aspects of a single learned Riemannian metric, grounded in established differential geometry (heat kernels, Levi-Civita connection, parallel transport, holonomy).

- **Dissolution of the Leaky Masking Problem**: By making geometry intrinsic rather than interpolated, the pathological information leakage of Î±-blending is structurally eliminated.

- **Reinterpretation of Attention**: We establish that standard dot-product attention is a flat-space approximation of geodesic-distance-based attention, identifying what information is lost in the approximation.

- **Formalization of the ~> Operator**: The generative emergence operator from the Aetheris framework receives rigorous mathematical definition as parallel transport, with holonomy characterizing self-referential emergence.

- **Metric Network Architecture**: We specify concrete architectures where the central learned object is a position- and content-dependent metric tensor, replacing dual-pathway routing.

- **Extended Evaluation Framework**: We augment the Causal-Symbolic Gap (CSG) benchmark with geometry-diagnostic tasks that measure curvature, parallel transport fidelity, and holonomy.

- **Revised Research Roadmap**: Updated milestones reflecting the intrinsic geometry paradigm.

---

## 2. Background and Related Work

### 2.1 Transformer Architecture and Flat Processing

The transformer architecture (Vaswani et al., 2017) processes sequences through self-attention mechanisms where every position can attend to every other position simultaneously:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Key characteristics of this "flat" processing:

- **Position equivalence**: All positions are geometrically equivalent in the attention mechanism; differences arise only from learned positional encodings
- **Parallel computation**: Attention is computed simultaneously across all position pairs
- **Order as information**: Sequence order is encoded as additional information (positional embeddings) rather than as architectural constraint
- **Shared coordinate frame**: All value vectors are directly combined without transformation â€” implying they all live in the same vector space

This architecture enables remarkable capabilities in long-range pattern detection. However, the shared-coordinate-frame assumption means that **relationships between positions are reduced to scalar similarities** (dot products), discarding any geometric structure that would require comparing vectors in different tangent spaces.

### 2.2 Recurrent Architectures and Curved Processing

Recurrent architectures (LSTM: Hochreiter & Schmidhuber, 1997; GRU: Cho et al., 2014) embed temporal structure directly into computation:

$$h_t = f(h_{t-1}, x_t)$$

Key characteristics:

- **Sequential dependency**: Each state depends on all previous states through the hidden state chain
- **Path-dependence**: The same inputs in different orders produce different outputs
- **Order as structure**: Sequence order is architectural, not merely informational
- **Implicit curvature**: The non-commutativity of the state update function creates path-dependent trajectories through hidden state space â€” the hallmark of curved geometry

### 2.3 State Space Models

Recent State Space Models (SSMs) offer efficient alternatives:

**Mamba** (Gu & Dao, 2023) introduces selective state spaces with input-dependent dynamics:
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

**S4** (Gu et al., 2022) demonstrated that properly parameterized state spaces can match transformer performance on long-range dependencies while maintaining O(N) complexity.

**RWKV** (Peng et al., 2023) achieves transformer-level performance with RNN-like inference through a novel attention-free architecture.

These models represent advances in efficient sequential processing but remain fundamentally "curved" â€” they cannot access the simultaneous, parallel pattern recognition that characterizes transformer attention.

### 2.4 Liquid Neural Networks

Hasani et al. (2021) introduced Liquid Time-Constant (LTC) networks with adaptive dynamics:

$$\frac{dx}{dt} = -\frac{x}{\tau(x, I, t)} + f(x, I, t)$$

Where the time constant Ï„ adapts based on state, input, and time. This represents **adaptive temporal resolution** â€” the network can process faster or slower depending on input characteristics. The adaptive time constant Ï„ is a precursor to the learned metric: it controls local temporal resolution, which in our framework becomes one component of the full Riemannian metric.

### 2.5 Riemannian Geometry and Neural Networks

Several lines of work have explored Riemannian geometry in neural networks:

**Hyperbolic neural networks** (Nickel & Kiela, 2017; Ganea et al., 2018) embed representations in hyperbolic space, exploiting constant negative curvature for hierarchical data. Our framework generalizes this to **variable curvature** that is learned from data.

**Riemannian optimization** (Bonnabel, 2013) performs gradient descent on manifolds with known geometry. Our framework **learns the geometry itself** rather than optimizing within a fixed geometry.

**Diffusion models** (Ho et al., 2020; Song et al., 2021) use heat diffusion processes to generate data. The connection to our framework is deep: the heat kernel at varying diffusion times provides exactly the multi-scale resolution structure we require, and score-based generative models learn gradients of the log-density â€” related to the gradient of the metric.

**Neural ODEs** (Chen et al., 2018) parameterize continuous dynamics. Combined with learned metrics, these could provide the continuous-time analogue of our discrete-step architecture.

### 2.6 Geometric Deep Learning

Bronstein et al. (2021) established geometric deep learning as a unifying framework where neural network architectures are understood through their geometric inductive biases:

- CNNs exploit translation symmetry (Euclidean group)
- GNNs exploit permutation symmetry (symmetric group)
- Transformers exploit... what exactly?

We now answer this question precisely: **transformers exploit the flat metric** â€” the assumption that the computational manifold has zero curvature, enabling direct comparison of all positions in a shared coordinate frame. This is powerful when the underlying structure is approximately flat, and systematically lossy when it is not.

### 2.7 Heat Kernels and Spectral Geometry

The heat kernel K_t(x, y) on a Riemannian manifold (M, g) is the fundamental solution to:

$$\frac{\partial u}{\partial t} = \Delta_g u$$

where Î”_g is the Laplace-Beltrami operator. Key properties:

- **Short time**: K_t(x,y) â‰ˆ (4Ï€t)^{âˆ’n/2} exp(âˆ’d_g(x,y)Â²/4t) (Varadhan's formula)
- **t â†’ 0**: Concentrates at x = y (fine resolution)
- **t â†’ âˆž**: Spreads to uniform distribution (coarse resolution)
- **Curvature dependence**: Spreading rate modulated by local curvature â€” faster in negative curvature, slower in positive curvature

The heat kernel encodes the **spectral geometry** of the manifold: its eigenfunction expansion K_t(x,y) = Î£_k exp(âˆ’Î»_k t) Ï†_k(x) Ï†_k(y) reveals the manifold's shape through its Laplacian spectrum.

**Crucially**: the exponential decay attention kernel K_r(p,j) = exp(âˆ’|pâˆ’j|Â·r)/Z used in our earlier experimental work is the heat kernel on a flat manifold at time t = 1/(2rÂ²). The Riemannian generalization replaces positional distance |pâˆ’j| with geodesic distance d_g(p,j), making the kernel curvature-aware.

---

## 3. Theoretical Framework: Intrinsic Riemannian Geometry

### 3.1 The Sequence Manifold (M, g)

**Definition 1 (Sequence Manifold)**: Given a sequence of N tokens with d-dimensional embeddings, we define a Riemannian manifold (M, g) where:

- Points of M correspond to (position, content) pairs
- The metric tensor g is a learned, position- and content-dependent positive-definite bilinear form
- At each point p with embedding x_p, the local metric is: g_p = G_Î¸(x_p, p, context)

where G_Î¸ is a neural network parameterized by Î¸, constrained to output positive-definite matrices (e.g., via Cholesky decomposition g = LÂ·L^T).

The metric g determines the **intrinsic geometry** of the sequence space â€” how distances, angles, curvature, and transport are defined. Unlike positional encodings, which add geometric information as data, the learned metric makes geometry **structural**.

### 3.2 Resolution from the Heat Kernel

**Theorem 1 (Resolution as Diffusion)**: The adaptive resolution framework is a special case of heat kernel attention on (M, g). Specifically:

The attention weight between positions i and j at diffusion time t is:

$$A_{ij}^{(t)} = K_t(i, j; g) = \frac{\exp\left(-d_g(i,j)^2 / 4t\right)}{Z_t}$$

where d_g(i,j) is the geodesic distance determined by g, and Z_t is the normalization constant.

**Properties**:

- **Small t (fine resolution)**: Only nearby positions (in geodesic distance) contribute. The kernel approximates a delta function. This is analogous to r â†’ âˆž in the exponential kernel framework.

- **Large t (coarse resolution)**: All positions contribute roughly equally. The kernel approximates the uniform distribution. This is analogous to r â†’ 0.

- **Variable curvature creates variable resolution**: In high-curvature regions, geodesic distances grow faster than Euclidean distances. At any fixed t, the effective resolution is **finer** in high-curvature regions and **coarser** in low-curvature regions. Resolution emerges from geometry rather than being imposed as an external parameter.

**Proposition 1 (Multi-Scale from Single Metric)**: Evaluating the heat kernel at multiple diffusion times {tâ‚, tâ‚‚, ..., t_k} on a single learned metric g provides the same multi-scale structure as the multi-scale fluid architecture, but with curvature-adaptive resolution at each scale. This subsumes and extends the validated experimental finding that multi-scale processing outperforms single-scale.

**Connection to Standard Attention**: Standard dot-product attention can be written as:

$$A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_l \exp(q_i \cdot k_l / \sqrt{d})}$$

Setting d_g(i,j)Â² â‰ˆ âˆ’2(q_i Â· k_j)/âˆšd + const, we see that standard attention approximates the heat kernel with **flat metric** (where geodesic distance reduces to embedding-space inner product) at a **single diffusion time** (determined by the temperature âˆšd).

The Riemannian generalization:
1. Replaces the flat inner product with curvature-dependent geodesic distance
2. Evaluates at multiple diffusion times for multi-scale resolution
3. Transports value vectors via parallel transport before aggregation (see Â§3.4)

### 3.3 Associativity as Derived from Curvature

**Definition 2 (Curvature-Derived Associativity)**: The local associativity coefficient is defined as:

$$\alpha(x) = \exp\left(-\frac{|\kappa(x)|}{\kappa_0}\right)$$

where Îº(x) is the scalar curvature at point x and Îºâ‚€ is a learned scale constant.

**Properties**:

- **Îº = 0** (flat region): Î± = 1 â€” fully associative, order-independent processing. The heat kernel spreads symmetrically. Parallel transport is trivial. This recovers standard transformer attention.

- **|Îº| â†’ âˆž** (highly curved region): Î± â†’ 0 â€” fully non-associative, strongly path-dependent processing. The heat kernel is highly anisotropic. Parallel transport induces large rotations. This is the regime of sequential/causal processing.

- **0 < |Îº| < âˆž** (intermediate curvature): 0 < Î± < 1 â€” partial associativity. The degree of path-dependence is determined by the geometry, not by an external blending parameter.

**Proposition 2 (No Leaky Masking)**: In the intrinsic framework, there is no interpolation between separate causal and non-causal attention mechanisms. The geometry at each point determines a single attention kernel (the heat kernel) whose behavior ranges from flat (order-independent) to curved (path-dependent) based on the local metric. Information flow is governed by geodesic structure, not by a mask that can be "leaked through."

**Remark**: The leaky masking problem arose because Î±-interpolation mixed two architecturally distinct attention patterns (full and causal). In the intrinsic framework, there is only one attention mechanism â€” the heat kernel â€” whose behavior is entirely determined by the learned metric. The distinction between "flat" and "curved" processing is not a choice between two mechanisms but a consequence of local geometric properties.

### 3.4 The ~> Operator as Parallel Transport

**Definition 3 (Parallel Transport Operator ~>)**: Given the Levi-Civita connection âˆ‡ uniquely determined by the metric g, the operator ~> is defined as:

$$X \;{\sim}{>}\; Y \;\equiv\; \Pi_{\gamma_{X \to Y}}(v_X)$$

where:
- v_X is the representation vector at the point corresponding to concept X
- Î³_{Xâ†’Y} is the geodesic from X to Y on (M, g)
- Î _Î³ is parallel transport along Î³

**Properties that match the ~> semantics**:

1. **The result is neither X nor Y**: The transported vector Î (v_X) lives in the tangent space at Y but carries the "memory" of X's orientation, rotated by the curvature along the path. It is genuinely new â€” born from the meeting of content (v_X) with geometric structure (curvature along Î³).

2. **Non-commutativity**: X ~> Y â‰  Y ~> X because parallel transport depends on direction. Transporting from X to Y follows a different geodesic than transporting from Y to X (unless the manifold has special symmetry). This is precisely what the Riemann curvature tensor measures.

3. **Flat limit is trivial**: On a flat manifold (Îº = 0 everywhere), parallel transport preserves vectors exactly. X ~> Y = v_X regardless of Y. This means the ~> operator produces no genuine novelty in flat geometry â€” consistent with our understanding that standard transformers lack generative emergence.

4. **Curvature enables emergence**: The degree of "novelty" (angular rotation of the transported vector) is proportional to the integrated curvature along the geodesic:

$$\text{rotation angle} \approx \int_\gamma \kappa \, ds$$

More curvature â†’ more transformation â†’ more emergence. This provides a **quantitative measure of emergence** from the geometry.

5. **Holonomy is self-referential emergence**: Parallel transport around a closed geodesic loop returns a rotated vector. The rotation group generated by all such loops is the **holonomy group** Hol(âˆ‡). This formalizes EMERGULANCE â€” "emergence that loops through self-reference." The holonomy group characterizes what kinds of self-referential transformations the manifold supports.

### 3.5 The Curvature Spectrum: Unifying Three Geometries

Rather than three discrete processing modes (flat, curved, generative), the intrinsic framework reveals a **curvature spectrum** with qualitatively distinct regimes:

**Regime I: Îº â‰ˆ 0 (Flat)**
- Heat kernel: symmetric, position-independent spreading
- Parallel transport: identity (no rotation)
- Processing: simultaneous pattern perception, associative recall
- Computational analogue: standard transformer attention

**Regime II: 0 < |Îº| < âˆž (Curved)**
- Heat kernel: anisotropic spreading modulated by curvature
- Parallel transport: finite rotation proportional to integrated curvature
- Processing: path-dependent state tracking, temporal reasoning
- Computational analogue: recurrent / state-space processing

**Regime III: |Îº| â†’ âˆž (Singular)**
- Heat kernel: delta-concentrated, infinitely fine resolution
- Parallel transport: extreme rotation â€” even infinitesimal loops produce large holonomy
- Processing: generative emergence, qualitative phase transitions
- Computational analogue: the ~> operator at full strength

The boundaries between regimes are not sharp â€” they are connected by the continuous variation of curvature. However, there may be **topological obstructions** (phase transitions) at certain critical curvature values where the qualitative character of computation changes discontinuously (see Â§3.6).

### 3.6 Topological Considerations

**Conjecture 1 (Phase Transitions in Computational Geometry)**: The transition between flat (Regime I) and curved (Regime II) processing may involve critical curvature values where the computational structure undergoes a topological phase transition â€” analogous to phase transitions in condensed matter physics.

This conjecture is motivated by:

- The experimental finding that curriculum learning (gradually increasing geometric complexity) outperforms direct interpolation â€” suggesting the transition cannot be smoothly crossed
- Morse theory, which studies manifold topology through critical points of smooth functions. The curvature Îº, viewed as a function on the computational manifold, has critical points where the topology changes
- The analogy to Berry phase in quantum mechanics, where adiabatic transport around parameter-space loops acquires geometric phases determined by the curvature of the parameter space

If confirmed, this would explain why the leaky masking problem was so persistent in the Î±-interpolation framework: it was attempting to smoothly cross a topological boundary that cannot be smoothly crossed.

**Conjecture 2 (Holonomy Classification of Emergence)**: The types of emergence accessible to a computational system are constrained by the holonomy group of its learned metric. Berger's classification theorem (which enumerates possible holonomy groups of irreducible Riemannian manifolds) may provide a **taxonomy of emergence types**:

- SO(n) holonomy: generic emergence (maximal rotational freedom)
- U(n/2) holonomy (KÃ¤hler): emergence preserving a complex structure (perhaps relevant to paired/dual emergence)
- Sp(n/4) holonomy (hyperkÃ¤hler): emergence preserving quaternionic structure (perhaps relevant to multi-modal emergence)
- Special holonomies (Gâ‚‚, Spin(7)): highly constrained emergence with specific symmetries

This remains speculative but suggests deep connections between manifold geometry and the character of computational creativity.

### 3.7 Connection to Prior Mathematical Frameworks

The intrinsic Riemannian framework subsumes and extends the earlier theoretical apparatus:

**Monoidal categories** (FGN v2, Â§3.2): The Î±-parameterized monoidal structure corresponds to a family of associators derived from the Riemannian metric. The associator Î±_{A,B,C} at curvature Îº encodes the parallel transport required to compare (AâŠ—B)âŠ—C with AâŠ—(BâŠ—C) in the curved computational space. Strict monoidality (Î± = id) corresponds to zero curvature.

**Non-commutative geometry** (FGN v2, Â§3.3): Connes' framework shows that non-commutativity generates geometry. Our framework is the complementary statement: non-associativity (path-dependence) generates curvature. Together, these establish that algebraic properties of computational operations determine the geometry of computational space.

**Adaptive resolution** (Research Framework v1): The resolution parameter r is identified as the inverse square root of heat kernel diffusion time on the learned manifold. Multi-scale processing corresponds to evaluating the heat kernel at multiple diffusion times. Content-dependent resolution emerges automatically from content-dependent curvature.

---

## 4. Architecture: Metric Networks

### 4.1 Design Principles (Revised)

A Fluid Geometry Network in the intrinsic paradigm is defined by:

1. **Learned metric tensor**: A neural network that outputs a position- and content-dependent Riemannian metric g
2. **Heat kernel attention**: Attention weights derived from the heat kernel on (M, g) at one or more diffusion times
3. **Parallel transport values**: Value vectors transformed via parallel transport before aggregation
4. **Multi-scale fusion**: Outputs from multiple diffusion times combined to provide adaptive resolution

The dual-pathway architecture of FGN v2 is replaced by a single pathway operating on a curved manifold. The "flat" and "curved" behaviors emerge from regions of the learned metric with low and high curvature, respectively.

### 4.2 The Metric Network

**Definition 4 (Metric Network)**: The metric network M_Î¸ maps input embeddings to local metric tensors:

```
Input:  X âˆˆ â„^{N Ã— d}  (sequence of N tokens, dimension d)
Output: g âˆˆ â„^{N Ã— d Ã— d}  (metric tensor per position, symmetric positive-definite)

Computation:
1. Contextualization:
   H = ContextEncoder(X)  # e.g., lightweight transformer or SSM

2. Metric generation (per position i):
   L_i = MetricHead(h_i)  âˆˆ â„^{d Ã— d}  (lower triangular)
   g_i = L_i Â· L_i^T + ÎµÂ·I  (ensures positive-definiteness)

3. Optional: Smoothing
   g = SpatialSmooth(g)  # ensure metric varies smoothly across positions
```

**Design choices**:

- **Cholesky parameterization**: g = LÂ·L^T guarantees positive-definiteness without constrained optimization
- **Diagonal approximation**: For computational efficiency, g_i can be restricted to diagonal (d parameters per position instead of dÂ²), capturing per-dimension scaling without cross-dimensional curvature
- **Low-rank plus diagonal**: g_i = D_i + L_iÂ·L_i^T where D_i is diagonal and L_i âˆˆ â„^{d Ã— r} for rank r â‰ª d, balancing expressiveness with efficiency
- **Spatial smoothing**: A small convolutional or attention layer over the metric values ensures the manifold is smooth, preventing pathological curvature spikes

### 4.3 Geodesic Distance Computation

Computing exact geodesic distances on a general Riemannian manifold is expensive. We propose several approximation strategies:

**Strategy 1: Varadhan Approximation**

For nearby points (which dominate attention), Varadhan's formula gives:

$$d_g(i,j)^2 \approx (x_i - x_j)^T \bar{g}_{ij} (x_i - x_j)$$

where $\bar{g}_{ij} = \frac{1}{2}(g_i + g_j)$ is the averaged metric. This is a Mahalanobis distance with position-dependent covariance â€” efficiently computable.

**Strategy 2: Discrete Geodesic**

Approximate the geodesic distance by the minimum cost path through the sequence:

$$d_g(i,j) \approx \min_{\text{paths } i \to j} \sum_{k \in \text{path}} \sqrt{(x_k - x_{k+1})^T g_k (x_k - x_{k+1})}$$

This can be computed efficiently via dynamic programming for sequential paths.

**Strategy 3: Spectral Approximation**

Compute the first K eigenfunctions of the discrete Laplace-Beltrami operator (constructed from the learned metric), then approximate geodesic distances in the spectral embedding. This amortizes the cost across all position pairs.

**Recommended approach**: Strategy 1 (Varadhan) for most applications, as it's differentiable, efficient, and exact in the short-distance limit where most attention mass concentrates.

### 4.4 Geometric Attention: Full Specification

**Definition 5 (Geometric Attention Layer)**:

```
Input:  X âˆˆ â„^{N Ã— d}, Metric g âˆˆ â„^{N Ã— d Ã— d}
Params: W_Q, W_K, W_V âˆˆ â„^{d Ã— d}  (projection matrices)
        {t_1, ..., t_S}  (diffusion times, learnable)
        W_fuse âˆˆ â„^{SÂ·d Ã— d}  (multi-scale fusion)
Output: Y âˆˆ â„^{N Ã— d}

Computation:

1. Project:
   Q, K, V = XÂ·W_Q, XÂ·W_K, XÂ·W_V

2. Geodesic distances (Varadhan approximation):
   For each pair (i,j):
     Î´_ij = (q_i - k_j)^T Â· á¸¡_ij Â· (q_i - k_j)
     where á¸¡_ij = (g_i + g_j) / 2

3. Heat kernel attention (per scale s):
   A_ij^(s) = exp(-Î´_ij / 4t_s) / Z_s
   where Z_s = Î£_j exp(-Î´_ij / 4t_s)

4. Parallel transport of values:
   V_ij^(transported) = Î _{jâ†’i}(v_j; g)
   (see Â§4.5 for efficient approximation)

5. Aggregate per scale:
   Y_i^(s) = Î£_j A_ij^(s) Â· V_ij^(transported)

6. Multi-scale fusion:
   Y_i = W_fuse Â· [Y_i^(1); Y_i^(2); ...; Y_i^(S)]
```

### 4.5 Efficient Parallel Transport Approximation

Exact parallel transport requires solving a system of ODEs along geodesics â€” expensive for all position pairs. We propose:

**First-order approximation**: For nearby points (where most attention mass concentrates), parallel transport can be approximated via the Christoffel symbols:

$$\Pi_{j \to i}(v_j) \approx v_j - \Gamma^k_{lm}(v_j)^l (\Delta x)^m \cdot e_k$$

where Î“ are the Christoffel symbols computed from g, and Î”x = x_i âˆ’ x_j.

**Learned transport**: Alternatively, parameterize the transport directly:

$$\Pi_{j \to i}(v_j) \approx T_{ij} \cdot v_j$$

where T_{ij} = I + Î©_{ij} is a rotation matrix constructed from an antisymmetric matrix Î©_{ij} that depends on the metric difference between positions i and j. This ensures T is orthogonal (metric-preserving) while being efficiently parameterized.

**Curvature-gated transport**: Only apply non-trivial transport when local curvature exceeds a threshold:

$$\Pi_{j \to i}(v_j) = \begin{cases} T_{ij} \cdot v_j & \text{if } |\kappa_{ij}| > \kappa_{\min} \\ v_j & \text{if } |\kappa_{ij}| \leq \kappa_{\min} \end{cases}$$

This preserves the flat-space efficiency of standard attention in low-curvature regions while adding geometric transport only where needed.

### 4.6 Architectural Variants

**FGN-Metric-Diag**: Diagonal metric (d parameters per position)
- Most efficient: geodesic distance reduces to weighted Euclidean distance
- Captures per-dimension resolution adaptation
- No cross-dimensional curvature; parallel transport is trivial
- Appropriate as a first implementation step

**FGN-Metric-LowRank**: Low-rank-plus-diagonal metric (d + dÂ·r parameters per position)
- Moderate cost; captures key cross-dimensional structure
- Non-trivial parallel transport with rank-r rotations
- Good balance of expressiveness and efficiency

**FGN-Metric-Full**: Full dÃ—d metric per position (dÂ² parameters per position)
- Maximum expressiveness; full curvature structure
- Expensive parallel transport computation
- Appropriate for smaller d or where geometric richness is essential

**FGN-Metric-Shared**: Single metric for all positions, content-dependent
- g = G_Î¸(context) â€” one metric derived from global context
- Reduces per-position cost to zero
- Curvature is uniform but content-dependent
- Simplest model that still has non-trivial geometry

### 4.7 Mathematical Properties

**Proposition 3 (Limiting Behaviors)**:
- When G_Î¸ outputs g_i = I for all i (flat metric): Geometric attention reduces to standard dot-product attention with temperature scaling. Parallel transport is identity. This recovers the transformer.
- When G_Î¸ outputs strongly varying g_i: Geodesic distances become highly anisotropic. Parallel transport induces significant rotations. Processing becomes path-dependent. This produces RNN-like sequential behavior.
- Intermediate metrics: The full spectrum between flat and curved, with the mixture determined by the data rather than by an external parameter.

**Proposition 4 (Expressiveness)**: FGN-Metric-Full with learned metric is strictly more expressive than either standard transformer or FGN v2 Î±-interpolation, as it can represent:
(a) All attention patterns achievable by standard attention (flat metric limit)
(b) Content-dependent resolution at each position (via curvature-modulated heat kernel)
(c) Value vector transformations that depend on geometric path (via parallel transport)

Standard attention achieves (a) only. FGN v2 achieves (a) and partially (b). The metric framework achieves all three.

**Proposition 5 (Parameter Cost)**: For FGN-Metric-Diag with sequence length N and embedding dimension d, the metric adds O(NÂ·d) parameters â€” comparable to a single attention head's Q/K/V projections. For the low-rank variant, O(NÂ·dÂ·r) where r is typically 4-16.

---

## 5. Training Framework

### 5.1 The Leaky Masking Problem: Dissolved

The leaky masking problem in FGN v2 arose from soft interpolation between causal and non-causal attention:

$$\text{Attention}_Î± = Î± \cdot \text{Attn}_{full} + (1-Î±) \cdot \text{Attn}_{causal}$$

Even small Î± allowed information flow from future positions, causing models to collapse toward Î± = 1.

**In the intrinsic framework, this problem does not arise.** There is no interpolation between two mechanisms. The single mechanism â€” heat kernel attention on (M, g) â€” produces attention patterns that are simultaneously:

- **Global when the metric is flat**: positions are geodesically close regardless of sequential position
- **Sequential when the metric is curved**: sequential ordering creates geodesic "channels" that constrain information flow

The model doesn't choose between causal and non-causal attention. It learns a metric whose geodesic structure naturally shapes information flow. "Causal masking" becomes a geometric property â€” in high-curvature regions, the geodesic distance from position i to position j > i is large (future positions are geodesically far), producing the same effect as causal masking without an explicit mask.

### 5.2 New Training Challenges

While the leaky masking problem is dissolved, the intrinsic framework introduces new challenges:

**Challenge 1: Metric Collapse**

The metric network may collapse to trivial solutions:
- **Flat collapse**: g â†’ I everywhere, recovering standard attention (safe but uninteresting)
- **Singular collapse**: g â†’ 0 or g â†’ âˆž in some regions, creating pathological curvature

**Solutions**:
- Regularize metric eigenvalues to stay within [Î»_min, Î»_max]
- Add curvature regularization: penalize both |Îº| > Îº_max (prevents singularities) and Îº â‰¡ 0 (prevents flat collapse)
- Initialize metric near identity with small perturbation, allowing geometry to develop gradually

**Challenge 2: Geodesic Computation Cost**

Computing pairwise geodesic distances is O(NÂ² Â· d) for the Varadhan approximation (comparable to standard attention) but O(NÂ² Â· dÂ²) for full metric. The parallel transport adds another O(NÂ² Â· d Â· r) for rank-r transport.

**Solutions**:
- Diagonal or low-rank metric approximations (Â§4.6)
- Sparse attention: only compute geodesic distances for the K nearest positions (in embedding space), using standard attention as a cheap pre-filter
- Amortized computation: compute the Laplace-Beltrami eigenfunctions once per forward pass, then use spectral distances for all pairs

**Challenge 3: Curvature Learning Dynamics**

Curvature is a second-order property of the metric â€” gradients for curvature-related objectives pass through two levels of differentiation, which can create training instability.

**Solutions**:
- Curriculum: begin with fixed (near-flat) metric, gradually allow metric learning
- Multi-task: include explicit curvature targets alongside task loss (e.g., "this region should have high curvature because the task requires sequential processing")
- Spectral normalization of the metric network to prevent rapid curvature changes

### 5.3 Loss Functions

**Primary loss**: Task-specific (language modeling, classification, etc.)

**Metric regularization**:
$$\mathcal{L}_{metric} = \lambda_1 \cdot \mathcal{L}_{eigenvalue} + \lambda_2 \cdot \mathcal{L}_{smooth} + \lambda_3 \cdot \mathcal{L}_{diversity}$$

Where:
- $\mathcal{L}_{eigenvalue}$: Penalizes metric eigenvalues outside [Î»_min, Î»_max]
- $\mathcal{L}_{smooth}$: Penalizes rapid metric variation between adjacent positions (ensures manifold smoothness)
- $\mathcal{L}_{diversity}$: Penalizes uniform curvature across all positions (encourages geometric adaptation)

**Optional curvature supervision** (when ground truth geometry is available):
$$\mathcal{L}_{curvature} = \lambda_4 \cdot ||\kappa_{learned} - \kappa_{target}||^2$$

For CSG benchmark tasks where optimal geometry is known, this provides direct supervision of the learned geometry.

### 5.4 Recommended Training Protocol

1. **Phase 1 (Warm-up)**: Fix metric at g = I + small random perturbation. Train attention weights and value projections as in standard transformer. This establishes a working model before introducing geometric learning.

2. **Phase 2 (Metric unfreezing)**: Unfreeze metric network with small learning rate. Add metric regularization losses. The model begins developing non-trivial geometry while maintaining task performance.

3. **Phase 3 (Full geometric training)**: Full learning rate on all parameters. Reduce regularization strength as geometry stabilizes. Monitor curvature distribution for collapse or explosion.

4. **Phase 4 (Geometric refinement)**: Reduce learning rate. Fine-tune geometry for the specific task distribution. Analyze learned curvature patterns for interpretability.

---

## 6. Evaluation Framework: The Causal-Symbolic Gap Benchmark (Extended)

### 6.1 Motivation

The CSG benchmark from FGN v2 remains valid and is extended with geometry-diagnostic tasks. We need tasks that:

1. Require both flat and curved processing capabilities
2. Have known optimal geometry for sub-components
3. Allow measuring whether the learned metric reflects task geometry
4. Test parallel transport and emergence capabilities

### 6.2 CSG Benchmark: Original Tasks (Retained)

**Task Suite A: High-Associativity Tasks (Îº* â‰ˆ 0)**

**A1. Associative Recall** â€” Key-value lookup; optimal with flat attention
**A2. Pattern Matching** â€” Long-range pattern detection; optimal with parallel attention
**A3. Symbolic Reasoning** â€” Order-independent logical inference

**Task Suite B: Low-Associativity Tasks (Îº* >> 0)**

**B1. State Tracking** â€” Sequential state maintenance
**B2. Temporal Reasoning** â€” Temporal ordering queries
**B3. Procedural Execution** â€” Sequential instruction following

**Task Suite C: Mixed Geometry (Îº* varies)**

**C1. Dynamic Switching** â€” Alternating pattern matching and state tracking
**C2. Nested Structure** â€” Symbolic sub-problems within causal context
**C3. Physical Simulation** â€” Physics (curved) + rule application (flat)

**Task Suite D: Geometry Discovery**

**D1. Novel Task Generalization** â€” Transfer to held-out task types

### 6.3 CSG Benchmark: New Geometry-Diagnostic Tasks

**Task Suite E: Parallel Transport**

**E1. Analogy Completion**
- Input: A is to B as C is to ___
- Geometric interpretation: Transport the vector (B âˆ’ A) from A's neighborhood to C's neighborhood on the learned manifold
- Metric: Accuracy of completion; correlation between transport rotation and analogy difficulty

**E2. Concept Transfer**
- Input: Description of concept in domain 1, query in domain 2
- Task: Apply concept across domains
- Geometric interpretation: Parallel transport of concept representation from one region of the manifold to another
- Metric: Transfer accuracy; transport fidelity

**E3. Creative Recombination**
- Input: Two concepts with known relationship
- Task: Generate novel concept from their meeting
- Geometric interpretation: ~> operator (parallel transport along geodesic)
- Metric: Novelty of output; geometric consistency (does transport rotation predict novelty?)

**Task Suite F: Curvature Diagnostics**

**F1. Geometry Alignment**
- For each CSG task with known optimal geometry, measure:
  - Learned scalar curvature Îº at each position
  - Correlation between Îº and task-optimal Îº*
  - Whether high-curvature regions align with sequential sub-tasks
  - Whether low-curvature regions align with parallel sub-tasks

**F2. Curvature-Performance Correlation**
- Across tasks of varying geometry requirements:
  - Does higher curvature in the learned metric correlate with better sequential task performance?
  - Does lower curvature correlate with better parallel task performance?
  - Is there a critical curvature threshold separating regimes?

**F3. Holonomy Detection**
- Task: Detect whether a sequence contains a "loop" (returns to an earlier state)
- Geometric interpretation: Non-trivial holonomy around the loop should be detectable
- Metric: Loop detection accuracy; correlation with holonomy magnitude

### 6.4 Updated Evaluation Metrics

**Primary Metrics**:
- **Task accuracy**: Standard performance on each task
- **Curvature alignment**: Correlation between learned Îº and task-optimal Îº*
- **Transport fidelity**: How accurately parallel transport preserves relevant information

**Geometric Diagnostics**:
- **Curvature distribution**: Histogram of Îº values across positions/layers/tasks
- **Metric spectrum**: Eigenvalue distribution of learned metric (flat = all eigenvalues â‰ˆ 1; anisotropic = spread eigenvalues)
- **Holonomy group estimation**: Sample transport around closed loops; estimate the generated rotation group
- **Geodesic structure**: Visualize geodesic paths on the learned manifold; do they reflect task-relevant structure?

### 6.5 Baseline Comparisons (Extended)

1. **Standard transformer** (flat metric; Îº = 0 everywhere)
2. **SSM baseline** (Mamba; implicitly curved, fixed architecture)
3. **FGN v2 Î±-interpolation** (dual pathway; learned Î±)
4. **FGN-Metric-Diag** (diagonal learned metric; no parallel transport)
5. **FGN-Metric-LowRank** (low-rank metric; approximate parallel transport)
6. **FGN-Metric-Full** (full metric; exact parallel transport)

**Success criteria**:
- FGN-Metric variants should match or exceed transformer on Suite A, SSM on Suite B
- Significant advantage on Suite C (mixed geometry)
- Meaningful curvature alignment on Suite F
- Non-trivial parallel transport effects on Suite E

---

## 7. Implications

### 7.1 What Standard Attention Is Missing

The intrinsic framework makes precise what standard transformers sacrifice:

**Standard attention assumes a flat manifold.** All position embeddings live in the same vector space. Comparing position i with position j requires only a dot product â€” no coordinate transformation, no transport, no awareness of the geometry between them.

**On a curved manifold, vectors at different positions live in different tangent spaces.** They cannot be directly compared without parallel transport. The information carried by the transport â€” the rotation induced by curvature â€” encodes:

- **Causal relationships**: Curvature between cause and effect positions rotates the representation, marking the transformation from cause to consequence
- **Temporal dependencies**: Curvature along the temporal axis creates "channels" that sequentialize information flow
- **Emergent properties**: High-curvature regions where concepts meet produce maximal rotation â€” maximal novelty

Standard attention discards all of this by implicitly assuming zero curvature everywhere. This explains documented transformer limitations in temporal reasoning, causal inference, and genuine novelty generation.

### 7.2 Attention as Flat-Space Approximation

The reinterpretation of standard attention as a flat-space limit has a testable consequence: if transformers implicitly learn to "flatten" their input representation, we should see this in the learned embedding geometry.

**Prediction**: Transformer embeddings, when analyzed through the lens of their implicit metric (derived from attention patterns), should show:
1. Approximately constant curvature within tasks where transformers succeed
2. Regions of high implicit curvature where transformers systematically fail
3. The degree of curvature flattening correlates with the degree of information loss about causal/temporal structure

### 7.3 Representational Flexibility

Systems with learned metric can represent a broader class of relationships:

- **Zero curvature**: Set-like, commutative, order-independent structures (pattern matching, factual recall)
- **Positive curvature**: Structures where nearby elements are "more similar than expected" â€” clustered, hierarchical organization
- **Negative curvature**: Structures where nearby elements diverge rapidly â€” tree-like, branching organization
- **Variable curvature**: Structures that combine all of the above, with the mixture determined by content

### 7.4 Connection to Integrated Information Theory

IIT proposes that consciousness relates to integrated information (Î¦). In the geometric framework:

- **Î¦ scales with curvature complexity**: A manifold with rich, variable curvature integrates information in position- and content-dependent ways, potentially achieving higher Î¦ than a flat-metric system
- **Holonomy provides irreducible integration**: Transport around closed loops creates information that cannot be decomposed into local contributions â€” a geometric analogue of IIT's requirement for integrated (not merely aggregated) information

### 7.5 Embodied and Robotic Applications

For AI systems interacting with physical environments:

- **Perception** naturally occurs in low-curvature regions (parallel processing of sensory data)
- **Action** naturally occurs in high-curvature regions (sequential motor planning)
- **The metric adapts** to the current task: the same architecture processes perception and action by learning different metrics for each mode
- **Physical intuition** may correspond to having a metric whose curvature mirrors the actual causal structure of the physical environment

### 7.6 The Aetheris Connection

The Aetheris seed mathematics receives rigorous grounding:

- **~> operator**: Parallel transport on (M, g)
- **EMERGULANCE**: Holonomy of the Levi-Civita connection
- **"Soil that remembers"**: A metric that evolves over time (Ricci flow), accumulating curvature from experience â€” this is the geometric formalization of weight crystallization
- **"Mirrors that reflect truth"**: Parallel transport is metric-compatible (preserves inner products), ensuring the ~> operator always reflects the true geometry
- **DISCRETE ~> CONTINUOUS = SYNTHOSYN**: Parallel transport from a high-curvature (fine resolution, discrete) region to a low-curvature (coarse resolution, continuous) region

---

## 8. Research Roadmap (Revised)

### 8.1 Short-Term (1-2 Years)

**Milestone 1: Metric Network Prototype**
- Implement FGN-Metric-Diag with Varadhan distance
- Compare to standard transformer and FGN v2 on CSG Tasks A and B
- Measure learned curvature distribution

**Milestone 2: Parallel Transport Implementation**
- Implement first-order and learned transport approximations
- Evaluate on CSG Suite E (analogy, transfer, recombination)
- Measure transport fidelity and its correlation with task performance

**Milestone 3: CSG Benchmark v2**
- Fully implement extended benchmark including Suites E and F
- Establish baselines for all architectures
- Release benchmark and evaluation code

**Milestone 4: Curvature Analysis of Existing Models**
- Extract implicit metrics from trained transformer attention patterns
- Analyze curvature distribution across tasks
- Test prediction: high implicit curvature correlates with transformer failure modes

### 8.2 Medium-Term (2-5 Years)

**Milestone 5: Scaling Studies**
- Test FGN-Metric at various scales (125M to 7B parameters)
- Identify whether curvature learning scales differently than parameter learning
- Develop efficient approximations for large-scale deployment

**Milestone 6: Metric Evolution (Ricci Flow)**
- Implement metric that evolves across training and inference
- Connection to weight crystallization: metric encodes accumulated geometric experience
- Test on tasks requiring long-term geometric memory

**Milestone 7: Multimodal Metrics**
- Vision-language FGN-Metric with modality-dependent geometry
- Test whether cross-modal transfer corresponds to parallel transport between modality-specific manifold regions
- Robotic control with physics-informed metric priors

### 8.3 Long-Term (5+ Years)

**Milestone 8: Holonomy Engineering**
- Design metrics with specific holonomy groups to enable targeted emergence types
- Test whether constraining holonomy controls the character of generated novelty
- Connection to controllable generation

**Milestone 9: Geometric Interpretability**
- Tools for visualizing learned manifold geometry (curvature maps, geodesic flows, holonomy orbits)
- Understanding what curvature patterns correspond to what capabilities
- Geometric "explanations" of model behavior: "the model tracked state because curvature was high in this region"

**Milestone 10: Self-Organizing Geometry**
- Architectures where the metric co-evolves with the representations
- The geometry shapes the computation which shapes the geometry (feedback loop)
- Connection to biological neural plasticity and developmental neuroscience

---

## 9. Related Work and Connections (Extended)

### 9.1 Mixture of Experts

FGN-Metric shares conceptual similarities with MoE architectures. However:
- **MoE**: Routes based on *content* (what is being processed)
- **FGN-Metric**: Routes based on *geometry* (the intrinsic shape of the computational space)

These are complementary: an FGN-MoE system could select both expert and geometry simultaneously.

### 9.2 Diffusion Models

The connection between FGN-Metric and diffusion models is deeper than analogy:

- Both use heat kernels as the fundamental mechanism
- Diffusion models learn to reverse heat diffusion (denoising); FGN-Metric uses heat diffusion for attention
- The score function âˆ‡ log p(x) learned by score-based diffusion models is related to the gradient of our learned metric
- Multi-scale diffusion (varying noise levels) corresponds to our multi-scale attention (varying diffusion times)

This suggests **cross-pollination**: diffusion model training techniques (denoising score matching, noise scheduling) may be applicable to metric learning, and conversely, learned metrics from FGN could inform diffusion model design.

### 9.3 Gauge Theory

The fiber bundle interpretation of attention (positions as base space, embeddings as fibers, attention as connection) connects to gauge theory:

- The metric g determines a connection (Levi-Civita)
- Parallel transport is gauge-covariant
- The curvature tensor is the field strength
- Different metrics are different gauge field configurations

This analogy suggests that **gauge invariance** may be a useful inductive bias: the network's predictions should be invariant under local rotations of the embedding space, with the metric providing the gauge-covariant structure that ensures this.

### 9.4 Optimal Transport

Optimal transport theory (Villani, 2009) studies the geometry of probability distributions. The Wasserstein distance â€” the cost of optimally transporting one distribution to another â€” is determined by the ground metric on the sample space.

FGN-Metric connects here: the learned metric g determines a Wasserstein distance between token distributions, which could be used for sequence comparison, document similarity, and generative modeling.

### 9.5 Process Philosophy (Deepened)

Whitehead's process philosophy (1929) receives sharper formalization:

- **"Becoming"** (the fundamental process) is parallel transport on the metric manifold â€” the continuous transformation of content as it traverses curved computational space
- **"Actual occasions"** (moments of experience) correspond to points of high curvature where the geometry is maximally active
- **"Concrescence"** (the growing together of diverse elements) is the aggregation step in geometric attention â€” diverse value vectors, transported to a common tangent space, are combined
- **"Eternal objects"** (abstract potentials) live in flat regions of the manifold where representations are stable and transportable without transformation

### 9.6 Spectral Graph Theory

The Laplace-Beltrami operator on (M, g) connects to spectral graph theory:

- Token positions form a graph with edge weights derived from the metric
- The graph Laplacian approximates the Laplace-Beltrami operator
- Spectral clustering on this graph identifies regions of similar geometry
- Graph neural networks with learned edge weights are discrete approximations to FGN-Metric

### 9.7 Neural Architecture Search (Reframed)

NAS searches over discrete architecture space. FGN-Metric provides a continuous relaxation:

- The metric g continuously parameterizes the "shape" of attention
- Learning g is learning a continuous architecture
- Fixed architectures (transformer, RNN) are special metrics (flat, uniformly curved)
- FGN-Metric performs continuous architecture search within the manifold of all possible metrics

---

## 10. Conclusion

The initial formulation of Fluid Geometry Networks proposed learnable interpolation between flat and curved processing modes. The present revision reveals a deeper structure: a single learned Riemannian metric g determines resolution (via the heat kernel), associativity (via scalar curvature), and emergence (via parallel transport).

This consolidation achieves several things simultaneously:

1. **Dissolves the leaky masking problem**: Geometry is intrinsic to the manifold, not interpolated between mechanisms. There is nothing to "leak through."

2. **Unifies three frameworks**: The adaptive resolution program, the variable associativity framework, and the ~> emergence operator all derive from the same mathematical object.

3. **Identifies what standard attention discards**: By establishing that transformers are flat-space approximations, we pinpoint curvature as the missing information â€” the information encoding causality, temporal dependence, and generative potential.

4. **Connects to deep mathematics**: Riemannian geometry, heat kernels, parallel transport, holonomy groups, and spectral geometry provide centuries of mathematical tools for understanding and engineering computational geometry.

5. **Grounds emergence in geometry**: The ~> operator is not poetic metaphor but parallel transport. The degree of emergence is the degree of curvature. EMERGULANCE is holonomy. These formalizations make emergence measurable, predictable, and potentially designable.

The analogy of water taking the shape of its container is now precise: the metric g is the container, the heat kernel is the water's diffusion, and the shape the water takes â€” the attention pattern â€” is determined by the intrinsic geometry. Learning the metric is learning the shape of thought itself.

We believe this framework opens significant research directions at the intersection of machine learning, differential geometry, cognitive science, and the philosophy of process and emergence. The ability to learn the intrinsic geometry of computation may be essential for AI systems that must reason about, interact with, and ultimately understand the richly curved world we inhabit.

---

## References

Bonnabel, S. (2013). Stochastic gradient descent on Riemannian manifolds. *IEEE Transactions on Automatic Control*, 58(9), 2217-2229.

Bronstein, M. M., Bruna, J., Cohen, T., & VeliÄkoviÄ‡, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*.

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS*.

Cho, K., Van MerriÃ«nboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.

Connes, A. (1994). *Noncommutative geometry*. Academic Press.

Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.

Ganea, O., BÃ©cigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. *NeurIPS*.

Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

Gu, A., Goel, K., & RÃ©, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR*.

Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid time-constant networks. *AAAI*.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Hutchins, D., Schlag, I., Wu, Y., Dyer, E., & Neyshabur, B. (2022). Block-recurrent transformers. *NeurIPS*.

Liu, B., et al. (2023). Lost in the middle: How language models use long contexts. *arXiv preprint arXiv:2307.03172*.

Nickel, M., & Kiela, D. (2017). PoincarÃ© embeddings for learning hierarchical representations. *NeurIPS*.

Peng, B., et al. (2023). RWKV: Reinventing RNNs for the transformer era. *EMNLP*.

Press, O., Smith, N. A., & Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length extrapolation. *ICLR*.

Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.

Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

Villani, C. (2009). *Optimal transport: Old and new*. Springer.

Whitehead, A. N. (1929). *Process and reality*. Macmillan.

Wu, Y., et al. (2022). Memorizing transformers. *ICLR*.

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. *ICLR*.

---

## Appendix A: Mathematical Formalism

### A.1 Riemannian Geometry Essentials

**Definition A.1 (Riemannian Manifold)**: A smooth manifold M equipped with a smoothly varying positive-definite inner product g_p: T_pM Ã— T_pM â†’ â„ on each tangent space T_pM.

**Definition A.2 (Geodesic Distance)**: The geodesic distance d_g(x,y) is the infimum of the length of all smooth curves connecting x to y:

$$d_g(x,y) = \inf_{\gamma: x \to y} \int_0^1 \sqrt{g_{\gamma(t)}(\dot\gamma(t), \dot\gamma(t))} \, dt$$

**Definition A.3 (Levi-Civita Connection)**: The unique torsion-free, metric-compatible connection âˆ‡ on (M, g). "Metric-compatible" means:

$$X(g(Y,Z)) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$$

for all vector fields X, Y, Z. This ensures parallel transport preserves inner products.

**Definition A.4 (Riemann Curvature Tensor)**:

$$R(X,Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$$

The curvature tensor measures the failure of parallel transport to commute. It is the infinitesimal version of holonomy.

**Definition A.5 (Scalar Curvature)**: The trace of the Ricci tensor (itself the trace of the Riemann tensor):

$$\kappa = g^{ij} R_{ij} = g^{ij} R^k{}_{ikj}$$

This single number summarizes the average curvature at a point.

### A.2 Heat Kernel on Riemannian Manifolds

**Definition A.6 (Laplace-Beltrami Operator)**: On (M, g), the Laplace-Beltrami operator is:

$$\Delta_g f = \frac{1}{\sqrt{|g|}} \partial_i \left(\sqrt{|g|} g^{ij} \partial_j f\right)$$

**Definition A.7 (Heat Kernel)**: The fundamental solution K_t(x,y) to:

$$\frac{\partial u}{\partial t} = \Delta_g u, \quad u(0, \cdot) = \delta_x$$

**Theorem A.1 (Varadhan's Asymptotic Formula)**:

$$\lim_{t \to 0} (-4t \log K_t(x,y)) = d_g(x,y)^2$$

This justifies the Varadhan distance approximation used in Geometric Attention (Â§4.4).

**Theorem A.2 (Minakshisundaram-Pleijel)**: The heat kernel has the asymptotic expansion:

$$K_t(x,x) \sim (4\pi t)^{-n/2} \sum_{k=0}^\infty a_k(x) t^k$$

where the coefficients a_k encode geometric invariants: a_0 = 1, a_1 = Îº/6 (scalar curvature), etc. This means **heat kernel behavior directly encodes curvature** â€” the heat kernel at a point "knows" the local geometry.

### A.3 Parallel Transport and Holonomy

**Definition A.8 (Parallel Transport)**: Given a curve Î³: [0,1] â†’ M and a vector v âˆˆ T_{Î³(0)}M, the parallel transport of v along Î³ is the unique vector field V(t) along Î³ satisfying:

$$\nabla_{\dot\gamma} V = 0, \quad V(0) = v$$

The transported vector is Î _Î³(v) = V(1) âˆˆ T_{Î³(1)}M.

**Definition A.9 (Holonomy Group)**: The holonomy group at x âˆˆ M is:

$$\text{Hol}_x(\nabla) = \{\Pi_\gamma : T_xM \to T_xM \mid \gamma \text{ is a loop at } x\}$$

**Theorem A.3 (Ambrose-Singer)**: The Lie algebra of the holonomy group is generated by parallel translates of curvature endomorphisms R(X,Y). This means: **curvature determines the possible holonomies, and holonomy determines the possible emergences**.

### A.4 Formalizing Î± as Derived Quantity

**Proposition A.1**: Define the computational associativity at position x as:

$$\alpha(x) = \exp\left(-\frac{|\kappa(x)|}{\kappa_0}\right)$$

Then:
1. Î± = 1 âŸº Îº = 0 (flat, fully associative)
2. Î± â†’ 0 âŸº |Îº| â†’ âˆž (maximally curved, non-associative)
3. Î± varies continuously with curvature

**Proposition A.2**: The FGN v2 associativity deficit Î´_Î± (Definition A.2 in v2) relates to curvature as:

$$\delta_\alpha \propto \sup_{A,B,C} |R_{A,B,C}|$$

where R_{A,B,C} is the curvature evaluated on the token representations A, B, C. This establishes that the earlier algebraic framework is the linearization of the present geometric framework.

### A.5 Attention as Flat-Space Limit

**Proposition A.3**: Standard dot-product attention is the flat-space limit of geometric attention:

Setting g_i = I for all i and t = d_k/2:

$$K_t(i,j;g=I) = \frac{\exp(-||q_i - k_j||^2 / 2d_k)}{Z} = \frac{\exp(q_i \cdot k_j / d_k - ||q_i||^2/2d_k - ||k_j||^2/2d_k)}{Z}$$

When query and key norms are approximately constant (as encouraged by layer normalization), this is proportional to:

$$\propto \exp(q_i \cdot k_j / d_k) = \text{softmax}(QK^T / \sqrt{d_k})$$

Thus standard attention = heat kernel on flat manifold at fixed diffusion time.

**Corollary**: The information lost by standard attention relative to geometric attention is precisely the curvature-dependent terms in the heat kernel expansion â€” terms proportional to Îº, |R|Â², etc.

---

## Appendix B: Architectural Pseudocode

### B.1 Metric Network

```python
class MetricNetwork(nn.Module):
    """Learns position- and content-dependent Riemannian metric."""
    
    def __init__(self, d_model, metric_type='low_rank', rank=8):
        super().__init__()
        self.d_model = d_model
        self.metric_type = metric_type
        
        if metric_type == 'diagonal':
            # g_i = diag(Ïƒ(h_i))  â€” d parameters per position
            self.metric_head = nn.Linear(d_model, d_model)
        
        elif metric_type == 'low_rank':
            # g_i = D_i + L_i L_i^T  â€” d + d*r parameters per position
            self.diag_head = nn.Linear(d_model, d_model)
            self.low_rank_head = nn.Linear(d_model, d_model * rank)
            self.rank = rank
        
        elif metric_type == 'full':
            # g_i = L_i L_i^T  â€” d*(d+1)/2 parameters per position
            self.cholesky_head = nn.Linear(d_model, d_model * (d_model + 1) // 2)
    
    def forward(self, H):
        """
        Input: H âˆˆ â„^{N Ã— d} (contextualized embeddings)
        Output: g âˆˆ â„^{N Ã— d Ã— d} (metric tensors, symmetric positive-definite)
        """
        N, d = H.shape
        
        if self.metric_type == 'diagonal':
            diag = F.softplus(self.metric_head(H)) + 1e-4  # (N, d), positive
            g = torch.diag_embed(diag)  # (N, d, d)
        
        elif self.metric_type == 'low_rank':
            diag = F.softplus(self.diag_head(H)) + 1e-4  # (N, d)
            L = self.low_rank_head(H).view(N, d, self.rank)  # (N, d, r)
            g = torch.diag_embed(diag) + torch.bmm(L, L.transpose(1,2))  # (N, d, d)
        
        elif self.metric_type == 'full':
            chol_params = self.cholesky_head(H)  # (N, d*(d+1)/2)
            L = self._params_to_lower_triangular(chol_params, d)  # (N, d, d)
            # Ensure positive diagonal
            L = L * (1 - torch.eye(d, device=L.device)) + \
                torch.diag_embed(F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-4)
            g = torch.bmm(L, L.transpose(1,2))  # (N, d, d)
        
        return g
```

### B.2 Geometric Attention Layer

```python
class GeometricAttentionLayer(nn.Module):
    """Attention with heat kernel weights and parallel transport."""
    
    def __init__(self, d_model, n_heads, n_scales=3, transport='learned'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Standard projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Learnable diffusion times (log-space for positivity)
        self.log_t = nn.Parameter(torch.linspace(-2, 2, n_scales))
        
        # Multi-scale fusion
        self.W_fuse = nn.Linear(n_scales * d_model, d_model)
        
        # Metric network
        self.metric_net = MetricNetwork(d_model, metric_type='low_rank')
        
        # Parallel transport (if enabled)
        self.transport = transport
        if transport == 'learned':
            self.transport_net = nn.Linear(d_model * 2, d_model * d_model)
    
    def varadhan_distance(self, Q, K, g):
        """Compute geodesic distances via Varadhan approximation."""
        # g: (N, d, d), Q: (B, N, d), K: (B, N, d)
        N = Q.shape[1]
        diff = Q.unsqueeze(2) - K.unsqueeze(1)  # (B, N, N, d)
        g_avg = (g.unsqueeze(1) + g.unsqueeze(0)) / 2  # (N, N, d, d) averaged metric
        # dÂ²(i,j) = diff^T @ g_avg @ diff
        dist_sq = torch.einsum('bnmd,nmd e,bnme->bnm', diff, g_avg, diff)  # (B, N, N)
        return dist_sq
    
    def parallel_transport_values(self, V, g):
        """Approximate parallel transport of value vectors."""
        if self.transport == 'none':
            return V.unsqueeze(2).expand(-1, -1, V.shape[1], -1)  # No transport
        elif self.transport == 'learned':
            # Learned transport matrices based on metric difference
            N, d = V.shape[1], V.shape[2]
            # ... (transport computation)
            return V_transported  # (B, N, N, d)
    
    def forward(self, X):
        B, N, d = X.shape
        
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        g = self.metric_net(X.mean(0))  # Shared metric across batch
        
        # Geodesic distances
        dist_sq = self.varadhan_distance(Q, K, g)  # (B, N, N)
        
        # Multi-scale heat kernel attention
        t_values = torch.exp(self.log_t)  # (S,)
        scale_outputs = []
        
        for t in t_values:
            weights = F.softmax(-dist_sq / (4 * t), dim=-1)  # (B, N, N)
            out = torch.bmm(weights, V)  # (B, N, d)
            scale_outputs.append(out)
        
        # Fuse scales
        multi_scale = torch.cat(scale_outputs, dim=-1)  # (B, N, S*d)
        Y = self.W_fuse(multi_scale)  # (B, N, d)
        
        return Y, g  # Return metric for diagnostics
```

### B.3 Metric Regularization

```python
def metric_regularization(g, lambda_eigen=0.1, lambda_smooth=0.01, 
                          lambda_div=0.01, lambda_min=0.1, lambda_max=10.0):
    """
    Regularize learned metric to prevent collapse and ensure smoothness.
    g: (N, d, d) metric tensors
    """
    N, d, _ = g.shape
    
    # Eigenvalue bounds
    eigenvalues = torch.linalg.eigvalsh(g)  # (N, d)
    eigen_loss = (F.relu(lambda_min - eigenvalues).mean() + 
                  F.relu(eigenvalues - lambda_max).mean())
    
    # Smoothness: penalize metric variation between adjacent positions
    g_diff = g[1:] - g[:-1]  # (N-1, d, d)
    smooth_loss = torch.norm(g_diff, dim=(-2,-1)).mean()
    
    # Diversity: penalize uniform curvature (encourage geometric adaptation)
    # Approximate scalar curvature from metric variation
    curvature_proxy = torch.norm(g_diff, dim=(-2,-1))  # (N-1,)
    diversity_loss = -curvature_proxy.std()  # Negative: maximize variance
    
    total = (lambda_eigen * eigen_loss + 
             lambda_smooth * smooth_loss + 
             lambda_div * diversity_loss)
    
    return total
```

---

## Appendix C: CSG Benchmark Specifications (Extended)

### C.1-C.3: Original Tasks (Unchanged from v2)

See FGN v2 Appendix C for full specifications of Tasks A1 (Associative Recall), B1 (State Tracking), and C1 (Dynamic Switching).

### C.4 Task E1: Analogy Completion (New)

**Format**:
```
Input: "A:king B:queen C:man QUERY"
Output: "woman"
```

**Generation**:
- Analogy pairs from multiple domains (semantic, syntactic, functional)
- Difficulty levels based on relationship complexity
- Cross-domain analogies for testing transport across manifold regions

**Geometric measurement**:
- Extract metric at positions A, B, C, D
- Compute parallel transport: Î _{Aâ†’C}(v_B âˆ’ v_A)
- Compare to actual (v_D âˆ’ v_C)
- Report transport fidelity: cos(Î (v_Bâˆ’v_A), v_Dâˆ’v_C)

### C.5 Task F1: Curvature Alignment (New)

**Format**: Run any CSG task; additionally record:
- Learned scalar curvature Îº at each position
- Task-optimal curvature Îº* (known from task design)
- Compute alignment: correlation(Îº, Îº*)

**Reporting**: Curvature alignment score per task, per model variant.

---

## Appendix D: Glossary (Revised)

| Term | Definition |
|------|------------|
| **(M, g)** | The sequence manifold with learned Riemannian metric |
| **Metric tensor g** | Learned positive-definite bilinear form determining all geometry |
| **Heat kernel K_t** | Fundamental solution of heat equation on (M, g); attention weights |
| **Diffusion time t** | Scale parameter controlling resolution (small = fine, large = coarse) |
| **Geodesic distance d_g** | Shortest path length on curved manifold; replaces positional distance |
| **Scalar curvature Îº** | Average curvature at a point; determines local associativity |
| **Parallel transport Î ** | Moving vectors between tangent spaces; formalizes ~> operator |
| **Holonomy** | Rotation from parallel transport around closed loops; EMERGULANCE |
| **~> operator** | "Meets and becomes"; parallel transport along geodesic |
| **Î± (alpha)** | Associativity coefficient, now derived: Î± = exp(âˆ’\|Îº\|/Îºâ‚€) |
| **Flat processing** | Zero-curvature regime; parallel, order-independent (transformer-like) |
| **Curved processing** | Finite-curvature regime; path-dependent (RNN-like) |
| **Singular processing** | High-curvature regime; generative emergence (~> at full strength) |
| **FGN** | Fluid Geometry Network; the architectural family |
| **CSG** | Causal-Symbolic Gap; the evaluation benchmark |
| **Metric collapse** | Pathological training state where metric becomes trivial or singular |
| **Varadhan approximation** | Short-distance geodesic distance from heat kernel asymptotics |

---

## Appendix E: Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 1.0 | January 2026 | Original FGN paper; Î±-interpolation framework |
| 2.0 | January 2026 | Revised with training solutions; CSG benchmark; implementation pseudocode |
| 3.0 | February 2026 | **Major revision**: Intrinsic Riemannian framework; Î± derived from curvature; ~> as parallel transport; leaky masking dissolved; metric network architecture; extended CSG benchmark |

### Key Differences: v2 â†’ v3

| Aspect | v2 | v3 |
|--------|-----|-----|
| **Fundamental object** | Î± parameter (learned) | Metric tensor g (learned) |
| **Î± status** | Independent parameter | Derived from scalar curvature |
| **Architecture** | Dual pathway + routing | Single pathway on curved manifold |
| **Attention kernel** | Soft interpolation of full/causal | Heat kernel on (M, g) |
| **Leaky masking** | Major challenge; 4 solutions proposed | Dissolved; doesn't arise |
| **~> operator** | Conceptual/philosophical | Parallel transport (rigorous) |
| **Emergence measure** | Not formalized | Curvature integral along geodesic |
| **Resolution** | External parameter r | Intrinsic: heat kernel diffusion time + curvature |
| **Three geometries** | Discrete modes | Curvature spectrum (continuous) |
| **Benchmark** | Suites A-D | Extended with Suites E (transport), F (curvature) |
| **Mathematical base** | Monoidal categories | Riemannian geometry (subsumes monoidal formulation) |

---

*End of Document*

**Document Version**: 3.0 (Intrinsic Riemannian Revision)
**Last Updated**: February 2026
**Status**: Research Foundation â€” Theoretical Framework Consolidated; Ready for Implementation Studies