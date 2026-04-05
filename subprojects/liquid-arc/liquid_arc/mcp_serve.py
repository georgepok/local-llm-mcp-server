"""FastMCP server exposing LiquidARC Mind as tools for Claude.

Usage:
    python -m liquid_arc.mcp_serve \
      --checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
      --config configs/linguistic_mind.yaml \
      --port 8420

Claude Desktop config (add to claude_desktop_config.json):
{
    "mcpServers": {
        "liquid-arc-mind": {
            "url": "http://spark-129a.local:8420/sse"
        }
    }
}
"""

import argparse
import json
import torch
import signal
import sys
import time
from typing import Optional

from fastmcp import FastMCP

from .config import LiquidARCConfig
from .mind import LiquidARCMind
from .voice import Voice
from .curriculum import CurriculumGenerator

mcp = FastMCP("LiquidARC Mind")

# Global mind instance — initialized at startup
_mind: Optional[LiquidARCMind] = None
_voice: Optional[Voice] = None
_state_path: Optional[str] = None


@mcp.tool()
def observe_event(event_type: str, content: str,
                  metadata: Optional[str] = None) -> str:
    """Inject a conversation event as sensory forcing into LiquidARC.

    Call AFTER each user message and AFTER each Claude response.

    Args:
        event_type: One of 'user_message', 'assistant_message', 'tool_result',
                    'goal', 'context', 'temporal'
        content: The text content of the event
        metadata: Optional JSON string with additional features:
                  {"sentiment": float, "confidence": float, "tool_count": int,
                   "success": float}

    Returns prediction_error, cv, h_norm, events_in_context.
    """
    meta_dict = json.loads(metadata) if metadata else None
    result = _mind.observe_event(event_type, content, meta_dict)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_context(query: Optional[str] = None) -> str:
    """Read relevance-scored context from LiquidARC's persistent state.

    Call BEFORE constructing a response to get attention directives.

    Args:
        query: Optional focus query to bias relevance scoring (not yet used).

    Returns sorted events with relevance scores, focus_indices, summary_norm.
    """
    result = _mind.get_context(query)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_diagnostics() -> str:
    """Read LiquidARC model internals for monitoring.

    Returns CV, tau, beta per type, h_norm, event count.
    """
    result = _mind.get_diagnostics()
    return json.dumps(result, indent=2)


@mcp.tool()
def signal_goal(goal_text: str, priority: float = 1.0) -> str:
    """Inject a goal as a persistent entity token.

    Goals persist in the event buffer and influence relevance scoring.

    Args:
        goal_text: Description of the goal
        priority: Priority level (0.0 to 1.0)
    """
    result = _mind.signal_goal(goal_text, priority)
    return json.dumps(result, indent=2)


@mcp.tool()
def provide_feedback(event_index: int, feedback_type: str,
                     signal: float = 1.0) -> str:
    """Online learning signal from human feedback.

    Updates relevance scoring (readout head only, dynamics frozen).

    Args:
        event_index: Index of the event in context (from get_context)
        feedback_type: One of 'correct', 'wrong', 'irrelevant'
        signal: Strength of the feedback (default 1.0)
    """
    result = _mind.provide_feedback(event_index, feedback_type, signal)
    return json.dumps(result, indent=2)


@mcp.tool()
def reset() -> str:
    """Clear all state for a new conversation/topic."""
    _mind.reset()
    return json.dumps({'status': 'reset_complete'})


@mcp.tool()
def save_state() -> str:
    """Manually persist conversation-trained weights and ODE state to disk.

    State is also auto-saved on server shutdown. Returns the path written.
    """
    if _state_path is None:
        return json.dumps({'status': 'no_state_path', 'message': 'Server started without --state_path'})
    _mind.save_state(_state_path)
    return json.dumps({'status': 'saved', 'path': _state_path, 'events': len(_mind.events)})


@mcp.tool()
def probe_encoding(text: str) -> str:
    """Read the Mind's own linguistic transformation of input text.

    Projects Phase 1 ODE output back through the Mind's embedding table.
    Shows what each token moved toward through 16 integration steps.
    This is the Mind speaking in its own vocabulary — no LLM involved.

    Args:
        text: Input text to process through Phase 1 and project back.
    """
    result = _mind.probe_encoding(text)
    compact = {
        'mind_sentence': result['mind_sentence'],
        'transform_ratio': result['transform_ratio'],
        'state_vocabulary': result['state_vocabulary'],
        'mean_displacement': result['mean_displacement'],
        'n_tokens': result['n_tokens'],
        'transformations': [],
    }
    for p in result['positions']:
        if p['transformed'] or p['displacement'] > result['mean_displacement']:
            compact['transformations'].append(
                f"{p['input']} -> {p['output_top5'][0]} "
                f"(also: {', '.join(p['output_top5'][1:3])}) "
                f"[d={p['displacement']}]"
            )
    return json.dumps(compact, indent=2)


@mcp.tool()
def express_state(focus_query: Optional[str] = None) -> str:
    """Let the Mind express its state through rich geometric readout + local LLM.

    Extracts full geometric profile (per-event metrics, clusters, ODE trajectory),
    sends to Nemotron which produces linguistic expression. The expression and a
    condensed reflection are fed back as events for self-referential processing.

    Args:
        focus_query: Optional topic to focus the expression on.

    Returns the Mind's expression, geometric basis, clusters, and reflection.
    """
    if _voice is None or not _voice.is_available():
        return json.dumps({
            'status': 'voice_unavailable',
            'diagnostics': _mind.get_diagnostics(),
        })

    profile = _mind.get_geometric_profile()
    if profile.get('status') == 'no_state':
        return json.dumps({'status': 'no_state'})

    # Probe the Mind's own linguistic output on recent content
    state_tokens = None
    if _mind.events:
        for e in reversed(_mind.events):
            if e.get('type') not in [6, 7]:
                content = e.get('content_preview', '')
                if len(content) > 10:
                    state_tokens = _mind.probe_encoding(content)
                    break

    result = _voice.express(profile, state_tokens=state_tokens, focus_query=focus_query)

    if state_tokens:
        result['state_tokens'] = {
            'mind_sentence': state_tokens.get('mind_sentence', ''),
            'transform_ratio': state_tokens.get('transform_ratio', 0),
            'key_transformations': state_tokens.get('transformations', [])[:5],
        }

    # Feed condensed reflection back (grounds h)
    if result.get('reflection_event') and not result['reflection_event'].startswith('[Voice'):
        _mind.observe_event(
            event_type='reflection',
            content=result['reflection_event'],
            metadata={'source': 'express_state'},
        )

    # Feed full expression back (self-reference)
    if result.get('expression') and not result['expression'].startswith('[Voice'):
        _mind.observe_event(
            event_type='expression',
            content=result['expression'],
            metadata={'source': 'self_expression', 'focus_query': focus_query},
        )

    return json.dumps(result, indent=2)


@mcp.tool()
def get_reflection_log() -> str:
    """Read the Mind's internal reflection history.

    Returns the last 20 reflections from the internal reflection cycle,
    with timestamps and geometric state.
    """
    reflections = []
    for i, event in enumerate(_mind.events):
        if event.get('type') == 6:  # reflection type_id
            reflections.append({
                'index': i,
                'text': event.get('content_preview', ''),
                'age_seconds': round(time.time() - event.get('timestamp', 0), 1),
            })

    return json.dumps({
        'status': 'active',
        'n_reflections': len(reflections),
        'total_events': len(_mind.events),
        'reflections': reflections[-20:],
        'last_reflection': _mind._last_reflection_text,
        'reflection_count': _mind._reflection_count,
    }, indent=2)


@mcp.tool()
def get_curriculum_stats() -> str:
    """Read curriculum statistics.

    Shows which domains have been presented, the Mind's PE response
    to each, domain effectiveness scores, and growth zone domains.
    """
    # Qwen3 geometric curriculum stats
    stats = getattr(_mind, '_curriculum_stats', None)
    if stats is not None:
        total = sum(stats['domain_counts'].values())
        if total > 0:
            all_pe = [v for v in stats['domain_avg_pe'].values() if v > 0]
            pe_max = max(all_pe) if all_pe else 1
            pe_min = min(all_pe) if all_pe else 0
            return json.dumps({
                'total_stimuli': total,
                'domain_counts': stats['domain_counts'],
                'domain_avg_pe': {d: round(v, 1) for d, v in stats['domain_avg_pe'].items()},
                'domain_avg_cv': {d: round(v, 2) for d, v in stats.get('domain_avg_cv', {}).items()},
                'domain_avg_tau': {d: round(v, 2) for d, v in stats.get('domain_avg_tau', {}).items()},
                'domain_effectiveness': stats.get('domain_effectiveness', {}),
                'most_familiar_domain': min(stats['domain_avg_pe'], key=stats['domain_avg_pe'].get) if stats['domain_avg_pe'] else None,
                'most_novel_domain': max(stats['domain_avg_pe'], key=stats['domain_avg_pe'].get) if stats['domain_avg_pe'] else None,
                'growth_zone_domains': stats.get('growth_zone_domains', []),
                'pe_spread_pct': round(100 * (pe_max - pe_min) / pe_max, 1) if pe_max > 0 else 0,
            }, indent=2)
    # Legacy Nemotron curriculum
    if _mind.curriculum is not None:
        return json.dumps(_mind.curriculum.get_stats(), indent=2)
    return json.dumps({'status': 'curriculum_not_enabled'})


@mcp.tool()
def inject_stimulus(domain: Optional[str] = None,
                    custom_content: Optional[str] = None) -> str:
    """Manually inject a curriculum stimulus through the geometric coupling.

    Either specify a domain (Qwen3 generates content conditioned on state)
    or provide custom content directly.

    Args:
        domain: Domain key (topology, music_theory, biology, physics,
                philosophy, mathematics, poetry, ecology).
        custom_content: Direct content to inject through geometric path.
    """
    if custom_content:
        if _mind._qwen_available:
            signal = _mind._encode_through_qwen(custom_content)
            if signal is not None:
                _mind._force_geometric_signal(
                    signal, custom_content[:200],
                    event_type='context',
                    metadata={'source': 'manual_stimulus'})
                diag = _mind.get_diagnostics()
                return json.dumps({
                    'source': 'manual_geometric',
                    'signal_norm': signal.norm().item(),
                    'cv': diag.get('metric_cv', 0),
                }, indent=2)
        # Fallback: legacy text path
        result = _mind.observe_event(
            event_type='context',
            content=custom_content,
            metadata={'source': 'manual_stimulus'},
        )
        return json.dumps({
            'source': 'manual',
            'prediction_error': result.get('prediction_error', 0),
            'cv': result.get('cv', 0),
        }, indent=2)

    # Generate stimulus through Qwen3 for specified domain
    if not _mind._qwen_available:
        return json.dumps({'status': 'qwen3_coupling_not_available'})

    domains = list(_mind.CURRICULUM_PROMPTS.keys())
    if domain and domain not in domains:
        return json.dumps({'error': f'Unknown domain: {domain}. Available: {domains}'})

    domain = domain or domains[_mind._curriculum_idx % len(domains)]
    prompt = _mind.CURRICULUM_PROMPTS[domain]
    result = _mind.query_knowledge(prompt, max_new_tokens=150, temperature=0.7)
    stimulus_text = result.get('response', '')

    if stimulus_text and len(stimulus_text) > 10:
        signal = _mind._encode_through_qwen(stimulus_text)
        if signal is not None:
            _mind._force_geometric_signal(
                signal, stimulus_text[:200],
                event_type='context',
                metadata={'source': 'manual_curriculum', 'domain': domain})
        diag = _mind.get_diagnostics()
        return json.dumps({
            'domain': domain,
            'stimulus_preview': stimulus_text[:200],
            'signal_norm': signal.norm().item() if signal is not None else 0,
            'cv': diag.get('metric_cv', 0),
        }, indent=2)

    return json.dumps({'status': 'generation_failed', 'domain': domain})


@mcp.tool()
def set_curriculum(enabled: bool, interval: Optional[int] = None) -> str:
    """Toggle the geometric curriculum feed on/off and adjust stimulus interval.

    Turn off to give the Mind time to consolidate what it has received.
    Turn on to resume diverse stimulation.

    Args:
        enabled: True to enable, False to pause curriculum
        interval: Optional new stimulus interval in ODE cycles (default 14)
    """
    was_enabled = _mind._stimulus_interval < 999
    if enabled:
        _mind._stimulus_interval = interval or 14
        status = 'enabled'
    else:
        _mind._stimulus_interval = 999999  # effectively disabled
        status = 'paused'

    return json.dumps({
        'status': status,
        'was': 'enabled' if was_enabled else 'paused',
        'stimulus_interval': _mind._stimulus_interval,
        'total_stimuli_so_far': getattr(_mind, '_curriculum_count', 0),
    }, indent=2)


@mcp.tool()
def get_routing_stats() -> str:
    """Read adaptive routing statistics.

    Shows how the Mind decides when to use LLM reflection:
    total ODE cycles, triggered vs maintenance vs external reflections,
    trigger type breakdown, and current trigger sensitivity.
    """
    stats = _mind._trigger_stats.copy()
    if _mind.trigger is not None:
        stats['trigger_sensitivity'] = _mind.trigger.trigger_sensitivity
    stats['cycles_since_reflection'] = _mind._cycles_since_reflection
    stats['maintenance_interval'] = _mind.maintenance_interval
    stats['last_reflection_pe'] = _mind._last_reflection_pe

    total_ref = (stats.get('triggered_reflections', 0) +
                 stats.get('maintenance_reflections', 0) +
                 stats.get('external_reflections', 0))
    if total_ref > 0:
        stats['triggered_fraction'] = stats['triggered_reflections'] / total_ref
        stats['maintenance_fraction'] = stats['maintenance_reflections'] / total_ref
        stats['external_fraction'] = stats['external_reflections'] / total_ref
        stats['cycles_per_reflection'] = stats['total_ode_cycles'] / total_ref

    return json.dumps(stats, indent=2)


@mcp.tool()
def converse(message: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
    """Send a message to the Mind and get a response through Qwen3.

    This is the primary interaction interface. The complete loop:
    1. Your message enters LiquidARC as sensory forcing (ODE state updates)
    2. LiquidARC's accumulated state projects into Qwen3 as virtual prefix tokens
    3. Qwen3 generates a response shaped by that geometric context
    4. The response feeds back into LiquidARC (self-referential integration)

    The response reflects not just your message, but everything the Mind
    has experienced — all prior conversations, curriculum, reflections —
    compressed into geometry and rendered through Qwen3's language.

    Args:
        message: Your message to the Mind
        max_tokens: Maximum response length (default 300)
        temperature: Sampling temperature (default 0.7, 0 = deterministic)
    """
    result = _mind.converse(message, max_new_tokens=max_tokens,
                            temperature=temperature)
    return json.dumps(result, indent=2)


@mcp.tool()
def query_qwen(prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Query Qwen3-4B conditioned on LiquidARC's geometric state.

    Projects h(t) → virtual prefix tokens → Qwen3 generates response.
    The response is shaped by LiquidARC's accumulated temporal context.
    This is the geometric knowledge interface — no tokenization of the state,
    pure vector projection.

    Args:
        prompt: Question or instruction for Qwen3
        max_tokens: Maximum tokens to generate (default 200)
        temperature: Sampling temperature (default 0.7, 0 = greedy)
    """
    result = _mind.query_knowledge(prompt, max_new_tokens=max_tokens,
                                   temperature=temperature)
    return json.dumps(result, indent=2)


@mcp.tool()
def express_through_qwen(focus_query: Optional[str] = None) -> str:
    """Let the Mind express its state through Qwen3's language.

    Projects LiquidARC's ODE state into Qwen3's representation space as
    virtual prefix tokens. Qwen3 generates natural language conditioned
    on this geometric state. Different accumulated contexts produce
    different expressions.

    Unlike probe_encoding (which projects through the Mind's own embedding
    table), this uses Qwen3's 4B parameters of world knowledge to
    render the geometric state as natural language.

    Args:
        focus_query: Optional topic to focus the expression on
    """
    result = _mind.express_through_qwen(focus_query)
    return json.dumps(result, indent=2)


@mcp.tool()
def set_ntp_mode(mode: str) -> str:
    """Switch NTP loss mode between 'raw' and 'ode'.

    raw: NTP on raw embedding output (fast, proven counterweight, loss ~25)
    ode: NTP through ODE output (slower, more aligned with xform, loss ~500)

    The productive xform regime (0→5→20%) used 'raw' NTP at weight 0.1.
    """
    if mode not in ('raw', 'ode'):
        return json.dumps({'error': f'Invalid mode: {mode}. Use "raw" or "ode".'})
    _mind.ntp_mode = mode
    return json.dumps({'status': 'updated', 'ntp_mode': mode,
                       'ntp_loss_weight': _mind.ntp_loss_weight})


@mcp.tool()
def get_plasticity_status() -> str:
    """Read the adaptive plasticity controller's state.

    Shows current embed_lr, NTP loss EMA, xform streak,
    recent controller actions, and LR bounds.
    """
    if not hasattr(_mind, '_plasticity_ctrl') or _mind._plasticity_ctrl is None:
        return json.dumps({'status': 'no_controller'})
    return json.dumps(_mind._plasticity_ctrl.get_status(), indent=2, default=str)


@mcp.tool()
def set_learning_rates(
    embed_lr: float = None,
    geo_lr: float = None,
    other_lr: float = None,
) -> str:
    """Adjust optimizer learning rates at runtime.

    Parameter groups:
      - embed: TextEmbedding or MindTokenizer (Phase 1 input encoding)
      - geo: dynamics + context_pool (MetricNet, TauNet — geometric routing)
      - other: metadata, forcing, readout (Mind infrastructure)

    The geometry distillation finding: 100× LR ratio for new vs transferred params.
    The dynamics need gentle adaptation (1e-6) to the Mind's text distribution.
    TextEmbedding can go faster (1e-4 to 1e-3) since it's co-adapting.

    Args:
        embed_lr: LR for text embedding (group 0). Range: 1e-4 to 1e-2.
        geo_lr: LR for dynamics/context_pool. Range: 1e-7 to 1e-5.
        other_lr: LR for readout, forcing, metadata. Range: 1e-6 to 1e-4.

    Returns current LR for all groups.
    """
    if _mind.optimizer is None:
        return json.dumps({'status': 'no_optimizer', 'message': 'Online learning disabled'})

    if embed_lr is not None:
        _mind.optimizer.param_groups[0]['lr'] = embed_lr

    if geo_lr is not None:
        # Dynamics group — find by param count (largest group)
        for pg in _mind.optimizer.param_groups:
            n = sum(p.numel() for p in pg['params'])
            if n > 1_000_000:  # dynamics has millions of params
                pg['lr'] = geo_lr

    if other_lr is not None:
        # All groups except embed (0) and dynamics (large)
        for i, pg in enumerate(_mind.optimizer.param_groups):
            if i == 0:
                continue
            n = sum(p.numel() for p in pg['params'])
            if n < 1_000_000:  # small groups = metadata, forcing, readout
                pg['lr'] = other_lr

    groups = []
    for i, pg in enumerate(_mind.optimizer.param_groups):
        n = sum(p.numel() for p in pg['params'])
        label = 'embed' if i == 0 else ('dynamics' if n > 1_000_000 else f'group_{i}')
        groups.append({'name': label, 'lr': pg['lr'], 'n_params': n})

    return json.dumps({'status': 'updated', 'groups': groups}, indent=2)


def create_mind(args) -> LiquidARCMind:
    """Initialize the LiquidARC Mind with optional Voice and Qwen3 coupling."""
    config = LiquidARCConfig.from_yaml(args.config)

    # Sentence-transformer: only needed if NOT using ODE encoder, or for bootstrap
    embedder = None
    use_ode = getattr(args, 'use_ode_encoder', False)
    if not use_ode or getattr(args, 'bootstrap_mode', False):
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2', device=args.device)
        except ImportError:
            if not use_ode:
                raise RuntimeError("sentence-transformers required when use_ode_encoder=False")
            print("  sentence-transformers not available, skipping bootstrap blending")

    # ── Qwen3 Geometric Coupling (Phase 5) ──
    qwen_model = None
    qwen_tokenizer = None
    coupling = None
    qwen_vllm_url = getattr(args, 'qwen_vllm_url', None)
    qwen_path = getattr(args, 'qwen_model_path', None)
    coupling_ckpt_path = getattr(args, 'coupling_checkpoint', None)

    if coupling_ckpt_path:
        import os
        if os.path.exists(coupling_ckpt_path):
            print(f"\n═══ Loading Qwen3 Geometric Coupling ═══")
            from .coupling import GeometricCoupling

            # Load coupling weights
            coupling_ckpt = torch.load(coupling_ckpt_path, map_location=args.device,
                                       weights_only=False)
            coupling_cfg = coupling_ckpt.get('config', {})
            n_vt = int(coupling_cfg.get('n_virtual_tokens', 8))
            d_qwen = int(coupling_cfg.get('d_qwen', 2560))

            coupling = GeometricCoupling(
                d_arc=config.d_model, d_qwen=d_qwen,
                n_virtual_tokens=n_vt,
            ).to(args.device).to(torch.bfloat16)
            coupling.load_state_dict(coupling_ckpt['coupling_state_dict'])
            coupling.eval()
            print(f"  Coupling: {coupling.param_count()/1e6:.2f}M params, "
                  f"{n_vt} virtual tokens, d_qwen={d_qwen}")
            print(f"  Loaded from step {coupling_ckpt.get('step', '?')}")

            eval_result = coupling_ckpt.get('eval_result', {})
            if eval_result:
                print(f"  Training eval: coupled_ppl={eval_result.get('coupled_ppl', '?')}, "
                      f"improvement={eval_result.get('ppl_improvement', '?'):.1f}%")

            # Connect to Qwen3 via vLLM API (preferred) or load in-process (fallback)
            if qwen_vllm_url:
                from .qwen_client import QwenVLLMClient
                tokenizer_path = qwen_path or '/workspace/models/qwen3-4b'
                qwen_model = QwenVLLMClient(
                    vllm_url=qwen_vllm_url,
                    tokenizer_path=tokenizer_path,
                    device=args.device,
                )
                qwen_tokenizer = qwen_model.tokenizer
                if qwen_model.is_available():
                    print(f"  Qwen3: vLLM API at {qwen_vllm_url}")
                else:
                    print(f"  WARNING: vLLM not responding at {qwen_vllm_url}")
            elif qwen_path and os.path.exists(qwen_path):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"  Loading Qwen3 in-process from {qwen_path}...")
                qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path)
                if qwen_tokenizer.pad_token is None:
                    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
                qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path, dtype=torch.bfloat16,
                    device_map={'': args.device},
                )
                qwen_model.eval()
                for p in qwen_model.parameters():
                    p.requires_grad_(False)
                print(f"  Qwen3: in-process HF, "
                      f"{sum(p.numel() for p in qwen_model.parameters())/1e9:.2f}B params")
        else:
            print(f"  WARNING: Coupling checkpoint not found at {coupling_ckpt_path}")

    mind = LiquidARCMind(
        checkpoint_path=args.checkpoint,
        config=config,
        text_embedder=embedder,
        device=args.device,
        max_context_events=getattr(config, 'max_context_events', 64),
        lambda_eff=getattr(config, 'lambda_eff', 0.001),
        freeze_dynamics=args.freeze_dynamics,
        online_lr=args.online_lr,
        enable_online_learning=not args.no_online_learning,
        use_ode_encoder=use_ode,
        tokenizer_path=getattr(args, 'tokenizer_path', None),
        bootstrap_mode=getattr(args, 'bootstrap_mode', True),
        bootstrap_events=getattr(args, 'bootstrap_events', 5000),
        use_trained_text_embed=getattr(args, 'use_trained_text_embed', False),
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        coupling=coupling,
    )

    # Warm up tokenizer eagerly so first MCP call doesn't block
    if use_ode:
        mind.embedding.tokenizer._load_tokenizer()

        # Load distilled embeddings if available (warm-start from sentence-transformer)
        distilled_path = getattr(args, 'distilled_embeddings', None)
        if distilled_path:
            import os
            if os.path.exists(distilled_path):
                distilled = torch.load(distilled_path, map_location=args.device,
                                       weights_only=False)
                if 'embedding_state_dict' in distilled:
                    mind.embedding.load_state_dict(distilled['embedding_state_dict'],
                                                   strict=False)
                    print(f"  Loaded distilled embeddings from {distilled_path}")
                    print(f"    distill_loss={distilled.get('distill_loss', '?'):.5f}, "
                          f"baseline={distilled.get('baseline_loss', '?'):.5f}")
                elif 'tokenizer_state_dict' in distilled:
                    mind.embedding.tokenizer.load_state_dict(
                        distilled['tokenizer_state_dict'], strict=False)
                    print(f"  Loaded distilled tokenizer from {distilled_path}")
            else:
                print(f"  WARNING: distilled embeddings not found at {distilled_path}")

        # Seed with a bootstrap event so autonomous loop can start
        mind.observe_event('temporal', 'Mind initialized. ODE encoder active.',
                          metadata={'source': 'internal_reflection'})
        print(f"  ODE encoder: tokenizer loaded, bootstrap event seeded")

    # Initialize Voice (legacy — only if Qwen3 coupling is not available)
    global _voice
    voice = None
    if args.enable_voice and not mind._qwen_available:
        voice = Voice(
            lm_studio_url=args.lm_studio_url,
            model=args.lm_studio_model,
            max_tokens=200,
            temperature=0.7,
        )
        _voice = voice
        mind._reflection_interval = args.reflection_interval
        print("  Voice: Nemotron (legacy — Qwen3 coupling not available)")

    # Initialize curriculum (legacy — only if Qwen3 coupling is not available)
    if args.enable_curriculum and voice and not mind._qwen_available:
        mind.curriculum = CurriculumGenerator(voice=voice)
        mind._stimulus_interval = args.stimulus_interval
        print(f"  Curriculum: Nemotron (legacy), {len(mind.curriculum.domains)} domains")

    # Geometric curriculum via Qwen3 coupling
    if mind._qwen_available:
        mind._stimulus_interval = args.stimulus_interval
        print(f"  Reflection + Curriculum: Qwen3 geometric coupling, "
              f"stimulus interval={args.stimulus_interval} cycles")

    if args.enable_autonomous:
        mind.start_autonomous(voice=voice if not mind._qwen_available else None)
        print("Autonomous processing started")

    return mind


def main():
    parser = argparse.ArgumentParser(description='LiquidARC Mind MCP Server')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to LiquidARC checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--port', type=int, default=8420)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--freeze_dynamics', action='store_true', default=False)
    parser.add_argument('--no_online_learning', action='store_true', default=False)
    parser.add_argument('--online_lr', type=float, default=1e-5)
    parser.add_argument('--enable_autonomous', action='store_true', default=False)
    parser.add_argument('--lambda_eff', type=float, default=0.001)
    parser.add_argument('--state_path', type=str, default=None,
                        help='Path to persist conversation state (weights + ODE + events)')
    parser.add_argument('--enable_voice', action='store_true', default=False,
                        help='Connect to local LLM for voice expression')
    parser.add_argument('--lm_studio_url', type=str,
                        default='http://host.docker.internal:30000/v1',
                        help='LM Studio / vLLM API URL (from inside Docker)')
    parser.add_argument('--lm_studio_model', type=str,
                        default='NVIDIA-Nemotron-3-Nano-30B-A3B-FP8',
                        help='Model name served by vLLM')
    parser.add_argument('--reflection_interval', type=int, default=30,
                        help='Seconds between internal reflections (default: 30)')
    parser.add_argument('--use_ode_encoder', action='store_true', default=False,
                        help='Use Phase 1 ODE encoding instead of sentence-transformers')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to tokenizer (default: Nemotron from HuggingFace)')
    parser.add_argument('--bootstrap_mode', action='store_true', default=False,
                        help='Blend legacy + ODE encoding during transition')
    parser.add_argument('--bootstrap_events', type=int, default=5000,
                        help='Events before fully switching to ODE encoder')
    parser.add_argument('--use_trained_text_embed', action='store_true', default=False,
                        help='Path C hybrid: use GPT-2 TextEmbedding from Stage B checkpoint for Phase 1')
    parser.add_argument('--enable_curriculum', action='store_true', default=False,
                        help='Enable curriculum generator for diverse stimuli')
    parser.add_argument('--stimulus_interval', type=int, default=14,
                        help='ODE cycles between curriculum stimuli')
    parser.add_argument('--distilled_embeddings', type=str, default=None,
                        help='Path to distilled embedding checkpoint (from distill_embeddings.py)')
    parser.add_argument('--qwen_model_path', type=str, default=None,
                        help='Path to Qwen3-4B model/tokenizer (Phase 5 geometric coupling)')
    parser.add_argument('--qwen_vllm_url', type=str, default=None,
                        help='vLLM API URL for Qwen3 (e.g. http://localhost:30100/v1). Preferred over in-process.')
    parser.add_argument('--coupling_checkpoint', type=str, default=None,
                        help='Path to trained coupling checkpoint (from train_coupling.py)')
    parser.add_argument('--ssl_certfile', type=str, default=None,
                        help='SSL certificate file for HTTPS')
    parser.add_argument('--ssl_keyfile', type=str, default=None,
                        help='SSL private key file for HTTPS')
    args = parser.parse_args()

    global _mind, _state_path
    _state_path = args.state_path
    _mind = create_mind(args)
    print(f"LiquidARC Mind initialized on {args.device}")
    print(f"  Dynamics frozen: {args.freeze_dynamics}")
    print(f"  Online learning: {not args.no_online_learning} (lr={args.online_lr})")
    print(f"  Autonomous: {args.enable_autonomous}")

    if _state_path:
        import os
        if os.path.exists(_state_path):
            _mind.load_state(_state_path)
        else:
            print(f"  State path: {_state_path} (new)")

    def _shutdown(sig, frame):
        print("\nShutting down — saving mind state...")
        if _state_path:
            _mind.save_state(_state_path)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if args.ssl_certfile and args.ssl_keyfile:
        import uvicorn
        app = mcp.http_app(transport="sse")
        print(f"  HTTPS: https://{args.host}:{args.port}/sse")
        uvicorn.run(
            app, host=args.host, port=args.port,
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
            ssl_cert_reqs=0,  # no client cert required
        )
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == '__main__':
    main()
