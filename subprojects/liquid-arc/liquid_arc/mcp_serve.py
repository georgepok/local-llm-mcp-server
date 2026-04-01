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
import signal
import sys
from typing import Optional

from fastmcp import FastMCP

from .config import LiquidARCConfig
from .mind import LiquidARCMind

mcp = FastMCP("LiquidARC Mind")

# Global mind instance — initialized at startup
_mind: Optional[LiquidARCMind] = None
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


def create_mind(args) -> LiquidARCMind:
    """Initialize the LiquidARC Mind from checkpoint and config."""
    from sentence_transformers import SentenceTransformer

    config = LiquidARCConfig.from_yaml(args.config)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=args.device)

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
    )

    if args.enable_autonomous:
        mind.start_autonomous()
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

    mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == '__main__':
    main()
