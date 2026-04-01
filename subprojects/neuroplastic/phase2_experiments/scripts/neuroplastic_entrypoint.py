"""Neuroplastic vLLM Entrypoint

Wraps the standard vLLM OpenAI API server with neuroplastic weight modification
endpoints. Monkey-patches EngineCore before the server starts, then hooks
init_app_state to add HTTP routes after the FastAPI app is created.

Usage (inside the vLLM container):
    python3 /workspace/neuroplastic_entrypoint.py \
        --host 0.0.0.0 --port 30000 --model /workspace/model \
        --served-model-name NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
        --trust-remote-code --max-model-len 32768 --max-num-seqs 8 \
        --gpu-memory-utilization 0.4 --enable-prefix-caching
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroplastic")

# Step 1: Install monkey-patches on EngineCore BEFORE any vLLM server code runs.
# This patches the CLASS, so all future instances get the methods.
logger.info("Installing neuroplastic EngineCore methods...")
from neuroplastic_plugin import install_engine_core_methods, install_api_routes
install_engine_core_methods()

# Step 2: Patch build_app to add our HTTP routes after the FastAPI app is created.
import vllm.entrypoints.openai.api_server as _api_mod

_original_build_app = _api_mod.build_app

def _patched_build_app(args, **kwargs):
    app = _original_build_app(args, **kwargs)
    install_api_routes(app)
    logger.info("Neuroplastic routes installed via build_app hook")
    return app

_api_mod.build_app = _patched_build_app

if __name__ == "__main__":
    logger.info("Starting vLLM server with neuroplastic extensions...")
    # Rewrite argv[0] so vLLM doesn't get confused
    sys.argv[0] = "vllm.entrypoints.openai.api_server"

    import asyncio
    from vllm.entrypoints.openai.api_server import run_server, make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(sys.argv[1:])
    asyncio.run(run_server(args))
