"""
check_deployment.py — Inspect vLLM deployment and output deployment_config.json.

Works in two modes:
  1. On the host (spark-129a.local or via SSH): uses docker inspect / docker stats
  2. Inside the container: reads /proc/self/cgroup, /proc/meminfo, calls the vLLM API

Usage:
    # On host machine:
    python check_deployment.py
    python check_deployment.py --container vllm-nemotron-serve
    python check_deployment.py --output deployment_config.json

    # Inside the container:
    python check_deployment.py --inside-container

    # Against a remote vLLM endpoint:
    python check_deployment.py --api-url http://spark-129a.local:30000
"""

import argparse
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_CONTAINER = "vllm-nemotron-serve"
DEFAULT_API_URL = "http://spark-129a.local:30000"
DEFAULT_OUTPUT = "deployment_config.json"


def run_cmd(cmd: list[str], timeout: int = 15) -> tuple[str, str, int]:
    """Run a subprocess command, return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s: {' '.join(cmd)}", 1
    except FileNotFoundError:
        return "", f"Command not found: {cmd[0]}", 127
    except Exception as e:
        return "", str(e), 1


def api_get(url: str, timeout: int = 10) -> dict | None:
    """Make a GET request and return parsed JSON, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"  API request failed: {url} — {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"  API JSON parse failed: {url} — {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  API error: {url} — {e}", file=sys.stderr)
        return None


def check_api_health(api_url: str) -> dict:
    """Check vLLM API health and get model info."""
    info = {}

    print(f"  Checking API health at {api_url}...")
    health = api_get(f"{api_url}/health")
    info["health_ok"] = health is not None

    models = api_get(f"{api_url}/v1/models")
    if models and "data" in models:
        info["models"] = models["data"]

    return info


def inspect_container(container_name: str) -> dict:
    """Run docker inspect on a container and parse relevant fields."""
    print(f"  Running docker inspect {container_name}...")
    stdout, stderr, rc = run_cmd(["docker", "inspect", container_name])

    if rc != 0:
        print(f"  WARNING: docker inspect failed (rc={rc}): {stderr}", file=sys.stderr)
        return {"error": stderr}

    try:
        data = json.loads(stdout)
        if not data:
            return {"error": "empty inspect output"}
        container = data[0]
    except (json.JSONDecodeError, IndexError) as e:
        return {"error": f"parse error: {e}"}

    # Extract useful fields
    result = {}
    result["id"] = container.get("Id", "")[:12]
    result["name"] = container.get("Name", "").lstrip("/")
    result["status"] = container.get("State", {}).get("Status")
    result["started_at"] = container.get("State", {}).get("StartedAt")
    result["image"] = container.get("Config", {}).get("Image")
    result["image_id"] = container.get("Image", "")[:12]

    # Entrypoint / command
    result["entrypoint"] = container.get("Config", {}).get("Entrypoint")
    result["cmd"] = container.get("Config", {}).get("Cmd")

    # Environment variables (filter sensitive ones, keep vLLM-relevant)
    env_list = container.get("Config", {}).get("Env") or []
    env_dict = {}
    relevant_prefixes = ("VLLM_", "TRITON_", "CUDA_", "HF_", "MODEL_", "PORT")
    for item in env_list:
        if "=" in item:
            key, _, val = item.partition("=")
            if any(key.startswith(p) for p in relevant_prefixes):
                env_dict[key] = val
    result["relevant_env"] = env_dict

    # Mounts
    mounts = container.get("Mounts") or []
    result["mounts"] = [
        {
            "source": m.get("Source"),
            "destination": m.get("Destination"),
            "mode": m.get("Mode"),
        }
        for m in mounts
    ]

    # Port bindings
    port_bindings = container.get("HostConfig", {}).get("PortBindings") or {}
    result["port_bindings"] = port_bindings

    # GPUs / device requests
    device_requests = container.get("HostConfig", {}).get("DeviceRequests") or []
    result["gpu_device_requests"] = device_requests

    # Resource limits
    host_config = container.get("HostConfig", {})
    result["memory_limit_bytes"] = host_config.get("Memory")
    result["shm_size_bytes"] = host_config.get("ShmSize")

    # Restart policy
    result["restart_policy"] = host_config.get("RestartPolicy")

    return result


def get_docker_stats(container_name: str) -> dict:
    """Get current resource usage via docker stats --no-stream."""
    print(f"  Running docker stats {container_name}...")
    stdout, stderr, rc = run_cmd(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_name],
        timeout=20,
    )
    if rc != 0:
        return {"error": stderr}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw": stdout}


def get_container_logs_tail(container_name: str, lines: int = 50) -> list[str]:
    """Get the last N lines of container logs."""
    print(f"  Reading last {lines} lines of container logs...")
    stdout, stderr, rc = run_cmd(
        ["docker", "logs", "--tail", str(lines), container_name],
        timeout=20,
    )
    if rc != 0:
        return [f"ERROR: {stderr}"]
    # Combine stdout and stderr (vLLM logs to stderr)
    combined = stdout + "\n" + stderr
    return [line for line in combined.splitlines() if line.strip()]


def extract_vllm_args_from_logs(log_lines: list[str]) -> dict:
    """Parse vLLM startup args from log output."""
    args = {}
    for line in log_lines:
        # Look for common vLLM startup log patterns
        if "model=" in line.lower() or "model_path" in line.lower():
            args["_model_log_line"] = line.strip()
        if "gpu_memory_utilization" in line:
            args["_gpu_mem_util_log"] = line.strip()
        if "tensor_parallel_size" in line:
            args["_tp_size_log"] = line.strip()
        if "Loading model weights" in line or "loading model" in line.lower():
            args["_loading_log"] = line.strip()
        if "KV Cache" in line or "kv_cache" in line.lower():
            args["_kv_cache_log"] = line.strip()
        if "GiB" in line and ("model" in line.lower() or "memory" in line.lower()):
            args["_memory_log"] = line.strip()
    return args


def read_proc_meminfo() -> dict:
    """Read /proc/meminfo when running inside a container."""
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return {}
    result = {}
    try:
        with open(meminfo_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val_kb = int(parts[1])
                    result[key + "_kb"] = val_kb
                    result[key + "_gib"] = round(val_kb / (1024 * 1024), 2)
    except Exception as e:
        result["error"] = str(e)
    return result


def detect_inside_container() -> bool:
    """Heuristic: check if we are running inside a Docker container."""
    # Check for .dockerenv
    if Path("/.dockerenv").exists():
        return True
    # Check cgroup
    cgroup_path = Path("/proc/1/cgroup")
    if cgroup_path.exists():
        try:
            content = cgroup_path.read_text()
            if "docker" in content or "containerd" in content:
                return True
        except Exception:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check vLLM deployment and output deployment_config.json"
    )
    parser.add_argument(
        "--container",
        default=DEFAULT_CONTAINER,
        help=f"Docker container name (default: {DEFAULT_CONTAINER})",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"vLLM API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--inside-container",
        action="store_true",
        help="Force inside-container mode (skip docker commands)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip API health checks",
    )
    parser.add_argument(
        "--logs-lines",
        type=int,
        default=100,
        help="Number of log lines to retrieve (default: 100)",
    )
    args = parser.parse_args()

    inside_container = args.inside_container or detect_inside_container()
    print(f"Mode: {'inside-container' if inside_container else 'host'}")

    result: dict = {
        "collection_mode": "inside-container" if inside_container else "host",
        "hostname": platform.node(),
        "platform": platform.platform(),
    }

    # API checks (work from anywhere)
    if not args.no_api:
        print(f"\nChecking vLLM API at {args.api_url}...")
        result["api"] = {
            "base_url": args.api_url,
            **check_api_health(args.api_url),
        }

    if inside_container:
        # Inside container: read system info directly
        print("\nReading system info from /proc...")
        result["meminfo"] = read_proc_meminfo()

        # Check GPU via nvidia-smi
        print("  Running nvidia-smi...")
        stdout, stderr, rc = run_cmd(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=15,
        )
        if rc == 0:
            lines = [l.strip() for l in stdout.splitlines() if l.strip()]
            result["gpus"] = [
                dict(zip(["name", "memory_total_mb", "memory_used_mb", "memory_free_mb", "utilization_pct"],
                         [v.strip() for v in line.split(",")]))
                for line in lines
            ]
        else:
            result["gpus"] = {"error": stderr}

        # Model path inside container
        model_path = Path("/workspace/model")
        result["model_path_container"] = str(model_path)
        result["model_path_exists"] = model_path.exists()
        if model_path.exists():
            config_path = model_path / "config.json"
            result["config_json_exists"] = config_path.exists()

    else:
        # Host mode: use docker commands
        print(f"\nInspecting container: {args.container}...")
        result["container_inspect"] = inspect_container(args.container)

        print(f"\nGetting container stats...")
        result["container_stats"] = get_docker_stats(args.container)

        print(f"\nReading container logs (last {args.logs_lines} lines)...")
        log_lines = get_container_logs_tail(args.container, args.logs_lines)
        result["log_tail"] = log_lines
        result["vllm_args_from_logs"] = extract_vllm_args_from_logs(log_lines)

        # nvidia-smi on host
        print("  Running nvidia-smi on host...")
        stdout, stderr, rc = run_cmd(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=15,
        )
        if rc == 0:
            lines = [l.strip() for l in stdout.splitlines() if l.strip()]
            result["gpus"] = [
                dict(zip(["name", "memory_total_mb", "memory_used_mb", "memory_free_mb", "utilization_pct"],
                         [v.strip() for v in line.split(",")]))
                for line in lines
            ]
        else:
            result["gpus"] = {"error": stderr}

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nWrote deployment_config.json to: {output_path}")

    # Print summary
    if "api" in result:
        health = result["api"].get("health_ok")
        models = result["api"].get("models")
        print(f"  API health: {'OK' if health else 'FAILED'}")
        if models:
            for m in models:
                print(f"  Loaded model: {m.get('id')}")

    if "container_inspect" in result:
        ci = result["container_inspect"]
        if "error" not in ci:
            print(f"  Container: {ci.get('name')} — {ci.get('status')}")
            print(f"  Image: {ci.get('image')}")

    if "gpus" in result and isinstance(result["gpus"], list):
        for gpu in result["gpus"]:
            name = gpu.get("name", "?")
            total = gpu.get("memory_total_mb", "?")
            used = gpu.get("memory_used_mb", "?")
            print(f"  GPU: {name} — {used}/{total} MB used")


if __name__ == "__main__":
    main()
