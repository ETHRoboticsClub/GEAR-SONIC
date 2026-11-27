"""Serve a fine-tuned GR00T policy to JPEG clients (``run_vla_inference.py --image-codec jpeg``).

Run it with the Python of an Isaac-GR00T install, next to the model:

    python gear_sonic/scripts/serve_gr00t_jpeg.py --model-path checkpoints/MODEL \\
        --embodiment-tag UNITREE_G1_SONIC --port 5550

It is ``gr00t/eval/run_gr00t_server.py`` for a ``--model-path``, with the policy wrapped in
``JpegDecodingPolicy``: a request whose ``video`` entries are JPEG is decoded first, and one
with raw arrays (GR00T's own ``PolicyClient``) is served unchanged. Only
``gear_sonic/utils/inference/jpeg_policy.py`` is needed beside this file; gear_sonic itself
need not be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


def _jpeg_policy():
    """``jpeg_policy`` from this checkout, without importing the gear_sonic package."""
    path = Path(__file__).resolve().parents[1] / "utils" / "inference" / "jpeg_policy.py"
    spec = importlib.util.spec_from_file_location("jpeg_policy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["jpeg_policy"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ServeConfig:
    model_path: str
    """The fine-tuned checkpoint directory."""

    embodiment_tag: str = "UNITREE_G1_SONIC"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 5550


def main(config: ServeConfig) -> None:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    jpeg = _jpeg_policy()
    tag = EmbodimentTag.resolve(config.embodiment_tag)
    print(f"Loading {config.model_path} ({tag}) on {config.device}...", flush=True)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=config.model_path, device=config.device)
    with PolicyServer(jpeg.JpegDecodingPolicy(policy), host=config.host, port=config.port) as server:
        server.run()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ServeConfig))
