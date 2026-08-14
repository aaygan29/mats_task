"""Run the REAL Qwen3-4B pipeline on a Modal A10G (24GB) and pull artifacts back.

Usage (Modal already authenticated on this machine):
    python -m modal run modal_run.py                      # uses configs/main.yaml
    python -m modal run modal_run.py --config configs/main.yaml

It copies this repo into the container, runs src.run_all into /tmp/out, zips
results/ + figures/ + checkpoints/jvecs.pt, and writes them back into the local repo.
Expected cost: ~$1-3 (A10G ~ $1.10/hr, ~10-30 min incl. model download).
"""
import io
import os
import zipfile
import modal

REPO = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.44", "numpy", "matplotlib",
                 "pyyaml", "safetensors", "huggingface_hub", "accelerate")
    .add_local_dir(os.path.join(REPO, "src"), "/root/mats_task/src")
    .add_local_dir(os.path.join(REPO, "data"), "/root/mats_task/data")
    .add_local_dir(os.path.join(REPO, "configs"), "/root/mats_task/configs")
)

app = modal.App("mats-jlens")


@app.function(image=image, gpu="A10G", timeout=3600)
def run_pipeline(config_rel: str) -> bytes:
    import sys
    sys.path.insert(0, "/root/mats_task")
    os.chdir("/root/mats_task")
    from src import run_all
    out_dir = "/tmp/out"
    run_all.main(os.path.join("/root/mats_task", config_rel), out_dir=out_dir, force=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("results", "figures"):
            base = os.path.join(out_dir, sub)
            for root, _, files in os.walk(base):
                for fn in files:
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.relpath(fp, out_dir))
        jp = os.path.join(out_dir, "checkpoints", "jvecs.pt")
        if os.path.exists(jp):
            z.write(jp, os.path.relpath(jp, out_dir))
    return buf.getvalue()


@app.local_entrypoint()
def main(config: str = "configs/main.yaml"):
    data = run_pipeline.remote(config)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(REPO)
    print(f"[modal_run] wrote {len(data)} bytes of artifacts into {REPO}")
