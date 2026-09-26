"""New forward passes from a local raw checkpoint; never loads saved activations."""
import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import torch
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="IProgress not found.*")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging
logging.disable_progress_bar()

DEFAULT = (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/"
           "snapshots/a10cc1512eabd3dde888204e902eca88bddb4951")
PROMPTS = (
    "Explain why a solid state drive can read one large contiguous block faster than many small scattered blocks.",
    "A scientist compares retained activation importance with projection error. Explain why these two quantities need not rank masks in the same order.",
)


def fresh_activations(layers=(12,), seed=20260926):
    path = Path(os.environ.get("VLM_CHECKPOINT", DEFAULT)).expanduser().resolve()
    if not (path / "config.json").exists():
        raise FileNotFoundError("Set VLM_CHECKPOINT to a local SmolLM2-compatible raw checkpoint; no synthetic fallback")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    samples = {layer: [] for layer in layers}
    handles = []
    weights = {}
    for layer in layers:
        projection = model.model.layers[layer].mlp.down_proj
        weights[layer] = projection.weight.detach().cpu().numpy().T.copy()
        def hook(module, inputs, output, layer=layer):
            samples[layer].append(inputs[0].detach().float().cpu().reshape(-1, weights[layer].shape[0]).numpy().copy())
        handles.append(projection.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        with file.open("rb") as source:
            for block in iter(lambda: source.read(2**20), b""):
                digest.update(block)
    meta = dict(checkpoint=str(path), weights_sha256=digest.hexdigest(), layers=list(layers),
                prompts=list(PROMPTS), dtype="float32", device="cpu", threads=4,
                source="new local LLM forward passes; text inputs; no VLM task-accuracy claim")
    return {layer: (np.concatenate(samples[layer]).astype(float), weights[layer].astype(float)) for layer in layers}, meta
