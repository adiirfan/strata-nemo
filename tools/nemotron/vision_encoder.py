"""vision_encoder.py --model DIR [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]: Nemotron 3.5 Bolt's image
encoder for Strata's server, in place of strata-vision and with its protocol: READY, then per line
`ENC <image> <out>` -> `OK <tokens> <nx> <ny>` (or `ERR <why>`), `QUIT`. --mmproj is accepted and ignored (the
server's vision config always passes one).

It runs NVIDIA's own RADIO encoder and image preprocessing from the model folder (PyTorch, transformers 5.x), then
the vision projector as the model does it: pixel shuffle (2x2 patches into one token), RMSNorm, linear, relu^2,
linear. Each output file is one strata-vision record: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd float32,
the rows strata-nemo --serve puts in place of the prompt's <image> tokens.
"""
import argparse
import json
import pathlib
import struct
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mmproj")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=0)
ap.add_argument("--min-tokens", type=int, default=0)
args = ap.parse_args()

import torch
from PIL import Image
from safetensors import safe_open

if args.threads:
    torch.set_num_threads(args.threads)
M = pathlib.Path(args.model)
# the model folder's code as a package, so its relative imports resolve
pkg = types.ModuleType("nemo_omni")
pkg.__path__ = [str(M)]
sys.modules["nemo_omni"] = pkg
from nemo_omni.configuration_radio import RadioConfig  # noqa: E402
from nemo_omni.modeling_radio import RadioModel  # noqa: E402
from nemo_omni.image_processing_nemotron_h_omni import NemotronH_Omni_Reasoning_V3ImageProcessor  # noqa: E402

cfg = json.loads((M / "config.json").read_text())
pp = json.loads((M / "preprocessor_config.json").read_text())
dev = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

vision = RadioModel(RadioConfig(**cfg["vision_config"]))
vision.make_preprocessor_external()          # the preprocessing normalizes, as in the model
# the checkpoint's timm-style names, renamed as modeling_radio's conversion mapping does (qkv split in three)
RENAME = [("radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"),
          ("radio_model.model.patch_generator.embedder", "embeddings.patch_projection"),
          ("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding"),
          ("radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"),
          ("radio_model.model.blocks", "encoder.layer"), ("attn.proj", "attention.output.dense"),
          ("radio_model.input_conditioner", "input_conditioner"), ("radio_model.summary_idxs", "summary_idxs")]
state, proj = {}, {}
for f in sorted(M.glob("model-*.safetensors")):
    with safe_open(str(f), "pt") as h:
        for k in h.keys():
            if k.startswith("mlp1."):
                proj[k] = h.get_tensor(k)
            if not k.startswith("vision_model."):
                continue
            name = k[len("vision_model."):]
            for a, b in RENAME:
                name = name.replace(a, b)
            t = h.get_tensor(k)
            if ".attn.qkv." in name:
                for part, piece in zip(("query", "key", "value"), t.chunk(3, dim=0)):
                    state[name.replace("attn.qkv", "attention.attention." + part)] = piece
            else:
                state[name] = t
missing, unexpected = vision.load_state_dict(state, strict=False)
missing = [k for k in missing if not k.endswith("lambda1")]           # layer scales: 1.0, not in the checkpoint
unexpected = [k for k in unexpected if not k.startswith("input_conditioner")]
if missing or unexpected:
    sys.exit(f"vision weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
vision = vision.to(dev, dt).eval()
norm_w = proj["mlp1.0.weight"].to(dev, torch.float32)
lin1 = proj["mlp1.1.weight"].to(dev, dt)
lin2 = proj["mlp1.3.weight"].to(dev, dt)

ip = NemotronH_Omni_Reasoning_V3ImageProcessor(**{k: v for k, v in pp.items()
                                                  if k not in ("auto_map", "image_processor_type", "_downsample_factor")})
# the server's budget, in tokens (one token = 2 x 2 patches)
if args.max_tokens > 0:
    ip.max_num_patches = max(ip.min_num_patches, min(ip.max_num_patches, args.max_tokens * 4))
if args.min_tokens > 0:
    ip.min_num_patches = min(ip.max_num_patches, args.min_tokens * 4)


def pixel_shuffle(x, s=0.5):                  # (n, w, h, c) -> (n, h*s, w*s, c/s^2), ps_version v2
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * s), int(c / s)).permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * s), int(w * s), int(c / (s * s)))
    return x.permute(0, 2, 1, 3).contiguous()


@torch.no_grad()
def encode(path, out):
    img = Image.open(path)
    img.load()
    b = ip(images=[img], return_tensors="pt")
    pv = b["pixel_values"].to(dev, dt)
    H, W = pv.shape[-2:]
    h, w = H // vision.patch_size, W // vision.patch_size
    feats = vision(pv).features                                       # (1, h*w, 1280)
    x = pixel_shuffle(feats.reshape(1, h, w, -1))                     # (1, h/2, w/2, 5120)
    x = x.reshape(-1, x.shape[-1])
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * norm_w
    y = torch.relu(xf.to(dt) @ lin1.T).pow(2) @ lin2.T                 # (n, 4096)
    y = y.float().cpu().contiguous()
    n, nx, ny = y.shape[0], w // 2, h // 2
    with open(out, "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, n, nx, ny, y.shape[1]))
        f.write(y.numpy().tobytes())
    return n, nx, ny


print("READY", flush=True)
for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    if parts[0] == "QUIT":
        break
    if parts[0] != "ENC" or len(parts) != 3:
        print("ERR expected: ENC <image> <out>", flush=True)
        continue
    try:
        n, nx, ny = encode(parts[1], parts[2])
        if dev.type == "cuda":                 # the activations' memory back to the card: the engine shares it
            torch.cuda.empty_cache()
        print(f"OK {n} {nx} {ny}", flush=True)
    except Exception as e:  # noqa: BLE001 - one bad image must not end the encoder
        print("ERR " + " ".join(str(e).split())[:300], flush=True)
