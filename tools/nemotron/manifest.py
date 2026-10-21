"""manifest.py MODEL_DIR: write <model dir>/strata-manifest.txt, the tensor list strata-nemo loads from, so the engine
needs no JSON reader. One line per tensor, space-separated:

    name file data_offset nbytes dtype format dim0,dim1,...

data_offset is from the start of the file (the 8-byte header length + the header + the tensor's own start). format is
the stored layout the engine must use: IQ2_XXS / Q8_0 (ggml blocks), NVFP4 (packed weight; its .weight_scale and
.weight_scale_2 tensors are listed under their own names as E4M3 / F32), or the plain dtype (BF16, F32, F8_E4M3)."""
import json, struct, sys, glob, os

if len(sys.argv) != 2:
    sys.exit("usage: manifest.py MODEL_DIR (the model or the drafter's safetensors folder)")
root = sys.argv[1]
qc = json.load(open(f"{root}/hf_quant_config.json"))["quantization"]
# the target lists every quantized layer; the drafter has one global algorithm with exclusions, so there a U8 weight
# with a weight_scale beside it is NVFP4 (found per tensor below)
algo = {k: v["quant_algo"] for k, v in qc.get("quantized_layers", {}).items()}
global_algo = qc.get("quant_algo") if "quantized_layers" not in qc else None
lines = []
files = sorted(glob.glob(f"{root}/model-*.safetensors")) or [f"{root}/model.safetensors"]
for f in files:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for name, v in h.items():
        if name == "__metadata__" or (not global_algo and not name.startswith("language_model.")):
            continue
        a, b = v["data_offsets"]
        mod = name[: -len(".weight")] if name.endswith(".weight") else None
        a_ = algo.get(mod)
        if global_algo == "W4A16_NVFP4" and v["dtype"] == "U8" and (mod + ".weight_scale") in h:
            a_ = "W4A16_NVFP4"
        fmt = {"W4A16_NVFP4": "NVFP4"}.get(a_, a_) or v["dtype"]
        shape = ",".join(str(d) for d in v["shape"]) or "1"
        lines.append(f"{name} {os.path.basename(f)} {8 + n + a} {b - a} {v['dtype']} {fmt} {shape}")
lines.sort()
open(f"{root}/strata-manifest.txt", "w").write("\n".join(lines) + "\n")
print(len(lines), "tensors ->", f"{root}/strata-manifest.txt")
