"""export_tokenizer.py MODEL_DIR OUT_DIR: the model's Hugging Face tokenizer.json in the form Strata's server reads
(vocab.json, merges.txt, token_type.json, tokenizer.json with the pre-tokenizer's split pattern) plus its
chat_template.jinja. Added tokens marked special become control tokens (matched in a prompt only where the server
asks for specials), the others user-defined (always matched), as llama.cpp types them."""
import json, pathlib, shutil, sys

if len(sys.argv) != 3:
    sys.exit("usage: export_tokenizer.py MODEL_DIR OUT_DIR")
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
t = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
m = t["model"]
if m["type"] != "BPE":
    sys.exit(f"a {m['type']} tokenizer: only byte-level BPE is supported")
pre = t.get("pre_tokenizer") or {}
parts = pre.get("pretokenizers", [pre])
split = next((p for p in parts if p.get("type") == "Split"), None)
if not split or not any(p.get("type") == "ByteLevel" for p in parts):
    sys.exit("expected a Split pre-tokenizer followed by ByteLevel")
pattern = split["pattern"]["Regex"]
vocab = dict(m["vocab"])
types = [1] * (max(vocab.values()) + 1)
special_ids = {}
for a in t.get("added_tokens", []):
    vocab[a["content"]] = a["id"]
    if a["id"] >= len(types):
        types += [1] * (a["id"] + 1 - len(types))
    types[a["id"]] = 3 if a.get("special") else 4
    if a["content"] in ("<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"):
        special_ids[a["content"]] = a["id"]
merges = [mg if isinstance(mg, str) else " ".join(mg) for mg in m["merges"]]
out.mkdir(parents=True, exist_ok=True)
(out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
(out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
(out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
(out / "tokenizer.json").write_text(json.dumps({
    "model": "gpt2", "pre": "nemotron", "vocab_size": len(types), "n_merges": len(merges), "special_ids": special_ids,
    "add_bos_token": False, "pre_pattern": pattern, "pre_pattern_source": "the model's tokenizer.json (Split)"},
    ensure_ascii=False, indent=1), encoding="utf-8")
shutil.copy(src / "chat_template.jinja", out / "chat_template.jinja")
print(f"{len(types)} tokens, {len(merges)} merges -> {out}")
