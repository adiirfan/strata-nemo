# strata-nemo: Nemotron 3.5 Bolt and Recursive Flint on one Radeon AI PRO R9700

`strata-nemo` runs NVIDIA's
[Nemotron 3.5 Bolt 74B-A9B (24 GB)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) entirely on
one 32 GB Radeon AI PRO R9700, with NVIDIA's
[LiLiCorr draft model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) for faster
writing. It reads up to the model's full 262,144-token context.

It is a separate program inside a copy of [Strata](https://github.com/Niko1221/Strata) (commit `7695859a`, with the
R9700 and low-RAM patches from [strata-patches](https://github.com/adiirfan/strata-patches)). Strata's own README is
now [README-STRATA.md](README-STRATA.md). The new code is in `src/nemotron/`, with some new kernels in
`src/kernels/cuda/iq_kernels.cu` and `src/prefill/`.

A second program runs [Recursive Flint](https://huggingface.co/RecursiveSI/flint): see
[Recursive Flint](#recursive-flint) below.

## Recursive Flint

`engine/strata-flint` runs Recursive Flint 40B-A2.5B from its
[Q4_1 GGUF](https://huggingface.co/RecursiveSI/flint-int4-gguf) (26 GB) entirely on the R9700, with the model's full
262,144-token context. The code is in `src/flint/`.

### Results

Tokens per second, one run per row, on the card at a 230 W limit, with the context sized for 262,144 tokens.
Writing is 256 tokens, greedy, continuing the prompt.

| Prompt | Reading | Writing | Reading, 8-bit keys | Writing, 8-bit keys |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 4,640 | 219 | 4,562 | 216 |
| 4K | 6,913 | 213 | 6,750 | 211 |
| 16K | 6,471 | 190 | 6,278 | 194 |
| 32K | 5,662 | 168 | 5,454 | 175 |
| 64K | 4,503 | 138 | 4,290 | 150 |
| 128K | 3,178 | 104 | 2,969 | 120 |
| 256K (260,000 tokens) | 2,017 | 70.0 | 1,852 | 85.8 |

- **Card memory at its peak:** 32.4-33.7 GB of the card's 34.2 GB (31.4 of its 31.9 GiB, the "32 GB" on the box), or
  30.5-31.8 GB with 8-bit keys. Nothing spills into system memory. The program itself holds 1.5 GB of RAM, mostly the
  1.2 GB token table, which stays in RAM on purpose.
- **llama.cpp on the same card** (with Flint support, 16-bit cache): 79 tokens/s writing, reading 5,388 at 4K and
  4,843 at 16K.
- **8-bit keys** (`FLINT_K8=1`, off by default) store the long-history layers' keys in 8 bits as well as their
  values. Writing at long context gets faster and 1.8 GB of card memory is freed, but the accuracy loss doubles (below).
- **Accuracy,** against Flint computed exactly in 64-bit floats (`tools/flint/ref_all.py`), over 512 positions of
  real text: KL divergence 0.008 with the default 8-bit values, 0.010 with a 16-bit cache, 0.018 with 8-bit keys too.
  The engine picks the same top token as the exact model 95.3% of the time (94.3% with 8-bit keys). For scale,
  rounding the layers' inputs to 8 bits, as llama.cpp does, costs 0.068 on a 300-token text.
- **Guessing ahead** (on in the server): when the text repeats something earlier in the conversation, the server
  guesses the next 1-3 tokens and checks them in one step. Copying a function out of a 66,000-token prompt goes from
  128 to about 190 tokens/s, with the same output as plain greedy writing. On fresh text it guesses only when that
  pays, so the speed is unchanged. `FLINT_SPEC=0` turns it off.
- **Why writing slows down with length:** 14 of the model's 49 layers compare each new token with every token before
  it, and Flint spreads that attention widely, so every word reads the whole stored history: 1.9-2.8 GB at 128K. The
  engine reads it at about 87% of the card's memory bandwidth.

### Through the server: `tools/strata_bench.py`

Medians of 3 runs, thinking off, guessing on, a 262,144-token context, same card at 230 W.

| Test | Result |
| --- | ---: |
| Chat: writing speed | 239.2 tokens/s (233-240) |
| Chat: first token | 0.07 s |
| 4K prompt: reading | 6,610 tokens/s, first token 0.62 s |
| 16K prompt: reading | 6,313 tokens/s, first token 2.61 s |
| 32K prompt: reading | 5,569 tokens/s, first token 5.91 s |
| Agent session (6 turns, 4 tools): prompt reused from cache | 66% |
| Agent session: first token per turn | 0.49 s |
| Agent session: writing speed | 208.2 tokens/s |

### Build and run

```sh
tools/r9700/build-flint.sh <ROCm 10 SDK dir>          # writes engine/strata-flint
engine/strata-flint --model flint-int4-q4_1.gguf --ids @prompt_ids.txt --new 256 --max-ctx 262144
```

Through the server, first write the tokenizer folder from the GGUF, then a config such as `strata-flint.json`:

```sh
python3 tools/strata_tokenizer.py --gguf flint-int4-q4_1.gguf --out <tokenizer dir>
python -m serve.server --engine strata --config strata-flint.json --port 8080
```

```json
{
  "exe": "<repo>/engine/strata-flint",
  "args": ["--model", "<model dir>/flint-int4-q4_1.gguf", "--max-context", "262144"],
  "tokenizer": "<tokenizer dir>",
  "model_family": "flint",
  "model_name": "flint",
  "lib_dirs": ["<ROCm 10 SDK dir>/lib", "<ROCm 10 SDK dir>/lib/rocm_sysdeps/lib", "<ROCm 10 SDK dir>/lib/llvm/lib"],
  "port": 8080,
  "backend": "hip",
  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
}
```

The prompts above are the same documentation text as the Nemotron table below. `tools/flint/` holds the accuracy
kit: the exact reference, a KL comparison of two logit files, and a launcher.

## Nemotron results

Tokens per second, one run per row, on the card at a 230 W limit (default 300 W).

| Prompt | Reading | Writing | Writing with the drafter | Tokens a round ¹ | Card memory | With the drafter |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 4,323 | 79.9 | 122 | 3.3 | 26.0 GB | 27.6 GB |
| 4K | 5,753 | 79.3 | 118 | 3.2 | 26.0 GB | 27.7 GB |
| 16K | 5,433 | 78.3 | 112 | 3.1 | 26.1 GB | 27.8 GB |
| 32K | 5,097 | 76.9 | 185 | 5.7 | 26.3 GB | 27.9 GB |
| 64K | 4,555 | 74.1 | 127 | 3.8 | 26.5 GB | 28.2 GB |
| 128K | 3,730 | 68.8 | 154 | 4.9 | 27.1 GB | 28.7 GB |
| 256K (261,800 tokens) | 2,746 | 60.7 | 163 | 6.0 | 28.2 GB | 29.8 GB |

¹ Tokens written per round with the drafter: the drafter guesses up to 7 tokens, the model checks them in one step and
keeps the ones it agrees with, plus one of its own. How many it keeps depends on how predictable the text is, not on
the prompt's length, so the drafter's speed moves around from row to row.

- **Short prompts with the drafter:** 123 tokens/s writing a story, 210 answering a worked-arithmetic question
  (prompts of 26-33 tokens, 400 tokens written).
- **Quality:** next-token perplexity 4.5656 on 8K tokens of the model's card and 7.5771 on 8K tokens of Strata's
  `docs/DETAILS.md`. The model card scores 4.5612 with every product kept in 16-bit, about 0.1% better.
- **Same answer with or without the drafter:** with the drafter, the arithmetic answer above is word for word the
  same as plain greedy writing.
- **Why reading slows down with length:** 80 of the model's 88 layers cost the same for every token. The 8 attention
  layers compare each new token with every one before it, so at 256K they take about half the reading time.

## Nemotron through the server: `tools/strata_bench.py`

Strata's own benchmark, run against the server (`python -m serve.server --engine strata --config ...`) with the
drafter on and a 262,144-token context. Medians of 3 runs, thinking off, same card at 230 W.

| Test | Result |
| --- | ---: |
| Chat: writing speed | 125.5 tokens/s (107-223) |
| Chat: first token | 0.13 s |
| 4K prompt: reading | 4,890 tokens/s, first token 0.84 s |
| 16K prompt: reading | 5,203 tokens/s, first token 3.18 s |
| 32K prompt: reading | 4,988 tokens/s, first token 6.62 s |
| Agent session (6 turns, 4 tools): prompt reused from cache | 69% |
| Agent session: first token per turn | 0.65 s |
| Agent session: writing speed | 104.6 tokens/s |

The chat writing speed ranges widely because it depends on how many of the drafter's guesses the model keeps.

## How these were measured

- **Program:** `engine/strata-nemo` from the command line (the table above; the server section has its own).
- **Prompts:** real text, not repeated: the ONNX Runtime and llama.cpp documentation, joined and cut to each size.
  Then 256 tokens written greedily, continuing the text.
- **Memory:** the card's peak use during the run, with the context sized to the prompt (`--max-ctx` = prompt + 344).
  A full 262,144-token context uses the 256K row's memory whatever the prompt.

**Desktop:**
- AMD Ryzen 5 7500F (6 cores), 96 GB DDR5-5600. Nothing is held in RAM: the model lives on the card.
- Radeon AI PRO R9700 32 GB at a 230 W power limit (default 300 W).
- Debian 13, kernel 7.1.8, ROCm 10.2.0a20261009 (AMD TheRock).

## Build and run

Build against AMD TheRock ROCm 10 (`pip install "rocm[libraries,devel,device-gfx1201]"` from
`https://nightly.repo.amd.com/rocm/whl-next/`; the SDK folder is its `_rocm_sdk_devel`):

```sh
tools/r9700/build-nemo.sh <ROCm 10 SDK dir>          # writes engine/strata-nemo
```

Download the model and the drafter from Hugging Face, then write the tensor list each one needs:

```sh
python3 tools/nemotron/manifest.py <model dir>
python3 tools/nemotron/manifest.py <drafter dir>
```

The program reads and writes token ids, not text. Turn a prompt into ids with the model's `tokenizer.json` (apply the
chat template yourself for a chat), for example:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("<model dir>/tokenizer.json")
open("prompt_ids.txt", "w").write(",".join(map(str, tok.encode(text, add_special_tokens=False).ids)))
```

Then:

```sh
# write 256 tokens
engine/strata-nemo --model <model dir> --ids @prompt_ids.txt --new 256 --max-ctx 262144
# the same with the drafter
engine/strata-nemo --model <model dir> --ids @prompt_ids.txt --new 256 --max-ctx 262144 --mode spec --draft <drafter dir>
# next-token perplexity over a text
engine/strata-nemo --model <model dir> --ids @text_ids.txt --ppl --max-ctx 9000
```

The written token ids go to standard output, the speeds to standard error. `--ids` also takes the ids inline
(`--ids 1784,8961,1307`).

## How it works

- **Everything on the card:** every weight stays in its stored format (NVFP4 state-update projections, IQ2_XXS
  experts, Q8_0 attention and output layer).
- **Reading** goes in chunks of 4,096 tokens: the state-update projections and the shared expert as 8-bit (FP8)
  matrix products, the routed experts unpacked straight into the matrix cores' registers, the state-update scan and
  the attention on the 16-bit matrix cores.
- **Writing** replays each step as a recorded graph. One token at a time, the chosen experts are read in one launch
  and multiplied in FP16 pairs, the router picks them in one launch, and the shared expert runs beside them on a
  second stream. The drafter's checks group the tokens' picks by expert, so each expert is read once.
- **The drafter** guesses 7 tokens a round. The engine checks only as many as recent rounds suggest will pay off, so
  hard text costs fewer wasted checks and easy text still checks all 7.
- **Switches:** most newer paths have an environment variable that turns the older one back on. The commit messages
  name them.

## Limits

- Token ids in and out: no tokenizer, chat template or server inside the program.
- Greedy writing only.
- Built and tested for the R9700 (gfx1201) only.

## License

The code is under Strata's MIT license (`LICENSE`). The models are not included; NVIDIA's license for them applies.
