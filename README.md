# strata-nemo: Nemotron 3.5 Bolt, Recursive Flint and Ling-3.0-flash on one Radeon AI PRO R9700

`strata-nemo` adds three programs to a copy of [Strata](https://github.com/Niko1221/Strata) (commit `7695859a`, with
the R9700 and low-RAM patches from [strata-patches](https://github.com/adiirfan/strata-patches)). Each runs one
mixture-of-experts model on a 32 GB Radeon AI PRO R9700, and each works behind Strata's server, which offers OpenAI-
and Anthropic-compatible APIs and a web app. Strata's own README is now [README-STRATA.md](README-STRATA.md).

| Program | Model | Weights | Where they live | Context | Code |
| --- | --- | --- | --- | ---: | --- |
| `engine/strata-nemo` | [Nemotron 3.5 Bolt 74B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB), with the [LiLiCorr drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) | 24 GB | all on the card | 262,144 | `src/nemotron/` |
| `engine/strata-flint` | [Recursive Flint 40B-A2.5B](https://huggingface.co/RecursiveSI/flint), [Q4_1 GGUF](https://huggingface.co/RecursiveSI/flint-int4-gguf) | 26 GB | all on the card | 262,144 | `src/flint/` |
| `engine/strata-ling` | [Ling-3.0-flash-VL 124B-A5.5B](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), [Q4_0 GGUF](https://huggingface.co/bartowski/Ling-3.0-flash-VL-GGUF) | 67 GB | card + system RAM | 131,072 | `src/ling/` |

Ling is too big for the card. Its attention, shared weights and 23 GB of its experts stay on the card. The processor
computes the other experts from system RAM, and the card's set changes as the conversation goes on (see
[Ling-3.0-flash](#ling-30-flash)). Flint also runs this way on smaller cards: on an RTX 3080 (10 GB) with 4 GB of its
experts on the card.

## Results through the server

Strata's own benchmark, `tools/strata_bench.py`, against the server (`python -m serve.server --engine strata --config
...`). Medians, thinking off. The R9700 at a 230 W power limit. Speeds are in tokens per second.

| Test | Nemotron | Flint | Ling | Flint, RTX 3080 |
| --- | ---: | ---: | ---: | ---: |
| Chat: writing | 125.5 ¹ | 239.2 | 64.0 ² | 78.1 |
| Chat: first token | 0.13 s | 0.07 s | 0.91 s | 0.32 s |
| 4K prompt: reading | 4,890 | 6,610 | 1,055 | 2,185 |
| 4K prompt: first token | 0.84 s | 0.62 s | 3.90 s | 1.88 s |
| 16K prompt: reading | 5,203 | 6,313 | 931 | 1,484 |
| 16K prompt: first token | 3.18 s | 2.61 s | 17.6 s | 11.06 s |
| 32K prompt: reading | 4,988 | 5,569 | 645 | ³ |
| 32K prompt: first token | 6.62 s | 5.91 s | 50.9 s | ³ |
| Agent session (6 turns, 4 tools): prompt reused | 69% | 66% | 69% | 68% |
| Agent session: first token per turn | 0.65 s | 0.49 s | 5.63 s | 2.21 s |
| Agent session: writing | 104.6 | 208.2 | 62.2 | 72.1 |
| Runs per test | 3 | 3 | 2 | 3 |
| Context the server was started with | 262,144 | 262,144 | 65,536 | 20,480 |

¹ With the drafter. The range is wide (107-223) because it depends on how many of the drafter's guesses the model
keeps. Flint's writing has the server's guessing on (see [Recursive Flint](#recursive-flint)).

² The range is 44-71. The first chat runs before the card holds the experts chat uses; the later ones are faster.

³ Longer than the 20,480-token context the RTX 3080 server was started with.

## Nemotron 3.5 Bolt

Tokens per second, one run per row, from the command line.

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

- **Card memory** here has the context sized to the prompt (`--max-ctx` = prompt + 344). A full 262,144-token context
  uses the 256K row's memory whatever the prompt.
- **Short prompts with the drafter:** 123 tokens/s writing a story, 210 answering a worked-arithmetic question
  (prompts of 26-33 tokens, 400 tokens written).
- **Quality:** next-token perplexity 4.5656 on 8K tokens of the model's card and 7.5771 on 8K tokens of Strata's
  `docs/DETAILS.md`. The model card scores 4.5612 with every product kept in 16-bit, about 0.1% better.
- **Same answer with or without the drafter:** with the drafter, the arithmetic answer above is word for word the
  same as plain greedy writing.
- **Why reading slows down with length:** 80 of the model's 88 layers cost the same for every token. The 8 attention
  layers compare each new token with every one before it, so at 256K they take about half the reading time.

## Recursive Flint

Tokens per second, one run per row, from the command line, with the context sized for 262,144 tokens.

| Prompt | Reading | Writing | Reading, 8-bit keys | Writing, 8-bit keys |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 4,640 | 219 | 4,562 | 216 |
| 4K | 6,913 | 213 | 6,750 | 211 |
| 16K | 6,471 | 190 | 6,278 | 194 |
| 32K | 5,662 | 168 | 5,454 | 175 |
| 64K | 4,503 | 138 | 4,290 | 150 |
| 128K | 3,178 | 104 | 2,969 | 120 |
| 256K (260,000 tokens) | 2,017 | 70.0 | 1,852 | 85.8 |

- **Card memory at its peak:** 30.1-31.4 GB of the card's 32 GB (31.9 GB usable), or 28.4-29.7 GB with 8-bit keys.
  Nothing spills into system memory. The program itself holds 1.5 GB of RAM, mostly the 1.2 GB token table, which
  stays in RAM on purpose.
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
- **On an RTX 3080** (`FLINT_PLAN` with 4 GB of experts, a 20,480-token context): the server results above. The
  processor computes the experts the card does not hold, the same split as Ling's.

## Ling-3.0-flash

Ling-3.0-flash-VL is 124B parameters with 5.5B used per token: 42 layers, 35 of them a linear "delta" attention with a
running state and 7 of them full attention over a compressed history, and 512 experts a layer, 8 used per token. This
program reads text only; the picture reader is not wired in.

- **What sits where:** the card holds the attention, the shared expert, the output layer and 23 GB of the routed
  experts (26.0 GB in all). The rest of the experts stay in the GGUF, read through the file mapping, so it wants
  about 67 GB of free RAM to keep them out of the disk. Each layer's experts on the card and on the processor run at
  the same time, and the card adds the processor's share when it is ready.
- **The card's experts follow the conversation.** Which experts a text uses depends on the kind of text. A set picked
  from code and documentation covered 87% of what that text used, but only 37% of what everyday chat used; a set
  picked from half of the chat covered 79% of the other half. So the engine counts each layer's routing as it goes,
  and an expert used more than the layer's least-used one on the card takes its place: up to 2 after each written
  token, and up to 64 a layer for each prompt chunk, copied from where the prompt path already put them on the card.
  Over 24 chat questions, writing went from about 44 to about 70 tokens/s with it. `LING_ADAPT=0` keeps the starting
  set fixed.
- **Command line:** about 79 tokens/s writing after a 1,500-token code prompt, read at about 300 tokens/s. Each prompt
  chunk copies the experts it uses that the card lacks, which costs much the same for a short chunk as a long one, so
  longer prompts read faster (1,055 tokens/s at 4K through the server).
- **Accuracy,** against Ling computed exactly in 64-bit floats (`tools/ling/ref_ling.py`, dequantizing the same
  GGUF): KL divergence about 1e-11 on the writing path and 3e-8 on the prompt path, the same top token at every
  position. The prompt path's matrix-core products split each input into two 16-bit halves, so they stay this close.
- **Context:** 131,072 tokens, the model's own limit (its GGUF says so, and it carries no setting for stretching it
  further). All working memory is set aside at start: at 131,072 tokens 0.75 GB of the card stays free, at 65,536
  (the benchmark above) 1.8 GB.
- **Conversation reuse:** the delta layers' running state cannot be rewound to an earlier token, so the engine saves
  it one token before the end of each prompt (`LING_SNAPS`, 4 kept). A turn that extends the last one goes on from
  where it stopped; otherwise from the latest saved state inside the shared part.

## How these were measured

- **Programs:** from the command line for the per-model tables; through the server for the shared table.
- **Prompts:** real text, not repeated: the ONNX Runtime and llama.cpp documentation, joined and cut to each size.
  Then 256 tokens written greedily, continuing the text.
- **Card memory:** the card's peak use during the run (GB here as on the card's box: 1,073,741,824 bytes).

**Desktop:**
- AMD Ryzen 5 7500F (6 cores), 96 GB DDR5-5600.
- Radeon AI PRO R9700 32 GB at a 230 W power limit (default 300 W).
- Debian 13, kernel 7.1.8, ROCm 10.2.0a20261009 (AMD TheRock).
- The RTX 3080 runs in the same desktop, on PCIe 4.0 x8, built with clang 19 and CUDA 12.5.

## Build

Build against AMD TheRock ROCm 10 (`pip install "rocm[libraries,devel,device-gfx1201]"` from
`https://nightly.repo.amd.com/rocm/whl-next/`; the SDK folder is its `_rocm_sdk_devel`):

```sh
tools/r9700/build-nemo.sh <ROCm 10 SDK dir>     # writes engine/strata-nemo
tools/r9700/build-flint.sh <ROCm 10 SDK dir>    # writes engine/strata-flint
tools/r9700/build-ling.sh <ROCm 10 SDK dir>     # writes engine/strata-ling
tools/cuda/build-flint.sh                       # Flint for NVIDIA: writes engine/strata-flint-cuda
```

## Run from the command line

The programs read and write token ids, not text. Turn a prompt into ids with the model's tokenizer (apply the chat
template yourself for a chat), for example:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("<model dir>/tokenizer.json")
open("prompt_ids.txt", "w").write(",".join(map(str, tok.encode(text, add_special_tokens=False).ids)))
```

Then:

```sh
# Nemotron: first write the tensor list the model and the drafter each need
python3 tools/nemotron/manifest.py <model dir>
python3 tools/nemotron/manifest.py <drafter dir>
engine/strata-nemo --model <model dir> --ids @prompt_ids.txt --new 256 --max-ctx 262144
engine/strata-nemo --model <model dir> --ids @prompt_ids.txt --new 256 --max-ctx 262144 --mode spec --draft <drafter dir>
engine/strata-nemo --model <model dir> --ids @text_ids.txt --ppl --max-ctx 9000      # perplexity over a text

# Flint
engine/strata-flint --model flint-int4-q4_1.gguf --ids @prompt_ids.txt --new 256 --max-ctx 262144

# Ling: LING_PLAN names the experts the card starts with (below)
LING_PLAN=plan-23gb.txt engine/strata-ling --model Ling-3.0-flash-VL-Q4_0-00001-of-00002.gguf \
    --ids @prompt_ids.txt --new 256 --max-ctx 131072
```

The written token ids go to standard output, the speeds to standard error. `--ids` also takes the ids inline
(`--ids 1784,8961,1307`).

**Expert plans** (Ling, and Flint on a smaller card): run some text with `LING_ROUTE_DUMP=<file>` (Flint:
`FLINT_ROUTE_DUMP`) to record the routing, then pick the most used experts up to a size:

```sh
python3 tools/flint/make_plan.py plan-23gb.txt routes1.bin routes2.bin --model ling --experts 23GB
```

Without a plan, Ling starts with no experts on the card and the processor computes them all.

## Run through the server

Write the model's tokenizer folder, then a config for it:

```sh
python3 tools/nemotron/export_tokenizer.py <model dir> <tokenizer dir>                 # Nemotron
python3 tools/strata_tokenizer.py --gguf flint-int4-q4_1.gguf --out <dir>                # Flint: <dir>/tokenizer
python3 tools/strata_tokenizer.py --gguf Ling-3.0-flash-VL-Q4_0-00001-of-00002.gguf --out <dir>   # Ling
python -m serve.server --engine strata --config strata-ling.json --port 8080
```

```json
{
  "exe": "<repo>/engine/strata-ling",
  "args": ["--model", "<model dir>/Ling-3.0-flash-VL-Q4_0-00001-of-00002.gguf", "--max-context", "131072"],
  "tokenizer": "<dir>/tokenizer",
  "model_family": "ling",
  "model_name": "ling-3.0-flash",
  "lib_dirs": ["<ROCm 10 SDK dir>/lib", "<ROCm 10 SDK dir>/lib/rocm_sysdeps/lib", "<ROCm 10 SDK dir>/lib/llvm/lib"],
  "port": 8080,
  "backend": "hip",
  "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
  "env": {"LING_PLAN": "<model dir>/plan-23gb.txt"}
}
```

What changes for the other two:

| Field | Nemotron | Flint |
| --- | --- | --- |
| `exe` | `engine/strata-nemo` | `engine/strata-flint` (`strata-flint-cuda` on NVIDIA, with `"backend": "cuda"`) |
| `args` | `--model <model dir> --draft <drafter dir> --max-context 262144` | `--model <dir>/flint-int4-q4_1.gguf --max-context 262144` |
| `model_family` | `nemotron` | `flint` |
| `sampling` | temperature 1.0, top_p 0.95 | temperature 1.0, top_p 0.95, top_k 20 |
| `env` | | on a smaller card: `FLINT_PLAN` |

## How it works

**Nemotron:**
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

**Flint and Ling, with experts in RAM:**
- **One graph a token.** The card writes a layer's expert input and choices into memory the processor can see and
  raises a flag. The processor's workers (one pinned to each physical core) compute the experts the card does not
  hold, and a card kernel waits for their flag and adds the result. No call from the host is needed in the middle, so
  the whole step stays one recorded graph.
- **Prompts** copy the experts each chunk uses to the card and run them there as grouped matrix products.

**Ling:**
- **Weights are rearranged at load** into forms the card reads in whole 16- or 32-byte pieces (the 4-bit codes
  first, then the scales). Writing reads one 32-value block of a row per thread.
- **Full-attention layers** keep the compressed history (512 + 64 values a token) and read it once for all 32 heads
  per token.

**Switches:** most newer paths have an environment variable that turns the older one back on. The commit messages
name them.

## Limits

- The programs take and give token ids. The tokenizer, chat template and API are Strata's Python server (`serve/`).
- Nemotron's command line writes greedily only.
- Nemotron and Ling are built and tested for the R9700 (gfx1201) only. Flint also builds for NVIDIA cards and was
  tested on an RTX 3080.
- Ling reads text only, up to 131,072 tokens; reading a 32K prompt takes about 50 s.

## License

The code is under Strata's MIT license (`LICENSE`). The models are not included; their own licenses apply.
