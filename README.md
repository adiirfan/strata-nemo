# strata-nemo: Nemotron 3.5 Bolt on one Radeon AI PRO R9700

`strata-nemo` runs NVIDIA's
[Nemotron 3.5 Bolt 74B-A9B (24 GB)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) entirely on
one 32 GB Radeon AI PRO R9700, with NVIDIA's
[LiLiCorr draft model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) for faster
writing. It reads up to the model's full 262,144-token context.

It is a separate program inside a copy of [Strata](https://github.com/Niko1221/Strata) (commit `7695859a`, with the
R9700 and low-RAM patches from [strata-patches](https://github.com/adiirfan/strata-patches)). Strata's own README is
now [README-STRATA.md](README-STRATA.md). The new code is in `src/nemotron/`, with some new kernels in
`src/kernels/cuda/iq_kernels.cu` and `src/prefill/`.

## Results

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

## How these were measured

- **Program:** `engine/strata-nemo` from the command line (there is no server yet).
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
