# SPEX — Speculative Exploration for Tree-of-Thought Reasoning

This repository is the artifact for the OSDI '26 paper

> **Breaking the Reward Barrier: Accelerating Tree-of-Thought Reasoning via
> Speculative Exploration**
>

SPEX is a system that accelerates Tree-of-Thought (ToT) reasoning by
overlapping reward-guided exploration with speculative branch expansion.
It implements three techniques:

1. **Intra-query speculative path selection** — predict and pre-expand
   high-potential branches before their reward arrives.
2. **Inter-query budget allocation** — globally bound the in-flight
   speculative work across concurrent queries so spec never starves the
   primary path.
3. **Adaptive early termination** — prune deep branches in skewed search
   trees once enough confidence has accumulated.

SPEX is implemented on top of a patched [SGLang](https://github.com/sgl-project/sglang).
We provide two reference ToT drivers, one per paradigm:

| Driver           | Paradigm | Paper algorithm | Entry point             |
|------------------|----------|-----------------|-------------------------|
| `bfs_async.py`   | BFS      | REBASE / ETS    | `scripts/bfs_async.sh`  |
| `dfs_async.py`   | DFS      | RSTAR-MCTS      | `scripts/dfs_async.sh`  |

Both drivers share the producer–consumer execution framework in
`utils.py` (request queue, `BudgetCoordinator`, trace dumper, …).

---

## 1. Repository layout

```
.
├── bfs_async.py             # BFS / REBASE driver (async, with intra+inter spec)
├── dfs_async.py             # DFS / RSTAR-MCTS driver (async, with intra+inter spec)
├── utils.py                 # shared producer–consumer + BudgetCoordinator + trace
├── math_evaluate.py         # answer-grading helpers
├── hype-parameters/bfs.yaml # base hyper-parameters (width, max_tokens, …)
├── math500.jsonl            # MATH-500 dataset (500 problems)
├── scripts/
│   ├── run_policy.sh        # launch policy LLM server (sglang)
│   ├── run_reward.sh        # launch reward-model server (sglang, patched)
│   ├── bfs_async.sh         # entry point for BFS driver
│   ├── dfs_async.sh         # entry point for DFS driver
│   ├── analyze_trace.py     # per-query trace analyser
│   └── analyze_mt_trace.py  # multi-thread global timeline analyser
└── evaluate/                # answer extractors and graders (used by drivers)
```

The patched **SGLang** is shipped as a git submodule under `sglang/`
(see `.gitmodules`), pointing at branch `reward-model` of
[`thu-wyz/sglang`](https://github.com/thu-wyz/sglang). §3.1 below installs it.

---

## 2. Models and datasets

| Component   | Model                              | Default location                    |
|-------------|------------------------------------|-------------------------------------|
| Policy      | Llemma-metamath-7b                 | `/opt/models/Llemma-metamath-7b`    |
| Reward      | Llemma-reward-model                | `/opt/models/Llemma-reward-model`   |

Other models reported in the paper (Llemma-34b, DeepSeek-R1-8b,
Qwen3-30B-A3B) are launched the same way — point `run_policy.sh` at the
checkpoint of choice. HuggingFace links:

- [tkitsers/Llemma-metamath-7b](https://huggingface.co/tkitsers/Llemma-metamath-7b)
- [tkitsers/Llemma-metamath-34b](https://huggingface.co/tkitsers/Llemma-metamath-34b)
- [tkitsers/Llemma-reward-model](https://huggingface.co/tkitsers/Llemma-reward-model)

Datasets used in the paper:

| Family   | Datasets                                          |
|----------|---------------------------------------------------|
| Llemma   | MATH-500 (`math500.jsonl`), GSM8K                 |
| DeepSeek | AIME 2024, AIME 2025, BRUMO, HMMT                 |
| Qwen     | OlymMATH, AIME, …                                 |

The shipped `math500.jsonl` mirrors the
[prm800k MATH split](https://github.com/openai/prm800k/tree/main/prm800k/math_splits/test.jsonl).

---

## 3. Setup

### 3.1 Conda environment

The patched SGLang fork ([`thu-wyz/sglang`](https://github.com/thu-wyz/sglang),
branch `reward-model`) is shipped as a git submodule under `sglang/`.
It is the same fork used by
[Wu et al. 2024 (Inference Scaling Laws)](https://arxiv.org/abs/2408.00724)
and adds the process-reward-model API that REBASE / ToT search relies on.

```bash
# 1. Clone the artifact with the patched SGLang submodule
git clone --recurse-submodules git@github.com:shuzhangzhong/SPEX.git
cd SPEX

# 2. Conda env
conda create -n spex python=3.10 -y
conda activate spex

# 3. Install the patched SGLang
cd sglang/python
pip install .
cd ../..
pip install outlines==0.0.44

# 4. Driver dependencies
pip install pyyaml aiohttp tqdm pulp scipy numpy torch
```

After `pip install .`, `import sglang` resolves out of the conda env's
site-packages — no `PYTHONPATH` setup is required.

### 3.2 Launch servers

Two GPUs are needed by default — one for the policy LLM, one for the
reward model. Edit `CUDA_VISIBLE_DEVICES` / `PORT` in the scripts as
needed.

```bash
# Terminal 1 — policy on GPU 0, port 12345
bash scripts/run_policy.sh

# Terminal 2 — reward on GPU 1, port 23456
bash scripts/run_reward.sh
```

Wait until both servers print their `Server is ready` line before
running any driver.

---

## 4. Running SPEX

Both drivers read the same set of environment variables; the only
difference is the algorithm-specific flags forwarded after the wrapper.

### 4.1 BFS

```bash
NUM_THREADS=4 WIDTH=8 SPECULATIVE=1 \
  TOTAL_PARALLEL_BUDGET=48 \
  TRACE_DIR=./traces/bfs_demo \
  bash scripts/bfs_async.sh
```

Wrapper flags:

| Env var                | Default                          | Meaning                                                |
|------------------------|----------------------------------|--------------------------------------------------------|
| `POLICY` / `REWARD`    | `12345` / `23456`                | local server ports                                     |
| `DATA_DIR`             | `./math500.jsonl`                | input dataset (jsonl with `problem` / `solution`)      |
| `OUT_DIR`              | `./exp_results/`                 | append-only run log directory                          |
| `PARA_PATH`            | `./hype-parameters/bfs.yaml`     | search hyper-parameters                                |
| `NUM_THREADS`          | `1`                              | concurrent queries                                     |
| `WIDTH`                | `8`                              | per-step expansion width (REBASE *W*)                  |
| `SPECULATIVE`          | `1`                              | enable intra-query speculation (set to `0` for vanilla REBASE) |
| `EARLY_TERMINATION`    | `1`                              | enable adaptive early termination                      |
| `ETS`                  | `0`                              | use ETS variant of REBASE                              |
| `TRACE_DIR`            | unset                            | per-query trace dump dir (see §6)                      |
| `TOTAL_PARALLEL_BUDGET`| `1.5 × NUM_THREADS × WIDTH`      | inter-query budget *M* (see §5.2)                      |

### 4.2 DFS

```bash
NUM_THREADS=4 WID=4 NUM_SPEC=4 \
  NUM_PATH_LIST="5 10" \
  EXTRA_FLAGS="--speculative --early_termination" \
  TOTAL_PARALLEL_BUDGET=64 \
  TRACE_DIR_BASE=./traces/dfs_demo \
  MAX_QUESTIONS=20 \
  bash scripts/dfs_async.sh
```

Wrapper flags:

| Env var                | Default                          | Meaning                                                |
|------------------------|----------------------------------|--------------------------------------------------------|
| `POLICY` / `REWARD`    | `12345` / `23456`                | local server ports                                     |
| `NUM_PATH_LIST`        | `5 10 20`                        | values of `--num_path` (paths per query) to sweep      |
| `NUM_THREADS`          | `10`                             | concurrent queries                                     |
| `WID`                  | `4`                              | branching factor per node                              |
| `NUM_SPEC`             | `4`                              | speculative children per main child (intra-query)      |
| `EXTRA_FLAGS`          | `--speculative --early_termination`     | append `--rm prm` for PRM, `--rm orm` for outcome RM   |
| `MAX_QUESTIONS`        | `0` (= all)                      | early-cutoff for shorter runs                          |
| `TRACE_DIR_BASE`       | unset                            | trace dir prefix; suffixed `_np<num_path>` per sweep   |
| `TOTAL_PARALLEL_BUDGET`| `0.8 × NUM_THREADS × WID × (NUM_SPEC+1)` | inter-query budget *M* (see §5.2)                |

---

## 5. SPEX knobs

### 5.1 Intra-query (per-tree)

All knobs are env vars consumed by `bfs_async.py` (BFS) /
`dfs_async.py` (DFS). They preserve legacy behaviour when unset.

| Env var                       | Driver | Meaning                                                       | Default |
|-------------------------------|--------|---------------------------------------------------------------|---------|
| `SPEC_LOOKAHEAD`              | BFS    | layers ahead `spec_all` walks (`d+1 .. d+1+L`)                | `2`     |
| `SPEC_BUDGET_SCALE`           | BFS    | `spec_widths = ceil(scale × \|cand\|)`                        | `1.2`   |
| `SPEC_LAYER_CAP_MULT`         | BFS    | per-layer cap = `mult × width`                                | `1e9`   |
| `SPEC_TOP_K_FRAC`             | BFS    | keep only top-K candidates by reward                          | `1.0`   |
| `SPEC_DEPTH_DECAY`            | BFS    | per-depth decay applied to spec budget                        | `1.0`   |
| `SPEC_USEFUL_GATE`            | BFS    | enable "useful spec ratio" EMA gate                           | `0`     |
| `SPEC_USEFUL_MIN`             | BFS    | minimum useful-EMA before spec is allowed                     | `0.25`  |
| `SPEC_USEFUL_MIN_FACTOR`      | BFS    | factor multiplied into the EMA threshold                      | `0.25`  |
| `DFS_PRE_FRAC` / `DFS_PRE_MAX`| DFS    | fraction / hard cap of pre-explored child paths               | unset   |

A reasonable starting point for BFS at `nt=1`:

```bash
SPEC_LOOKAHEAD=3 SPEC_USEFUL_GATE=1 \
SPEC_USEFUL_MIN=0.25 SPEC_USEFUL_MIN_FACTOR=0.4 \
  bash scripts/bfs_async.sh
```

### 5.2 Inter-query (across all threads): `TOTAL_PARALLEL_BUDGET`

`utils.BudgetCoordinator` enforces a global cap *M* on the number of
in-flight requests across **all** concurrent queries. Main-path
requests always claim a slot; speculative requests claim a slot **only
if** `main_active + spec_active < M`. Otherwise spec is rejected (the
request is dropped silently and producers stay free for someone else).

Recommended *M*:

| Algorithm | Rule of thumb                                  | Why                                            |
|-----------|------------------------------------------------|------------------------------------------------|
| BFS       | `M ≈ 1.5 × num_threads × width`                | leaves headroom for spec without overshoot     |
| DFS       | `M ≈ 0.8 × num_threads × wid × (num_spec + 1)` | DFS rarely saturates compute; be permissive    |

When `TOTAL_PARALLEL_BUDGET` is **unset**, no global cap applies and
each tree manages its own budget locally (legacy behaviour).

The driver prints a one-line summary at the end of the run:

```
[budget-stats] total=64 peak_total=63 peak_main=16 peak_spec=58 spec_claims=2901 spec_rejects=42
```

### 5.3 Adaptive early termination (`--early_termination`)

Pass `--early_termination` (BFS) or include it in `EXTRA_FLAGS` (DFS) to enable
the §3 adaptive early-termination policy. It prunes deep branches once
the running confidence on the best path exceeds a learned threshold.

---

## 6. Citation

If you use this artifact, please cite:

```bibtex
@inproceedings{spex_osdi26,
  title     = {Breaking the Reward Barrier: Accelerating Tree-of-Thought
               Reasoning via Speculative Exploration},
  booktitle = {Proceedings of the 19th USENIX Symposium on Operating
               Systems Design and Implementation (OSDI '26)},
  author        = {Shuzhang Zhong and Haochen Huang and Shengxuan Qiu and Pengfei Zuo and Runsheng Wang and Meng Li},
  year      = {2026},
}
```

The REBASE baseline this work builds on:

```bibtex
@misc{wu2024inferencescalinglawsempirical,
  title         = {Inference Scaling Laws: An Empirical Analysis of
                   Compute-Optimal Inference for Problem-Solving with
                   Language Models},
  author        = {Yangzhen Wu and Zhiqing Sun and Shanda Li and Sean
                   Welleck and Yiming Yang},
  year          = {2024},
  eprint        = {2408.00724},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2408.00724},
}
```
