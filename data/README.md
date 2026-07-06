# Data

This project analyses a **public neural recording dataset** but **does not
redistribute it**. No raw `.h5` / `.hdf5` files are included in this repository,
and `data/raw/` is git-ignored.

## Dataset

> Tili et al. 2025, *"Mirror Neurons in Monkey Frontal and Parietal Areas."*

The dataset contains single-unit spiking activity recorded from macaque areas
**AIP**, **F5**, and **F6** during an observation/execution Go/No-Go task.
Obtain it from the original public source and cite the authors — do not
re-host it here.

## Where to put the data

Place the raw HDF5 files so that the following structure exists:

```
data/raw/HDF5/
├── Spikes/
│   ├── AIP/
│   ├── F5/
│   └── F6/
└── Events/
    ├── AIP/
    ├── F5/
    └── F6/
```

Equivalently, point the pipeline at an existing copy anywhere on disk by
setting an environment variable:

```bash
# Linux / macOS
export DATA_ROOT=/absolute/path/to/HDF5

# Windows PowerShell
$env:DATA_ROOT = "D:\datasets\mirror\HDF5"
```

or by passing `--data-root` / `--base-dir` on the command line, e.g.:

```bash
python run_all.py --data-root /absolute/path/to/HDF5
python pipeline/01_batch_session_object_analysis.py --base-dir /absolute/path/to/HDF5
```

`DATA_ROOT` (or `--data-root`) should point at the directory that **contains
`Spikes/` and `Events/`**.

## File-naming convention the pipeline expects

Event files:

```
Monkey<A|B>_Session<NN>_Context<N>_Condition<N>_Object<N>_Event<N>.h5
```

Spike files:

```
Monkey<A|B>_Session<NN>_Spk_<unit>.h5
```

The pipeline maps task conditions as follows (see `pipeline/00_inventory.py`
and `src/core_utils.py`):

- **Observation (OBS)** trials: `Context2`
- **OBS Go**: `Condition1` · **OBS No-Go**: `Condition2`
- Analyses focus on **Event1** and **Event3**.

## What you do NOT need raw data for

The derived result tables in [`../results/`](../results) are shipped with the
repo, so the meta-summaries, statistics, and figures can be reproduced
**without** the raw HDF5 files. Raw data are only needed to regenerate the
per-session decoding tables from scratch (pipeline steps 01–03, 06, 10).
