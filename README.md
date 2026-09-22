# Multi-Camera Person Re-Identification & Tracking System

A system for detecting, tracking, and re-identifying people across multiple
camera feeds. This repository is being built in phases; **Phase 1 (project
setup) is complete**. Detection, tracking, re-identification, matching,
topology/temporal reasoning, trajectory building, database persistence,
analytics, and the dashboard are **not yet implemented**.

## Project status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Project setup, config, logging, env/device checks | ✅ Done |
| 2 | Detection (YOLO) + single-camera tracking (ByteTrack) | ⏳ Not started |
| 3 | Re-identification (OSNet) + cross-camera matching (FAISS) | ⏳ Not started |
| 4 | Topology, temporal reasoning, trajectory building | ⏳ Not started |
| 5 | Database, analytics, dashboard | ⏳ Not started |

## Project structure

```
.
├── data/
│   ├── videos/            # input video files (gitignored)
│   └── reid/               # re-id gallery/query images (gitignored)
├── models/                 # model weights (gitignored)
├── src/
│   ├── config.py            # YAML config loader
│   ├── logging_setup.py     # logging configuration
│   ├── env_check.py         # environment + CPU/CUDA device checks
│   ├── detection/           # [Phase 2] person detection (YOLO)
│   ├── tracking/            # [Phase 2] single-camera tracking (ByteTrack)
│   ├── reid/                # [Phase 3] re-id embeddings (OSNet)
│   ├── matching/            # [Phase 3] cross-camera matching (FAISS)
│   ├── topology/            # [Phase 4] camera topology
│   ├── temporal/            # [Phase 4] temporal reasoning
│   ├── trajectory/          # [Phase 4] trajectory construction
│   ├── database/            # [Phase 5] persistence layer
│   └── analytics/           # [Phase 5] analytics/reporting
├── training/                # model training scripts (future)
├── dashboard/                # web dashboard (future)
├── configs/
│   ├── config.yaml           # main app configuration
│   └── reid.yaml             # re-id configuration (placeholder)
├── tests/                    # unit tests
├── notebooks/                 # exploration notebooks
├── main.py                    # application entry point
├── requirements.txt
└── .gitignore
```

## Requirements

- Python 3.9+
- See `requirements.txt` (Phase 1 only needs PyYAML and pytest; heavier ML
  dependencies such as `torch`, `ultralytics`, `torchreid`, and `faiss`
  will be added in later phases).

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the project

```bash
python main.py
```

This will:
1. Load `configs/config.yaml` (merged with `configs/reid.yaml`).
2. Set up console + rotating file logging (`logs/app.log` by default).
3. Run environment checks (Python version, required packages).
4. Detect whether CUDA is available (falls back cleanly to CPU, and to a
   "torch not installed" message if `torch` isn't installed yet).
5. Ensure the configured data directories exist.
6. Print a startup summary and exit with code `0` on success, `1` if the
   environment check failed.

### Useful flags

```bash
python main.py --config configs/config.yaml --reid-config configs/reid.yaml
python main.py --log-level DEBUG
```

### Configuration

All paths in `configs/config.yaml` are relative to the project root by
default (see `src/config.py: Config.get_path`), so the project can be
cloned and run from any location without editing absolute paths.

Configuration values can also be overridden with environment variables of
the form:

```bash
APP__LOGGING__LEVEL=DEBUG python main.py
```

which overrides `logging.level` in the loaded config.

## Running tests

```bash
pytest -v
```

or, with coverage:

```bash
pytest -v --cov=src
```

## Roadmap (not yet implemented)

- **Phase 2**: YOLO-based person detection, ByteTrack single-camera tracking
- **Phase 3**: OSNet re-identification embeddings, FAISS-based cross-camera matching
- **Phase 4**: Camera topology modeling, temporal reasoning, trajectory construction
- **Phase 5**: Person crop extraction and Re-ID data preparation
# Multi-Camera Person Re-Identification and Tracking

## Phase 4: local ByteTrack IDs

Phase 4 connects the existing YOLO person detector to Ultralytics ByteTrack.
It produces **local** track IDs: an ID is only meaningful within the one
camera/video being processed. It is not a cross-camera or global identity.

Run a CPU-friendly sample (the default configuration automatically selects CPU
when CUDA is unavailable):

```powershell
python -m src.tracking.tracker --input data/videos/cam01.mp4 --max-frames 20 --annotate
```

The command writes `outputs/tracking/<CAMERA>_tracks.csv` and, when requested,
`outputs/tracking_annotated/<CAMERA>_tracked.mp4`. No Re-ID, FAISS, or
cross-camera matching is included in this phase.

## Phase 5: person crop preparation

Phase 5 runs the existing YOLO and local ByteTrack pipeline, then saves sampled,
quality-filtered person crops for a future embedding model. Track IDs are local
to one input video/camera; they are not global identities. Crops are stored as
`outputs/crops/<CAMERA>/track_<ID>/frame_<NUMBER>.jpg`, with a `crops.csv`
metadata file per camera. OSNet/Re-ID, FAISS, and cross-camera matching are not
implemented here.

```powershell
python -m src.crops.crop_extractor --input data/videos/cam01.mp4 --max-frames 100
```

## Phase 6: OSNet appearance embeddings

Phase 6 runs pretrained OSNet **inference only** on Phase 5 crops. OSNet turns
each crop into a numerical appearance embedding; it does not decide whether two
people are the same and it does not train or fine-tune model weights. The
embedder selects CUDA automatically when available and otherwise runs on CPU.
It saves a per-local-track `embeddings.npy` plus an `embeddings.csv` mapping each
vector index to its camera, local track, frame, and crop path under
`outputs/embeddings/<CAMERA>/`. L2 normalization is enabled by default, so later
FAISS/similarity matching can use cosine-style comparisons consistently.

```powershell
python -m src.reid.embedder --input outputs/crops/CAM01 --max-images 5
```

The first real run may download the pretrained OSNet weights through
`torchreid`. Training/fine-tuning belongs to Phase 7; FAISS and cross-camera
matching belong to later phases.
