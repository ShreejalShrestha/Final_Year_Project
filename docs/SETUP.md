# Setup notes

## Python version

Developed against **Python 3.11 / 3.12** for the CV stack. Python 3.13 works for
the Django app, but `insightface` and `mediapipe` wheels can lag on the newest
Python — if `pip install -r requirements.txt` fails on 3.13, create the
environment with 3.12:

```bash
py -3.12 -m venv .venv
```

## Two install tiers

The `requirements.txt` file has a core (web) tier and a CV tier. To work on the
Django side only:

```bash
pip install "Django>=5.0,<5.2" "psycopg[binary]" Pillow numpy scipy python-dotenv
```

Add the CV tier when you need the live pipeline:

```bash
pip install opencv-python onnxruntime insightface mediapipe
```

### NVIDIA GPU (recommended — CPU tops out around 4–8 FPS)

`onnxruntime-gpu` alone is not enough on Windows; it needs matching CUDA/cuDNN
runtime DLLs, and the newest `onnxruntime-gpu` (1.29) wants CUDA 13. The
verified route (RTX 4060, Win 11, Python 3.13) uses the CUDA 12 build and the
`nvidia-*-cu12` wheels — no system CUDA toolkit:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu   # never keep both, and drop the CPU one
pip install -r requirements-gpu.txt
```

`cv_pipeline/detection.py` calls `onnxruntime.preload_dlls()` **and** prepends
the wheel `bin/` dirs to `PATH` (cuDNN 9 loads its sub-libraries itself, so
`add_dll_directory` alone is not enough). Then in `.env`:

```
CV_DEVICE=cuda
CV_MODEL_NAME=buffalo_l      # full-accuracy model; fine on GPU
CV_PROCESS_EVERY_N=1
```

Verify the CUDA provider actually initialises (no silent CPU fallback):

```bash
python -c "import cv_pipeline.detection as d; d._preload_cuda_dlls(); import onnxruntime as ort; import numpy as np; print(ort.get_available_providers())"
```

The dashboard log prints `InsightFace ready (buffalo_l, ..., providers=['CUDAExecutionProvider', ...])`
on a successful GPU start.

### Windows / `insightface` build errors

`insightface` compiles a small Cython extension. If it fails:

1. Install "Microsoft C++ Build Tools" (Desktop C++ workload), **or**
2. `pip install insightface --only-binary :all:` to force a prebuilt wheel, **or**
3. `conda install -c conda-forge insightface onnxruntime`.

## PostgreSQL (default database)

Create the database once (in `psql`, or pgAdmin's Query Tool):

```sql
CREATE DATABASE smart_classroom ENCODING 'UTF8';
```

Then set the `DB_*` values in `.env` (`DB_ENGINE=postgresql`, `DB_USER`,
`DB_PASSWORD`, `DB_HOST`, `DB_PORT=5432`) and run `python manage.py migrate`.
The driver is `psycopg[binary]` (in `requirements.txt`).

`DB_ENGINE=sqlite` still works for a quick throw-away run, and `DB_ENGINE=mysql`
(with `pip install mysqlclient`) is kept as an alternative.

### Moving existing SQLite data across

```bash
# 1. with DB_ENGINE=sqlite in .env
python manage.py dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.permission --indent 2 -o data.json
# 2. switch DB_ENGINE=postgresql, create the DB, then:
python manage.py migrate
python manage.py loaddata data.json
```

## Model weights

* **InsightFace `buffalo_l`** (~330 MB) downloads automatically to
  `~/.insightface/models/` on first recognition.
* **MediaPipe `face_landmarker.task`** (~3 MB) downloads to
  `cv_pipeline/models/` on first landmark analysis.

Both are cached; the first pipeline start is slow.

## Running the live feed

The dashboard's camera loop runs in a background thread of the Django process,
so use the built-in server for development:

```bash
python manage.py runserver
```

For a shared demo, run with `--nothreading` disabled (default is threaded) and a
single worker. `gunicorn`/`uwsgi` multi-worker deployments would run one camera
loop per worker — out of scope for the prototype (one classroom, one camera).


## Phone-photo / replay rejection

Monitoring now requires MiniFASNetV2 anti-spoof classification **and** a complete
open/closed/open blink before a track is counted or attendance is marked.
Landmark movement alone never establishes liveness. Keep the face in view and
blink naturally; another blink is required every 15 seconds. Faces smaller than
60 pixels on either side are withheld because the classifier needs useful detail.
Landmarks run every processed frame by default to capture short blinks.

The approximately 1.7 MB ONNX file downloads automatically on first monitoring
startup to `CV_MODEL_DIR/MiniFASNetV2.onnx`. For offline installation, download
[the upstream ONNX release](https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx)
and put it there. The model uses the existing ONNX Runtime dependency on CPU;
no additional packages are required. Download/load errors stop startup with an
error; inference errors reject the face rather than bypassing the gate.

Implementation and weights originate from
[yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing), based on
[MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).
The input is an 80x80 BGR float32 crop with 2.7x context, without RGB conversion
or division by 255; real is output class 1. Defaults are in `PipelineConfig`.
Three consecutive real scores >= 0.90 are required. A failed score revokes
acceptance immediately; evidence older than 0.75 seconds cannot authorize a face.
Already saved attendance is not automatically deleted when a track is revoked.

Validate on consented clips from the deployment camera: real faces, stationary
and tilted phone photos, prints, and replay videos, under different lighting,
screen brightness and distances. Record spoof acceptance and real-person rejection
before adjusting thresholds. Unit tests and successful model inference do not
establish accuracy on your camera. RGB-only classification can still be fooled;
this change does not guarantee rejection of every screen or replay.
