FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
    PYTHONPATH=/opt/MuseTalk:/opt/MuseTalk/musetalk/utils
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg libgl1 libglib2.0-0 build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/MuseTalk
RUN git init && git remote add origin https://github.com/TMElyralab/MuseTalk.git && git fetch --depth 1 origin 0a89dec45a0192b824e3cf4daf96c239440c5ed8 && git checkout --detach FETCH_HEAD
RUN python3 -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0 && python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --only-binary=mmcv mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
COPY requirements-worker.txt /opt/worker/requirements-worker.txt
RUN python3 -m pip install -r /opt/worker/requirements-worker.txt && python3 -m pip check
COPY download_models.py model_revisions.json patch_upstream.py /opt/worker/
RUN python3 /opt/worker/patch_upstream.py && python3 /opt/worker/download_models.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN python3 -c "import scripts.inference; print('CPU import and model dependency check passed')"
COPY handler.py engine.py /opt/worker/
COPY test_input.json /opt/worker/test_input.json
WORKDIR /opt/worker
CMD ["python3", "-u", "handler.py"]
