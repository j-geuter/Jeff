# Jeff decision-model server (TypeSafe System One wire format, POST /v1/systemone).
# The CUDA runtime comes with the torch wheel; the host needs an NVIDIA driver for CUDA 13.0 (R580 or newer).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
# A C compiler: Triton (used by torch for some GPU ops, e.g. in Gemma's rotary embedding) compiles a small driver
# helper with it on the first request. Without it the server fails at its first forward pass.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install -r requirements.txt && pip install --no-deps .
EXPOSE 8013
# Default: serve a model folder mounted at /model (weights fetched beforehand; works offline).
# Or pass a Hugging Face repo id instead of /model (then the container needs network access).
ENTRYPOINT ["jeff-serve"]
CMD ["/model", "--host", "0.0.0.0", "--port", "8013"]
