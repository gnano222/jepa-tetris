FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Pre-install Python deps so startup only needs `pip install -e .`
COPY requirements.txt pyproject.toml ./
# Install everything except torch (already in base image)
RUN pip install --no-cache-dir \
        numpy "numpy>=1.26,<3.0" \
        matplotlib pillow tqdm pytest \
        runpod \
    && pip cache purge

# Source code is pulled at startup via git — not baked into image
CMD ["/start.sh"]
