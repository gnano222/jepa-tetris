FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Pre-install Python deps so startup only needs `pip install -e .`
# torch is already in the base image — do not install it here
RUN pip install --no-cache-dir \
        "numpy>=1.26,<3.0" \
        "matplotlib>=3.8" \
        "pillow>=10.0" \
        "tqdm>=4.66" \
        "runpod>=1.7,<2.0" \
    && pip cache purge

# Source code is pulled at startup via git — not baked into image
CMD ["/start.sh"]
