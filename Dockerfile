# ==========================================
# STAGE 1: BUILDER (Heavy lifting, highly cached)
# ==========================================
FROM python:3.12-slim AS builder

# 1. Install system tools and dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata \
    bash \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up a global virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Upgrade foundational pip tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools build virtualenv

# 4. Install stable dependencies (Crucial: Copy ONLY requirements)
WORKDIR /build
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip install secretstorage jeepney cryptography


# ==========================================
# STAGE 2: RUNNER (Fast execution, changes every build)
# ==========================================
FROM python:3.12-slim AS runner

# 1. Install runtime-only OS dependencies (like FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Re-create your production user account
ARG GROUP_NAME=docker
ARG USER_NAME=docker
ARG USER_UID=1000
ARG GROUP_GID=1000
RUN groupadd --gid ${GROUP_GID} ${GROUP_NAME} \
    && useradd --uid ${USER_UID} --gid ${GROUP_GID} --create-home --home-dir /home/${USER_NAME} ${USER_NAME}

# 3. COPY the pre-built virtual environment from Stage 1
COPY --from=builder --chown=docker:docker /opt/venv /opt/venv

# 4. Set environment paths to point to the copied venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV YOLO_CONFIG_DIR="/home/docker/.config/Ultralytics"
ENV TORCHINDUCTOR_CACHE_DIR="/home/docker/.cache/torchinductor"
ENV TORCH_HOME="/home/docker/.cache/torch"
ARG TZ="America/New_York"
ENV TZ=${TZ}

# 5. Copy and install your rapidly changing wheel file
WORKDIR /home/docker
COPY dist/*.whl ./dist/
RUN pip install dist/*.whl && \
    rm -rf dist

# 6. Copy frontend and model assets with correct permissions
COPY --chown=docker:docker pynvr/model/*.pt ./pynvr/model/
COPY --chown=docker:docker pynvr/frontend_dist ./pynvr/frontend_dist

# 7. Give ownership of the virtual environment to your runtime user

USER docker
EXPOSE 7860

ENTRYPOINT [ "pynvr" ]

