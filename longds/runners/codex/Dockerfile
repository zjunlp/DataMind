ARG BASE_IMAGE=executor-prebuilt
FROM ${BASE_IMAGE}

ARG CODEX_VERSION=latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm --version

RUN npm install -g "@openai/codex@${CODEX_VERSION}" \
    && codex --version

RUN mkdir -p /workspace /codex-home \
    && chmod 0777 /workspace /codex-home

ENV HOME=/workspace/.home
WORKDIR /workspace

CMD ["codex", "--version"]
