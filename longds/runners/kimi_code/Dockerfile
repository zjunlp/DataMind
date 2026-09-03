ARG BASE_IMAGE=executor-prebuilt
FROM ${BASE_IMAGE}

ARG KIMI_CODE_VERSION=latest

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

RUN npm install -g "@moonshot-ai/kimi-code@${KIMI_CODE_VERSION}" \
    && kimi --version

RUN mkdir -p /workspace && chmod 0777 /workspace

ENV HOME=/workspace/.home
WORKDIR /workspace

CMD ["kimi", "--version"]
