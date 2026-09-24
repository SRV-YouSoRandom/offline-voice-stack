FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git .

RUN cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=ON \
    && cmake --build build --config Release --target llama-server -j "$(nproc)"

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/build/bin /app/bin

ENV LD_LIBRARY_PATH=/app/bin

EXPOSE 8002

ENTRYPOINT ["/app/bin/llama-server"]