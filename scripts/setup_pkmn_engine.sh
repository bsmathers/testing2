#!/usr/bin/env bash
set -e

# Setup script for @pkmn/engine native Zig build (Linux/macOS)
# Builds pkmn-showdown.node with -Dshowdown -Dlog for 50x faster battle simulation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PKMN_DIR="${ROOT_DIR}/pkmn-engine"

echo "=== Metamon @pkmn/engine Setup ==="

# 1. Check or install Zig
if command -v zig &> /dev/null; then
    echo "Found Zig: $(zig version)"
else
    echo "Zig not found in PATH. Downloading standalone Zig for Linux x86_64..."
    ZIG_VERSION="0.14.0"
    ZIG_TAR="zig-linux-x86_64-${ZIG_VERSION}.tar.xz"
    ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_TAR}"
    
    mkdir -p "${ROOT_DIR}/.zig"
    curl -L "${ZIG_URL}" -o "/tmp/${ZIG_TAR}"
    tar -xf "/tmp/${ZIG_TAR}" -C "${ROOT_DIR}/.zig" --strip-components=1
    rm "/tmp/${ZIG_TAR}"
    export PATH="${ROOT_DIR}/.zig:${PATH}"
    echo "Installed Zig: $(zig version)"
fi

# 2. Clone pkmn/engine if needed
if [ ! -d "${PKMN_DIR}" ]; then
    echo "Cloning pkmn/engine..."
    git clone --depth 1 https://github.com/pkmn/engine.git "${PKMN_DIR}"
fi

cd "${PKMN_DIR}"

# 3. Build native addon with -Dshowdown -Dlog
echo "Building @pkmn/engine native addon (-Dshowdown -Dlog)..."
zig build -Dshowdown -Dlog

# 4. Install npm dependencies in vectorized env
echo "Setting up Node.js packages..."
cd "${ROOT_DIR}/metamon/env/vectorized"
npm install --save-dev @pkmn/data @pkmn/protocol || true

# 5. Link built addon into vectorized env
if [ -f "${PKMN_DIR}/zig-out/lib/pkmn-showdown.node" ]; then
    cp "${PKMN_DIR}/zig-out/lib/pkmn-showdown.node" "${ROOT_DIR}/metamon/env/vectorized/"
    echo "Successfully built and copied pkmn-showdown.node to metamon/env/vectorized/"
elif [ -f "${PKMN_DIR}/build/lib/pkmn-showdown.node" ]; then
    cp "${PKMN_DIR}/build/lib/pkmn-showdown.node" "${ROOT_DIR}/metamon/env/vectorized/"
    echo "Successfully built and copied pkmn-showdown.node to metamon/env/vectorized/"
fi

echo "=== @pkmn/engine setup complete! ==="
