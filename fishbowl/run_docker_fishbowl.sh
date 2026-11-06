#!/usr/bin/env bash
# SOMA-Fishbowl Docker + Shared FrameBuffer & DevMem Channel Bridge
# Mounts /dev/fb0_rgba.raw, /dev/mem_channels/devmem_arena.bin, /bin, /src, and Host Rust Toolchain into Docker!
set -e
CONTAINER_NAME="soma-fishbowl-os"
IMAGE="${SOMA_DOCKER_IMAGE:-ubuntu:24.04}"

echo "🐳 [SOMA-Docker-FB] Launching container $CONTAINER_NAME with shared Framebuffer & 64KB DevMem Bus..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" \
  --hostname soma-kernel \
  -v "/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/fishbowl/rootfs/dev:/dev/soma_bus" \
  -v "/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/fishbowl/rootfs/bin:/bin/soma" \
  -v "/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/fishbowl/rootfs/src:/src" \
  -v "/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/fishbowl/rootfs/etc:/etc/soma" \
  -v "/home/omegacd/.cargo:/opt/rust/cargo:ro" \
  -v "/home/omegacd/.rustup:/opt/rust/rustup:ro" \
  -e FISHBOWL_FB_PATH="/dev/soma_bus/fb0_rgba.raw" \
  -e SOMA_DEVMEM_PATH="/dev/soma_bus/mem_channels/devmem_arena.bin" \
  -e SOMA_UI_CFG_PATH="/etc/soma/soma_ui_config.cfg" \
  -e PATH="/bin/soma:/opt/rust/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e RUSTUP_HOME="/opt/rust/rustup" \
  "$IMAGE" tail -f /dev/null

echo "✅ Container $CONTAINER_NAME is UP and bound to Fishbowl Framebuffer: /home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/fishbowl/rootfs/dev/fb0_rgba.raw"
echo "   Run inside Docker OS: docker exec -it $CONTAINER_NAME bash"
