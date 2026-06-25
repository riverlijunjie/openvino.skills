#!/usr/bin/env bash
#
# install_unitrace.sh — ensure Intel's `unitrace` profiler is available on PATH.
#
# The KernelFoundry eval pipeline profiles SYCL/OpenCL/Triton-on-Intel kernels by
# wrapping the run command with `unitrace ... --group {ComputeBasic,MemoryProfile,
# VectorEngineProfile}` (see eval_pipeline/profiler_command.py). The binary is
# resolved from $KERNELFOUNDRY_unitrace_cmd (default: "unitrace") and is assumed to
# already be on PATH — nothing builds it on demand. This script closes that gap:
# it is idempotent and safe to run before any Intel-GPU optimization run.
#
# Behaviour:
#   1. If `unitrace` is already callable (PATH or $KERNELFOUNDRY_unitrace_cmd) -> done.
#   2. Else if a binary was already built in the pti-gpu clone -> just symlink it.
#   3. Else clone https://github.com/intel/pti-gpu.git, cmake+make, then symlink.
# The resulting binary is symlinked into $INSTALL_BIN (default ~/.local/bin), which
# you should ensure is on PATH.
#
# Usage:
#   bash install_unitrace.sh                 # auto: detect / build / symlink
#   PTI_GPU_DIR=/path/to/pti-gpu bash install_unitrace.sh   # reuse/clone here
#   INSTALL_BIN=/some/bin bash install_unitrace.sh          # symlink target
#
# Env overrides:
#   PTI_GPU_DIR   where pti-gpu lives / will be cloned (default: <repo root>/pti-gpu)
#   PTI_GPU_REPO  git URL (default: https://github.com/intel/pti-gpu.git)
#   INSTALL_BIN   dir to symlink the binary into (default: $HOME/.local/bin)
#   JOBS          parallel make jobs (default: nproc)

set -euo pipefail

PTI_GPU_REPO="${PTI_GPU_REPO:-https://github.com/intel/pti-gpu.git}"
INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

log() { printf '[install_unitrace] %s\n' "$*" >&2; }

# 1. Already available?
existing_cmd="${KERNELFOUNDRY_unitrace_cmd:-unitrace}"
if command -v "$existing_cmd" >/dev/null 2>&1; then
    log "unitrace already on PATH: $(command -v "$existing_cmd")"
    exit 0
fi

# Resolve a default location for the pti-gpu clone: next to this skill's repo root.
# This script lives at <repo>/skills/dev_kf_distill_master_opt/scripts/, so climb up.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd "$script_dir/../../.." && pwd)"
PTI_GPU_DIR="${PTI_GPU_DIR:-$default_root/pti-gpu}"

unitrace_src="$PTI_GPU_DIR/tools/unitrace"
build_dir="$unitrace_src/build"
built_bin="$build_dir/unitrace"

# 2. Already built in the clone? Just (re)symlink it.
if [[ -x "$built_bin" ]]; then
    log "found existing build at $built_bin"
else
    # 3a. Clone if needed.
    if [[ ! -d "$unitrace_src" ]]; then
        log "cloning $PTI_GPU_REPO -> $PTI_GPU_DIR"
        git clone --depth 1 "$PTI_GPU_REPO" "$PTI_GPU_DIR"
    else
        log "reusing existing pti-gpu checkout at $PTI_GPU_DIR"
    fi

    # 3b. Build.
    log "building unitrace (cmake + make -j$JOBS) in $build_dir"
    mkdir -p "$build_dir"
    (
        cd "$build_dir"
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF ..
        make "-j$JOBS"
    )
fi

if [[ ! -x "$built_bin" ]]; then
    log "ERROR: build finished but $built_bin is missing/not executable" >&2
    exit 1
fi

# Symlink into INSTALL_BIN.
mkdir -p "$INSTALL_BIN"
ln -sf "$built_bin" "$INSTALL_BIN/unitrace"
log "symlinked $INSTALL_BIN/unitrace -> $built_bin"

# Allow non-root GPU profiling on Intel i915 if possible (best-effort, non-fatal).
if [[ -f /proc/sys/dev/i915/perf_stream_paranoid ]]; then
    sudo sh -c 'echo 0 > /proc/sys/dev/i915/perf_stream_paranoid' 2>/dev/null \
        && log "set i915 perf_stream_paranoid=0 (non-root profiling enabled)" \
        || log "note: could not set i915 perf_stream_paranoid (need root for non-root profiling)"
fi

if ! command -v unitrace >/dev/null 2>&1; then
    log "NOTE: $INSTALL_BIN is not on PATH. Add it, e.g.:"
    log "      export PATH=\"$INSTALL_BIN:\$PATH\""
    log "      (or export KERNELFOUNDRY_unitrace_cmd=$INSTALL_BIN/unitrace)"
fi

log "done."
