#!/usr/bin/env bash
# Build & run the custom OpenCL hw probes on the BMG remote (per SKILL §4).
set -e
HOST=openvino-ci-74@10.239.140.155
PASS=openvino
DST=/tmp/hw_probe
LOCAL=$(dirname "$(readlink -f "$0")")

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$HOST" "mkdir -p $DST"
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no "$LOCAL/gpu_info.c" "$LOCAL/mem_bw.c" "$HOST:$DST/"
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$HOST" "cd $DST && \
  gcc gpu_info.c -lOpenCL -O2 -o gpu_info && \
  gcc mem_bw.c   -lOpenCL -O2 -o mem_bw && \
  echo === gpu_info === && ./gpu_info && \
  echo === mem_bw  1 GiB === && ./mem_bw 1024 20 && \
  echo === mem_bw  2 GiB === && ./mem_bw 2048 10"
