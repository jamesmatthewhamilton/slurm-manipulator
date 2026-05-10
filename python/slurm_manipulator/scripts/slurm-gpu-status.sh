#!/bin/bash
# Print GPU partition status on a Slurm cluster.
#
# Usage:
#   bash slurm-gpu-status.sh <user@host>

set -eu

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <user@host>" >&2
    exit 2
fi
SSH_TARGET="$1"

echo "Connecting to ${SSH_TARGET}..."
echo ""

ssh -o ConnectTimeout=10 "$SSH_TARGET" bash -s <<'REMOTE'
echo "=========================================="
echo "  Slurm GPU Status — $(date)"
echo "=========================================="
echo ""

declare -A VRAM_MAP
VRAM_MAP[gpu-rtxpro-blackwell]="96GB"
VRAM_MAP[gpu-h200]="141GB"
VRAM_MAP[gpu-h100]="80GB"
VRAM_MAP[gpu-a100]="80GB"
VRAM_MAP[gpu-l40s]="48GB"
VRAM_MAP[gpu-rtx6000]="48GB"
VRAM_MAP[gpu-v100]="32GB"

echo "── GPU Partitions ──"
echo ""
gpu_parts=$(sinfo -h -o "%P" 2>/dev/null | grep -i gpu | tr -d '*' | sort -u)
if [ -z "$gpu_parts" ]; then
  echo "  (no GPU partitions found)"
else
  (echo "PARTITION AVAIL NODES STATE CPUS MEMORY GRES"
   sinfo -p "$(echo $gpu_parts | tr ' ' ',')" -h -o "%P %a %D %T %c %m %G" 2>/dev/null
  ) | column -t
fi
echo ""

echo "── GPU Availability ──"
echo ""
count_gpus() {
  local part=$1 state_flag=$2
  sinfo -p "$part" -h $state_flag -N -o "%G" 2>/dev/null \
    | sed 's/(.*)//' \
    | awk -F: '/^gpu/{s+=$NF}END{print s+0}'
}

printf "  %-24s  %6s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s\n" \
  "Partition" "VRAM" "Idle Nodes" "Idle GPUs" "Mixed Nodes" "Mixed GPUs" "Alloc Nodes" "Alloc GPUs" "Drain Nodes" "Drain GPUs" "Total Nodes" "Total GPUs"
printf "  %-24s  %6s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s\n" \
  "---------" "----" "----------" "---------" "-----------" "----------" "-----------" "----------" "-----------" "----------" "-----------" "----------"

for part in $gpu_parts; do
  total_n=$(sinfo -p "$part" -h -o "%D" 2>/dev/null | awk '{s+=$1}END{print s+0}')
  idle_n=$(sinfo -p "$part" -h -t idle -o "%D" 2>/dev/null | awk '{s+=$1}END{print s+0}')
  mix_n=$(sinfo -p "$part" -h -t mix -o "%D" 2>/dev/null | awk '{s+=$1}END{print s+0}')
  alloc_n=$(sinfo -p "$part" -h -t alloc -o "%D" 2>/dev/null | awk '{s+=$1}END{print s+0}')
  drain_n=$(sinfo -p "$part" -h -t drain -o "%D" 2>/dev/null | awk '{s+=$1}END{print s+0}')

  total_g=$(count_gpus "$part" "")
  idle_g=$(count_gpus "$part" "-t idle")
  mix_g=$(count_gpus "$part" "-t mix")
  alloc_g=$(count_gpus "$part" "-t alloc")
  drain_g=$(count_gpus "$part" "-t drain")

  vram=${VRAM_MAP[$part]:-"?"}

  printf "  %-24s  %6s  %10s  %9s  %11s  %10s  %11s  %10s  %11s  %10s  %11s  %10s\n" \
    "$part" "$vram" \
    "$idle_n" "$idle_g" \
    "$mix_n" "$mix_g" \
    "$alloc_n" "$alloc_g" \
    "$drain_n" "$drain_g" \
    "$total_n" "$total_g"
done
echo ""

echo "── Your Running/Pending GPU Jobs ──"
echo ""
jobs=$(squeue -u "$USER" -h -o "%i %j %T %P %D %b %N %M" 2>/dev/null)
if [ -n "$jobs" ]; then
  (echo "JOBID NAME STATE PARTITION NODES GPUS NODELIST RUNTIME"
   echo "$jobs"
  ) | column -t
else
  echo "  (no GPU jobs)"
fi
echo ""
REMOTE
