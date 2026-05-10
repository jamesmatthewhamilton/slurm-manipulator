#!/bin/bash
# Cancel all Slurm jobs for the current user on a Slurm cluster.
#
# Usage:
#   bash slurm-kill-all.sh <user@host>
#
# Example:
#   bash slurm-kill-all.sh alice@login.example-hpc.edu

set -eu

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <user@host>" >&2
    exit 2
fi
SSH_TARGET="$1"

echo "Connecting to ${SSH_TARGET}..."
echo ""

ssh -o ConnectTimeout=10 "$SSH_TARGET" bash -s <<'REMOTE'
echo "── Current jobs ──"
jobs=$(squeue -u "$USER" -h -o "%i %j %T %P %M" 2>/dev/null)
if [ -z "$jobs" ]; then
  echo "  No jobs running."
  exit 0
fi

(echo "JOBID NAME STATE PARTITION RUNTIME"
 echo "$jobs"
) | column -t
echo ""

echo "Cancelling all jobs..."
scancel -u "$USER"
echo "Done."
REMOTE
