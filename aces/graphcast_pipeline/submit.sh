#!/bin/bash -l
#SBATCH --job-name=graphcast_pipeline
#SBATCH --partition=data-science
#SBATCH --qos=data-science
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load python3/2024

ENV_NAME=env_graphcast
ENV_PATH="$HOME/envs/$ENV_NAME"

# -----------------------
# Usage
# -----------------------
usage() {
  cat <<EOF
Usage: sbatch $0 [options]

Required:
  --config FILE              Period config YAML (resolved under ./configs/ if not absolute)

Optional:
  --output-dir DIR           (default: from graphcast.yml)
  --tag STR                  (default: slurm_run_<JOB_ID>)

Examples:
  sbatch $0 --config periods_job1.yml
  sbatch $0 --config periods_job1.yml --output-dir /raid/apaudel/GRAPHCAST_OUTPUTS/test
  sbatch $0 --config periods_job2.yml --tag jan_run
EOF
}

# -----------------------
# Defaults
# -----------------------
CONFIG=""
OUTPUT_DIR=""
TAG=""

# -----------------------
# Parse args
# -----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# -----------------------
# Validate
# -----------------------
if [[ -z "$CONFIG" ]]; then
  echo "ERROR: --config is required"
  usage
  exit 1
fi

if [[ -z "$TAG" ]]; then
  TAG="slurm_run_${SLURM_JOB_ID}"
fi

mkdir -p logs

# -----------------------
# Diagnostics
# -----------------------
echo "============================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Job Name      : $SLURM_JOB_NAME"
echo "Cores         : $SLURM_CPUS_PER_TASK"
echo "GPU Allocated : $SLURM_GPUS_ON_NODE"
echo "Submit Dir    : $SLURM_SUBMIT_DIR"
echo "Node(s)       : $SLURM_NODELIST"
echo "Config        : $CONFIG"
echo "Output Dir    : ${OUTPUT_DIR:-<from config>}"
echo "Tag           : $TAG"
echo "============================================="

nvidia-smi

cd "$SLURM_SUBMIT_DIR" || exit 1

# -----------------------
# Build command
# -----------------------
CMD="python main.py --config $CONFIG --tag $TAG"
if [[ -n "$OUTPUT_DIR" ]]; then
  CMD="$CMD --output-dir $OUTPUT_DIR"
fi

echo "Running: crun.python3 -p $ENV_PATH $CMD"

crun.python3 -p "$ENV_PATH" $CMD

echo "Finished at: $(date)"
