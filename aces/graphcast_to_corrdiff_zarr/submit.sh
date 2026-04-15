#!/bin/bash -l
#SBATCH --job-name=gc2zarr
#SBATCH --partition=data-science
#SBATCH --qos=data-science
#SBATCH --output=log/graphcast_to_zarr_%j.log
#SBATCH --error=log/graphcast_to_zarr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
##SBATCH --time=08:00:00

module load python3/2024

# -----------------------
# Usage
# -----------------------
usage() {
  cat <<EOF
Usage: sbatch $0 [options]

Optional:
  --config FILE              YAML config (default: configs/default.yml)
  --graphcast-nc-dir DIR     Override: directory of GraphCast .nc files
  --base-zarr-path DIR       Override: base CorrDiff Zarr store
  --zarr-output-path DIR     Override: output Zarr path

Examples:
  sbatch $0
  sbatch $0 --config configs/default.yml
  sbatch $0 --graphcast-nc-dir /path/to/nc --zarr-output-path /path/out.zarr
EOF
}

CONFIG="configs/default.yml"
GC_DIR=""
BASE_ZARR=""
OUT_ZARR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)            CONFIG="$2"; shift 2;;
    --graphcast-nc-dir)  GC_DIR="$2"; shift 2;;
    --base-zarr-path)    BASE_ZARR="$2"; shift 2;;
    --zarr-output-path)  OUT_ZARR="$2"; shift 2;;
    -h|--help)           usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p log

# -----------------------
# Env (mirror Aurora pipeline)
# -----------------------
ENV_NAME=wrf-python
ENV_PATH=$HOME/envs/$ENV_NAME

export LD_PRELOAD=$ENV_PATH/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$ENV_PATH/lib:$LD_LIBRARY_PATH

# -----------------------
# Diagnostics
# -----------------------
echo "============================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Job Name      : $SLURM_JOB_NAME"
echo "Cores         : $SLURM_CPUS_PER_TASK"
echo "Submit Dir    : $SLURM_SUBMIT_DIR"
echo "Node(s)       : $SLURM_NODELIST"
echo "Config        : $CONFIG"
echo "============================================="

nvidia-smi || echo "No NVIDIA GPU (this is fine)."
crun.python3 -p "$ENV_PATH" python --version

cd "$SLURM_SUBMIT_DIR" || exit 1

# -----------------------
# Build command
# -----------------------
CMD="python pipeline.py --config $CONFIG"
[[ -n "$GC_DIR"    ]] && CMD="$CMD --graphcast_nc_dir $GC_DIR"
[[ -n "$BASE_ZARR" ]] && CMD="$CMD --base_zarr_path $BASE_ZARR"
[[ -n "$OUT_ZARR"  ]] && CMD="$CMD --zarr_output_path $OUT_ZARR"

echo "Starting at: $(date)"
echo "Running: crun.python3 -p $ENV_PATH $CMD"
crun.python3 -p "$ENV_PATH" $CMD
echo "Finished at: $(date)"
