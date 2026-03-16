#!/usr/bin/env bash
set -euo pipefail

PARTITION="data-science"
QOS="data-science"
CPUS="1"

ENV_PREFIX=""   # required

J_PORT="8890"       # preferred compute-local port (auto-adjust if busy)
R_PORT="18890"      # preferred login-node reverse port (auto-increment if busy)
L_PORT="8891"       # suggested Mac-local port (cannot auto-check from Wahab)

NOTEBOOK_DIR="${HOME}"  # default root shown in JupyterLab; user can override

LOGIN_INTERNAL="$(hostname -s)"
LOGIN_EXTERNAL="wahab.hpc.odu.edu"

SSH_USER="${USER}"

usage() {
  cat <<EOF
Usage: $0 --env-prefix PATH [options]

Required:
  --env-prefix PATH        Conda prefix for: crun.python3 -p PATH

Options:
  --notebook-dir PATH          JupyterLab root directory [${NOTEBOOK_DIR}]
  --j-port PORT            Preferred Jupyter port on compute node (auto-pick if busy) [${J_PORT}]
  --r-port PORT            Preferred reverse port on login node (auto-increment if busy) [${R_PORT}]
  --l-port PORT            Suggested local port on Mac (printed) [${L_PORT}]
  --login-internal HOST    Internal login host (default: $(hostname -s))
  --partition NAME         Slurm partition [${PARTITION}]
  --qos NAME               Slurm qos [${QOS}]
  -c, --cpus N             CPUs for allocation [${CPUS}]

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-prefix) ENV_PREFIX="$2"; shift 2;;
    --notebook-dir) NOTEBOOK_DIR="$2"; shift 2;;
    --j-port) J_PORT="$2"; shift 2;;
    --r-port) R_PORT="$2"; shift 2;;
    --l-port) L_PORT="$2"; shift 2;;
    --login-internal) LOGIN_INTERNAL="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --qos) QOS="$2"; shift 2;;
    -c|--cpus) CPUS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[error] unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "${ENV_PREFIX}" ]]; then echo "[error] --env-prefix is required"; usage; exit 2; fi

CRUN=(crun.python3 -p "${ENV_PREFIX}")

echo "[info] login node (internal): ${LOGIN_INTERNAL}"
echo "[info] requesting allocation: -p ${PARTITION} --qos=${QOS} -c ${CPUS}"
echo "[info] jupyter: preferred compute 127.0.0.1:${J_PORT}"
echo "[info] reverse tunnel: preferred ${LOGIN_INTERNAL}:127.0.0.1:${R_PORT} -> compute:127.0.0.1:${J_PORT}"
echo "[info] notebook dir: ${NOTEBOOK_DIR}"
echo

salloc -p "${PARTITION}" --qos="${QOS}" -c "${CPUS}" \
  srun --ntasks=1 --cpus-per-task="${CPUS}" --pty bash -lc "
    set -euo pipefail
    module load python3/2024
    echo '[info] compute host:' \$(hostname)

    port_listening() {
      python3 - <<PY
import socket, sys
p=int(\"\$1\")
s=socket.socket(); s.settimeout(0.2)
rc=s.connect_ex(('127.0.0.1', p))
s.close()
sys.exit(0 if rc==0 else 1)
PY
    }

    pick_free_port() {
      python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
    }

    J_PORT_RUNTIME='${J_PORT}'
    if port_listening \"\${J_PORT_RUNTIME}\"; then
      echo \"[warn] compute port \${J_PORT_RUNTIME} already in use; picking a free port\"
      J_PORT_RUNTIME=\"\$(pick_free_port)\"
      echo \"[info] new compute port: \${J_PORT_RUNTIME}\"
    fi

    J_LOG=\"jupyter_\${J_PORT_RUNTIME}.log\"
    echo \"[step] starting jupyter lab on 127.0.0.1:\${J_PORT_RUNTIME} (log: \${J_LOG})\"
    ${CRUN[*]} jupyter lab --no-browser --ip=127.0.0.1 --port \"\${J_PORT_RUNTIME}\" --notebook-dir='${NOTEBOOK_DIR}' >\"\${J_LOG}\" 2>&1 &
    J_PID=\$!

    # Wait briefly for an HTTP URL to appear (generic; avoids token parsing)
    for i in {1..60}; do
      if grep -Eq 'https?://[^[:space:]]+' \"\${J_LOG}\"; then break; fi
      sleep 1
    done

    JUPYTER_URL=\"\$(grep -Eo 'https?://[^[:space:]]+' \"\${J_LOG}\" | tail -n 1 || true)\"

    echo
    echo '============================================================'
    if [[ -n \"\${JUPYTER_URL}\" ]]; then
      echo \"[JUPYTER URL] \${JUPYTER_URL}\"
      echo \"[note] The URL above includes the token (if Jupyter requires it).\"
      echo \"[LOCAL URL]  http://localhost:${L_PORT}/lab\"
    else
      echo \"[warn] URL not found yet. Check: tail -n 80 \${J_LOG}\"
    fi
    echo '============================================================'
    echo

    # Establish reverse tunnel (login port may be in use -> auto-increment)
    R_PORT_TRY='${R_PORT}'
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      echo
      echo \"[step] reverse tunnel attempt \${attempt}: ${LOGIN_INTERNAL}:127.0.0.1:\${R_PORT_TRY} -> compute:127.0.0.1:\${J_PORT_RUNTIME}\"

      echo '============================================================'
      echo '[LOCAL MAC] Run this in a NEW terminal on your Mac:'
      echo
      echo \"ssh -N -L ${L_PORT}:127.0.0.1:\${R_PORT_TRY} ${SSH_USER}@${LOGIN_EXTERNAL}\"
      echo
      echo '[LOCAL MAC] Then open in browser:'
      echo \"http://localhost:${L_PORT}/lab\"
      echo \"[LOCAL MAC] If you see Address already in use, change ${L_PORT} to ${L_PORT}+1 (e.g., $(( ${L_PORT} + 1 )))\"
      echo '============================================================'
      echo

      if ssh -o ExitOnForwardFailure=yes -N -R \"\${R_PORT_TRY}:127.0.0.1:\${J_PORT_RUNTIME}\" ${SSH_USER}@${LOGIN_INTERNAL}; then
        exit 0
      fi

      echo \"[warn] reverse tunnel failed (likely ${LOGIN_INTERNAL}:\${R_PORT_TRY} in use). Trying next port.\"
      R_PORT_TRY=\$((R_PORT_TRY + 1))
    done

    echo '[error] could not establish reverse tunnel after multiple attempts'
    echo '[info] stopping jupyter'
    kill \$J_PID 2>/dev/null || true
    exit 1
  "