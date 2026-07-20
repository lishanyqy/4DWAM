#!/usr/bin/env bash
# Start the LIFT2 VA client on the robot (source ROS first).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NEEDS_ROS_PYTHON=1
for argument in "$@"; do
  if [[ "${argument}" == "--dry_run_mock" || "${argument}" == "--probe_server" ]]; then
    NEEDS_ROS_PYTHON=0
    break
  fi
done

# Keep the dedicated 4dwam Conda interpreter, but don't inherit Conda's global
# native-library search overrides in ROS mode. In particular, LD_LIBRARY_PATH
# may cause cv_bridge to load a Conda libffi beside the system libp11-kit and
# fail with an undefined LIBFFI_BASE_7.0 symbol. The interpreter remains the
# Conda interpreter; only the inherited loader/module overrides are reset.
if [[ "${NEEDS_ROS_PYTHON}" == "1" ]]; then
  echo "[client-launch] resetting inherited Python/native-library paths for ROS"
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD
  unset PYTHONPATH
  unset PYTHONHOME
  unset PKG_CONFIG_PATH
  unset CMAKE_PREFIX_PATH
fi

if [[ "${DEBUG_LAUNCH:-0}" == "1" ]]; then
  set -x
  echo "[client-launch] script_dir=${SCRIPT_DIR}"
  echo "[client-launch] shell=$(command -v bash)"
  echo "[client-launch] python=$(command -v python3)"
fi

ROS_SETUP="${ROS_SETUP:-/home/arx/Desktop/LIFT/R5/ROS/R5_ws/devel/setup.bash}"
if [[ -f "${ROS_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
else
  echo "Warning: ROS setup not found at ${ROS_SETUP}"
  echo "Set ROS_SETUP=/path/to/devel/setup.bash if needed."
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${CLIENT_PYTHON:-${ROS_PYTHON:-/home/arx/miniconda3/envs/4dwam/bin/python}}"

if [[ "${NEEDS_ROS_PYTHON}" == "1" ]]; then
  # Conda Python has a RUNPATH into its own lib directory, so unsetting
  # LD_LIBRARY_PATH alone may still let a Conda libffi satisfy dependencies of
  # the system libp11-kit loaded by cv_bridge. Preload the system libffi.so.7;
  # Conda may still load a different SONAME (for example libffi.so.8) for its
  # own packages without replacing the ABI required by libp11-kit.
  SYSTEM_LIBFFI="${SYSTEM_LIBFFI:-}"
  if [[ -z "${SYSTEM_LIBFFI}" ]]; then
    for candidate_path in \
      /usr/lib/x86_64-linux-gnu/libffi.so.7 \
      /lib/x86_64-linux-gnu/libffi.so.7; do
      if [[ -f "${candidate_path}" ]]; then
        SYSTEM_LIBFFI="${candidate_path}"
        break
      fi
    done
  fi
  if [[ -z "${SYSTEM_LIBFFI}" ]] && command -v ldconfig >/dev/null 2>&1; then
    SYSTEM_LIBFFI="$(ldconfig -p 2>/dev/null | awk '$1 == "libffi.so.7" {print $NF; exit}')"
  fi
  if [[ -z "${SYSTEM_LIBFFI}" || ! -f "${SYSTEM_LIBFFI}" ]]; then
    echo "Cannot locate the system libffi.so.7 required by ROS cv_bridge." >&2
    echo "Install libffi7 or set SYSTEM_LIBFFI=/path/to/libffi.so.7." >&2
    exit 1
  fi
  export LD_PRELOAD="${SYSTEM_LIBFFI}"
  echo "[client-launch] preloading system libffi=${SYSTEM_LIBFFI}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not executable: ${PYTHON_BIN}" >&2
  echo "Set CLIENT_PYTHON=/path/to/python if needed." >&2
  exit 1
fi

echo "[client-launch] python=${PYTHON_BIN} needs_ros=${NEEDS_ROS_PYTHON}"
if [[ "${NEEDS_ROS_PYTHON}" == "1" ]]; then
  ROS_IMPORT_ERROR=""
  if ! ROS_IMPORT_ERROR="$("${PYTHON_BIN}" -c 'import numpy as np; major_version = int(np.__version__.split(".", 1)[0]); assert major_version < 2, f"ROS Noetic cv_bridge requires NumPy <2, found {np.__version__}"; import rospy; from cv_bridge.boost.cv_bridge_boost import getCvType; from arm_control.msg import PosCmd' 2>&1)"; then
    echo "ROS Python native-module check failed with ${PYTHON_BIN}." >&2
    echo "${ROS_IMPORT_ERROR}" >&2
    echo "Check NumPy<2, cv_bridge, arm_control, and SYSTEM_LIBFFI." >&2
    echo "Expected interpreter: /home/arx/miniconda3/envs/4dwam/bin/python" >&2
    echo "Use DEBUG_LAUNCH=1 to inspect the cleaned environment." >&2
    exit 1
  fi
fi

exec "${PYTHON_BIN}" -u deploy/client_lift2_va.py --profile lift2_va_default "$@"
