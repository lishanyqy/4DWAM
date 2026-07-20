#!/usr/bin/env bash
# Optional helper: start CAN / body / arms / cameras on the robot before launch.sh.
# Paths mirror openpi-on-LIFT2 defaults; adjust for your machine.

set -euo pipefail

LIFT2_ROOT="${LIFT2_ROOT:-/home/arx/Desktop/openpi-on-LIFT2}"

if [[ ! -d "${LIFT2_ROOT}" ]]; then
  echo "LIFT2_ROOT does not exist: ${LIFT2_ROOT}"
  echo "Export LIFT2_ROOT to your robot software root, or start ROS stacks manually."
  exit 1
fi

cd "${LIFT2_ROOT}"

if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal --title="CAN1" -- bash -c "cd '${LIFT2_ROOT}/ARX_CAN/arx_can' && sudo ./arx_can1.sh; exec bash"
  sleep 0.1
  gnome-terminal --title="CAN3" -- bash -c "cd '${LIFT2_ROOT}/ARX_CAN/arx_can' && sudo ./arx_can3.sh; exec bash"
  sleep 0.1
  gnome-terminal --title="CAN5" -- bash -c "cd '${LIFT2_ROOT}/ARX_CAN/arx_can' && sudo ./arx_can5.sh; exec bash"
  sleep 1
  gnome-terminal --title="LIFT Body" -- bash -c "cd '${LIFT2_ROOT}/body' && source devel/setup.bash && roslaunch '${LIFT2_ROOT}/body/src/ARX_LIFT_ros/arx_lift_controller/launch/lift.launch'; exec bash"
  sleep 1
  gnome-terminal --title="R5 Arms" -- bash -c "cd '${LIFT2_ROOT}/R5_ws' && source devel/setup.bash && roslaunch '${LIFT2_ROOT}/R5_ws/src/arx_r5_ros/arx_r5_controller/launch/open_double_arm_xvla.launch'; exec bash"
  sleep 2
  if [[ -f "${LIFT2_ROOT}/realsense_camera/realsense.sh" ]]; then
    gnome-terminal --title="RealSense" -- bash -c "cd '${LIFT2_ROOT}/realsense_camera' && bash realsense.sh; exec bash"
  fi
  echo "All services launched in gnome-terminals."
else
  echo "gnome-terminal not available; start CAN/body/arms/cameras manually."
fi

echo "Then run:  cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) && ./launch.sh --task wrench"
