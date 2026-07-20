# real_robot: LingBot-VA / 4DWAM on LIFT2

Remote inference package:

- **GPU server**: algorithm (`wan_va_server`), VA protocol
- **Robot client**: ROS + absolute EEF execution (openpi-on-LIFT2 I/O)

```text
evaluation/real_robot/
  README.md
  PROTOCOL.md
  server/          # GPU only
  client/          # copy this folder to the robot
  tests/           # no ROS required
```

## Defaults

| Item | Value |
|------|-------|
| Port | **7777** |
| Publish rate | **60 Hz** |
| Client profile file | **`client/launch_profiles.yaml`** |
| Bridge | `segment_relative_va` (relative → absolute re-anchor) |
| Camera | `tshape` (256×320 high, 128×160 wrists) |

See [PROTOCOL.md](PROTOCOL.md) for full I/O.

---

## 1. GPU server

### Prerequisites

- Python env: repository `uv` project under `lingbot-va` (see repo `AGENTS.md`)
- Checkpoint with `transformer/`; set `attn_mode` to `torch` or `flashattn` in
  `<checkpoint>/transformer/config.json` (not `flex`)
- Base model root for VAE + text encoder (default HF cache path in infer cfg)
- Matching `action_norm_stats.json` for the checkpoint

### Launch

```bash
cd /soft/wangxi/4DWAM/lingbot-va

# LingBot-VA LIFT2 (default)
bash evaluation/real_robot/server/launch_server.sh

# 4DWAM LIFT2 weights
CONFIG_NAME=4dwam_lift2_infer bash evaluation/real_robot/server/launch_server.sh

# Override paths
LIFT2_VA_CHECKPOINT=/path/to/checkpoint_step_XXXX \
LINGBOT_VA_BASE_MODEL=/path/to/lingbot-va-base \
PORT=7777 \
bash evaluation/real_robot/server/launch_server.sh
```

Registered config names:

- `lift2_merged_infer`
- `4dwam_lift2_infer`

### Server dry-run (from any machine with network access)

With the server running:

```bash
cd evaluation/real_robot/client
python3 deploy/client_lift2_va.py \
  --profile dry_run \
  --dry_run_mock \
  --host <GPU_IP> \
  --port 7777
```

This uses a live GPU server, but does not require ROS, cameras, or a robot. The
client constructs the full protocol loop with synthetic RGB frames and a fake
EEF execution:

```text
metadata
→ reset(prompt)
→ infer(initial_fake_frame)
→ fake-expand/execute action
→ generate sparse fake keyframes
→ compute_kv_cache(fake_keyframes, state=action)
→ infer(initial_fake_frame)
→ ...
```

The default is two chunks, which verifies that the second `infer` happens after
the first KV-cache update. A successful run ends with `PASS`; for the standard
`(16, 2, 16)` action it reports 16 fake execution steps and 4 keyframes for the
first chunk, followed by 32 steps and 8 keyframes for the second chunk.
Use `--dry_run_mock_chunks N` to test a different number of complete chunks.

### Direct VPN connectivity debug

If the GPU server has a reachable VPN address such as `192.168.215.55`, the
robot client can connect to that address directly; SSH port forwarding is not
required. Verify the route and the server health endpoint from the same
machine where the client will run:

```bash
ip route get 192.168.215.55
curl --max-time 3 http://192.168.215.55:7777/healthz
```

The second command should print `OK`. It is preferable to `nc -z` or a raw
`socket.create_connection`, because `/healthz` sends a valid HTTP request and
does not generate a false WebSocket handshake error in the server log.

Then run the client with the VPN address and bounded startup diagnostics:

```bash
uv --directory /soft/wangxi/4DWAM/lingbot-va run python \
  evaluation/real_robot/client/deploy/client_lift2_va.py \
  --profile lift2_va_default \
  --host 192.168.215.55 \
  --port 7777 \
  --dry_run_mock \
  --server_wait_timeout_seconds 30 \
  --verbose
```

Before running ROS or model inference, test only the WebSocket handshake and
metadata. This is the most direct way to distinguish a network/client startup
problem from a GPU inference problem:

```bash
uv --directory /soft/wangxi/4DWAM/lingbot-va run python \
  evaluation/real_robot/client/deploy/client_lift2_va.py \
  --profile lift2_va_default \
  --host 127.0.0.1 \
  --port 7777 \
  --probe_server \
  --server_wait_timeout_seconds 10 \
  --connect_timeout_seconds 3 \
  --metadata_timeout_seconds 5 \
  --connect_retry_seconds 1 \
  --verbose
```

For an SSH `-L` tunnel, run this command on the same machine where the SSH
tunnel is listening. A successful probe prints `WebSocket handshake
succeeded` and `PASS`, and the server should log a normal WebSocket client
connection. Unlike `/healthz`, the probe exercises the WebSocket handler and
receives the server metadata.

`--server_wait_timeout_seconds 0` means unlimited total waiting. Individual
WebSocket handshakes and metadata reads still have finite timeouts. `Ctrl+C`
is handled explicitly and should terminate the client with exit code `130`.

On the GPU server, verify that the process is listening on all interfaces:

```bash
ss -ltnp | rg ':7777'
```

The inference config uses `host='0.0.0.0'`. If `/healthz` works but the client
does not connect, check that the robot is running the updated
`client/dwam_client/websocket_va_policy.py` and that no HTTP/SOCKS proxy is
being applied; the client disables proxy use for this direct robot/VPN link.

---

## 2. Robot client

Copy **`evaluation/real_robot/client/`** to the robot PC (self-contained).

### Install (robot)

```bash
pip install -r requirements-robot.txt
# ROS (rospy, cv_bridge, arm_control, sensor_msgs) from robot workspace
```

### Configure

Edit `client/launch_profiles.yaml`:

```yaml
profiles:
  lift2_va_default:
    host: <GPU_SERVER_IP>
    port: 7777
    publish_rate: 60
    auto_init: true
    init_duration: 3.0
    left_init_pose: [0, 0, 0, 0, 0, 0, 1]
    right_init_pose: [0, 0, 0, 0, 0, 0, 1]
    ...
```

The reset-pose gripper values are normalized to `[0, 1]`; commands are
converted to the robot's configured raw range before publishing. Use
`wait_after_init: true` (or `--wait_after_init`) during the first hardware
check, and keep the emergency stop accessible. `--no_auto_init` disables the
startup motion.

### Start robot stacks (optional helper)

```bash
# Adjust LIFT2_ROOT if needed
./run_lift2.sh
```

Or start CAN / body / dual-arm / RealSense the same way as openpi-on-LIFT2.

### Run client

```bash
./launch.sh --task wrench
# or
python3 deploy/client_lift2_va.py --profile lift2_va_default --host <GPU_IP> --task wrench

# Log absolute EEF without publishing commands
./launch.sh --dry_run --task wrench

# Override rate / port
./launch.sh --publish_rate 60 --port 7777 --host <GPU_SERVER_IP>
```

### Client loop (summary)

```text
wait for dual-arm state
smoothly move to reset pose
reset(prompt)
cache init EEF
loop:
  infer -> action (C,F,H)
  expand + re-anchor + publish @ 60 Hz  (sleep between steps for rate)
  sample keyframes every H//4 micro-steps via get_latest_images (peek)
  if no keyframes: raise (do not call kv with empty list)
  compute_kv_cache(keyframes, state=action)
```

The `--dry_run_mock` path follows the same loop without ROS. It replaces the
camera and arm data with deterministic synthetic values, skips wall-clock
publish-rate sleeping because no commands are sent to hardware, and keeps the
WebSocket requests and server responses real.

Images are **RGB** (same as openpi-on-LIFT2). Quaternions are **xyzw** (SciPy / training `get_relative_pose`).

---

## 3. Offline unit tests (no ROS / no GPU)

```bash
cd /soft/wangxi/4DWAM/lingbot-va
python3 -m unittest evaluation.real_robot.tests.test_msgpack_roundtrip \
  evaluation.real_robot.tests.test_keyframe_schedule \
  evaluation.real_robot.tests.test_va_action_bridge \
  evaluation.real_robot.tests.test_startup_reset -v
```

Or:

```bash
python3 evaluation/real_robot/tests/test_va_action_bridge.py -v
python3 evaluation/real_robot/tests/test_msgpack_roundtrip.py -v
python3 evaluation/real_robot/tests/test_keyframe_schedule.py -v
python3 evaluation/real_robot/tests/test_startup_reset.py -v
```

---

## 4. Acceptance checklist

- [ ] `client/launch_profiles.yaml` is the only default client profile source
- [ ] Default **port=7777** on both server launch and client profile
- [ ] Default **publish_rate=60**
- [ ] Server responds to `reset` / `infer` / `compute_kv_cache`
- [ ] `action` field present; shape matches cfg (e.g. 16×2×16)
- [ ] No openpi RTC; no `apply_eef_delta`
- [ ] Re-anchor uses **episode init** pose only
- [ ] Client waits for arm state and reaches the configured reset pose before `reset(prompt)`
- [ ] Keyframes: first chunk 4, later chunks 8 for F=2,H=16,divisor=4
- [ ] ROS publishes absolute 14D with gripper raw ≈ `[0, 5]`

---

## 5. Alignment map

| Layer | Source of truth |
|-------|-----------------|
| Server protocol / action semantics | `wan_va_server` + robotwin eval client |
| Training representation | LIFT2 conversion + `lerobot_latent_dataset` segment-relative |
| Robot sensors / actuators | openpi-on-LIFT2 `RosOperator` + `PosCmd` |
| Bridge | `client/deploy/utils/va_action_bridge.py` |

**One line:** GPU speaks LingBot-VA; robot speaks openpi ROS; `client/` converts between them.
