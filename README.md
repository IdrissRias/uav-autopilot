# UAV Autopilot Architecture (X-Plane 12)

This project provides a sim-agnostic autopilot core with an X-Plane 12 UDP adapter (RREF read + DREF write). It is structured for a future MAVLink adapter without rewriting the core guidance/control logic.

## Quick Start

1. Ensure X-Plane 12 is running locally on macOS.
2. Confirm X-Plane is configured to receive UDP on port 49000 (default).
3. From this folder:

```bash
python -m uav.main --config src/uav/config/default.yaml
```

## X-Plane UDP Notes

- X-Plane listens on UDP port 49000 by default.
- The autopilot binds a local UDP port (default 49005). If you see an error like "Address already in use", change `local_port` in `src/uav/config/default.yaml`.

## Safety Warning

This autopilot writes control inputs (throttle/roll/pitch/yaw). **Controls will move**. Use at your own risk and be ready to disable or quit the script.

## Configuration

Edit `src/uav/config/default.yaml` to adjust:

- ports (`xplane_ip`, `xplane_port`, `local_port`)
- loop rate
- controller gains
- safety limits
- start mode
- auto start and destination gating
- targets (altitude/heading/airspeed)

## Logging

Logs are written to `logs/` as CSV files at ~10 Hz. Each record contains time, mode, targets, telemetry, and actuator outputs.

## Architecture

- `sim/` contains the X-Plane UDP adapter and sim-agnostic interfaces.
- `core/` contains modes, guidance, control, safety, and the main autopilot loop.
- `logging/` handles telemetry/actuator recording.
- `tests/` includes unit tests for PID, safety limits, and mode transitions.
