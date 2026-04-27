# UAV Project — Claude Code Context
Load this file at the start of your Claude Code session so Claude has full context.
---
## What This Project Is
A personal long-term UAV project. The goal is to build and test UAV control systems,
starting in simulation before moving to real hardware. Speed of execution matters.
Financial constraints are real. This is a dream project.
---
## Stack Decisions (Already Made)
- **Simulator**: X-Plane 12 (already installed, already running on this machine)
- **Plugin**: XPlane Connect (NASA) — installed in X-Plane 12 plugins folder, bridges Python ↔ X-Plane via UDP on port 49009
- **Language**: Python
- **RL Framework**: Stable Baselines3 (PPO algorithm) + Gymnasium
- **IDE**: Claude Code

Why X-Plane over ArduPilot SITL: chosen for its visual fidelity — you can watch the
agent fly in a realistic 3D environment in real time.

Why RL over scripted flight: instead of hand-coding takeoff/landing logic, a PPO agent
learns to fly through trial and error. Reward functions define the goal, the agent
figures out how to achieve it.

---
## Project Structure
```
uav-project/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── xplane_client.py     # Low-level UDP client — reads/writes X-Plane DataRefs
│   ├── env/
│   │   ├── __init__.py
│   │   ├── xplane_env.py        # Base Gymnasium environment (subclass for each task)
│   │   ├── takeoff_env.py       # Task: take off and climb to 300m AGL
│   │   └── landing_env.py       # Task: descend and land smoothly
│   └── agents/                  # Empty — for future custom policy architectures
├── models/                      # Saved PPO model checkpoints (.zip files go here)
├── tools/                       # Utility scripts
├── tests/                       # Test suite
├── train.py                     # Main training entry point
├── evaluate.py                  # Run a trained agent and watch it fly
├── pyproject.toml               # Project config and dependencies
├── CONTEXT.md                   # This file
└── README.md                    # Full setup and usage guide
```
---
## How It Works
```
X-Plane 12 (running locally)
      ↕  UDP port 49009
XPlaneClient (src/core/xplane_client.py)
  — reads DataRefs: lat, lon, alt_agl, airspeed, vspeed, pitch, roll, heading
  — writes DataRefs: throttle, elevator, aileron, rudder, gear, flaps, brakes
  — POSI packet: teleport aircraft to any lat/lon/altitude instantly
      ↕
XPlaneEnv (src/env/xplane_env.py) — Gymnasium-compatible base class
  observation space: 9 normalized floats [alt, speed, vspeed, pitch, roll, heading, throttle, elevator, aileron]
  action space:      4 continuous floats [throttle, elevator, aileron, rudder]
      ↕
TakeoffEnv / LandingEnv — task-specific reward functions
      ↕
PPO Agent (Stable Baselines3)
  — learns which actions maximise cumulative reward
  — trains for N timesteps, saves checkpoints every 10k steps
  — can be resumed from any checkpoint
```
---
## How to Run
### First time setup
```bash
pip install gymnasium stable-baselines3 numpy tensorboard
```
### Verify X-Plane connection (run this first)
```python
import socket, struct, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(3.0)
packet = b"RREF\x00" + struct.pack("<ii", 1, 1) + b"sim/flightmodel/position/latitude".ljust(400, b"\x00")
sock.sendto(packet, ("127.0.0.1", 49009))
time.sleep(0.2)
data, _ = sock.recvfrom(1024)
print("Lat:", struct.unpack_from("<if", data, 5)[1])
```
Expected: prints current aircraft latitude. If timeout → check XPlane Connect plugin is loaded.

### Train the takeoff agent
```bash
python train.py --task takeoff --timesteps 200000
```
### Train the landing agent
```bash
python train.py --task landing --timesteps 200000
```
### Watch a trained agent fly
```bash
python evaluate.py --task takeoff --model models/ppo_takeoff_final --episodes 5
```
### Monitor training in TensorBoard
```bash
tensorboard --logdir logs/
```
---
## Key DataRefs Used
| DataRef | Description |
|---------|-------------|
| `sim/flightmodel/position/latitude` | Current latitude (read only) |
| `sim/flightmodel/position/longitude` | Current longitude (read only) |
| `sim/flightmodel/position/elevation` | Altitude MSL in metres (read only) |
| `sim/flightmodel/position/y_agl` | Altitude above ground in metres (read only) |
| `sim/flightmodel/position/indicated_airspeed` | Airspeed in knots (read only) |
| `sim/flightmodel/position/vh_ind_fpm` | Vertical speed in fpm (read only) |
| `sim/flightmodel/position/theta` | Pitch in degrees (read only) |
| `sim/flightmodel/position/phi` | Roll in degrees (read only) |
| `sim/flightmodel/position/psi` | Heading 0–360 degrees (read only) |
| `sim/cockpit2/engine/actuators/throttle_ratio_all` | Throttle 0.0–1.0 (write) |
| `sim/joystick/yoke_pitch_ratio` | Elevator -1.0–1.0 (write) |
| `sim/joystick/yoke_roll_ratio` | Aileron -1.0–1.0 (write) |
| `sim/joystick/yoke_heading_ratio` | Rudder -1.0–1.0 (write) |
| `sim/cockpit/switches/gear_handle_status` | Gear 1=down 0=up (write) |
| `sim/cockpit2/controls/flap_handle_deploy_ratio` | Flaps 0.0–1.0 (write) |
| `sim/cockpit2/controls/parking_brake_ratio` | Brakes 0.0–1.0 (write) |

---
## What's NOT Built Yet (Next Steps)
1. **Navigation environment** (`src/env/navigation_env.py`)
   - Agent flies from point A to point B autonomously
   - Observation includes: distance to target, bearing error
   - Reward: closing distance to waypoint
   - Requires airport parser (see below)

2. **Airport parser** (`tools/parse_airports.py`)
   - Parses X-Plane's `apt.dat` file to extract all airport lat/lon coords
   - Uses Haversine formula to find nearest airport to current position
   - Feeds target waypoint into navigation environment
   - apt.dat lives at: `[X-Plane 12]/Resources/default scenery/default apt dat/Earth nav data/apt.dat`

3. **Training hasn't run yet** — the code is written and correct in structure but
   hasn't been tested against a live X-Plane instance. Small DataRef tweaks may
   be needed once you run it for the first time.

---
## Known Issues / Watch Out For
- **DataRef names**: some may need minor corrections once tested live — X-Plane 12
  occasionally differs from X-Plane 11 DataRef paths. Use DataRefTool plugin
  (free, datareftool.com) to verify any DataRef name if a value reads as 0.
- **Control frequency**: currently running at ~10 Hz (0.1s sleep per step). If
  training feels slow, X-Plane's time acceleration (set via DataRef
  `sim/time/sim_speed`) can speed up the sim during training.
- **Airport coordinates** in takeoff_env.py and landing_env.py are set to KSFO
  (San Francisco). Change these to match whatever airport you have loaded in X-Plane.

---
## Long-Term Direction
Simulation → Real Hardware pipeline:
1. Train and validate control policies in X-Plane 12
2. Port working policies to ArduPilot or PX4 autopilot firmware
3. Deploy to real UAV hardware via MAVLink protocol
