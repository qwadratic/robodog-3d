# 🐕 Robodog 3D — LiDAR Space Reconstruction

Interactive Three.js viewer for a LiDAR-reconstructed indoor space from a **Unitree Go2** robot dog.

## 🚀 [Live Demo](https://qwadratic.github.io/robodog-3d/)

![Screenshot](screenshot.png)

## What is this?

A robot dog walked through an indoor space for 4 minutes with a solid-state LiDAR (15 Hz, 41.5M points total). From that single scan, we reconstructed a 3D architectural model and deployed it as an interactive first-person walkthrough.

### Pipeline

1. **Raw data**: 2GB MCAP file (ROS2 rosbag) → 41.5M deskewed LiDAR points + SLAM odometry
2. **Downsample**: 1cm voxel grid → 2.16M unique points
3. **Classify**: surface normals + height → floor / wall / ceiling / furniture
4. **Reconstruct**: voxelized walls (every occupied cell), flat floor/ceiling, schematic furniture
5. **Ghost walls**: point cloud boundary heuristic — where density drops abruptly = wall
6. **Export**: GLB model (1MB) + minimap + collision data → Three.js viewer

### Features

- **6,028 wall cells** — voxelized with occupancy probability (bright = confident, dark = uncertain)
- **11 ghost walls** (inferred from point cloud density boundaries)
- **289 furniture objects** — schematic: floor shadows + wireframe edges + top cap
- **Flat floor** with ambient occlusion near walls
- **Flat ceiling** only in enclosed rooms (34.2 m²)
- **Human-height camera** (1.6m eye level)
- **Minimap** with real-time position + FOV cone
- **Collision detection** against real walls
- **Low coverage warning** in barely-scanned areas

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Click** | Lock mouse (enter first-person) |
| **WASD** | Move |
| **Mouse** | Look around |
| **Shift** | Sprint |
| **P** | Toggle point cloud overlay (loaded on demand) |
| **G** | Toggle ghost walls (inferred boundaries) |
| **ESC** | Release mouse |

## Tech

- Single `index.html` — no build step
- Three.js from CDN (importmap + modulepreload)
- PointerLockControls for first-person navigation
- Assets loaded in parallel with `<link rel="preload">`
- Point cloud lazy-loaded only on P key press
- Total payload: ~4MB

## Data source

- **Robot**: Unitree Go2 quadruped
- **LiDAR**: Unitree L1 solid-state, 15 Hz
- **Recording**: 4 minutes, 39.8m path, ~38m² covered
- **Format**: MCAP (ROS2 rosbag2, libmcap 1.3.1)

## Reconstruction scripts

The `scripts/` folder contains the Python pipeline that generates the 3D model from raw LiDAR data:

| Script | Purpose |
|--------|--------|
| `extract_floorplan.py` | Read MCAP, accumulate 41.5M points, downsample (default 1cm), save NPZ |
| `build_clean_model.py` | Classify points, build probabilistic voxelized walls + floor + ceiling + furniture |
| `export_ghost_walls.py` | Detect ghost walls from point cloud boundaries + export collision data |

Requires: `pip install open3d mcap mcap-ros2-support scipy numpy matplotlib pillow`

## Local development

```bash
git clone https://github.com/qwadratic/robodog-3d.git
cd robodog-3d
python3 -m http.server 8000
# Open http://localhost:8000
```

---

*Built by [@qwadratic](https://github.com/qwadratic)*
