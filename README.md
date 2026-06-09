# 🐕 Robodog 3D — LiDAR Space Reconstruction

Interactive 3D walkthrough of an indoor space, reconstructed from a **Unitree Go2** robot dog's LiDAR scan.

## 🚀 [Live Demo](https://qwadratic.github.io/robodog-3d/)

![LiDAR floor plan reconstruction](screenshot.png)

## What is this?

A quadruped robot walked through an office/apartment for **4 minutes** carrying a solid-state LiDAR (Unitree L1, 15 Hz). The raw scan — 41.5 million 3D points — was processed into a navigable architectural model: walls, floor, ceiling, and schematic furniture outlines.

No cameras were used. Everything you see comes from a single LiDAR sensor at 40cm height.

### What the robot recorded

| Topic | Rate | Description |
|-------|------|-------------|
| `/utlidar/cloud_deskewed` | 15 Hz | 3D point clouds (motion-compensated) |
| `/sportmodestate` | 50 Hz | SLAM odometry (position + orientation) |
| `/imu` | 500 Hz | IMU accelerometer + gyroscope |
| `/jointstate` | 500 Hz | 12 joint positions (4 legs × 3 joints) |

The MCAP file (2 GB) is available on [Google Drive](https://drive.google.com/drive/folders/1X2RnhCLFHmyrKIzM3l7SOrvbRpVZP4eA).

### Reconstruction pipeline

```
41.5M raw points
  → 1cm voxel downsample → 2.16M points
  → normal estimation → classify floor / wall / ceiling / furniture
  → height-span filter: only floor-to-ceiling surfaces are walls (979 cells)
  → greedy meshing: merge adjacent coplanar faces into flat quads
  → occupancy probability: point density → wall brightness
  → schematic furniture: floor shadows + wireframe edges + top cap
  → GLB export → Three.js viewer
```

### Key numbers

| Metric | Value |
|--------|-------|
| Raw points | 41.5M |
| After 1cm voxel downsample | 2.16M |
| Real walls (height-filtered) | 979 cells |
| Furniture objects | 289 |
| Ceiling area (enclosed rooms) | 62.5 m² |
| Robot path length | 39.8 m |
| Scan duration | 4 min |
| Model file | 3.3 MB GLB |
| Total web payload | ~4 MB |

### Why walls are height-filtered

The LiDAR sees vertical surfaces on furniture (desks, chairs, shelves) the same way it sees walls. Without filtering, 6,028 cells were classified as "walls" — most of them furniture, choking every corridor. 

The fix: real walls span floor-to-ceiling (~2.1m). Furniture doesn't. Filtering by height span (>45% of room height) drops to **979 real wall cells** — matching the actual room layout.

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Click** | Enter first-person mode |
| **WASD** | Move |
| **Mouse** | Look around |
| **Shift** | Sprint |
| **R** | Robodog replay (original 4-min trajectory) |
| **P** | Point cloud overlay (lazy-loaded) |
| **F2** | Save screenshot |
| **ESC** | Release mouse |

## Tech stack

- **Vite** + **Three.js** (npm, tree-shaken, ~155KB gzipped)
- PointerLockControls for first-person navigation
- GLB model with vertex colors (occupancy probability → brightness)
- Point cloud: quantized Int16 + Uint8 binary format (352 KB)
- Robot replay: 206 keyframes with position + heading, interpolated
- GitHub Pages via CI build (`npm run build` → `dist/`)

## Reconstruction scripts

The `scripts/` folder contains the full pipeline. Download the MCAP from [Google Drive](https://drive.google.com/drive/folders/1X2RnhCLFHmyrKIzM3l7SOrvbRpVZP4eA) and place it in `data/`.

| Script | Purpose |
|--------|--------|
| `extract_floorplan.py [resolution]` | Read MCAP → accumulate + downsample points → save NPZ |
| `build_clean_model.py` | Classify → height-filter walls → greedy mesh → GLB + minimap |

```bash
pip install open3d mcap mcap-ros2-support scipy numpy matplotlib pillow

# Place MCAP in data/
python scripts/extract_floorplan.py 0.01   # 1cm voxel
python scripts/build_clean_model.py         # → public/assets/model.glb
```

## Local development

```bash
git clone https://github.com/qwadratic/robodog-3d.git
cd robodog-3d
npm install
npm run dev    # http://localhost:5173/robodog-3d/
```

## Data source

- **Robot**: [Unitree Go2](https://www.unitree.com/go2) quadruped
- **LiDAR**: Unitree L1 solid-state, 15 Hz, ~11K points/scan
- **Format**: [MCAP](https://mcap.dev/) (ROS2 rosbag2)
- **MCAP file**: [Google Drive](https://drive.google.com/drive/folders/1X2RnhCLFHmyrKIzM3l7SOrvbRpVZP4eA) (2 GB)

---

*Built by [@qwadratic](https://github.com/qwadratic)*
