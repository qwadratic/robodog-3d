# 🐕 Robodog 3D — LiDAR Space Reconstruction

Interactive 3D walkthrough of an indoor space reconstructed from a **Unitree Go2 robot dog's LiDAR sensor data**.

**[🔗 Live Demo](https://gerhardgustav.github.io/robodog-3d/)**

## What You're Looking At

A robot dog walked through an indoor space for 4 minutes, scanning with its onboard LiDAR (Unitree L1 solid-state LiDAR, 15 Hz). From those **41.5 million raw LiDAR points**, we:

1. **Accumulated** all scans into a unified 3D point cloud (678K points after voxel downsampling)
2. **Classified** every point as floor, wall, ceiling, or furniture using surface normals and height
3. **Reconstructed** an architectural 3D model with:
   - **287 wall segments** (PCA-oriented, gap-filled with endpoint connectivity)
   - **Checkerboard wood floor** with procedural grain and ambient occlusion
   - **Ceiling** only in enclosed rooms (detected via boundary wall analysis)
   - **40 furniture objects** from DBSCAN clustering
   - **Baseboards** at wall-floor junctions
4. **Exported** for web viewing with Three.js

## Controls

| Input | Action |
|-------|--------|
| `W` `A` `S` `D` | Walk around |
| `Mouse` | Look around |
| `Shift` | Sprint |
| `1` | Point cloud only |
| `2` | 3D model only |
| `3` | Both overlaid |
| `ESC` | Menu |

## Data Source

- **Robot:** Unitree Go2 quadruped
- **Sensor:** Unitree L1 solid-state LiDAR (15 Hz, ~11K pts/scan)
- **Recording:** 4 minutes, 39.8m path, ~38 m² area
- **Format:** MCAP (ROS2 rosbag2), 2.0 GB
- **Topics used:** `/utlidar/cloud_deskewed` (motion-corrected point clouds in odom frame), `/utlidar/robot_pose` (6DOF poses)

## Pipeline

```
MCAP recording (2.0 GB)
  → Point cloud accumulation (41.5M points)
  → Voxel downsampling (678K points, 3cm)
  → Surface normal estimation
  → Point classification (floor/wall/ceiling/object)
  → DBSCAN wall clustering (275 clusters)
  → PCA line fitting + collinear merging (137 segments)
  → Endpoint connectivity (150 connecting segments → 287 total)
  → Mesh generation (walls, floor, ceiling, baseboards, furniture)
  → GLB export (2.3 MB)
  → Three.js web viewer
```

## Tech Stack

- **Data processing:** Python, Open3D, SciPy, NumPy, mcap-ros2-support
- **3D model:** Open3D → GLB (glTF binary)
- **Web viewer:** Three.js (vanilla, CDN-loaded, no build step)
- **Hosting:** GitHub Pages

## Local Development

Just serve the directory with any HTTP server:

```bash
# Python
python3 -m http.server 8000

# Node
npx serve .
```

Then open `http://localhost:8000`

## Files

| File | Size | Description |
|------|------|-------------|
| `index.html` | 21 KB | Three.js viewer (self-contained) |
| `assets/model.glb` | 2.3 MB | Architectural 3D model |
| `assets/pointcloud.bin` | 1.2 MB | 78K height-colored points (binary) |
| `assets/trajectory.json` | 5 KB | Robot path (205 waypoints) |

## How It Was Made

This project started with a single 2GB MCAP file from a robot dog walk-around. Using pure LiDAR data (no camera), we extracted:

- Floor plans and room segmentation
- Wall detection and measurement  
- Ceiling height mapping
- Furniture/object detection
- Complete 3D reconstruction

The reconstruction is geometry-only (no color/texture from camera). Adding camera data would unlock photorealistic rendering via Gaussian Splatting or NeRF.

## License

MIT
