# 🐕 Robodog 3D — LiDAR Space Reconstruction

**Interactive Three.js viewer for LiDAR-reconstructed indoor spaces from a Unitree Go2 robot dog.**

## 🚀 [Live Demo](https://gerhardgustav.github.io/robodog-3d/)

![Robodog 3D Screenshot](https://github.com/gerhardgustav/robodog-3d/assets/demo-screenshot.png)

## 🎯 What is this?

This project reconstructs a 3D architectural model from raw LiDAR point cloud data collected by a **Unitree Go2** robot dog walking through an indoor space. The web viewer lets you explore the reconstructed environment in first-person mode with real-time rendering.

### Key Features

- **84,750 point cloud** with height-based coloring (blue→green→yellow→red)
- **21,252 triangle mesh** with procedural wall connectivity bridges  
- **Wood plank floors** with grain variation and ambient occlusion
- **Intelligent ceiling placement** only in enclosed rooms
- **Robot trajectory visualization** (205 waypoints from the actual walk)
- **First-person exploration** with WASD+mouse controls and flashlight
- **Multiple view modes**: Point Cloud / Model / Both overlaid

## 🎮 Controls

| Key | Action |
|-----|--------|
| **WASD** | Move around |
| **Mouse** | Look around (click to lock pointer) |
| **Shift** | Sprint |
| **1** | Point cloud only |
| **2** | Model only |
| **3** | Both overlaid |
| **ESC** | Exit pointer lock |

## 📊 Technical Details

### Data Processing Pipeline

1. **MCAP Scan Data** → 41.5M raw LiDAR points from Unitree Go2
2. **Spatial Analysis** → Surface normals, height clustering, wall detection
3. **Wall Connectivity** → PCA-based clustering + 30cm gap bridging  
4. **Mesh Generation** → Floor grids, ceiling enclosure detection, furniture boxes
5. **Web Export** → Downsampled point cloud + GLB model + trajectory JSON

### Model Architecture

- **137 base wall segments** + **57 connectivity bridges** = **194 total walls**
- **Wood floor**: 8cm resolution with procedural grain patterns
- **Smart ceiling**: Only rendered in enclosed room areas (20.7 m²)
- **40 furniture objects** detected via DBSCAN clustering
- **Baseboards** at every wall-floor junction

### File Structure

```
robodog-3d/
├── index.html              # Three.js viewer (zero build step)
├── assets/
│   ├── model.glb           # 3D architectural mesh (1.6 MB)
│   ├── pointcloud.bin      # Height-colored points (2.0 MB) 
│   ├── trajectory.json     # Robot waypoints (12.8 KB)
│   └── metadata.json       # Scene bounds and parameters
└── README.md
```

## 🔧 Local Development

```bash
# Clone the repository
git clone https://github.com/gerhardgustav/robodog-3d.git
cd robodog-3d

# Serve locally (required for CORS)
python -m http.server 8000
# or
npx serve

# Open http://localhost:8000
```

## 📡 Data Sources

- **Robot**: Unitree Go2 quadruped with integrated LiDAR
- **Recording**: 4-minute walkaround of indoor office space
- **Raw data**: 3,701 scans, 41.5M points, 618 m² explored area
- **Environment**: Floor 0.0m, ceiling 2.15m, mixed room layouts

## 🛠️ Technologies

- **Three.js** — WebGL 3D rendering and controls
- **Open3D** — Point cloud processing and mesh generation  
- **Python** — Spatial analysis and wall connectivity algorithms
- **GitHub Pages** — Zero-config deployment

## 🎨 Coordinate System

The data uses **Z-up coordinates** from the robot's SLAM system, converted to **Y-up** for Three.js:

```javascript
// MCAP → Three.js conversion
position.x = mcap.x      // X stays X  
position.y = mcap.z      // Z becomes Y (up)
position.z = -mcap.y     // Y becomes -Z (forward)
```

## 🚧 Known Limitations

- **No camera data**: Walls are procedurally textured (no photorealistic surfaces)
- **Geometric reconstruction only**: Missing windows, doors, detailed furniture
- **2.2m ceiling height**: Limited vertical exploration range

## 🔮 Future Improvements

- **Camera fusion**: Add RGB textures from synchronized camera data
- **Gaussian Splatting**: Neural radiance fields for photorealistic rendering
- **Room segmentation**: Semantic labeling (kitchen, bedroom, etc.)
- **Physics**: Collision detection and gravity simulation

---

**Built with LiDAR data from a robot dog's indoor exploration.** 🐕‍🦺

*Made by [@gerhardgustav](https://github.com/gerhardgustav)*