#!/usr/bin/env python3
"""Extract floor plan from Unitree Go2 LiDAR MCAP data.

Pipeline:
1. Accumulate deskewed point clouds (already in odom frame)
2. Slice at wall height (0.3m–2.0m) to isolate walls
3. Project to 2D bird's-eye-view
4. Rasterize to occupancy grid
5. Morphological cleanup → wall detection
6. Export: PNG floor plan + point cloud PLY + occupancy grid NPZ
"""

import sys
import struct
import numpy as np
from pathlib import Path
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory

# Resolve paths: MCAP and output live in robodog-telemetry/
_here = Path(__file__).resolve().parent
_telemetry = _here.parent.parent / 'robodog-telemetry'
if (_telemetry / 'output').exists() or _telemetry.exists():
    MCAP_FILE = _telemetry / "debug_big_walkaround_recovered.mcap"
    OUT_DIR = _telemetry / "output"
else:
    MCAP_FILE = Path("debug_big_walkaround_recovered.mcap")
    OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# --- Parameters ---
import sys
VOXEL_SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01  # default 1cm, override via CLI
GRID_RES = 0.02            # 2cm per pixel in floor plan
WALL_Z_MIN = 0.3           # min height for wall slice (above floor)
WALL_Z_MAX = 2.2           # max height for wall slice
FLOOR_Z_MIN = -0.1         # floor slice
FLOOR_Z_MAX = 0.15
MIN_POINTS_PER_CELL = 3    # occupancy threshold


def read_pointcloud2_xyz(msg) -> np.ndarray:
    """Extract XYZ from a PointCloud2 message. Handles variable point_step."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    data = np.frombuffer(msg.data, dtype=np.uint8)
    step = msg.point_step

    # Find x,y,z field offsets
    offsets = {}
    for f in msg.fields:
        if f.name in ('x', 'y', 'z'):
            offsets[f.name] = f.offset

    if len(offsets) < 3:
        # Fallback: assume x=0, y=4, z=8 (standard float32 layout)
        offsets = {'x': 0, 'y': 4, 'z': 8}

    points = np.zeros((n, 3), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        off = offsets[name]
        col = np.array([
            struct.unpack_from('<f', data, row * step + off)[0]
            for row in range(n)
        ], dtype=np.float32)
        points[:, i] = col

    # Filter NaN/inf
    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def read_pointcloud2_xyz_fast(msg) -> np.ndarray:
    """Fast vectorized XYZ extraction from PointCloud2."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)

    # Find offsets
    offsets = {}
    for f in msg.fields:
        if f.name in ('x', 'y', 'z'):
            offsets[f.name] = f.offset
    if len(offsets) < 3:
        offsets = {'x': 0, 'y': 4, 'z': 8}

    points = np.zeros((n, 3), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        off = offsets[name]
        # Extract every point_step bytes, starting at offset
        indices = np.arange(n) * step + off
        # Read 4 bytes (float32) at each index
        raw = np.array([data[idx:idx+4].view(np.float32)[0] for idx in indices])
        points[:, i] = raw

    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def read_pointcloud2_structured(msg) -> np.ndarray:
    """Fastest: reinterpret buffer with structured dtype."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    step = msg.point_step
    offsets = {}
    for f in msg.fields:
        if f.name in ('x', 'y', 'z'):
            offsets[f.name] = f.offset

    if len(offsets) < 3:
        offsets = {'x': 0, 'y': 4, 'z': 8}

    # Build structured dtype matching point_step
    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': ['<f4', '<f4', '<f4'],
                   'offsets': [offsets['x'], offsets['y'], offsets['z']],
                   'itemsize': step})

    arr = np.frombuffer(msg.data, dtype=dt, count=n)
    points = np.column_stack([arr['x'], arr['y'], arr['z']])
    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def accumulate_clouds(mcap_path: Path, topic: str = "/utlidar/cloud_deskewed") -> np.ndarray:
    """Read all point clouds from MCAP, return Nx3 array."""
    print(f"Reading {topic} from {mcap_path.name}...")
    decoder_factory = DecoderFactory()
    all_points = []
    count = 0

    with open(mcap_path, "rb") as f:
        sr = StreamReader(f, record_size_limit=1024*1024*1024)
        channels, schemas, decoders = {}, {}, {}

        for record in sr.records:
            rtype = type(record).__name__
            if rtype == 'Schema':
                schemas[record.id] = record
            elif rtype == 'Channel':
                channels[record.id] = record
                schema = schemas.get(record.schema_id)
                if schema:
                    try:
                        decoders[record.id] = decoder_factory.decoder_for(
                            record.message_encoding, schema)
                    except Exception:
                        pass
            elif rtype == 'Message':
                ch = channels.get(record.channel_id)
                if ch and ch.topic == topic:
                    decoder = decoders.get(record.channel_id)
                    if decoder:
                        try:
                            msg = decoder(record.data)
                            pts = read_pointcloud2_structured(msg)
                            if len(pts) > 0:
                                all_points.append(pts)
                                count += 1
                                if count % 500 == 0:
                                    total = sum(len(p) for p in all_points)
                                    print(f"  {count} scans, {total:,} points...")
                        except Exception as e:
                            pass

    cloud = np.vstack(all_points)
    print(f"  Total: {count} scans, {cloud.shape[0]:,} points")
    return cloud


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel grid downsampling without Open3D."""
    print(f"Voxel downsampling ({voxel_size}m)...")
    quantized = np.floor(points / voxel_size).astype(np.int32)
    # Unique voxels
    _, idx = np.unique(quantized, axis=0, return_index=True)
    result = points[idx]
    print(f"  {points.shape[0]:,} → {result.shape[0]:,} points")
    return result


def make_occupancy_grid(points_2d: np.ndarray, resolution: float, min_count: int = 1):
    """Rasterize 2D points to occupancy grid. Returns grid, origin."""
    x_min, y_min = points_2d.min(axis=0) - 0.5
    x_max, y_max = points_2d.max(axis=0) + 0.5

    w = int(np.ceil((x_max - x_min) / resolution))
    h = int(np.ceil((y_max - y_min) / resolution))
    print(f"  Grid: {w}×{h} pixels ({resolution}m/px)")

    grid = np.zeros((h, w), dtype=np.int32)
    ix = ((points_2d[:, 0] - x_min) / resolution).astype(int)
    iy = ((points_2d[:, 1] - y_min) / resolution).astype(int)

    # Clip to bounds
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    np.add.at(grid, (iy, ix), 1)

    occupied = (grid >= min_count).astype(np.uint8)
    return grid, occupied, (x_min, y_min, resolution)


def extract_trajectory(mcap_path: Path) -> np.ndarray:
    """Extract robot trajectory from odometry."""
    print("Extracting trajectory...")
    decoder_factory = DecoderFactory()
    poses = []

    with open(mcap_path, "rb") as f:
        sr = StreamReader(f, record_size_limit=1024*1024*1024)
        channels, schemas, decoders = {}, {}, {}

        for record in sr.records:
            rtype = type(record).__name__
            if rtype == 'Schema':
                schemas[record.id] = record
            elif rtype == 'Channel':
                channels[record.id] = record
                schema = schemas.get(record.schema_id)
                if schema:
                    try:
                        decoders[record.id] = decoder_factory.decoder_for(
                            record.message_encoding, schema)
                    except Exception:
                        pass
            elif rtype == 'Message':
                ch = channels.get(record.channel_id)
                if ch and ch.topic == '/utlidar/robot_pose':
                    decoder = decoders.get(record.channel_id)
                    if decoder:
                        try:
                            msg = decoder(record.data)
                            p = msg.pose.position
                            poses.append([p.x, p.y, p.z])
                        except:
                            pass

    traj = np.array(poses)
    print(f"  {len(traj)} poses")
    return traj


def generate_floorplan():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import ndimage

    # 1. Accumulate point clouds
    cloud = accumulate_clouds(MCAP_FILE)

    # 2. Voxel downsample
    cloud_ds = voxel_downsample(cloud, VOXEL_SIZE)

    # 3. Extract trajectory
    traj = extract_trajectory(MCAP_FILE)

    # 4. Save full cloud as PLY
    ply_path = OUT_DIR / "full_cloud.ply"
    print(f"Saving PLY → {ply_path}")
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cloud_ds)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in cloud_ds:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    print(f"  {ply_path.stat().st_size / 1e6:.1f} MB")

    # 5. Wall slice → floor plan
    print(f"\nSlicing walls: z ∈ [{WALL_Z_MIN}, {WALL_Z_MAX}]m...")
    wall_mask = (cloud_ds[:, 2] >= WALL_Z_MIN) & (cloud_ds[:, 2] <= WALL_Z_MAX)
    walls = cloud_ds[wall_mask][:, :2]
    print(f"  {len(walls):,} wall points")

    # 6. Floor slice
    print(f"Slicing floor: z ∈ [{FLOOR_Z_MIN}, {FLOOR_Z_MAX}]m...")
    floor_mask = (cloud_ds[:, 2] >= FLOOR_Z_MIN) & (cloud_ds[:, 2] <= FLOOR_Z_MAX)
    floor = cloud_ds[floor_mask][:, :2]
    print(f"  {len(floor):,} floor points")

    # 7. Build occupancy grids
    print("\nBuilding wall occupancy grid...")
    wall_counts, wall_occ, wall_meta = make_occupancy_grid(walls, GRID_RES, MIN_POINTS_PER_CELL)

    print("Building floor occupancy grid...")
    floor_counts, floor_occ, floor_meta = make_occupancy_grid(floor, GRID_RES, 2)

    # 8. Morphological cleanup on walls
    print("Morphological cleanup...")
    # Close small gaps
    wall_clean = ndimage.binary_closing(wall_occ, structure=np.ones((3, 3)), iterations=2)
    # Remove tiny noise blobs
    wall_clean = ndimage.binary_opening(wall_clean, structure=np.ones((2, 2)), iterations=1)
    wall_clean = wall_clean.astype(np.uint8)

    # 9. Render floor plan
    x_min, y_min, res = wall_meta

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # A: Raw wall density heatmap
    ax = axes[0, 0]
    density = np.log1p(wall_counts).astype(float)
    ax.imshow(density, cmap='hot', origin='lower', aspect='equal')
    ax.set_title('Wall Point Density (log scale)', fontsize=14)
    ax.axis('off')

    # B: Clean floor plan (walls = black, free = white)
    ax = axes[0, 1]
    # Combine: white background, gray for floor, black for walls
    canvas = np.ones_like(wall_clean, dtype=float)  # white
    # Map floor occupancy to same grid
    fx_min, fy_min, fres = floor_meta
    # Just show walls on white
    canvas[wall_clean == 1] = 0.0  # black walls
    ax.imshow(canvas, cmap='gray', origin='lower', aspect='equal', vmin=0, vmax=1)
    # Overlay trajectory
    traj_px_x = (traj[:, 0] - x_min) / res
    traj_px_y = (traj[:, 1] - y_min) / res
    ax.plot(traj_px_x, traj_px_y, 'b-', linewidth=0.5, alpha=0.7, label='Robot path')
    ax.plot(traj_px_x[0], traj_px_y[0], 'go', markersize=8, label='Start')
    ax.plot(traj_px_x[-1], traj_px_y[-1], 'ro', markersize=8, label='End')
    ax.legend(fontsize=10)
    ax.set_title('Floor Plan + Trajectory', fontsize=14)
    ax.axis('off')

    # C: Height-colored top-down view
    ax = axes[1, 0]
    # Use full cloud, project top-down, color by height
    h_min, h_max = -0.1, 2.5
    height_mask = (cloud_ds[:, 2] >= h_min) & (cloud_ds[:, 2] <= h_max)
    hcloud = cloud_ds[height_mask]
    scatter = ax.scatter(hcloud[:, 0], hcloud[:, 1], c=hcloud[:, 2],
                        cmap='viridis', s=0.01, alpha=0.3, vmin=h_min, vmax=h_max)
    ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1, alpha=0.8)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Height (m)', shrink=0.8)
    ax.set_title('Height-Colored Top-Down View', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # D: Cross-section / side view (X-Z)
    ax = axes[1, 1]
    # Sample for rendering speed
    sample_idx = np.random.choice(len(cloud_ds), min(500_000, len(cloud_ds)), replace=False)
    sample = cloud_ds[sample_idx]
    ax.scatter(sample[:, 0], sample[:, 2], c=sample[:, 1], cmap='coolwarm',
              s=0.01, alpha=0.2)
    ax.set_aspect('equal')
    ax.set_title('Side View (X-Z cross section)', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')

    plt.suptitle(f'Unitree Go2 LiDAR Floor Plan — {MCAP_FILE.name}\n'
                 f'{cloud.shape[0]:,} raw points → {cloud_ds.shape[0]:,} downsampled '
                 f'| Grid: {GRID_RES*100:.0f}cm/px | Duration: 4min',
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = OUT_DIR / "floorplan.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Floor plan saved → {png_path}")

    # Also save clean SVG-style floor plan (just walls)
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 16))
    ax2.imshow(canvas, cmap='gray', origin='lower', aspect='equal', vmin=0, vmax=1)
    ax2.plot(traj_px_x, traj_px_y, 'b-', linewidth=0.8, alpha=0.5)

    # Add scale bar
    scale_m = 1.0  # 1 meter
    scale_px = scale_m / res
    bar_x = wall_clean.shape[1] * 0.05
    bar_y = wall_clean.shape[0] * 0.05
    ax2.plot([bar_x, bar_x + scale_px], [bar_y, bar_y], 'k-', linewidth=3)
    ax2.text(bar_x + scale_px/2, bar_y + wall_clean.shape[0]*0.02,
            f'{scale_m:.0f}m', ha='center', fontsize=12, fontweight='bold')

    ax2.set_title('Floor Plan', fontsize=18)
    ax2.axis('off')

    clean_path = OUT_DIR / "floorplan_clean.png"
    plt.savefig(clean_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Clean floor plan saved → {clean_path}")

    # Save data
    npz_path = OUT_DIR / "floorplan_data.npz"
    np.savez_compressed(npz_path,
                       wall_grid=wall_counts,
                       wall_occupied=wall_clean,
                       floor_grid=floor_counts,
                       trajectory=traj,
                       cloud_downsampled=cloud_ds,
                       grid_origin=np.array([x_min, y_min]),
                       grid_resolution=np.array([res]))
    print(f"✅ Data saved → {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    plt.close('all')
    return cloud_ds, wall_clean, traj


if __name__ == "__main__":
    generate_floorplan()
