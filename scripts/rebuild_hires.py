#!/usr/bin/env python3
"""Rebuild floorplan_data.npz at maximum precision (0.5cm voxel).
Then re-run the full model build + ghost wall export pipeline.
"""

import numpy as np
from pathlib import Path
import time

OUT = Path(__file__).parent / "output"

print("=" * 60)
print("REBUILD AT MAX PRECISION")
print("=" * 60)

# ============================================================
# 1. Re-downsample from raw scan cache at 0.5cm
# ============================================================
print("\nLoading raw scan cache (41.5M points)...")
t0 = time.time()
cache = np.load(OUT / "scan_cache.npz", allow_pickle=True)
all_points = cache['all_points']
pose_pos = cache['pose_pos']
print(f"  {all_points.shape[0]:,} raw points loaded in {time.time()-t0:.1f}s")

# Voxel downsample at 1cm (0.5cm is 3.1M points which is slow for wall detection)
# 1cm gives 2.2M — good balance of detail vs speed
VOXEL = 0.01  # 1cm
print(f"\nVoxel downsampling at {VOXEL*100:.1f}cm...")
t0 = time.time()
quantized = np.floor(all_points / VOXEL).astype(np.int32)
_, unique_idx = np.unique(quantized, axis=0, return_index=True)
cloud_hires = all_points[unique_idx].astype(np.float32)
print(f"  {all_points.shape[0]:,} → {cloud_hires.shape[0]:,} points ({time.time()-t0:.1f}s)")

# Trajectory (use full resolution)
traj = pose_pos.astype(np.float32)

# Save updated floorplan_data.npz
print("\nSaving hi-res floorplan_data.npz...")

# Also need wall_occupied and grid data for compatibility
# Recompute occupancy grid
GRID_RES = 0.02
x_min, y_min = cloud_hires[:,0].min() - 0.5, cloud_hires[:,1].min() - 0.5
x_max, y_max = cloud_hires[:,0].max() + 0.5, cloud_hires[:,1].max() + 0.5

wall_mask = (cloud_hires[:,2] >= 0.3) & (cloud_hires[:,2] <= 2.0)
walls = cloud_hires[wall_mask][:,:2]
gw = int(np.ceil((x_max - x_min) / GRID_RES))
gh = int(np.ceil((y_max - y_min) / GRID_RES))
wall_grid = np.zeros((gh, gw), dtype=np.int32)
wix = np.clip(((walls[:,0] - x_min) / GRID_RES).astype(int), 0, gw-1)
wiy = np.clip(((walls[:,1] - y_min) / GRID_RES).astype(int), 0, gh-1)
np.add.at(wall_grid, (wiy, wix), 1)
wall_occupied = (wall_grid >= 3).astype(np.uint8)

np.savez_compressed(OUT / "floorplan_data.npz",
    cloud_downsampled=cloud_hires,
    trajectory=traj,
    wall_occupied=wall_occupied,
    grid_origin=np.array([x_min, y_min]),
    grid_resolution=np.array([GRID_RES]))
print(f"  ✅ floorplan_data.npz ({(OUT / 'floorplan_data.npz').stat().st_size/1e6:.1f} MB)")
print(f"  Cloud: {cloud_hires.shape[0]:,} points at {VOXEL*100:.1f}cm resolution")

print("\nDone! Now run:")
print("  python build_polished_model.py")
print("  python export_ghost_walls.py")
