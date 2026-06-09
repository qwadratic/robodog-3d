#!/usr/bin/env python3
"""Detect ghost walls from point cloud density boundaries.

Heuristic: where the point cloud at wall height has a sharp edge
(points on one side, nothing on the other), there's likely a wall.

Steps:
1. Build 2D occupancy grid from wall-height points
2. Find boundary cells (occupied with unoccupied neighbor)  
3. Filter out boundaries near existing detected walls
4. Trace connected boundary cells into line segments
5. Simplify with RDP algorithm
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree
import json

OUT = Path(__file__).parent / "output"
WEB = Path(__file__).parent.parent / "robodog-3d" / "assets"

print("Loading data...")
data = np.load(OUT / "floorplan_data.npz", allow_pickle=True)
cloud = data['cloud_downsampled']
traj = data['trajectory']

floor_z = float(np.median(cloud[cloud[:,2] < 0.2][:,2]))
ceil_z = float(np.median(cloud[cloud[:,2] > 1.8][:,2]))
print(f"  Floor: {floor_z:.2f}m | Ceiling: {ceil_z:.2f}m")

# ============================================================
# 1. Build wall-height occupancy grid
# ============================================================
RES = 0.10  # 10cm resolution
x_min, y_min = cloud[:,0].min() - 1, cloud[:,1].min() - 1
x_max, y_max = cloud[:,0].max() + 1, cloud[:,1].max() + 1
GW = int(np.ceil((x_max - x_min) / RES))
GH = int(np.ceil((y_max - y_min) / RES))

# All points projected to 2D (any height)
all_grid = np.zeros((GH, GW), dtype=np.int32)
aix = np.clip(((cloud[:,0] - x_min) / RES).astype(int), 0, GW-1)
aiy = np.clip(((cloud[:,1] - y_min) / RES).astype(int), 0, GH-1)
np.add.at(all_grid, (aiy, aix), 1)

# Wall-height points (0.3 to 1.8m)
wall_mask = (cloud[:,2] > floor_z + 0.3) & (cloud[:,2] < ceil_z - 0.2)
wall_pts = cloud[wall_mask][:,:2]
wall_grid = np.zeros((GH, GW), dtype=np.int32)
wix = np.clip(((wall_pts[:,0] - x_min) / RES).astype(int), 0, GW-1)
wiy = np.clip(((wall_pts[:,1] - y_min) / RES).astype(int), 0, GH-1)
np.add.at(wall_grid, (wiy, wix), 1)

# Dense wall regions (many points = actual wall surface)
wall_dense = wall_grid >= 5

# Any-height occupancy (where LiDAR saw SOMETHING)
occupied = all_grid >= 2
# Slight dilation to fill small gaps
occupied_filled = ndimage.binary_dilation(occupied, iterations=1)

# Trajectory coverage — where the robot walked
traj_grid = np.zeros((GH, GW), dtype=bool)
tix = np.clip(((traj[:,0] - x_min) / RES).astype(int), 0, GW-1)
tiy = np.clip(((traj[:,1] - y_min) / RES).astype(int), 0, GH-1)
traj_grid[tiy, tix] = True
traj_near = ndimage.binary_dilation(traj_grid, iterations=20)  # ~2m around trajectory

print(f"  Occupied cells: {occupied.sum()}")
print(f"  Dense wall cells: {wall_dense.sum()}")

# ============================================================
# 2. Find density boundary edges
# ============================================================
print("\nFinding point cloud boundaries...")

# Strategy: find the OUTLINE of the entire point cloud occupancy
# This outline IS where walls exist (point cloud ends = wall surface)
# Then subtract segments that overlap with already-detected real walls

# Use all-height occupancy filled and smoothed
occ_smooth = ndimage.binary_closing(occupied_filled, iterations=2)
occ_smooth = ndimage.binary_fill_holes(occ_smooth)

# Find boundary of the smooth occupancy
boundary = np.zeros((GH, GW), dtype=bool)
for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
    shifted = np.roll(np.roll(occ_smooth, dy, axis=0), dx, axis=1)
    boundary |= occ_smooth & ~shifted

# Keep boundaries near trajectory AND near any scanned points
boundary &= traj_near

print(f"  Total boundary cells: {boundary.sum()}")

# We'll mark which boundary cells are near REAL walls (to exclude later)
wall_dense_dilated = ndimage.binary_dilation(wall_dense, iterations=2)
boundary_no_existing = boundary.copy()  # keep ALL for now, filter segments later
print(f"  Boundary near existing walls: {(boundary & wall_dense_dilated).sum()}")
print(f"  Boundary NOT near existing walls: {(boundary & ~wall_dense_dilated).sum()}")

# ============================================================
# 3. Trace boundary cells into line segments
# ============================================================
print("\nTracing boundary segments...")

# Label ALL boundary components (we filter overlap with real walls per-segment later)
labeled, n_components = ndimage.label(boundary)
print(f"  Connected boundary components: {n_components}")

segments = []
for comp_id in range(1, n_components + 1):
    cells = np.argwhere(labeled == comp_id)  # [y, x] pairs
    if len(cells) < 3:  # need at least 3 cells = 30cm
        continue
    
    # Convert to world coordinates
    world_pts = np.column_stack([
        x_min + cells[:,1] * RES + RES/2,
        y_min + cells[:,0] * RES + RES/2,
    ])
    
    # Fit line via PCA
    center = world_pts.mean(axis=0)
    cov = np.cov(world_pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Check elongation — skip blobby clusters  
    elong = eigvals[1] / (eigvals[0] + 1e-8)
    if elong < 1.5:
        continue
    
    along = eigvecs[:,1]  # main direction
    
    # Project onto line
    projs = (world_pts - center) @ along
    p_min, p_max = projs.min(), projs.max()
    length = p_max - p_min
    
    if length < 0.25:  # min 25cm
        continue
    
    start = center + along * p_min
    end = center + along * p_max
    
    segments.append({
        'start': [round(float(start[0]), 2), round(float(start[1]), 2)],
        'end': [round(float(end[0]), 2), round(float(end[1]), 2)],
        'length': round(float(length), 2),
        'n_cells': len(cells),
    })

# Also add floor-edge boundaries: where floor exists but no points beyond
# Use the floor grid from floorplan_data
floor_mask_pts = (cloud[:,2] < floor_z + 0.15) & (cloud[:,2] > floor_z - 0.10)
floor_pts = cloud[floor_mask_pts][:,:2]
floor_grid = np.zeros((GH, GW), dtype=bool)
fix = np.clip(((floor_pts[:,0] - x_min) / RES).astype(int), 0, GW-1)
fiy = np.clip(((floor_pts[:,1] - y_min) / RES).astype(int), 0, GH-1)
floor_grid[fiy, fix] = True
floor_grid = ndimage.binary_fill_holes(floor_grid)

# Floor boundary
floor_boundary = np.zeros((GH, GW), dtype=bool)
for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
    shifted = np.roll(np.roll(floor_grid, dy, axis=0), dx, axis=1)
    floor_boundary |= floor_grid & ~shifted

# Floor boundary NOT near existing walls
floor_boundary_new = floor_boundary & ~wall_dense_dilated & traj_near

# Trace floor boundary segments
labeled_fb, n_fb = ndimage.label(floor_boundary_new)
for comp_id in range(1, n_fb + 1):
    cells = np.argwhere(labeled_fb == comp_id)
    if len(cells) < 5:
        continue
    
    world_pts = np.column_stack([
        x_min + cells[:,1] * RES + RES/2,
        y_min + cells[:,0] * RES + RES/2,
    ])
    
    center = world_pts.mean(axis=0)
    if len(world_pts) < 3:
        continue
    cov = np.cov(world_pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    if eigvals[1] / (eigvals[0] + 1e-8) < 1.5:
        continue
    
    along = eigvecs[:,1]
    projs = (world_pts - center) @ along
    p_min, p_max = projs.min(), projs.max()
    length = p_max - p_min
    
    if length < 0.4:
        continue
    
    start = center + along * p_min
    end = center + along * p_max
    
    segments.append({
        'start': [round(float(start[0]), 2), round(float(start[1]), 2)],
        'end': [round(float(end[0]), 2), round(float(end[1]), 2)],
        'length': round(float(length), 2),
        'n_cells': len(cells),
    })

# Deduplicate: remove segments that are very close to each other
if len(segments) > 1:
    unique = [segments[0]]
    for seg in segments[1:]:
        s = np.array(seg['start'])
        e = np.array(seg['end'])
        mid = (s + e) / 2
        
        too_close = False
        for existing in unique:
            es = np.array(existing['start'])
            ee = np.array(existing['end'])
            emid = (es + ee) / 2
            if np.linalg.norm(mid - emid) < 0.3:
                too_close = True
                break
        if not too_close:
            unique.append(seg)
    segments = unique

segments.sort(key=lambda s: s['length'], reverse=True)
print(f"\n  Ghost wall segments: {len(segments)}")
for s in segments[:10]:
    print(f"    {s['length']:.1f}m ({s['n_cells']} cells)")

# ============================================================
# 4. Also export collision data for REAL walls only (not furniture)
# ============================================================
print("\nExporting collision walls (real walls only)...")

# Load the wall segments from the polished model build
import open3d as o3d

# Re-detect walls using same logic as build_polished_model.py
normals_arr = np.zeros_like(cloud)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=25))
normals_arr = np.asarray(pcd.normals)
nz = np.abs(normals_arr[:,2])

wall_pts_3d = cloud[(cloud[:,2] > floor_z + 0.3) & (cloud[:,2] < ceil_z - 0.2) & (nz < 0.35)]
pcd_w = o3d.geometry.PointCloud()
pcd_w.points = o3d.utility.Vector3dVector(wall_pts_3d)
w_labels = np.array(pcd_w.cluster_dbscan(eps=0.10, min_points=12, print_progress=False))
n_wc = w_labels.max() + 1 if len(w_labels) > 0 else 0

real_walls = []
for ci in range(n_wc):
    cluster = wall_pts_3d[w_labels == ci]
    if len(cluster) < 15:
        continue
    xy = cluster[:,:2]
    center_2d = xy.mean(axis=0)
    cov_2d = np.cov(xy.T)
    eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)
    
    length_spread = np.sqrt(eigvals_2d[1]) * 4
    thickness_spread = np.sqrt(eigvals_2d[0]) * 4
    elongation_2d = eigvals_2d[1] / (eigvals_2d[0] + 1e-8)
    z_span = cluster[:,2].max() - cluster[:,2].min()
    
    if length_spread < 0.5 or thickness_spread > 0.25 or elongation_2d < 3.0 or z_span < 0.4:
        continue
    
    along_2d = eigvecs_2d[:,1]
    proj = (xy - center_2d) @ along_2d
    a_min, a_max = proj.min(), proj.max()
    
    start = center_2d + along_2d * a_min
    end = center_2d + along_2d * a_max
    
    real_walls.append({
        'start': [round(float(start[0]), 2), round(float(start[1]), 2)],
        'end': [round(float(end[0]), 2), round(float(end[1]), 2)],
        'length': round(float(np.linalg.norm(end - start)), 2),
    })

# Merge nearby collinear
merged = []
used = set()
for i, w1 in enumerate(real_walls):
    if i in used:
        continue
    group = [w1]
    used.add(i)
    s1 = np.array(w1['start']); e1 = np.array(w1['end'])
    d1 = e1 - s1; d1 /= np.linalg.norm(d1) + 1e-8
    
    for j, w2 in enumerate(real_walls):
        if j in used:
            continue
        s2 = np.array(w2['start']); e2 = np.array(w2['end'])
        d2 = e2 - s2; d2 /= np.linalg.norm(d2) + 1e-8
        if abs(np.dot(d1, d2)) < 0.9:
            continue
        mid1 = (s1 + e1) / 2; mid2 = (s2 + e2) / 2
        perp = mid2 - mid1 - np.dot(mid2 - mid1, d1) * d1
        if np.linalg.norm(perp) > 0.2:
            continue
        group.append(w2)
        used.add(j)
    
    all_pts = []
    for g in group:
        all_pts.extend([g['start'], g['end']])
    all_pts = np.array(all_pts)
    center = all_pts.mean(axis=0)
    projs = (all_pts - center) @ d1
    ms = center + d1 * projs.min()
    me = center + d1 * projs.max()
    ml = float(np.linalg.norm(me - ms))
    if ml > 0.3:
        merged.append({
            'start': [round(float(ms[0]), 2), round(float(ms[1]), 2)],
            'end': [round(float(me[0]), 2), round(float(me[1]), 2)],
            'length': round(ml, 2),
        })

print(f"  Real walls: {len(merged)}")

# ============================================================
# 5. Save
# ============================================================
# Ghost walls
ghost_path = WEB / "walls_assumed.json"
with open(ghost_path, 'w') as f:
    json.dump(segments, f, separators=(',',':'))
print(f"\n  ✅ {ghost_path} ({len(segments)} segments, {ghost_path.stat().st_size/1e3:.0f}KB)")

# Collision walls (real + ghost combined for collision, flagged)
collision_walls = []
for w in merged:
    w['type'] = 'real'
    collision_walls.append(w)
for s in segments:
    collision_walls.append({
        'start': s['start'],
        'end': s['end'],
        'length': s['length'],
        'type': 'ghost',
    })

coll_path = WEB / "walls_collision.json"
with open(coll_path, 'w') as f:
    json.dump(collision_walls, f, separators=(',',':'))
print(f"  ✅ {coll_path} ({len(collision_walls)} walls, {coll_path.stat().st_size/1e3:.0f}KB)")

# Also update minimap with ghost walls
print("\n  Updating minimap...")
from PIL import Image
minimap_meta = json.load(open(OUT / "minimap_meta.json"))
mm_img = Image.open(OUT / "minimap.png").convert('RGBA')
mm_arr = np.array(mm_img)

# Draw ghost walls in orange/cyan
MM_RES = minimap_meta['resolution']
mm_ox = minimap_meta['origin_x']
mm_oy = minimap_meta['origin_y']
mm_h, mm_w = mm_arr.shape[:2]

for seg in segments:
    s = np.array(seg['start'])
    e = np.array(seg['end'])
    n_steps = max(2, int(seg['length'] / MM_RES * 2))
    for t in np.linspace(0, 1, n_steps):
        p = s + (e - s) * t
        px = int((p[0] - mm_ox) / MM_RES)
        py = mm_h - 1 - int((p[1] - mm_oy) / MM_RES)  # flipped Y
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = px+dx, py+dy
                if 0 <= nx < mm_w and 0 <= ny < mm_h:
                    mm_arr[ny, nx] = [80, 180, 220, 200]  # cyan for ghost walls

mm_out = Image.fromarray(mm_arr, 'RGBA')
mm_out.save(WEB / "minimap.png")
print(f"  ✅ minimap.png updated with ghost walls")

print("\nDone!")
