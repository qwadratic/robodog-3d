#!/usr/bin/env python3
"""Render a publication-quality floor plan from the processed point cloud.

Outputs screenshot.png in the project root.
Run after extract_floorplan.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
import open3d as o3d

_project = Path(__file__).resolve().parent.parent
OUT = _project / 'data' / 'output'
if not OUT.exists():
    raise FileNotFoundError(f'Run extract_floorplan.py first. Expected: {OUT}')

data = np.load(OUT / "floorplan_data.npz", allow_pickle=True)
cloud = data['cloud_downsampled']
traj = data['trajectory']

floor_z = float(np.median(cloud[cloud[:,2] < 0.2][:,2]))
ceil_z = float(np.median(cloud[cloud[:,2] > 1.8][:,2]))
wall_h = ceil_z - floor_z

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=20))
nz = np.abs(np.asarray(pcd.normals)[:,2])

GRID = 0.10
x_min, y_min = cloud[:,0].min()-0.5, cloud[:,1].min()-0.5
x_max, y_max = cloud[:,0].max()+0.5, cloud[:,1].max()+0.5
GW = int(np.ceil((x_max-x_min)/GRID)); GH = int(np.ceil((y_max-y_min)/GRID))

def to_grid(pts):
    return (np.clip(((pts[:,0]-x_min)/GRID).astype(int),0,GW-1),
            np.clip(((pts[:,1]-y_min)/GRID).astype(int),0,GH-1))

# Floor
is_floor = (cloud[:,2] > floor_z-0.10) & (cloud[:,2] < floor_z+0.12) & (nz > 0.6)
floor_grid = np.zeros((GH, GW), dtype=bool)
fix, fiy = to_grid(cloud[is_floor][:,:2])
floor_grid[fiy, fix] = True
floor_grid = ndimage.binary_fill_holes(floor_grid)

# Walls (height-filtered)
is_wall = (~is_floor) & (nz < 0.4) & (cloud[:,2] > floor_z+0.15) & (cloud[:,2] < ceil_z-0.10)
wall_density = np.zeros((GH, GW), dtype=np.int32)
wix, wiy = to_grid(cloud[is_wall][:,:2])
np.add.at(wall_density, (wiy, wix), 1)
wall_candidates = wall_density >= 2

vert_pts = cloud[is_wall]
vert_ix, vert_iy = to_grid(vert_pts[:,:2])
z_min_grid = np.full((GH, GW), np.inf)
z_max_grid = np.full((GH, GW), -np.inf)
np.minimum.at(z_min_grid, (vert_iy, vert_ix), vert_pts[:,2])
np.maximum.at(z_max_grid, (vert_iy, vert_ix), vert_pts[:,2])
height_span = np.where(wall_candidates, z_max_grid - z_min_grid, 0)
wall_cells = wall_candidates & (height_span > wall_h * 0.45)

# Furniture
is_furn = (~is_floor) & (~is_wall) & (nz >= 0.4) & (cloud[:,2] > floor_z+0.08) & (cloud[:,2] < ceil_z-0.30)
furn_density = np.zeros((GH, GW), dtype=np.int32)
furnix, furniy = to_grid(cloud[is_furn][:,:2])
np.add.at(furn_density, (furniy, furnix), 1)
furn_cells = furn_density >= 3

# --- RENDER ---
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='#0c0c14')
ax.set_facecolor('#0c0c14')

canvas = np.full((GH, GW, 4), [12, 12, 20, 255], dtype=np.uint8)
canvas[floor_grid] = [35, 38, 45, 255]
canvas[furn_cells & floor_grid] = [55, 48, 38, 255]
canvas[wall_cells] = [180, 175, 165, 255]

ax.imshow(canvas, origin='lower', extent=[x_min, x_max, y_min, y_max], interpolation='nearest')
ax.plot(traj[:,0], traj[:,1], color='#4488ff', lw=1.5, alpha=0.7, zorder=3)
ax.plot(traj[0,0], traj[0,1], 'o', color='#7ae8b4', markersize=8, zorder=4)
ax.plot(traj[-1,0], traj[-1,1], 's', color='#ff6644', markersize=8, zorder=4)
ax.text(traj[0,0]+0.3, traj[0,1]+0.3, 'START', color='#7ae8b4', fontsize=9, fontweight='bold', zorder=5)
ax.text(traj[-1,0]+0.3, traj[-1,1]+0.3, 'END', color='#ff6644', fontsize=9, fontweight='bold', zorder=5)

ax.plot([-10, -8], [-14.5, -14.5], color='white', lw=2)
ax.text(-9, -15.2, '2m', color='white', fontsize=9, ha='center')

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.axis('off')

ax.text(0.5, 0.98, 'Unitree Go2 — LiDAR Floor Plan Reconstruction',
        transform=ax.transAxes, ha='center', va='top', color='#7ae8b4',
        fontsize=14, fontweight='bold', fontfamily='monospace')
ax.text(0.5, 0.94, f'41.5M points → 2.16M (1cm voxel) → {wall_cells.sum()} walls • {furn_cells.sum()} furniture • 4 min scan',
        transform=ax.transAxes, ha='center', va='top', color='#667788',
        fontsize=9, fontfamily='monospace')

plt.tight_layout(pad=0.5)
out_path = _project / 'screenshot.png'
plt.savefig(out_path, dpi=150, facecolor='#0c0c14', bbox_inches='tight')
print(f'✅ {out_path}')
