#!/usr/bin/env python3
"""Build a clean, presentable 3D model.

Philosophy: simple rules, honest geometry.
- Floor: flat plane at floor_z
- Ceiling: flat plane at ceil_z (inside rooms only)
- Walls: voxelized — every cell with wall data gets a thin surface
- Furniture: thin vertical lines / floor outlines (not boxes)
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree
import open3d as o3d
import json, time

OUT = Path(__file__).parent / "output"
WEB = Path(__file__).parent.parent / "robodog-3d" / "assets"

t0 = time.time()
print("Loading 2.16M point cloud...")
data = np.load(OUT / "floorplan_data.npz", allow_pickle=True)
cloud = data['cloud_downsampled']
traj = data['trajectory']

floor_z = float(np.median(cloud[cloud[:,2] < 0.2][:,2]))
ceil_z = float(np.median(cloud[cloud[:,2] > 1.8][:,2]))
wall_h = ceil_z - floor_z
print(f"  Floor: {floor_z:.2f}m | Ceiling: {ceil_z:.2f}m | Height: {wall_h:.2f}m")

# Normals
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=20))
normals = np.asarray(pcd.normals)
nz = np.abs(normals[:,2])

# ============================================================
# CLASSIFY POINTS (simple rules)
# ============================================================
print("\nClassifying points...")

# Floor: near floor_z, horizontal surface
is_floor = (cloud[:,2] > floor_z - 0.10) & (cloud[:,2] < floor_z + 0.12) & (nz > 0.6)

# Ceiling: near ceil_z
is_ceiling = (cloud[:,2] > ceil_z - 0.25) & (nz > 0.5)

# Wall: NOT floor, NOT ceiling, vertical normal, wall height zone
is_wall = (~is_floor) & (~is_ceiling) & (nz < 0.4) & \
          (cloud[:,2] > floor_z + 0.15) & (cloud[:,2] < ceil_z - 0.10)

# Furniture: above floor, not wall (horizontal-ish normal or short), in the furniture zone
is_furniture = (~is_floor) & (~is_ceiling) & (~is_wall) & \
               (cloud[:,2] > floor_z + 0.08) & (cloud[:,2] < ceil_z - 0.30)

print(f"  Floor: {is_floor.sum():,} | Wall: {is_wall.sum():,} | Ceiling: {is_ceiling.sum():,} | Furniture: {is_furniture.sum():,}")

# ============================================================
# GRIDS
# ============================================================
GRID = 0.10  # 10cm grid for structure
x_min, y_min = cloud[:,0].min() - 0.5, cloud[:,1].min() - 0.5
x_max, y_max = cloud[:,0].max() + 0.5, cloud[:,1].max() + 0.5
GW = int(np.ceil((x_max - x_min) / GRID))
GH = int(np.ceil((y_max - y_min) / GRID))

def to_grid(pts):
    ix = np.clip(((pts[:,0] - x_min) / GRID).astype(int), 0, GW-1)
    iy = np.clip(((pts[:,1] - y_min) / GRID).astype(int), 0, GH-1)
    return ix, iy

# Floor grid
floor_grid = np.zeros((GH, GW), dtype=bool)
fix, fiy = to_grid(cloud[is_floor][:,:2])
floor_grid[fiy, fix] = True
floor_grid = ndimage.binary_fill_holes(floor_grid)
floor_grid = ndimage.binary_closing(floor_grid, iterations=2)

# Trajectory proximity
traj_grid = np.zeros((GH, GW), dtype=bool)
tix, tiy = to_grid(traj[:,:2])
traj_grid[tiy, tix] = True
traj_near = ndimage.binary_dilation(traj_grid, iterations=25)
floor_valid = floor_grid & traj_near

# Wall density grid (how many wall points per cell)
wall_density = np.zeros((GH, GW), dtype=np.int32)
wix, wiy = to_grid(cloud[is_wall][:,:2])
np.add.at(wall_density, (wiy, wix), 1)
wall_cells = wall_density >= 3  # at least 3 points = real wall

# Furniture grid
furn_density = np.zeros((GH, GW), dtype=np.int32)
furnix, furniy = to_grid(cloud[is_furniture][:,:2])
np.add.at(furn_density, (furniy, furnix), 1)
furn_cells = furn_density >= 3

# Enclosed rooms (for ceiling)
free = floor_valid & ~wall_cells
labeled, n_regions = ndimage.label(free)
enclosed = np.zeros((GH, GW), dtype=bool)
for rid in range(1, n_regions + 1):
    region = labeled == rid
    if region.sum() < 20:
        continue
    dilated = ndimage.binary_dilation(region, iterations=2)
    boundary = dilated & ~region
    wall_ratio = (boundary & wall_cells).sum() / (boundary.sum() + 1)
    if wall_ratio > 0.2:
        enclosed[region] = True
enclosed &= traj_near

print(f"  Floor cells: {floor_valid.sum()} | Wall cells: {wall_cells.sum()} | Furniture cells: {furn_cells.sum()}")
print(f"  Enclosed (ceiling): {enclosed.sum()} ({enclosed.sum() * GRID * GRID:.1f} m²)")

# ============================================================
# BUILD MESHES
# ============================================================
all_verts = []
all_tris = []
all_colors = []
v_offset = 0

def add_mesh(v, t, c):
    global v_offset
    all_verts.append(v); all_tris.append(t + v_offset); all_colors.append(c)
    v_offset += len(v)

# --- FLOOR ---
print("\nBuilding floor...")
FLOOR_TILE = 0.12
np.random.seed(42)
fv, ft, fc = [], [], []
vi = 0
for gy in range(GH):
    for gx in range(GW):
        if not floor_valid[gy, gx]:
            continue
        x0 = x_min + gx * GRID
        y0 = y_min + gy * GRID
        fv.extend([[x0,y0,floor_z],[x0+GRID,y0,floor_z],[x0+GRID,y0+GRID,floor_z],[x0,y0+GRID,floor_z]])
        ft.extend([[vi,vi+1,vi+2],[vi,vi+2,vi+3]])
        # Subtle warm concrete — slight variation
        base = 0.22 + 0.015 * np.sin(gx*0.7 + gy*1.1) + np.random.uniform(-0.008, 0.008)
        c = [base + 0.02, base, base - 0.02]
        # Darken near walls
        if wall_cells[gy, gx] or (gy > 0 and wall_cells[gy-1, gx]) or \
           (gy < GH-1 and wall_cells[gy+1, gx]) or (gx > 0 and wall_cells[gy, gx-1]) or \
           (gx < GW-1 and wall_cells[gy, gx+1]):
            c = [c[0]*0.7, c[1]*0.7, c[2]*0.7]
        fc.extend([c,c,c,c])
        vi += 4
if fv:
    add_mesh(np.array(fv), np.array(ft), np.array(fc))
print(f"  {len(ft)} triangles")

# --- CEILING ---
print("Building ceiling...")
cv, ct, cc = [], [], []
vi = 0
for gy in range(GH):
    for gx in range(GW):
        if not enclosed[gy, gx]:
            continue
        x0 = x_min + gx * GRID
        y0 = y_min + gy * GRID
        cv.extend([[x0,y0,ceil_z],[x0+GRID,y0,ceil_z],[x0+GRID,y0+GRID,ceil_z],[x0,y0+GRID,ceil_z]])
        ct.extend([[vi,vi+2,vi+1],[vi,vi+3,vi+2]])
        c = [0.30, 0.29, 0.28]
        cc.extend([c,c,c,c])
        vi += 4
if cv:
    add_mesh(np.array(cv), np.array(ct), np.array(cc))
print(f"  {len(ct)} triangles")

# --- WALLS (voxelized) ---
print("Building walls (every occupied cell → surface)...")
wv, wt, wc = [], [], []
vi = 0
wall_color_base = np.array([0.55, 0.53, 0.50])

# For each wall cell, place a thin vertical quad facing the open side
for gy in range(GH):
    for gx in range(GW):
        if not wall_cells[gy, gx]:
            continue
        x0 = x_min + gx * GRID
        y0 = y_min + gy * GRID
        xc = x0 + GRID/2
        yc = y0 + GRID/2

        # Check which neighbors are NOT walls → face that direction
        faces_to_build = []
        if gx == 0 or not wall_cells[gy, gx-1]:
            faces_to_build.append('west')
        if gx == GW-1 or not wall_cells[gy, gx+1]:
            faces_to_build.append('east')
        if gy == 0 or not wall_cells[gy-1, gx]:
            faces_to_build.append('south')
        if gy == GH-1 or not wall_cells[gy+1, gx]:
            faces_to_build.append('north')

        if not faces_to_build:
            continue  # interior wall cell, no visible face

        # Color variation
        noise = 0.02 * np.sin(gx * 1.3 + gy * 0.9)
        col = wall_color_base + noise
        # Darken at floor/ceiling junctions
        col_bottom = col * 0.75
        col_top = col * 0.90

        for face in faces_to_build:
            if face == 'west':
                p = [[x0,y0,floor_z],[x0,y0+GRID,floor_z],[x0,y0+GRID,ceil_z],[x0,y0,ceil_z]]
            elif face == 'east':
                p = [[x0+GRID,y0,floor_z],[x0+GRID,y0+GRID,floor_z],[x0+GRID,y0+GRID,ceil_z],[x0+GRID,y0,ceil_z]]
            elif face == 'south':
                p = [[x0,y0,floor_z],[x0+GRID,y0,floor_z],[x0+GRID,y0,ceil_z],[x0,y0,ceil_z]]
            elif face == 'north':
                p = [[x0,y0+GRID,floor_z],[x0+GRID,y0+GRID,floor_z],[x0+GRID,y0+GRID,ceil_z],[x0,y0+GRID,ceil_z]]

            wv.extend(p)
            wt.extend([[vi,vi+1,vi+2],[vi,vi+2,vi+3]])
            wc.extend([col_bottom.tolist(), col_bottom.tolist(), col_top.tolist(), col_top.tolist()])
            vi += 4

if wv:
    add_mesh(np.array(wv), np.array(wt), np.array(wc))
print(f"  {wall_cells.sum()} wall cells → {len(wt)} triangles")

# --- FURNITURE (schematic: thin vertical lines + floor shadow) ---
print("Building furniture (schematic)...")
furn_pts = cloud[is_furniture]
if len(furn_pts) > 20:
    pcd_f = o3d.geometry.PointCloud()
    pcd_f.points = o3d.utility.Vector3dVector(furn_pts)
    f_labels = np.array(pcd_f.cluster_dbscan(eps=0.12, min_points=10, print_progress=False))
    n_fc = f_labels.max() + 1 if len(f_labels) > 0 else 0

    furn_palette = [
        [0.35, 0.25, 0.15], [0.25, 0.32, 0.22], [0.30, 0.28, 0.35],
        [0.38, 0.30, 0.18], [0.35, 0.22, 0.22], [0.28, 0.32, 0.32],
    ]

    n_furniture = 0
    fur_v, fur_t, fur_c = [], [], []
    fvi = 0
    for ci in range(n_fc):
        cluster = furn_pts[f_labels == ci]
        if len(cluster) < 8:
            continue
        bmin = cluster.min(axis=0)
        bmax = cluster.max(axis=0)
        dims = bmax - bmin
        if dims[0] * dims[1] < 0.01:
            continue

        col = np.array(furn_palette[n_furniture % len(furn_palette)])
        h = bmax[2]  # top of furniture
        cx, cy = (bmin[0]+bmax[0])/2, (bmin[1]+bmax[1])/2

        # Floor shadow: dark rectangle on floor
        pad = 0.02
        x0, y0 = bmin[0]-pad, bmin[1]-pad
        x1, y1 = bmax[0]+pad, bmax[1]+pad
        shadow_col = [0.12, 0.11, 0.10]
        fur_v.extend([[x0,y0,floor_z+0.002],[x1,y0,floor_z+0.002],
                      [x1,y1,floor_z+0.002],[x0,y1,floor_z+0.002]])
        fur_t.extend([[fvi,fvi+1,fvi+2],[fvi,fvi+2,fvi+3]])
        fur_c.extend([shadow_col]*4)
        fvi += 4

        # Thin vertical edges at corners (wireframe feel)
        edge_w = 0.015
        for ex, ey in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:
            fur_v.extend([
                [ex-edge_w, ey-edge_w, floor_z],
                [ex+edge_w, ey+edge_w, floor_z],
                [ex+edge_w, ey+edge_w, h],
                [ex-edge_w, ey-edge_w, h],
            ])
            ecol = (col * 0.6).tolist()
            ecol_top = (col * 0.8).tolist()
            fur_t.extend([[fvi,fvi+1,fvi+2],[fvi,fvi+2,fvi+3]])
            fur_c.extend([ecol, ecol, ecol_top, ecol_top])
            fvi += 4

        # Top surface (thin flat)
        top_col = col.tolist()
        fur_v.extend([[x0,y0,h],[x1,y0,h],[x1,y1,h],[x0,y1,h]])
        fur_t.extend([[fvi,fvi+1,fvi+2],[fvi,fvi+2,fvi+3]])
        fur_c.extend([top_col]*4)
        fvi += 4

        n_furniture += 1

    if fur_v:
        add_mesh(np.array(fur_v), np.array(fur_t), np.array(fur_c))
    print(f"  {n_furniture} objects (shadows + edges + top)")
else:
    n_furniture = 0
    print("  No furniture detected")

# ============================================================
# COMBINE + SAVE
# ============================================================
print("\nCombining...")
combined_v = np.vstack(all_verts)
combined_t = np.vstack(all_tris).astype(np.int32)
combined_c = np.vstack(all_colors)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(combined_v)
mesh.triangles = o3d.utility.Vector3iVector(combined_t)
mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(combined_c, 0, 1))
mesh.compute_vertex_normals()
print(f"  {len(combined_v):,} vertices, {len(combined_t):,} triangles")

# Save
for path in ["game_model.ply", "game_model.glb"]:
    out = OUT / path
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"  ✅ {out} ({out.stat().st_size/1e6:.1f} MB)")

# Copy to web
import shutil
shutil.copy(OUT / "game_model.glb", WEB / "model.glb")
print(f"  ✅ Copied to web assets")

# ============================================================
# MINIMAP
# ============================================================
print("\nGenerating minimap...")
from PIL import Image
MM_RES = 0.05
mm_w = int((x_max - x_min) / MM_RES)
mm_h = int((y_max - y_min) / MM_RES)
mm = np.zeros((mm_h, mm_w, 4), dtype=np.uint8)

# Floor
from scipy.ndimage import zoom
floor_hr = zoom(floor_valid.astype(float), MM_RES / GRID, order=0) > 0.5
fh, fw = min(floor_hr.shape[0], mm_h), min(floor_hr.shape[1], mm_w)
mm[:fh, :fw][floor_hr[:fh, :fw]] = [35, 37, 42, 200]

# Walls
wall_hr = zoom(wall_cells.astype(float), MM_RES / GRID, order=0) > 0.5
mm[:fh, :fw][wall_hr[:fh, :fw]] = [220, 215, 200, 255]

# Furniture shadows
furn_hr = zoom(furn_cells.astype(float), MM_RES / GRID, order=0) > 0.5
mm[:fh, :fw][furn_hr[:fh, :fw]] = [100, 85, 65, 180]

# Crop
rows = np.any(mm[:,:,3] > 0, axis=1)
cols = np.any(mm[:,:,3] > 0, axis=0)
if rows.any() and cols.any():
    r0, r1 = np.where(rows)[0][[0,-1]]
    c0, c1 = np.where(cols)[0][[0,-1]]
    pad = 8
    r0, r1 = max(0,r0-pad), min(mm_h-1,r1+pad)
    c0, c1 = max(0,c0-pad), min(mm_w-1,c1+pad)
    mm_crop = mm[r0:r1+1, c0:c1+1]
    crop_ox = x_min + c0 * MM_RES
    crop_oy = y_min + r0 * MM_RES
else:
    mm_crop = mm; crop_ox = x_min; crop_oy = y_min

img = Image.fromarray(np.flipud(mm_crop), 'RGBA')
img.save(WEB / "minimap.png")
with open(WEB / "minimap_meta.json", 'w') as f:
    json.dump({"origin_x": float(crop_ox), "origin_y": float(crop_oy), "resolution": MM_RES,
               "width": int(mm_crop.shape[1]), "height": int(mm_crop.shape[0]),
               "floor_z": float(floor_z), "ceiling_z": float(ceil_z)}, f)
print(f"  ✅ minimap.png ({mm_crop.shape[1]}x{mm_crop.shape[0]})")

# Wall collision + ghost walls
print("\nExporting collision data...")
# Wall collision: use wall_cells grid edges as segments
collision_walls = []
for gy in range(GH):
    for gx in range(GW):
        if not wall_cells[gy, gx]:
            continue
        x0 = float(x_min + gx * GRID)
        y0 = float(y_min + gy * GRID)
        g = float(GRID)
        if gx == 0 or not wall_cells[gy, gx-1]:
            collision_walls.append({'start':[round(x0,2),round(y0,2)],'end':[round(x0,2),round(y0+g,2)],'type':'real'})
        if gx==GW-1 or not wall_cells[gy,gx+1]:
            collision_walls.append({'start':[round(x0+g,2),round(y0,2)],'end':[round(x0+g,2),round(y0+g,2)],'type':'real'})
        if gy==0 or not wall_cells[gy-1,gx]:
            collision_walls.append({'start':[round(x0,2),round(y0,2)],'end':[round(x0+g,2),round(y0,2)],'type':'real'})
        if gy==GH-1 or not wall_cells[gy+1,gx]:
            collision_walls.append({'start':[round(x0,2),round(y0+g,2)],'end':[round(x0+g,2),round(y0+g,2)],'type':'real'})

# Deduplicate (many shared edges)
seen = set()
unique_walls = []
for w in collision_walls:
    key = (tuple(w['start']), tuple(w['end']))
    rkey = (tuple(w['end']), tuple(w['start']))
    if key not in seen and rkey not in seen:
        seen.add(key)
        unique_walls.append(w)

with open(WEB / "walls_collision.json", 'w') as f:
    json.dump(unique_walls, f, separators=(',',':'))
print(f"  ✅ {len(unique_walls)} collision segments")

# Ghost walls (keep existing)
print(f"\n✅ Done in {time.time()-t0:.0f}s")
print(f"  Model: {len(combined_v):,} verts, {len(combined_t):,} tris")
print(f"  Walls: {wall_cells.sum()} cells | Furniture: {n_furniture} objects")
