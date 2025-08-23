import open3d as o3d
import os
import time

# === Configuration ===
obj_dir = "/home/bingbing/Physics_Simulation/shell/build/output"
fps = 10  # Frames per second
view_direction = [0, 0.5, 0.4]  # View from +X axis
up_direction = [0, 0, 1]    # Z-up

# === Load and Sort OBJ Files ===
obj_files = sorted(
    [f for f in os.listdir(obj_dir) if f.startswith("frame_") and f.endswith(".obj")],
    key=lambda x: int(x.replace("frame_", "").replace(".obj", ""))
)

#obj_paths = [os.path.join(obj_dir, f) for f in obj_files]
obj_paths = [os.path.join(obj_dir, obj_files[i]) for i in range(0, len(obj_files), 1)]

if not obj_paths:
    print("No shell*.obj files found.")
    exit()

# === Open3D Visualization Setup ===
vis = o3d.visualization.Visualizer()
vis.create_window()
mesh = o3d.io.read_triangle_mesh(obj_paths[0])
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

# === Set Camera View ===
view_ctl = vis.get_view_control()
view_ctl.set_front(view_direction)
view_ctl.set_up(up_direction)
view_ctl.set_lookat(mesh.get_center())
view_ctl.set_zoom(0.8)

# === Animate Through Frames ===
for path in obj_paths:
    new_mesh = o3d.io.read_triangle_mesh(path)
    new_mesh.compute_vertex_normals()
    mesh.vertices = new_mesh.vertices
    mesh.triangles = new_mesh.triangles

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.001)

vis.destroy_window()