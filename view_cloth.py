import open3d as o3d
import os
import time

# === Configuration ===
obj_dir = "cloth_simulation_output"  # 布料仿真输出目录
fps = 5  # 播放帧率
view_direction = [0.5, -0.3, -0.8]  # 从右上方观察
up_direction = [0, 1, 0]    # Y-up (Taichi使用Y-up)

def main():
    # === Load and Sort OBJ Files ===
    if not os.path.exists(obj_dir):
        print(f"❌ 目录不存在: {obj_dir}")
        print("请先运行布料仿真来生成OBJ文件!")
        return
    
    obj_files = sorted(
        [f for f in os.listdir(obj_dir) if f.startswith("frame_") and f.endswith(".obj")],
        key=lambda x: int(x.replace("frame_", "").replace(".obj", ""))
    )

    obj_paths = [os.path.join(obj_dir, f) for f in obj_files]

    if not obj_paths:
        print(f"❌ 在 {obj_dir} 中没有找到 frame_*.obj 文件")
        print("请先运行布料仿真!")
        return

    print(f"🎬 找到 {len(obj_paths)} 个动画帧")
    print(f"📁 目录: {os.path.abspath(obj_dir)}")

    # === Open3D Visualization Setup ===
    vis = o3d.visualization.Visualizer()
    vis.create_window("Cloth Animation Player", width=1200, height=800)
    
    # 加载第一帧
    mesh = o3d.io.read_triangle_mesh(obj_paths[0])
    mesh.paint_uniform_color([0.8, 0.4, 0.4])  # 红色布料
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)

    # 添加坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)
    
    # 添加地面参考
    ground = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.005, depth=1.0)
    ground.translate([0.0, -0.1, 0.0])
    ground.paint_uniform_color([0.9, 0.9, 0.9])
    vis.add_geometry(ground)

    # === Set Camera View ===
    view_ctl = vis.get_view_control()
    view_ctl.set_front(view_direction)
    view_ctl.set_up(up_direction)
    view_ctl.set_lookat([0.45, 0.3, 0.45])  # 布料中心
    view_ctl.set_zoom(0.8)

    print("🎮 控制说明:")
    print("  - 鼠标左键拖拽: 旋转视角")
    print("  - 鼠标右键拖拽: 平移视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 按 ESC 或关闭窗口退出")
    print(f"\n▶️ 开始播放动画 (FPS: {fps})...")

    # === Animate Through Frames ===
    frame_time = 1.0 / fps
    loop_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            for i, path in enumerate(obj_paths):
                loop_start = time.time()
                
                # 加载新帧
                new_mesh = o3d.io.read_triangle_mesh(path)
                new_mesh.paint_uniform_color([0.8, 0.4, 0.4])
                new_mesh.compute_vertex_normals()
                
                # 更新网格
                mesh.vertices = new_mesh.vertices
                mesh.triangles = new_mesh.triangles
                mesh.vertex_normals = new_mesh.vertex_normals

                vis.update_geometry(mesh)
                
                if not vis.poll_events():
                    print("\n👋 窗口关闭，退出播放")
                    return
                    
                vis.update_renderer()
                
                # 控制帧率
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)
                
                # 显示进度
                if i % 25 == 0:
                    progress = (i + 1) / len(obj_paths) * 100
                    print(f"📺 播放进度: {progress:.1f}% (帧 {i+1}/{len(obj_paths)})")
            
            loop_count += 1
            total_time = time.time() - start_time
            print(f"🔄 第 {loop_count} 次循环完成 (用时 {total_time:.1f}s)")
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户停止播放")
    finally:
        vis.destroy_window()
        print("👋 播放结束")

if __name__ == "__main__":
    main()
