import open3d as o3d
import os
import time
import cv2
import numpy as np

# === Configuration ===
obj_dir = "cloth_simulation_output_0.005"  # 布料仿真输出目录
fps = 5  # 播放帧率
view_direction = [0.4, -0.3, -0.7]  # 从右上方观察
up_direction = [0, 1, 0]    # Y-up (Taichi使用Y-up)
save_video = True  # 是否保存视频
video_filename = "cloth_animation_0.005_3.mp4"  # 输出视频文件名
video_quality = 30  # 视频质量 (CRF值，越小质量越好)

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
    
    # 只取前50帧
    if len(obj_paths) > 50:
        obj_paths = obj_paths[:50]  # 只取前50帧
        print(f"📏 限制为前 {len(obj_paths)} 帧")

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
    
    # 设置更稳定的光照
    render_opt = vis.get_render_option()
    render_opt.light_on = True
    
    # 尝试设置背景颜色和其他选项，如果出错就跳过
    try:
        render_opt.mesh_show_back_face = True  # 显示背面，避免内侧过暗
        render_opt.background_color = np.asarray([0.05, 0.05, 0.05])  # 深色背景
    except AttributeError:
        print("⚠️ 某些渲染选项在当前Open3D版本中不可用，跳过设置")
    
    # 可选：使用更均匀的着色模式来减少光照变化（如果版本支持）
    try:
        # render_opt.mesh_shade_option = o3d.visualization.MeshShadeOption.FlatShade  # 平面着色
        # render_opt.mesh_color_option = o3d.visualization.MeshColorOption.Color  # 使用统一颜色
        pass  # 暂时注释掉这些选项
    except AttributeError:
        pass
    
    # # 添加地面参考
    # ground = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.005, depth=1.0)
    # ground.translate([0.0, -0.1, 0.0])
    # ground.paint_uniform_color([0.9, 0.9, 0.9])
    # vis.add_geometry(ground)

    # === Set Camera View ===
    view_ctl = vis.get_view_control()
    view_ctl.set_front(view_direction)
    view_ctl.set_up(up_direction)
    view_ctl.set_lookat([0.45, 0.3, 0.45])  # 布料中心
    view_ctl.set_zoom(0.8)

    # === 准备视频录制 ===
    if save_video:
        import cv2
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (1200, 800))
        print(f"🎥 将保存视频到: {video_filename}")

    print("🎮 控制说明:")
    print("  - 鼠标左键拖拽: 旋转视角")
    print("  - 鼠标右键拖拽: 平移视角") 
    print("  - 鼠标滚轮: 缩放")
    print("  - 按 ESC 或关闭窗口退出")
    if save_video:
        print(f"🎥 录制视频: {video_filename} (FPS: {fps})")
    print(f"\n▶️ 开始播放动画...")

    # === Animate Through Frames ===
    frame_time = 1.0 / fps
    loop_count = 0
    
    try:
        # 只录制一次循环
        max_loops = 1 if save_video else float('inf')
        
        while loop_count < max_loops:
            start_time = time.time()
            
            for i, path in enumerate(obj_paths):
                loop_start = time.time()
                
                # 加载新帧
                new_mesh = o3d.io.read_triangle_mesh(path)
                new_mesh.paint_uniform_color([0.8, 0.4, 0.4])
                
                # 更稳定的法线计算：使用面法线而不是顶点法线来减少光照闪烁
                new_mesh.compute_triangle_normals()
                new_mesh.compute_vertex_normals()
                
                # 可选：平滑法线以减少光照变化
                # new_mesh.filter_smooth_simple(number_of_iterations=1)
                # new_mesh.compute_vertex_normals()
                
                # 更新网格
                mesh.vertices = new_mesh.vertices
                mesh.triangles = new_mesh.triangles
                mesh.vertex_normals = new_mesh.vertex_normals

                vis.update_geometry(mesh)
                
                if not vis.poll_events():
                    print("\n👋 窗口关闭，退出播放")
                    if save_video:
                        video_writer.release()
                        print(f"✅ 视频已保存: {video_filename}")
                    return
                    
                vis.update_renderer()
                
                # 截取屏幕并保存到视频
                if save_video:
                    # 捕获屏幕
                    img = vis.capture_screen_float_buffer(False)
                    img = np.asarray(img)
                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Open3D使用RGB，OpenCV使用BGR
                    video_writer.write(img)
                
                # 控制帧率
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)
                
                # 显示进度
                if i % 5 == 0 or save_video:  # 录制视频时显示更频繁的进度
                    progress = (i + 1) / len(obj_paths) * 100
                    status = "🎥 录制" if save_video else "📺 播放"
                    print(f"{status}进度: {progress:.1f}% (帧 {i+1}/{len(obj_paths)})")
            
            loop_count += 1
            total_time = time.time() - start_time
            
            if save_video:
                print(f"✅ 视频录制完成! (用时 {total_time:.1f}s)")
                video_writer.release()
                print(f"📁 视频文件: {os.path.abspath(video_filename)}")
                break
            else:
                print(f"🔄 第 {loop_count} 次循环完成 (用时 {total_time:.1f}s)")
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户停止播放")
        if save_video:
            video_writer.release()
            print(f"✅ 视频已保存: {video_filename}")
    finally:
        vis.destroy_window()
        if save_video:
            print("🎥 视频录制结束")
        else:
            print("👋 播放结束")

if __name__ == "__main__":
    main()
