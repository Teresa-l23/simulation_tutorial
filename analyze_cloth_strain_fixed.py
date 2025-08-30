#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
布料拉伸分析脚本
统计每一帧的三角形面积、边长变化等，验证不可拉伸特性
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

def load_obj_mesh(filename):
    """加载OBJ文件并返回顶点和三角形数据"""
    vertices = []
    triangles = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                # OBJ文件索引从1开始，转换为从0开始
                triangles.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
    
    return np.array(vertices), np.array(triangles)

def compute_triangle_area(v1, v2, v3):
    """计算三角形面积"""
    # 使用叉积计算面积
    edge1 = v2 - v1
    edge2 = v3 - v1
    cross_product = np.cross(edge1, edge2)
    # 对于3D向量的叉积，需要计算向量的模长
    if cross_product.ndim == 1 and len(cross_product) == 3:
        area = 0.5 * np.linalg.norm(cross_product)
    else:
        # 如果是2D，直接使用标量结果
        area = 0.5 * abs(cross_product)
    return area

def compute_edge_length(v1, v2):
    """计算边长"""
    return np.linalg.norm(v2 - v1)

def analyze_triangle_areas(vertices, triangles):
    """分析三角形面积分布"""
    areas = []
    for tri in triangles:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        area = compute_triangle_area(v1, v2, v3)
        areas.append(area)
    
    areas = np.array(areas)
    return {
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'total_area': np.sum(areas),
        'areas': areas
    }

def analyze_edge_lengths(vertices, triangles, n_u=30, n_v=30):
    """分析边长分布 - 只分析网格边（水平和垂直），不包含对角线"""
    grid_edge_lengths = []
    
    # 分析水平边 (u方向)
    for v in range(n_v):
        for u in range(n_u - 1):
            i = v * n_u + u
            j = v * n_u + (u + 1)
            
            if i < len(vertices) and j < len(vertices):
                length = compute_edge_length(vertices[i], vertices[j])
                grid_edge_lengths.append(length)
    
    # 分析垂直边 (v方向)
    for v in range(n_v - 1):
        for u in range(n_u):
            i = v * n_u + u
            j = (v + 1) * n_u + u
            
            if i < len(vertices) and j < len(vertices):
                length = compute_edge_length(vertices[i], vertices[j])
                grid_edge_lengths.append(length)
    
    grid_edge_lengths = np.array(grid_edge_lengths)
    return {
        'min_length': np.min(grid_edge_lengths),
        'max_length': np.max(grid_edge_lengths),
        'mean_length': np.mean(grid_edge_lengths),
        'std_length': np.std(grid_edge_lengths),
        'num_edges': len(grid_edge_lengths),
        'lengths': grid_edge_lengths
    }

def compute_strain_tensor(vertices, triangles, initial_vertices, debug=False):
    """计算应变张量的主应变"""
    strains = []
    debug_info = {'large_strains': [], 'edge_ratios': []}
    
    for tri_idx, tri in enumerate(triangles):
        # 当前三角形顶点
        v0_curr = vertices[tri[0]]
        v1_curr = vertices[tri[1]] 
        v2_curr = vertices[tri[2]]
        
        # 初始三角形顶点
        v0_init = initial_vertices[tri[0]]
        v1_init = initial_vertices[tri[1]]
        v2_init = initial_vertices[tri[2]]
        
        # 计算初始和当前的边向量
        e1_init = v1_init - v0_init  # edge from v0 to v1
        e2_init = v2_init - v0_init  # edge from v0 to v2
        e3_init = v2_init - v1_init  # edge from v1 to v2
        
        e1_curr = v1_curr - v0_curr
        e2_curr = v2_curr - v0_curr  
        e3_curr = v2_curr - v1_curr
        
        # 计算所有三条边的应变
        try:
            edges_init = [e1_init, e2_init, e3_init]
            edges_curr = [e1_curr, e2_curr, e3_curr]
            edge_strains = []
            
            for e_init, e_curr in zip(edges_init, edges_curr):
                l_init = np.linalg.norm(e_init)
                l_curr = np.linalg.norm(e_curr)
                
                if l_init > 1e-10:
                    strain = (l_curr - l_init) / l_init
                    edge_strains.append(abs(strain))
                    
                    if debug and abs(strain) > 0.05:  # 大于5%的应变
                        debug_info['large_strains'].append({
                            'triangle': tri_idx,
                            'strain': strain,
                            'l_init': l_init,
                            'l_curr': l_curr,
                            'vertices': tri
                        })
                    
                    debug_info['edge_ratios'].append(l_curr / l_init)
                else:
                    edge_strains.append(0.0)
            
            if edge_strains:
                max_strain = max(edge_strains)
                min_strain = min(edge_strains)
                strains.append((max_strain, min_strain))
            else:
                strains.append((0.0, 0.0))
                
        except Exception as e:
            if debug:
                print(f"Error computing strain for triangle {tri_idx}: {e}")
            strains.append((0.0, 0.0))
    
    strains = np.array(strains)
    max_strains = strains[:, 0]
    min_strains = strains[:, 1]
    
    result = {
        'max_principal_strain': np.max(max_strains),
        'mean_max_strain': np.mean(max_strains),
        'std_max_strain': np.std(max_strains),
        'max_strains': max_strains,
        'min_strains': min_strains
    }
    
    if debug:
        result['debug_info'] = debug_info
        print(f"Debug: Found {len(debug_info['large_strains'])} triangles with strain > 5%")
        if debug_info['edge_ratios']:
            ratios = np.array(debug_info['edge_ratios'])
            print(f"Edge length ratios: min={np.min(ratios):.4f}, max={np.max(ratios):.4f}, mean={np.mean(ratios):.4f}")
    
    return result

def analyze_constraint_violations(vertices, initial_vertices, n_u=30, n_v=30, dx=0.03):
    """分析约束违反情况 - 检查网格边长是否保持恒定"""
    violations = []
    
    # 分析水平边 (u方向)
    for v in range(n_v):
        for u in range(n_u - 1):
            i = v * n_u + u
            j = v * n_u + (u + 1)
            
            if i < len(vertices) and j < len(vertices):
                # 当前边长
                curr_len = np.linalg.norm(vertices[j] - vertices[i])
                # 初始边长
                init_len = np.linalg.norm(initial_vertices[j] - initial_vertices[i])
                # 约束违反程度
                violation = abs(curr_len - init_len) / init_len
                violations.append(violation)
    
    # 分析垂直边 (v方向) 
    for v in range(n_v - 1):
        for u in range(n_u):
            i = v * n_u + u
            j = (v + 1) * n_u + u
            
            if i < len(vertices) and j < len(vertices):
                curr_len = np.linalg.norm(vertices[j] - vertices[i])
                init_len = np.linalg.norm(initial_vertices[j] - initial_vertices[i])
                violation = abs(curr_len - init_len) / init_len
                violations.append(violation)
    
    violations = np.array(violations)
    return {
        'max_violation': np.max(violations),
        'mean_violation': np.mean(violations),
        'std_violation': np.std(violations),
        'violations': violations
    }

def analyze_cloth_simulation(output_dir, max_frames=None):
    """分析整个布料仿真的拉伸情况"""
    
    if not os.path.exists(output_dir):
        print(f"❌ Directory does not exist: {output_dir}")
        return None
    
    # 获取所有OBJ文件
    obj_files = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".obj")],
                      key=lambda x: int(x.replace("frame_", "").replace(".obj", "")))
    
    if max_frames:
        obj_files = obj_files[:max_frames]
    
    if not obj_files:
        print(f"❌ No OBJ files found in {output_dir}")
        return None
    
    print(f"🔍 Analyzing {len(obj_files)} frames...")
    
    # 存储分析结果
    results = {
        'frames': [],
        'triangle_areas': [],
        'edge_lengths': [],
        'strain_analysis': [],
        'constraint_violations': [],
        'frame_numbers': []
    }
    
    initial_vertices = None
    
    for i, obj_file in enumerate(obj_files):
        filepath = os.path.join(output_dir, obj_file)
        frame_num = int(obj_file.replace("frame_", "").replace(".obj", ""))
        
        try:
            vertices, triangles = load_obj_mesh(filepath)
            
            if initial_vertices is None:
                initial_vertices = vertices.copy()
                print(f"📐 Initial mesh: {len(vertices)} vertices, {len(triangles)} triangles")
            
            # 分析三角形面积
            area_stats = analyze_triangle_areas(vertices, triangles)
            
            # 分析边长 (只分析网格边，不包含对角线)
            edge_stats = analyze_edge_lengths(vertices, triangles, n_u=30, n_v=30)
            
            # 分析应变 (在第一帧启用调试)
            debug_mode = (i == 0)
            strain_stats = compute_strain_tensor(vertices, triangles, initial_vertices, debug=debug_mode)
            
            # 分析约束违反
            constraint_stats = analyze_constraint_violations(vertices, initial_vertices)
            
            results['frames'].append(i)
            results['frame_numbers'].append(frame_num)
            results['triangle_areas'].append(area_stats)
            results['edge_lengths'].append(edge_stats)
            results['strain_analysis'].append(strain_stats)
            results['constraint_violations'].append(constraint_stats)
            
            if i % 10 == 0:
                print(f"📊 Processed frame {i+1}/{len(obj_files)}")
            
        except Exception as e:
            print(f"⚠️ Error processing file {obj_file}: {e}")
            continue
    
    return results

def plot_strain_analysis(results, output_dir):
    """绘制应变分析图表 - 分开保存四张图"""
    if not results or len(results['frames']) == 0:
        print("❌ No data to plot")
        return
    
    frames = np.array(results['frames'])
    
    # 提取数据
    min_areas = [stats['min_area'] for stats in results['triangle_areas']]
    max_areas = [stats['max_area'] for stats in results['triangle_areas']]
    mean_areas = [stats['mean_area'] for stats in results['triangle_areas']]
    
    min_lengths = [stats['min_length'] for stats in results['edge_lengths']]
    max_lengths = [stats['max_length'] for stats in results['edge_lengths']]
    mean_lengths = [stats['mean_length'] for stats in results['edge_lengths']]
    
    max_strains = [stats['max_principal_strain'] for stats in results['strain_analysis']]
    mean_max_strains = [stats['mean_max_strain'] for stats in results['strain_analysis']]
    
    max_violations = [stats['max_violation'] for stats in results['constraint_violations']]
    mean_violations = [stats['mean_violation'] for stats in results['constraint_violations']]
    
    # 计算理论值 (基于30x30网格，dx=0.03的默认值)
    n_u, n_v = 30, 30
    dx = 0.03
    
    # 理论三角形面积 (直角三角形，边长为dx)
    theoretical_triangle_area = 0.5 * dx * dx
    
    # 理论边长 - 只有网格边有约束
    theoretical_grid_edge = dx  # 网格边（水平和垂直边）
    
    # 图1: 三角形面积统计
    plt.figure(figsize=(10, 6))
    plt.plot(frames, min_areas, 'b-', label='Min Area', alpha=0.7, linewidth=2)
    plt.plot(frames, max_areas, 'r-', label='Max Area', alpha=0.7, linewidth=2)
    plt.plot(frames, mean_areas, 'g-', label='Mean Area', linewidth=3)
    
    # 添加理论面积参考线
    plt.axhline(y=theoretical_triangle_area, color='black', linestyle='--', 
                alpha=0.8, linewidth=2, label=f'Theoretical Area ({theoretical_triangle_area:.6f})')
    
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Area', fontsize=12)
    plt.title(f'Triangle Area Statistics - {output_dir}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_filename = f"{output_dir}_triangle_areas.png"
    plt.savefig(plot1_filename, dpi=150, bbox_inches='tight')
    print(f"📊 Chart 1 saved: {plot1_filename}")
    plt.close()
    
    # 图2: 边长统计
    plt.figure(figsize=(10, 6))
    plt.plot(frames, min_lengths, 'b-', label='Min Length', alpha=0.7, linewidth=2)
    plt.plot(frames, max_lengths, 'r-', label='Max Length', alpha=0.7, linewidth=2)
    plt.plot(frames, mean_lengths, 'g-', label='Mean Length', linewidth=3)
    
    # 只添加网格边的理论参考线（对角线无约束，不应该有参考线）
    plt.axhline(y=theoretical_grid_edge, color='black', linestyle='--', 
                alpha=0.8, linewidth=2, label=f'Theoretical Grid Edge ({theoretical_grid_edge:.3f})')
    
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Edge Length', fontsize=12)
    plt.title(f'Edge Length Statistics - {output_dir}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_filename = f"{output_dir}_edge_lengths.png"
    plt.savefig(plot2_filename, dpi=150, bbox_inches='tight')
    print(f"📊 Chart 2 saved: {plot2_filename}")
    plt.close()
    
    # 图3: 应变分析 - 使用双Y轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 主y轴显示最大应变
    color1 = 'tab:red'
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Max Principal Strain', color=color1, fontsize=12)
    line1 = ax1.plot(frames, max_strains, color=color1, linewidth=3, label='Max Principal Strain')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 次y轴显示平均最大应变
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Mean Max Strain', color=color2, fontsize=12)
    line2 = ax2.plot(frames, mean_max_strains, color=color2, linewidth=3, label='Mean Max Strain')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)
    
    ax1.set_title(f'Strain Analysis - {output_dir}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_filename = f"{output_dir}_strain_analysis.png"
    plt.savefig(plot3_filename, dpi=150, bbox_inches='tight')
    print(f"📊 Chart 3 saved: {plot3_filename}")
    plt.close()
    
    # 图4: 约束违反分析
    plt.figure(figsize=(10, 6))
    plt.plot(frames, np.array(max_violations) * 100, 'r-', label='Max Constraint Violation', linewidth=3)
    plt.plot(frames, np.array(mean_violations) * 100, 'orange', label='Mean Constraint Violation', linewidth=3)
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='1% threshold')
    plt.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Constraint Violation (%)', fontsize=12)
    plt.title(f'Constraint Violation Analysis - {output_dir}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot4_filename = f"{output_dir}_constraint_violations.png"
    plt.savefig(plot4_filename, dpi=150, bbox_inches='tight')
    print(f"📊 Chart 4 saved: {plot4_filename}")
    plt.close()
    
    print(f"📊 All 4 charts saved separately for {output_dir}")

def save_analysis_report(results, output_dir):
    """保存详细的分析报告"""
    if not results:
        return
    
    report_filename = f"{output_dir}_analysis_report.json"
    
    # 转换numpy数组为列表以便JSON序列化
    json_results = {
        'frames': results['frames'],
        'frame_numbers': results['frame_numbers'],
        'summary': {},
        'detailed_stats': []
    }
    
    # 计算总体统计
    if results['triangle_areas']:
        all_mean_areas = [stats['mean_area'] for stats in results['triangle_areas']]
        all_max_strains = [stats['max_principal_strain'] for stats in results['strain_analysis']]
        
        json_results['summary'] = {
            'total_frames': len(results['frames']),
            'area_stats': {
                'initial_mean_area': all_mean_areas[0] if all_mean_areas else 0,
                'final_mean_area': all_mean_areas[-1] if all_mean_areas else 0,
                'max_area_change': max(all_mean_areas) - min(all_mean_areas) if all_mean_areas else 0,
                'area_stability': np.std(all_mean_areas) if all_mean_areas else 0
            },
            'strain_stats': {
                'max_strain_overall': max(all_max_strains) if all_max_strains else 0,
                'mean_strain': np.mean(all_max_strains) if all_max_strains else 0,
                'strain_std': np.std(all_max_strains) if all_max_strains else 0
            }
        }
    
    # 保存每帧的详细统计
    for i, frame_num in enumerate(results['frame_numbers']):
        if i < len(results['triangle_areas']):
            area_stats = results['triangle_areas'][i]
            edge_stats = results['edge_lengths'][i]
            strain_stats = results['strain_analysis'][i]
            
            frame_data = {
                'frame_number': frame_num,
                'area_min': float(area_stats['min_area']),
                'area_max': float(area_stats['max_area']),
                'area_mean': float(area_stats['mean_area']),
                'area_std': float(area_stats['std_area']),
                'edge_min': float(edge_stats['min_length']),
                'edge_max': float(edge_stats['max_length']),
                'edge_mean': float(edge_stats['mean_length']),
                'edge_std': float(edge_stats['std_length']),
                'max_principal_strain': float(strain_stats['max_principal_strain']),
                'mean_max_strain': float(strain_stats['mean_max_strain'])
            }
            json_results['detailed_stats'].append(frame_data)
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed report saved: {report_filename}")

def print_summary(results, output_dir):
    """打印分析摘要"""
    if not results or not results['triangle_areas']:
        print("❌ No data to summarize")
        return
    
    print(f"\n🎯 Cloth Stretch Analysis Summary - {output_dir}")
    print("=" * 60)
    
    # 面积分析
    all_mean_areas = [stats['mean_area'] for stats in results['triangle_areas']]
    initial_area = all_mean_areas[0]
    final_area = all_mean_areas[-1]
    area_change = (final_area - initial_area) / initial_area * 100
    area_stability = np.std(all_mean_areas) / np.mean(all_mean_areas) * 100
    
    print(f"📐 Area Analysis:")
    print(f"   Initial mean area: {initial_area:.6f}")
    print(f"   Final mean area: {final_area:.6f}")
    print(f"   Relative change: {area_change:+.3f}%")
    print(f"   Stability (CV): {area_stability:.3f}%")
    
    # 应变分析  
    all_max_strains = [stats['max_principal_strain'] for stats in results['strain_analysis']]
    all_mean_strains = [stats['mean_max_strain'] for stats in results['strain_analysis']]
    max_strain = max(all_max_strains)
    mean_strain = np.mean(all_max_strains)
    max_mean_strain = max(all_mean_strains)
    
    print(f"\n📊 Strain Analysis:")
    print(f"   Max principal strain: {max_strain:.6f}")
    print(f"   Mean principal strain: {mean_strain:.6f}")
    print(f"   Max of mean strain: {max_mean_strain:.6f}")
    
    # 边长分析
    all_mean_lengths = [stats['mean_length'] for stats in results['edge_lengths']]
    length_stability = np.std(all_mean_lengths) / np.mean(all_mean_lengths) * 100
    
    # 约束违反分析
    all_max_violations = [stats['max_violation'] for stats in results['constraint_violations']]
    all_mean_violations = [stats['mean_violation'] for stats in results['constraint_violations']]
    max_constraint_violation = max(all_max_violations)
    mean_constraint_violation = np.mean(all_mean_violations)
    
    print(f"\n📏 Edge Length Analysis:")
    print(f"   Mean edge length stability (CV): {length_stability:.3f}%")
    
    print(f"\n🔗 Constraint Violation Analysis:")
    print(f"   Max constraint violation: {max_constraint_violation:.6f} ({max_constraint_violation*100:.3f}%)")
    print(f"   Mean constraint violation: {mean_constraint_violation:.6f} ({mean_constraint_violation*100:.3f}%)")
    
    # 不可拉伸性评估
    print(f"\n✅ Inextensibility Assessment:")
    if area_stability < 1.0:
        print(f"   Area stability: Excellent (< 1%)")
    elif area_stability < 5.0:
        print(f"   Area stability: Good (< 5%)")
    else:
        print(f"   Area stability: Fair (> 5%)")
    
    if max_strain < 0.01:
        print(f"   Strain control: Excellent (< 1%)")
    elif max_strain < 0.05:
        print(f"   Strain control: Good (< 5%)")
    else:
        print(f"   Strain control: Fair (> 5%)")
    
    print("=" * 60)

def main():
    """主函数"""
    
    # 可以分析的目录列表
    possible_dirs = [
        "cloth_simulation_output",
        "cloth_simulation_output_0.005",
        "cloth_simulation_output_0.01",
        "cloth_simulation_output_0.05"
    ]
    
    print("🔍 Looking for analyzable cloth simulation output directories...")
    
    found_dirs = []
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            obj_files = [f for f in os.listdir(dir_name) if f.startswith("frame_") and f.endswith(".obj")]
            if obj_files:
                found_dirs.append(dir_name)
                print(f"✅ Found directory: {dir_name} ({len(obj_files)} frames)")
    
    if not found_dirs:
        print("❌ No analyzable directories found")
        print("Please run cloth simulation first to generate OBJ files!")
        return
    
    # 分析每个找到的目录
    for output_dir in found_dirs:
        print(f"\n{'='*60}")
        print(f"🚀 Analyzing directory: {output_dir}")
        print(f"{'='*60}")
        
        results = analyze_cloth_simulation(output_dir, max_frames=50)
        
        if results:
            print_summary(results, output_dir)
            save_analysis_report(results, output_dir)
            plot_strain_analysis(results, output_dir)
        else:
            print(f"❌ Failed to analyze {output_dir}")

if __name__ == "__main__":
    main()
