import taichi as ti
import math
import numpy as np

# ------------------------------------------------------------
# Inextensible Cloth with Fast Projection (ICD ≡ SAP w/ FP)
# Minimal Taichi implementation (3D positions, quad grid).
# - Hard constraints on warp/weft (grid) edges: C = ||d||^2 / l - l = 0
# - Fast Projection: solve (J M^{-1} J^T) dλ = C / h^2 with matrix-free PCG
# - Then Δx = -h^2 M^{-1} J^T dλ, x ← x + Δx
# - External forces (gravity) are integrated explicitly to get x^0 (predict step)
# - Velocity update: v = (x - x_prev)/dt
# - Optional: simple ground collision (y >= 0)
# ------------------------------------------------------------

real = ti.f32
vec3 = ti.types.vector(3, real)

@ti.data_oriented
class InextensibleClothFP:
    def __init__(self, n_u=30, n_v=30, dx=0.03, dt=1/240, iters=15, pcg_iters=60,
                 tol=1e-3, g=(0.0, -9.8, 0.0), mass_per_vertex=0.02,
                 k_shear=500.0, k_bend=1.0,
                 strain_threshold=0.0):
        self.n_u = n_u
        self.n_v = n_v
        self.n = n_u * n_v
        self.dx = dx
        self.dt = float(dt)
        self.fp_outer_iters = iters
        self.pcg_max_iters = pcg_iters
        self.pcg_tol = tol
        self.g = vec3(g)
        self.mass = mass_per_vertex
        self.k_shear = k_shear
        self.k_bend = k_bend
        self.strain_threshold = strain_threshold 

        self.x = ti.Vector.field(3, dtype=real, shape=self.n)
        self.x_prev = ti.Vector.field(3, dtype=real, shape=self.n)
        self.v = ti.Vector.field(3, dtype=real, shape=self.n)
        self.f = ti.Vector.field(3, dtype=real, shape=self.n)
        self.inv_m = ti.field(dtype=real, shape=self.n)

        self.max_cons = (n_u - 1) * n_v + n_u * (n_v - 1)
        self.cnt_cons = ti.field(dtype=ti.i32, shape=())
        self.con_i = ti.field(dtype=ti.i32, shape=self.max_cons)
        self.con_j = ti.field(dtype=ti.i32, shape=self.max_cons)
        self.con_l0 = ti.field(dtype=real, shape=self.max_cons)

        self.C = ti.field(dtype=real, shape=self.max_cons)
        self.g_a = ti.Vector.field(3, dtype=real, shape=self.max_cons)
        self.g_b = ti.Vector.field(3, dtype=real, shape=self.max_cons)        # tensor/stress 模式: 每个 quad 的主应变/应力度量 + 全局最大值
        self.max_strain = ti.field(dtype=real, shape=())

        self.dlambda = ti.field(dtype=real, shape=self.max_cons)
        self.r = ti.field(dtype=real, shape=self.max_cons)
        self.p = ti.field(dtype=real, shape=self.max_cons)
        self.Ap = ti.field(dtype=real, shape=self.max_cons)
        self.tmp_vertex = ti.Vector.field(3, dtype=real, shape=self.n)
        self.colors = ti.Vector.field(3, dtype=real, shape=self.n)

        self.max_shear_edges = 2 * (n_u - 1) * (n_v - 1) 
        self.max_bend_quads = (n_u - 1) * (n_v - 1)
        self.cnt_shear_edges = ti.field(dtype=ti.i32, shape=())
        self.cnt_bend_quads = ti.field(dtype=ti.i32, shape=())
        
        self.shear_edge_i = ti.field(dtype=ti.i32, shape=self.max_shear_edges)
        self.shear_edge_j = ti.field(dtype=ti.i32, shape=self.max_shear_edges)
        self.shear_edge_l0 = ti.field(dtype=real, shape=self.max_shear_edges)
        
        self.bend_quad_i0 = ti.field(dtype=ti.i32, shape=self.max_bend_quads)
        self.bend_quad_i1 = ti.field(dtype=ti.i32, shape=self.max_bend_quads)
        self.bend_quad_i2 = ti.field(dtype=ti.i32, shape=self.max_bend_quads)
        self.bend_quad_i3 = ti.field(dtype=ti.i32, shape=self.max_bend_quads)

        self._init_states()
        self._build_topology()

    def idx(self, u, v):
        return v * self.n_u + u

    @ti.func
    def idx_func(self, u, v):
        return v * self.n_u + u

    def _build_topology(self):
        cons = []
        shear_edges = []
        bend_quads = []   
        for v in range(self.n_v):
            for u in range(self.n_u):
                i = v * self.n_u + u
                if u + 1 < self.n_u:
                    j = v * self.n_u + (u + 1)
                    cons.append((i, j, self.dx))
                if v + 1 < self.n_v:
                    j = (v + 1) * self.n_u + u
                    cons.append((i, j, self.dx))
                if u + 1 < self.n_u and v + 1 < self.n_v:
                    i0 = self.idx(u, v)
                    i1 = self.idx(u+1, v+1)
                    i2 = self.idx(u+1, v)
                    i3 = self.idx(u, v+1)
                    diag_dist = (2 * self.dx * self.dx) ** 0.5  # sqrt(2) * dx
                    shear_edges.append((i0, i1, diag_dist))
                    shear_edges.append((i2, i3, diag_dist))
                    bend_quads.append((i0, i2, i3, i1))

        self.cnt_cons[None] = len(cons)
        for cid, (i, j, l0) in enumerate(cons):
            self.con_i[cid] = i
            self.con_j[cid] = j
            self.con_l0[cid] = l0
        
        self.cnt_shear_edges[None] = len(shear_edges)
        for sid, (i, j, l0) in enumerate(shear_edges):
            self.shear_edge_i[sid] = i
            self.shear_edge_j[sid] = j
            self.shear_edge_l0[sid] = l0
        
        self.cnt_bend_quads[None] = len(bend_quads)
        for bid, (i0, i1, i2, i3) in enumerate(bend_quads):
            self.bend_quad_i0[bid] = i0
            self.bend_quad_i1[bid] = i1
            self.bend_quad_i2[bid] = i2
            self.bend_quad_i3[bid] = i3

    @ti.kernel
    def _init_states(self):
        for v in range(self.n_v):
            for u in range(self.n_u):
                i = v * self.n_u + u
                self.x[i] = vec3(u * self.dx, 0.5, v * self.dx)
                self.v[i] = vec3(0.0, 0.0, 0.0)
                self.inv_m[i] = 1.0 / self.mass
                self.colors[i] = vec3(0.7 + 0.3 * u / max(1, self.n_u - 1),
                                       0.6 + 0.3 * v / max(1, self.n_v - 1), 0.8)
        self.inv_m[self.idx_func(0, 0)] = 0.0
        self.inv_m[self.idx_func(self.n_u - 1, 0)] = 0.0

    # ------------------------------------------------------------
    @ti.kernel
    def clear_force(self):
        for i in range(self.n):
            self.f[i] = vec3(0.0, 0.0, 0.0)

    @ti.kernel
    def add_gravity(self):
        for i in range(self.n):
            self.f[i] += self.g * (1.0 / max(1e-8, self.inv_m[i]))

    @ti.kernel
    def add_shear_forces(self):
        for eid in range(self.cnt_shear_edges[None]):
            i = self.shear_edge_i[eid]
            j = self.shear_edge_j[eid]
            l0 = self.shear_edge_l0[eid]
            xi = self.x[i]
            xj = self.x[j]
            d = xj - xi
            l = d.norm()
            if l > 1e-8:
                f = self.k_shear * (l - l0) * d / l
                if self.inv_m[i] > 0:
                    self.f[i] += f
                if self.inv_m[j] > 0:
                    self.f[j] -= f

    @ti.func
    def sin_half_angle(n1, n2, e):
        dotn = n1.dot(n2)
        val = ti.sqrt(0.5 * max(0.0, 1.0 - dotn))
        sign = 1.0
        if (n1.cross(n2)).dot(e) < 0:
            sign = -1.0
        return sign * val
    
    @ti.func
    def bending_forces(p1, p2, p3, p4, k_bend: float):
        E = p2 - p3
        e_len = E.norm(1e-8)

        N1 = (p1 - p3).cross(p1 - p2)
        N2 = (p4 - p2).cross(p4 - p3)

        N1_len = N1.norm(1e-8)
        N2_len = N2.norm(1e-8)
        n1 = N1 / N1_len
        n2 = N2 / N2_len

        u1 = (e_len / (N1_len * N1_len)) * N1
        u4 = (e_len / (N2_len * N2_len)) * N2
        u2 = -((p1 - p3).dot(E) / e_len) * (N1 / (N1_len * N1_len)) \
        - ((p4 - p3).dot(E) / e_len) * (N2 / (N2_len * N2_len))
        u3 = ((p1 - p2).dot(E) / e_len) * (N1 / (N1_len * N1_len)) \
        +((p4 - p2).dot(E) / e_len) * (N2 / (N2_len * N2_len))

        coeff = k_bend * e_len * e_len / (N1_len + N2_len)
        elastic_term = InextensibleClothFP.sin_half_angle(n1, n2, E.normalized(1e-8))

        F1 = coeff * elastic_term * u1
        F2 = coeff * elastic_term * u2
        F3 = coeff * elastic_term * u3
        F4 = coeff * elastic_term * u4

        return F1, F2, F3, F4
    
    @ti.kernel
    def add_bending_forces(self):
        for bid in range(self.cnt_bend_quads[None]):
            i0 = self.bend_quad_i0[bid]
            i1 = self.bend_quad_i1[bid]  
            i2 = self.bend_quad_i2[bid]
            i3 = self.bend_quad_i3[bid]
            p0, p1, p2, p3 = self.x[i0], self.x[i1], self.x[i2], self.x[i3]
            f1, f2, f3, f4 = InextensibleClothFP.bending_forces(p0, p1, p2, p3, self.k_bend)
            self.f[i0] += f1
            self.f[i1] += f2
            self.f[i2] += f3
            self.f[i3] += f4
    
    @ti.kernel
    def explicit_predict(self, dt: real):
        for i in range(self.n):
            self.x_prev[i] = self.x[i]
        for i in range(self.n):
            if self.inv_m[i] > 0:
                self.v[i] += dt * self.inv_m[i] * self.f[i]
                self.x[i] += dt * self.v[i]

    @ti.kernel
    def compute_constraint_residuals_and_grads(self):
        for c in range(self.cnt_cons[None]):
            i = self.con_i[c]
            j = self.con_j[c]
            l0 = self.con_l0[c]
            xi = self.x[i]
            xj = self.x[j]
            d = xj - xi
            Cc = (d.dot(d)) / l0 - l0
            self.C[c] = Cc
            g = 2.0 * d / l0
            self.g_a[c] = -g
            self.g_b[c] = g

    @ti.kernel
    def compute_cell_max_strain(self):
        maxv = 0.0
        for q in range(self.max_bend_quads):
            W = self.n_u - 1
            u = q % W
            v = q // W
            i0 = v * self.n_u + u
            i1 = v * self.n_u + (u + 1)
            i2 = (v + 1) * self.n_u + u
            e_u = self.x[i1] - self.x[i0]
            e_v = self.x[i2] - self.x[i0]
            fxu = e_u.x / self.dx
            fzu = e_u.z / self.dx
            fxv = e_v.x / self.dx
            fzv = e_v.z / self.dx
            C00 = fxu * fxu + fzu * fzu
            C01 = fxu * fxv + fzu * fzv
            C11 = fxv * fxv + fzv * fzv
            E00 = 0.5 * (C00 - 1.0)
            E01 = 0.5 * C01
            E11 = 0.5 * (C11 - 1.0)
            diff = E00 - E11
            temp = ti.sqrt(diff * diff + 4.0 * E01 * E01)
            lam1 = 0.5 * (E00 + E11 + temp)
            lam2 = 0.5 * (E00 + E11 - temp)
            m = ti.max(ti.abs(lam1), ti.abs(lam2))
            if m > maxv:
                maxv = m
        self.max_strain[None] = maxv

    @ti.kernel
    def clear_tmp_vertex(self):
        for i in range(self.n):
            self.tmp_vertex[i] = vec3(0.0, 0.0, 0.0)

    @ti.kernel
    def Jt_times_z_accumulate(self, z: ti.template()):
        # tmp_vertex = sum_c J^T_c * z_c  (per-vertex 3D vector)
        for c in range(self.cnt_cons[None]):
            i = self.con_i[c]
            j = self.con_j[c]
            za = z[c]
            # J^T contribution: vertex i gets g_a[c] * z, vertex j gets g_b[c] * z
            ti.atomic_add(self.tmp_vertex[i], self.g_a[c] * za)
            ti.atomic_add(self.tmp_vertex[j], self.g_b[c] * za)

    @ti.kernel
    def Minv_times_tmp_vertex(self):
        # tmp_vertex <- Minv * tmp_vertex
        for i in range(self.n):
            self.tmp_vertex[i] *= self.inv_m[i]

    @ti.kernel
    def J_times_tmp_vertex(self, y: ti.template()):
        # y_c = J_c * tmp_vertex = g_a·tmp[i] + g_b·tmp[j]
        for c in range(self.cnt_cons[None]):
            i = self.con_i[c]
            j = self.con_j[c]
            y[c] = self.g_a[c].dot(self.tmp_vertex[i]) + self.g_b[c].dot(self.tmp_vertex[j])

    def apply_A(self, x_vec):
        # y = (J Minv J^T) x_vec  (matrix-free)
        self.clear_tmp_vertex()
        self.Jt_times_z_accumulate(x_vec)
        self.Minv_times_tmp_vertex()
        self.J_times_tmp_vertex(self.Ap)

    @ti.kernel
    def compute_delta_x_from_dlambda(self, dt: real):
        # Δx = -dt^2 * Minv * J^T * dλ  (matrix-free)
        for i in range(self.n):
            self.tmp_vertex[i] = vec3(0.0, 0.0, 0.0)
        for c in range(self.cnt_cons[None]):
            i = self.con_i[c]
            j = self.con_j[c]
            lam = self.dlambda[c]
            ti.atomic_add(self.tmp_vertex[i], self.g_a[c] * lam)
            ti.atomic_add(self.tmp_vertex[j], self.g_b[c] * lam)
        for i in range(self.n):
            if self.inv_m[i] > 0:
                self.x[i] += - (dt * dt) * self.inv_m[i] * self.tmp_vertex[i]

    @ti.kernel
    def apply_ground(self):
        # simple ground plane at y=0
        for i in range(self.n):
            if self.x[i].y < 0.0:
                self.x[i].y = 0.0
                if self.v[i].y < 0:
                    self.v[i].y = 0.0

    def pcg_solve_for_dlambda(self, rhs_scale):
        # Solve (J Minv J^T) dlambda = rhs = C / (dt^2)  (we pass rhs_scale = 1/(dt^2))
        # Initialize dlambda = 0, r = rhs, p = r
        cnt = self.cnt_cons[None]
        # r = C * rhs_scale
        @ti.kernel
        def init_system():
            for c in range(cnt):
                self.dlambda[c] = 0.0
                self.r[c] = self.C[c] * rhs_scale
                self.p[c] = self.r[c]
        init_system()

        # PCG loop
        rTr_old = 0.0
        @ti.kernel
        def reduce_rTr() -> real:
            s = 0.0
            for c in range(cnt):
                s += self.r[c] * self.r[c]
            return s

        @ti.kernel
        def axpy(vec_y: ti.template(), a: real, vec_x: ti.template()):
            for c in range(cnt):
                vec_y[c] += a * vec_x[c]

        @ti.kernel
        def xpay(vec_y: ti.template(), a: real, vec_x: ti.template()):
            # y = x + a*y  (used for p = r + beta * p)
            for c in range(cnt):
                vec_y[c] = vec_x[c] + a * vec_y[c]

        # early check
        rTr = reduce_rTr()
        if rTr < self.pcg_tol * self.pcg_tol:
            return

        for k in range(self.pcg_max_iters):
            # Ap = A * p
            self.apply_A(self.p)

            # alpha = rTr / (p^T Ap)
            pAp = 0.0
            @ti.kernel
            def reduce_pAp() -> real:
                s = 0.0
                for c in range(cnt):
                    s += self.p[c] * self.Ap[c]
                return s
            pAp = reduce_pAp()
            if abs(pAp) < 1e-12:
                break
            alpha = rTr / pAp

            # dlambda += alpha * p; r -= alpha * Ap
            axpy(self.dlambda, alpha, self.p)
            axpy(self.r, -alpha, self.Ap)

            rTr_new = reduce_rTr()
            if rTr_new < self.pcg_tol * self.pcg_tol:
                break
            beta = rTr_new / rTr
            xpay(self.p, beta, self.r)
            rTr = rTr_new

    def fast_projection(self):
        for it in range(self.fp_outer_iters):
            self.compute_constraint_residuals_and_grads()
            rhs_scale = 1.0 / (self.dt * self.dt)
            self.pcg_solve_for_dlambda(rhs_scale)
            self.compute_delta_x_from_dlambda(self.dt)
            # self.apply_ground() 
            if self.strain_threshold > 0.0 and (it + 1) % 3 == 0:
                self.compute_cell_max_strain()
                if self.max_strain[None] < self.strain_threshold:
                    print(it)
                    break           

    @ti.kernel
    def update_velocity(self, dt: real):
        for i in range(self.n):
            self.v[i] = (self.x[i] - self.x_prev[i]) / dt

    def step(self):
        self.clear_force()
        self.add_gravity()
        self.add_shear_forces()
        self.add_bending_forces()
        self.explicit_predict(self.dt)
        # self.apply_ground()
        self.fast_projection()
        self.update_velocity(self.dt)

import os

# ------------------------------------------------------------
# Example usage: Save simulation as OBJ files
# ------------------------------------------------------------
def save_mesh_to_obj(vertices, triangles, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for tri in triangles:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

def create_cloth_mesh_data(sim):
    vertices = sim.x.to_numpy()
    triangles = []
    for v in range(sim.n_v - 1):
        for u in range(sim.n_u - 1):
            i0 = v * sim.n_u + u
            i1 = v * sim.n_u + (u + 1)
            i2 = (v + 1) * sim.n_u + u
            i3 = (v + 1) * sim.n_u + (u + 1)
            triangles.append([i0, i1, i2])
            triangles.append([i1, i3, i2])
    return vertices, triangles

if __name__ == "__main__":
    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)

    sim = InextensibleClothFP(n_u=30, n_v=30, dx=0.03, dt=1/360,
                              iters=20, pcg_iters=50, tol=1e-6,
                              g=(0.0, -9.8, 0.0), mass_per_vertex=0.05,
                              k_shear=500.0, k_bend=0.5,
                              strain_threshold=-1)

    output_dir = "cloth_simulation_output"
    os.makedirs(output_dir, exist_ok=True)
        
    max_frames = 100
    save_every_n_frames = 2 
    
    frame_count = 0
    saved_count = 0
    
    try:
        while frame_count < max_frames:

            for _ in range(2): 
                sim.step()
            
            if frame_count % save_every_n_frames == 0:
                vertices, triangles = create_cloth_mesh_data(sim)
                filename = os.path.join(output_dir, f"frame_{saved_count:04d}.obj")
                save_mesh_to_obj(vertices, triangles, filename)
                saved_count += 1
                
                if saved_count % 5 == 0:
                    print(f"saved {saved_count} frames to {filename}")

            frame_count += 1
                
    except KeyboardInterrupt:
        print(f"\n{saved_count} frames saved")

    print(f"Total saved OBJ files: {saved_count}")