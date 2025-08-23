import taichi as ti

ti.init(arch=ti.gpu)

# Parameters
n = 30  # grid size
num_step=0
dx = 1.0 / n
inv_dx = 1.0 / dx
V = n * n
N = (n - 1) * (n - 1) * 2  # two triangles per square
k_shear = 0.5
k_strectch = 0.5
k_bending = 1.0

dt = 0.0001
kd = 0.0001


vertices = ti.Vector.field(3, dtype=ti.f32, shape=V)
vertices = ti.Vector.field(3, dtype=ti.f32, shape=V, needs_grad=True)
bending_energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

velocity = ti.Vector.field(3, dtype=ti.f32, shape=V)
f = ti.Vector.field(3, dtype=ti.f32, shape=V)
d = ti.Vector.field(3, dtype=ti.f32, shape = V)
indices = ti.field(int, shape= 3 * N)
uv = ti.Vector.field(2, dtype=ti.f32, shape=V)
initial_stretch_matrix = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N)
mass_inv = ti.field(dtype=ti.f32, shape=V)
mass = ti.field(dtype=ti.f32, shape=V)

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

gravity = ti.Vector([0.0,0.0,-9.8])

dwdu = ti.Vector.field(3, dtype=ti.f32, shape=N)
dwdv = ti.Vector.field(3, dtype=ti.f32, shape=N)
stiffness_K = ti.Matrix.field(3, 3, dtype=ti.f32, shape =(V, V))

max_bending_edges = N * 6
bending_indices = ti.Vector.field(4, dtype=ti.i32, shape=max_bending_edges)
rest_angles = ti.field(dtype=ti.f32, shape=max_bending_edges)
bending_count = ti.field(dtype=ti.i32, shape=())

class MPCG:
    def __init__(self):
        self.n = V * 3

        self.A = ti.field(dtype=ti.f32, shape=(self.n, self.n))
        self.P = ti.field(dtype=ti.f32, shape=self.n)

        self.r = ti.Vector.field(3, dtype=ti.f32, shape=V)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=V)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=V)
        self.c = ti.Vector.field(3, dtype=ti.f32, shape=V)
        self.s = ti.Vector.field(3, dtype=ti.f32, shape=V)
        self.q = ti.Vector.field(3, dtype=ti.f32, shape=V)

    @ti.kernel
    def construct_P(self):
        for i in range(self.n):
            self.P[i] = self.A[i, i]

    @ti.kernel
    def Ax(self, x: ti.template(), out: ti.template()):
        for i in range(self.n):
            out[i // 3][i % 3] = 0.0
            for j in range(self.n):
                out[i // 3][i % 3] += self.A[i, j] * x[j // 3][j % 3]

    @ti.kernel
    def dot_product(self, a: ti.template(), b: ti.template()) -> ti.f32:
        result = 0.0
        for i in range(V):
            result += a[i].dot(b[i])
        return result

    @ti.kernel
    def vector_add(self, a: ti.template(), b: ti.template(), alpha: ti.f32, out: ti.template()):
        for i in range(V):
            out[i] = a[i] + alpha * b[i]

    @ti.kernel
    def vector_sub(self, a: ti.template(), b: ti.template(), alpha: ti.f32, out: ti.template()):
        for i in range(V):
            out[i] = a[i] - alpha * b[i]

    @ti.kernel
    def apply_preconditioner(self, in_vec: ti.template(), out_vec: ti.template()):
        for i in range(self.n):
            out_vec[i // 3][i % 3] = in_vec[i // 3][i % 3] / self.P[i]

    def solve(self, rhs: ti.template(), x: ti.template(), max_iters=100, tol=1e-5):
        self.Ax(x, self.Ap)
        self.vector_sub(rhs, self.Ap, 1.0, self.r)
        self.apply_preconditioner(self.r, self.c)
        self.vector_copy(self.c, self.p)

        delta_new = self.dot_product(self.r, self.c)
        delta_0 = delta_new
        eps = tol ** 2

        for k in range(max_iters):
            self.Ax(self.p, self.q)
            alpha = delta_new / (self.dot_product(self.p, self.q) + 1e-8)

            self.vector_add(x, self.p, alpha, x)
            self.vector_sub(self.r, self.q, alpha, self.r)

            self.apply_preconditioner(self.r, self.s)
            delta_old = delta_new
            delta_new = self.dot_product(self.r, self.s)

            if delta_new < eps * delta_0:
                break

            beta = delta_new / (delta_old + 1e-8)
            for i in range(V):
                self.p[i] = self.s[i] + beta * self.p[i]

    @ti.kernel
    def vector_copy(self, a: ti.template(), b: ti.template()):
        for i in range(V):
            b[i] = a[i]

    @ti.kernel
    def fill_A_from_stiffness(self):
        for i, j in stiffness_K:
            for d1 in ti.static(range(3)):
                for d2 in ti.static(range(3)):
                    row = i * 3 + d1
                    col = j * 3 + d2
                    self.A[row, col] = stiffness_K[i, j][d1, d2]

@ti.kernel
def initialize_cloth():
    for i, j in ti.ndrange(n, n):
        idx = i * n + j
        u = i * dx - 0.5
        v = j * dx - 0.5

        uv[idx] = ti.Vector([u, v])
        offset = 0.5 * (ti.random(ti.f32) - 0.5)  # small random z displacement
        vertices[idx] = ti.Vector([u, v, 1.0])
        velocity[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])

    for i, j in ti.ndrange(n - 1, n - 1):
        base = (i * (n - 1) + j) * 2 * 3
        idx0 = i * n + j
        idx1 = (i + 1) * n + j
        idx2 = i * n + j + 1
        idx3 = (i + 1) * n + j + 1

        indices[base] = idx0
        indices[base + 1] = idx1
        indices[base + 2] = idx2
        indices[base + 3] = idx2
        indices[base + 4] = idx1
        indices[base + 5] = idx3

@ti.kernel
def compute_initial_strech_matrix():
    for i in initial_stretch_matrix:
        idx0 = indices[i * 3] 
        idx1 = indices[i * 3 + 1] 
        idx2 = indices[i * 3 + 2] 

        u0, v0 = uv[idx0]
        u1, v1 = uv[idx1]
        u2, v2 = uv[idx2]
        D_uv = ti.Matrix([
            [u1 - u0, u2 - u0],
            [v1 - v0, v2 - v0]
        ])

        initial_stretch_matrix[i] = D_uv.inverse()

@ti.kernel
def compute_mass(density: ti.f32):
    for i in range(V):
        mass[i] = 0.0  # reset

    for t in range(N):
        i0 = indices[t * 3]
        i1 = indices[t * 3 + 1]
        i2 = indices[t * 3 + 2]

        x0, x1, x2 = vertices[i0], vertices[i1], vertices[i2]
        area = 0.5 * (x1 - x0).cross(x2 - x0).norm()
        m = (density * area) / 3.0  # 1/3 mass to each vertex

        mass[i0] += m
        mass[i1] += m
        mass[i2] += m

    for i in range(V):
        mass_inv[i] = 1.0 / mass[i] if mass[i] > 1e-8 else 0.0
        if vertices[i].x == - 0.5:
            mass_inv[i] = 0.0

    

@ti.kernel
def compute_stretching_energy_and_hessian():
    I3 = ti.Matrix.identity(ti.f32, 3)

    for t in range(N):
        idx0 = indices[t * 3]
        idx1 = indices[t * 3 + 1]
        idx2 = indices[t * 3 + 2]

        index = [idx0, idx1, idx2]

        x0, x1, x2 = vertices[idx0], vertices[idx1], vertices[idx2]

        dx1 = x1 - x0
        dx2 = x2 - x0
        Dx = ti.Matrix.cols([dx1, dx2])  
        D_inv = initial_stretch_matrix[t]  
        F = Dx @ D_inv

        wu = F[:,0]
        wv = F[:,1]
        
        wu_norm = wu.norm()
        wv_norm = wv.norm()
        
        cu = wu_norm - 1.0
        cv = wv_norm - 1.0

        wu_hat = wu / wu_norm
        wv_hat = wv / wv_norm

        d00, d10 = D_inv[0, 0], D_inv[1, 0]
        d01, d11 = D_inv[0, 1], D_inv[1, 1]

        dcudx = [-wu_hat * (d00 + d10), wu_hat * d00, wu_hat * d10]
        dcvdx = [-wv_hat * (d01 + d11), wv_hat * d01, wv_hat * d11]

        for i in ti.static(range(3)):
            fi = - cu * dcudx[i] - cv * dcvdx[i]
            f[index[i]] += k_strectch * fi
            d[index[i]] -= kd * dcudx[i] * dcudx[i].dot(velocity[index[i]]) \
                + kd * dcvdx[i] * dcvdx[i].dot(velocity[index[i]])
            #print(f[index[i]],d[index[i]])
        
        P_u = (I3 - wu.outer_product(wu) / wu_norm**2) / wu_norm
        P_v = (I3 - wv.outer_product(wv) / wv_norm**2) / wv_norm
        
        for m in ti.static(range(3)):
            for n in ti.static(range(3)):
                gm_u = dcudx[m]
                gn_u = dcudx[n]
                gm_v = dcvdx[m]
                gn_v = dcvdx[n]

                K_u = -k_strectch * (gm_u.outer_product(gn_u) + cu * gm_u.outer_product(P_u @ gn_u))
                K_v = -k_strectch * (gm_v.outer_product(gn_v) + cv * gm_v.outer_product(P_v @ gn_v))

                stiffness_K[index[m], index[n]] -= (K_u + K_v) * dt * dt


@ti.kernel
def reset_bending_energy():
    bending_energy[None] = 0.0  # 非循环语句

@ti.kernel
def compute_bending_energy():
    for i in range(bending_count[None]):
        idx = bending_indices[i]
        i0, i1, i2, i3 = idx[0], idx[1], idx[2], idx[3]
        x0, x1, x2, x3 = vertices[i0], vertices[i1], vertices[i2], vertices[i3]

        # 计算边向量和法向量
        e0 = x1 - x0
        e3 = x2 - x1
        e4 = x3 - x1

        n1 = e0.cross(e3)
        n2 = -e0.cross(e4)

        # 归一化法向量
        n1_norm = n1.norm()
        n2_norm = n2.norm()
        n1_hat = n1 / n1_norm
        n2_hat = n2 / n2_norm

        # 计算法向量之间的夹角
        cos_theta = n1_hat.dot(n2_hat)
        cos_theta = ti.min(1.0, ti.max(-1.0, cos_theta))  # 限制在有效范围内
        theta = ti.acos(cos_theta)
        
        # 累加弯曲能量
        bending_energy[None] += theta




        
@ti.kernel
def compute_shear_energy():
    I3 = ti.Matrix.identity(ti.f32, 3)

    for t in range(N):
        idx0 = indices[t * 3]
        idx1 = indices[t * 3 + 1]
        idx2 = indices[t * 3 + 2]
        index = [idx0, idx1, idx2]

        x0, x1, x2 = vertices[idx0], vertices[idx1], vertices[idx2]

        dx1 = x1 - x0
        dx2 = x2 - x0
        Dx = ti.Matrix.cols([dx1, dx2])  
        D_inv = initial_stretch_matrix[t]  
        F = Dx @ D_inv

        wu = F[:,0]
        wv = F[:,1]

        c = wu.dot(wv)

        d00, d10 = D_inv[0, 0], D_inv[1, 0]
        d01, d11 = D_inv[0, 1], D_inv[1, 1]

        dwudx = [-(d00 + d10), d00, d10]
        dwvdx = [-(d01 + d11), d01, d11]

        dc_dx = [ dwudx[i] * wv + dwvdx[i] * wu for i in range(3) ]

        for i in ti.static(range(3)):
            f[index[i]] -= k_shear * c * dc_dx[i]
            d[index[i]] -= kd * dc_dx[i] * dc_dx[i].dot(velocity[index[i]])
        
    

def construct_bending_edges():
    edge_map = {}  # maps (min_idx, max_idx) -> list of triangle indices
    cnt = 0

    # Step 1: Build edge to triangle map
    for t in range(N):
        ia0 = indices[t * 3 + 0]
        ia1 = indices[t * 3 + 1]
        ia2 = indices[t * 3 + 2]

        for e0, e1 in [(ia0, ia1), (ia1, ia2), (ia2, ia0)]:
            edge = (min(e0, e1), max(e0, e1))
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(t)

    # Step 2: Find shared edges and create bending constraints
    for edge, tri_list in edge_map.items():
        if len(tri_list) == 2:
            t0, t1 = tri_list

            verts0 = [indices[t0 * 3 + i] for i in range(3)]
            verts1 = [indices[t1 * 3 + i] for i in range(3)]

            shared = list(edge)
            other = [v for v in verts0 if v not in shared] + [v for v in verts1 if v not in shared]

            if len(shared) == 2 and len(other) == 2 and cnt < bending_indices.shape[0]:
                bending_indices[cnt] = ti.Vector([shared[0], shared[1], other[0], other[1]])
                cnt += 1

    bending_count[None] = cnt


@ti.func
def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


@ti.kernel
def compute_bending_energy_test():
    eps = 1e-6
    I3 = ti.Matrix.identity(ti.f32, 3)
    for i in range(bending_count[None]):
        idx = bending_indices[i]
        i0, i1, i2, i3 = idx[0], idx[1], idx[2], idx[3]
        x0, x1, x2, x3 = vertices[i0], vertices[i1], vertices[i2], vertices[i3]

        e0 = x1 - x0
        e1 = x2 - x0
        e2 = x3 - x0
        e3 = x2 - x1
        e4 = x3 - x1

        n1 = e0.cross(e3)
        n1_hat = n1 / n1.norm()
        n2 = -e0.cross(e4)
        n2_hat = n2 / n2.norm()

        si = sgn(n1.cross(n2).dot(e0))
        if si != 0:
            n1 = n1 * si
            n2 = n2 * si

            cos_theta = n1.dot(n2) / (n1.norm() * n2.norm() + 1e-8)
            cos_theta = ti.min(1.0, ti.max(-1.0, cos_theta))  # Clamp to valid acos range
            theta = ti.acos(cos_theta)
            c = 0.0
            if abs(cos_theta) > 0.99:
                c = 0.0
            else:
                c = theta

            dcdx0 = -(x2 - x1).dot(e0) / e0.norm() * n1 / (n1.norm()**2) \
                    - (x3 - x1).dot(e0) / e0.norm() * n2 / (n2.norm()**2)

            dcdx1 = (x2 - x0).dot(e0) / e0.norm() * n1 / (n1.norm()**2) \
                    + (x3 - x0).dot(e0) / e0.norm() * n2 / (n2.norm()**2)

            dcdx2 = -e0.norm() * n1 / (n1.norm()**2)
            dcdx3 = -e0.norm() * n2 / (n2.norm()**2)

            f[i0] -= k_bending * c * dcdx0
            f[i1] -= k_bending * c * dcdx1
            f[i2] -= k_bending * c * dcdx2
            f[i3] -= k_bending * c * dcdx3


            d[i0] -= 0.05 * kd * dcdx0 * dcdx0.dot(velocity[i0])
            d[i1] -= 0.05 * kd * dcdx1 * dcdx1.dot(velocity[i1])
            d[i2] -= 0.05  * kd * dcdx2 * dcdx2.dot(velocity[i2])
            d[i3] -= 0.05 * kd * dcdx3 * dcdx3.dot(velocity[i3])
            

@ti.kernel
def step():
    for I in vertices:
        velocity[I] += mass_inv[I] * (f[I]+d[I]+mass[I]*gravity) * dt
        if mass_inv[I]==0:
            print("test")
    
    for i in vertices:
        offset_to_center = vertices[i] - ball_center[0]
        #if offset_to_center.norm() <= ball_radius:
            # Velocity projection
        #    normal = offset_to_center.normalized()
        #    velocity[i] = -velocity[i]
        vertices[i] += dt * velocity[i]
    

# Initialize cloth simulation
initialize_cloth()
print("init")
compute_initial_strech_matrix()
print("strech")
compute_mass(10.0)
print("mass")
construct_bending_edges()
#construct_bending_edges()

window = ti.ui.Window("Taichi Cloth Simulation", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

current_t = 0.0


@ti.kernel
def clear_forces():
    for i in range(V):
        f[i] = ti.Vector([0.0, 0.0, 0.0])
        d[i] = ti.Vector([0.0, 0.0, 0.0])

while window.running:

    for step_i in range(10):  # run multiple physics steps per frame
        clear_forces()
        #reset_bending_energy()
        compute_stretching_energy_and_hessian()
        
        compute_shear_energy()

        #bending_energy.grad[None] = 1
        #compute_bending_energy()
        #compute_bending_energy.grad()
        #for i in range(V):
        #    print(vertices.grad[i])

        compute_bending_energy_test()

        
        #input()
        step()
        num_step+=1
        current_t += dt

    # Camera view
    camera.position(1.0, 0.0,6.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)
    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
    scene.ambient_light((0.4, 0.4, 0.4))

    # Draw cloth mesh
    scene.mesh(vertices, indices, color=(0.5, 0.42, 0.8), two_sided=True)  
    #scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.0, 0.0, 0.0))
    canvas.scene(scene)
    if num_step%60==0:
        window.save_image(f"screenshot_{num_step}.png")

    window.show()
    