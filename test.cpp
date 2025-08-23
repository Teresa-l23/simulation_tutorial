#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
const double PERTURB_QUANT = 1e-6;
const double SCALE_STIFF = 10;
const double SCALE_DAMP_STIFF = 1;
const double SHEAR_STIFF = 10;
const double SHEAR_DAMP_STIFF = 1;
const double BENDING_STIFF = 0.0001;
const double BENDING_DAMP_STIFF = 0;
const double MASS = 1.0;
const double MAX_BEND = 0.000003;
const double TIMESTEP = 0.001; 
const double GRAVITY_ACCEL = 9.8;
const double AREA = 0.01;

struct Particle {
    Vector3d pos; 
    Vector2d uv;
    Vector3d vel;
    double m; 
    Vector3d force;
};
Vector2d springCondition(const Particle** p, Vector2d buv) {
    const Particle& pi = *p[0];
    const Particle& pj = *p[1];
    const Particle& pk = *p[2];
    
    Vector3d edge1 = pj.pos - pi.pos;
    Vector3d edge2 = pk.pos - pi.pos;
    
    Eigen::Matrix<double,3,2> dmatrix;
    dmatrix.col(0) = edge1;
    dmatrix.col(1) = edge2;
    
    Eigen::Matrix<double,2,2> duvmatrix;
    duvmatrix.col(0) = pj.uv - pi.uv;
    duvmatrix.col(1) = pk.uv - pi.uv;
    
    if (abs(duvmatrix.determinant()) < 1e-12) {
        duvmatrix = Matrix<double,2,2>::Zero();
    }
    
    duvmatrix = duvmatrix.inverse().eval();
    Eigen::Matrix<double,3,2> wuvdmatrix = dmatrix * duvmatrix;
    Vector2d wnorms = wuvdmatrix.colwise().norm();
    return AREA * (wnorms - buv);
}
Matrix<double,3,2> springPartial(Particle** p, int index, Vector2d buv) {
    Matrix<double,3,2> partial;
    Particle& pi = *p[index];
    double ori_pos[3] = {pi.pos[0],pi.pos[1],pi.pos[2]};
    
    for(int dim = 0; dim < 3; dim++) {
        pi.pos[dim] = ori_pos[dim] + PERTURB_QUANT;
        Vector2d cond1 = springCondition((const Particle**)p, buv);
        pi.pos[dim] = ori_pos[dim] - PERTURB_QUANT;
        Vector2d cond2 = springCondition((const Particle**)p, buv);
        
        partial.row(dim) = (cond1 - cond2) / (2 * PERTURB_QUANT);
        pi.pos[dim] = ori_pos[dim]; 
    }
    return partial;
}
void springForce(Particle** p, Vector2d buv) {
    Vector2d cond = springCondition((const Particle**)p,buv);
    for(int col = 0; col < 3; col++) {
        Matrix<double,3,2> partial = springPartial(p,col,buv);
        p[col]->force -= SCALE_STIFF * partial * cond;
        p[col]->force -= SCALE_DAMP_STIFF * partial * partial.transpose() * p[col]->vel;
    }
}
double shearCondition(const Particle** p) {
    const Particle& pi = *p[0];
    const Particle& pj = *p[1];
    const Particle& pk = *p[2];
    
    Vector3d edge1 = pj.pos - pi.pos;
    Vector3d edge2 = pk.pos - pi.pos;
    
    Eigen::Matrix<double,3,2> dmatrix;
    dmatrix.col(0) = edge1;
    dmatrix.col(1) = edge2;
    
    Eigen::Matrix<double,2,2> duvmatrix;
    duvmatrix.col(0) = pj.uv - pi.uv;
    duvmatrix.col(1) = pk.uv - pi.uv;
    
    if (abs(duvmatrix.determinant()) < 1e-12) {
       duvmatrix = Matrix<double,2,2>::Zero();
    }
    
    duvmatrix = duvmatrix.inverse().eval();
    Eigen::Matrix<double,3,2> wuvdmatrix = dmatrix * duvmatrix;
    return AREA * wuvdmatrix.col(0).dot(wuvdmatrix.col(1));
}
Vector3d shearPartial(Particle** p, int index) {
    Vector3d partial;
    Particle& pi = *p[index];
    double ori_pos[3] = {pi.pos[0],pi.pos[1],pi.pos[2]};
    
    for(int dim = 0; dim < 3; dim++) {
        pi.pos[dim] = ori_pos[dim] + PERTURB_QUANT;
        double cond1 = shearCondition((const Particle**)p);
        pi.pos[dim] = ori_pos[dim] - PERTURB_QUANT;
        double cond2 = shearCondition((const Particle**)p);

        partial[dim] = (cond1 - cond2) / (2 * PERTURB_QUANT);
        pi.pos[dim] = ori_pos[dim];
    }
    return partial;
}
void shearForce(Particle** p) {
    double cond = shearCondition((const Particle**)p);
    for(int col = 0; col < 3; col++) {
        Vector3d partial = shearPartial(p,col);
        p[col]->force -= SHEAR_STIFF * partial * cond;
        p[col]->force -= SHEAR_DAMP_STIFF * partial * partial.transpose() * p[col]->vel;
    }
}
double bendingCondition(const Particle** p) {
    const Particle& p1 = *p[0];
    const Particle& p2 = *p[1];
    const Particle& p3 = *p[2];
    const Particle& p4 = *p[3];

    Vector3d edge1 = p2.pos - p1.pos;
    Vector3d edge2 = p3.pos - p1.pos;
    Vector3d norm1 = edge1.cross(edge2);
    norm1.normalize();
    Vector3d edge3 = p3.pos - p4.pos;
    Vector3d edge4 = p2.pos - p4.pos;
    Vector3d norm2 = edge3.cross(edge4);
    norm2.normalize();

    Vector3d edge = p2.pos - p3.pos;
    edge.normalize();
    double st = norm1.cross(norm2).dot(edge);
    double ct = norm1.dot(norm2);
    return atan2(st, ct);
}
Vector3d bendingPartial(Particle** p, int index) {
    Vector3d partial;
    Particle& pi = *p[index];
    double ori_pos[3] = {pi.pos[0],pi.pos[1],pi.pos[2]};
    
    for(int dim = 0; dim < 3; dim++) {
        pi.pos[dim] = ori_pos[dim] + PERTURB_QUANT;
        double cond1 = bendingCondition((const Particle**)p);
        pi.pos[dim] = ori_pos[dim] - PERTURB_QUANT;
        double cond2 = bendingCondition((const Particle**)p);

        partial[dim] = (cond1 - cond2) / (2 * PERTURB_QUANT);
        pi.pos[dim] = ori_pos[dim]; 
    }
    return partial;
}
inline double clamp(double x, double low, double high) {
	if (x < low)  return low;
	if (x > high) return high;
	return x;
}
void bendingForce(Particle** p) {
    double cond = bendingCondition((const Particle**)p);
    for(int col = 0; col < 4; col++) {
        Vector3d partial = bendingPartial(p,col);
        Vector3d force = BENDING_STIFF * partial * cond;
        for(int j = 0; j < 3; j++) {
            force[j] = clamp(force[j], -MAX_BEND, MAX_BEND);
        }
        p[col]->force -= force;
        p[col]->force -= BENDING_DAMP_STIFF * partial * partial.transpose() * p[col]->vel;
    }
}
void visualizeGrid(const vector<vector<Particle>>& particle2D, int step) {
    const int GRID_SIZE = particle2D.size();
    cout << "\n=== Step " << step << " ===" << endl;
    cout << "Position (z-coordinate only):" << endl;
    
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            printf("%6.3f ", particle2D[i][j].pos[2]);
        }
        cout << endl;
    }
    
    cout << "\nCorner particles details:" << endl;
    cout << "Top-left (0,0): pos=(" << particle2D[0][0].pos.transpose() 
         << "), vel=(" << particle2D[0][0].vel.transpose() << ")" << endl;
    cout << "Top-right (0," << GRID_SIZE-1 << "): pos=(" << particle2D[0][GRID_SIZE-1].pos.transpose() 
         << "), vel=(" << particle2D[0][GRID_SIZE-1].vel.transpose() << ")" << endl;
    cout << "Bottom-left (" << GRID_SIZE-1 << ",0): pos=(" << particle2D[GRID_SIZE-1][0].pos.transpose() 
         << "), vel=(" << particle2D[GRID_SIZE-1][0].vel.transpose() << ")" << endl;
    cout << "Bottom-right (" << GRID_SIZE-1 << "," << GRID_SIZE-1 << "): pos=(" << particle2D[GRID_SIZE-1][GRID_SIZE-1].pos.transpose() 
         << "), vel=(" << particle2D[GRID_SIZE-1][GRID_SIZE-1].vel.transpose() << ")" << endl;
}

int main() {
   
    const int GRID_SIZE = 5; 
    const int MAX_STEPS = 1000; 
    const int DISPLAY_INTERVAL = 100;

    vector<vector<Particle>> particle2D(GRID_SIZE, vector<Particle>(GRID_SIZE));
    
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            particle2D[i][j].pos = Vector3d(i * 0.1, j * 0.1, 0);  // 缩小初始间距
            particle2D[i][j].uv = Vector2d(i * 0.1, j * 0.1);      // 对应的UV坐标
            particle2D[i][j].vel = Vector3d::Zero();   // 速度为0
            particle2D[i][j].force = Vector3d::Zero(); // 合力为0
            particle2D[i][j].m = MASS;                 // 质量为MASS
        }
    }

    cout << "Starting cloth simulation..." << endl;
    cout << "Grid size: " << GRID_SIZE << "x" << GRID_SIZE << endl;
    cout << "Max steps: " << MAX_STEPS << endl;
    cout << "Timestep: " << TIMESTEP << endl;
    cout << "Gravity: " << GRAVITY_ACCEL << endl;


    visualizeGrid(particle2D, 0);


    for (int step = 1; step <= MAX_STEPS; ++step) {
  
        for (int i = 0; i < GRID_SIZE; ++i) {
            for (int j = 0; j < GRID_SIZE; ++j) {
                particle2D[i][j].force = Vector3d::Zero();
            }
        }

        for (int i = 0; i < GRID_SIZE-1; ++i) {
            for (int j = 0; j < GRID_SIZE-1; ++j) {
                Particle* p_up[3] = {
                    &particle2D[i][j],
                    &particle2D[i][j+1],
                    &particle2D[i+1][j]
                };
                Particle* p_low[3] = {
                    &particle2D[i+1][j+1],
                    &particle2D[i+1][j],
                    &particle2D[i][j+1]
                };
                Vector2d buv = Vector2d(0.1, 0.1); 
                springForce(p_up, buv);
                springForce(p_low, buv);
                shearForce(p_up);
                shearForce(p_low);

                Particle* p_diag[4] = {
                    &particle2D[i][j],
                    &particle2D[i][j+1],
                    &particle2D[i+1][j],
                    &particle2D[i+1][j+1]
                };
                bendingForce(p_diag);
                if(j < GRID_SIZE-2) {
                    Particle* p_right[4] = {
                        &particle2D[i+1][j],
                        &particle2D[i][j+1],
                        &particle2D[i+1][j+1],
                        &particle2D[i][j+2]
                    };
                    bendingForce(p_right);
                }
                if(i < GRID_SIZE-2) {
                    Particle* p_top[4] = {
                        &particle2D[i][j+1],
                        &particle2D[i+1][j+1],
                        &particle2D[i+1][j],
                        &particle2D[i+2][j]
                    };
                    bendingForce(p_top);
                }
            }
        }

        for(int i = 0; i < GRID_SIZE; ++i) {
            for(int j = 0; j < GRID_SIZE; ++j) {
                Particle* p = &particle2D[i][j];
                
                if (p->force.norm() > 1000.0) {
                    p->force = p->force.normalized() * 1000.0;
                }
                
                p->vel += (p->force / p->m) * TIMESTEP; 
                p->vel[2] -= GRAVITY_ACCEL * TIMESTEP;
                
                if (p->vel.norm() > 10.0) {
                    p->vel = p->vel.normalized() * 10.0;
                }
                
                p->pos += p->vel * TIMESTEP;
            }
        }

        for (int j = 0; j < GRID_SIZE; ++j) {
            particle2D[0][j].pos = Vector3d(0, j * 0.1, 0); 
            particle2D[0][j].vel = Vector3d::Zero(); 
        }

        if (step % DISPLAY_INTERVAL == 0 || step == MAX_STEPS) {
            visualizeGrid(particle2D, step);
        }
        
        if (step % 5 == 0) {
            cout << "Progress: " << step << "/" << MAX_STEPS << " (" 
                 << (100 * step / MAX_STEPS) << "%)" << endl;
        }
    }

    cout << "\nSimulation completed!" << endl;
    return 0;
}
