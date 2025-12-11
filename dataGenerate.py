import numpy as np
import taichi as ti
from tqdm import tqdm
import h5py
import os

ti.init(arch=ti.gpu, debug=False, default_fp=ti.f32)

input_dir = "/home/wqz/fluid/data"
output_dir = "/home/wqz/fluid/data"
demox = 1
file_num = 3

dim = 3
n_grid = 128
steps = 200
dt = 5e-5


n_particles = 37908+37908+37908
fluid_particles = 37908*3
dx = 1 / n_grid
inv_dx = float(n_grid)

p_rho = 1.0
p_vol = dx ** dim
p_mass = p_rho * p_vol
gravity = 9.8 
bound = 3
E = 1.2e3
nu = 0.3
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

F_x = ti.Vector.field(3, float, n_particles)
F_v = ti.Vector.field(3, float, n_particles)
F_C = ti.Matrix.field(3, 3, float, n_particles)
F_F = ti.Matrix.field(dim, dim, float, n_particles)

material = ti.field(int, n_particles)

F_grid_v = ti.field(ti.types.vector(dim, float))
F_grid_m = ti.field(float)

neighbour = (3,) * dim

block0 = ti.root.pointer(ti.ijk, (4, 4, 4))
block1 = block0.pointer(ti.ijk, (8, 8, 8)) 
pixel = block1.dense(ti.ijk, (8, 8, 8))  
pixel.place(F_grid_v, F_grid_m)

@ti.kernel
def substep():
    for p in F_x:
        base = ti.cast((F_x[p] * inv_dx - 0.5), ti.i32)
        fx = F_x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        U, sig, V = ti.svd(F_F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            J *= sig[d, d]

        mu = mu_0
        la = lambda_0
        if material[p] == 0:
            F_F[p] = [[J, 0, 0], 
                      [0, 1, 0], 
                      [0, 0, 1]]
            mu = 0.0

        stress = 2 * mu * (F_F[p] - U @ V.transpose()) @ F_F[p].transpose() + ti.Matrix.identity(float, dim) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_idx = base + ti.cast(offset, ti.i32)
            F_grid_v[grid_idx] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[grid_idx] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
            F_grid_v[I][1] -= dt * 9.8
            if (I[0] < bound and F_grid_v[I][0] < 0.0) or (I[0] > n_grid-bound and F_grid_v[I][0] > 0.0):
                F_grid_v[I][0] = 0.0
            if (I[1] < bound and F_grid_v[I][1] < 0.0) or (I[1] > n_grid-bound and F_grid_v[I][1] > 0.0):
                F_grid_v[I][1] = 0.0
            if (I[2] < bound and F_grid_v[I][2] < 0.0) or (I[2] > n_grid-bound and F_grid_v[I][2] > 0.0):
                F_grid_v[I][2] = 0.0
    for p in F_x:
        base = ti.cast((F_x[p] * inv_dx - 0.5), ti.i32)
        fx = F_x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx)
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_idx = base + ti.cast(offset, ti.i32)
            g_v = F_grid_v[grid_idx]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C
        F_F[p] = (ti.Matrix.identity(float, dim) + dt * F_C[p]) @ F_F[p]

@ti.kernel
def init():
    for i in range(n_particles):
        F_v[i] = ti.Matrix([0.0, 0.0, 0.0])
        F_F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if i < fluid_particles:
            material[i] = 0
        else:
            material[i] = 1

class DynamicNormalizer:
    def __init__(self):
        # 粒子 (Particles) 的统计
        self.pos_min = None; self.pos_max = None
        self.vel_min = None; self.vel_max = None
        self.C_min = None;   self.C_max = None
        self.F_min = None;   self.F_max = None
        self.initialized_particle = False

    def update_particle_stats(self, features):
        """用 (N,24) 的粒子特征更新统计"""
        pos = features[:, 0:3]
        vel = features[:, 3:6]
        C = features[:, 6:15]
        F = features[:, 15:24]

        if not self.initialized_particle:
            self.pos_min, self.pos_max = pos.min(axis=0), pos.max(axis=0)
            self.vel_min, self.vel_max = vel.min(axis=0), vel.max(axis=0)
            self.C_min, self.C_max = C.min(axis=0), C.max(axis=0)
            self.F_min, self.F_max = F.min(axis=0), F.max(axis=0)
            self.initialized_particle = True
        else:
            self.pos_min = np.minimum(self.pos_min, pos.min(axis=0))
            self.pos_max = np.maximum(self.pos_max, pos.max(axis=0))
            self.vel_min = np.minimum(self.vel_min, vel.min(axis=0))
            self.vel_max = np.maximum(self.vel_max, vel.max(axis=0))
            self.C_min = np.minimum(self.C_min, C.min(axis=0))
            self.C_max = np.maximum(self.C_max, C.max(axis=0))
            self.F_min = np.minimum(self.F_min, F.min(axis=0))
            self.F_max = np.maximum(self.F_max, F.max(axis=0))

    def compute_particle_params(self, method='minmax', eps=1e-8, F_use_delta=True):
        """计算粒子的最终归一化参数 (使用 'minmax')"""
        if not self.initialized_particle:
            raise ValueError("粒子统计从未更新。")
        
        pos_scale = np.maximum(np.maximum(np.abs(self.pos_min), np.abs(self.pos_max)), eps)
        vel_scale = np.maximum(np.maximum(np.abs(self.vel_min), np.abs(self.vel_max)), eps)
        C_scale = np.maximum(np.maximum(np.abs(self.C_min), np.abs(self.C_max)), eps)
        F_scale = np.maximum(np.maximum(np.abs(self.F_min), np.abs(self.F_max)), eps)

        params = {
            'pos': pos_scale, 'vel': vel_scale, 'C': C_scale, 'F': F_scale,
            'F_use_delta': F_use_delta
        }
        return params

    def normalize_particle_features(self, features, params):
        """归一化粒子特征 (用于 X)"""
        normalized = np.zeros_like(features, dtype=np.float32)
        #normalized[:, 0:3] = features[:, 0:3] / params['pos']
        normalized[:, 0:3] = (features[:, 0:3] / params['pos']) * 2.0 - 1.0
        normalized[:, 3:6] = features[:, 3:6] / params['vel']
        normalized[:, 6:15] = features[:, 6:15] / params['C']
        
        F_flat = features[:, 15:24]
        if params.get('F_use_delta', True):
            I_flat = np.tile(np.eye(3).reshape(-1), (features.shape[0], 1))
            normalized[:, 15:24] = (F_flat - I_flat) / params['F']
        else:   
            normalized[:, 15:24] = F_flat / params['F']
        return normalized

    def denormalize_particle_features(self, normalized, params):    
        """
        反归一化粒子特征 (用于反归一化网络预测)
        您将在您的训练代码中需要这个函数。
        """
        orig = np.zeros_like(normalized)
        #orig[:, 0:3] = normalized[:, 0:3] * params['pos']
        orig[:, 0:3] = (normalized[:, 0:3] + 1.0) / 2.0 * params['pos']
        orig[:, 3:6] = normalized[:, 3:6] * params['vel']
        orig[:, 6:15] = normalized[:, 6:15] * params['C']
        if params.get('F_use_delta', True):
            I_flat = np.tile(np.eye(3).reshape(-1), (normalized.shape[0], 1))
            orig[:, 15:24] = normalized[:, 15:24] * params['F'] + I_flat
        else:
            orig[:, 15:24] = normalized[:, 15:24] * params['F']
        return orig

# ----------------------------------------------------------------------------
# 2. 辅助函数
# ----------------------------------------------------------------------------
def create_24d_features(pos, vel, C, F):
    """仅创建24维特征向量，不进行归一化"""
    features = np.column_stack([
        pos,                    # 3维
        vel,                    # 3维
        C.reshape(-1, 9),       # 9维
        F.reshape(-1, 9)        # 9维
    ])
    return features

def main():
    npy_dir = f"{output_dir}/demo{demox}/npy"
    os.makedirs(npy_dir, exist_ok=True)
    

    print("=== 阶段一：开始模拟并保存所有 NPY 原始文件 ===")
    
    normalizer_pass1 = DynamicNormalizer()
    
    data = np.genfromtxt(f"{input_dir}/demo{demox}/csv/obj1.csv", delimiter=' ', dtype=np.float32)
    for i in range(2, file_num+1):
        data_ = np.genfromtxt(f"{input_dir}/demo{demox}/csv/obj{i}.csv", delimiter=' ', dtype=np.float32)
        data = np.vstack((data, data_))

    F_x.from_numpy(data)
    init()
    
    # --- 处理 P(t=0) ---
    pos = F_x.to_numpy()
    vel = F_v.to_numpy()
    C = F_C.to_numpy() 
    F = F_F.to_numpy()
    
    np.save(f"{npy_dir}/F_x_0.npy", pos)
    np.save(f"{npy_dir}/F_v_0.npy", vel)
    np.save(f"{npy_dir}/F_C_0.npy", C)
    np.save(f"{npy_dir}/F_F_0.npy", F)
    
    features_0 = create_24d_features(pos, vel, C, F)
    normalizer_pass1.update_particle_stats(features_0)

    # --- 模拟循环 (t=0 to 999) ---
    for frame in tqdm(range(1000), desc="模拟阶段 (阶段一)"):
        for s in range(steps):
            block0.deactivate_all()
            substep()
        
        # 1. 获取并保存 P(t+1)
        pos = F_x.to_numpy()
        vel = F_v.to_numpy()
        C = F_C.to_numpy() 
        F_arr = F_F.to_numpy()
        
        filenum_p = (frame + 1) * steps # P(1) 在 200...
        np.save(f"{npy_dir}/F_x_{filenum_p}.npy", pos)
        np.save(f"{npy_dir}/F_v_{filenum_p}.npy", vel)
        np.save(f"{npy_dir}/F_C_{filenum_p}.npy", C)
        np.save(f"{npy_dir}/F_F_{filenum_p}.npy", F_arr)

        features_t_plus_1 = create_24d_features(pos, vel, C, F_arr)
        normalizer_pass1.update_particle_stats(features_t_plus_1)

    print("=== 阶段一完成。模拟结束，已保存所有 NPY 原始文件。 ===")

    print("=== 阶段二：开始归一化X(P(t)) 并 写入Y(P(t+1)原始值) ===")
    
    try:
        particle_params = normalizer_pass1.compute_particle_params(method='minmax')
        norm_param_path = f"{output_dir}/demo{demox}/normalization_params.npy"
        np.save(norm_param_path, particle_params)
        print(f"粒子归一化参数已保存到: {norm_param_path}")
    except ValueError as e:
        print(f"错误：无法计算归一化参数。{e}")
        return

    input_h5_path = f"{output_dir}/demo{demox}/input.h5"
    output_h5_path = f"{output_dir}/demo{demox}/output.h5"
    
    with h5py.File(input_h5_path, 'w') as f_in, h5py.File(output_h5_path, 'w') as f_out:
        
        for key, value in particle_params.items():
            if isinstance(value, (np.ndarray, list)):
                 f_in.attrs[f'norm_{key}'] = value
            else:
                 f_in.attrs[f'norm_{key}'] = np.array([value])
        
        f_in.attrs['info'] = "归一化的粒子数据 P(t) (Normalized P(t))"
        f_out.attrs['info'] = "原始物理粒子数据 P(t+1) (Raw P(t+1))"

        for idx in tqdm(range(1000), desc="归一化/写入阶段 (阶段二)"):
            
            filenum_p_t = idx * steps # t=0 -> 0; t=1 -> 200; ...
            
            try:
                pos_x = np.load(f"{npy_dir}/F_x_{filenum_p_t}.npy")
                vel_x = np.load(f"{npy_dir}/F_v_{filenum_p_t}.npy")
                C_x = np.load(f"{npy_dir}/F_C_{filenum_p_t}.npy")
                F_arr_x = np.load(f"{npy_dir}/F_F_{filenum_p_t}.npy")
            except FileNotFoundError:
                print(f"警告：找不到 X=P(t={idx}) 的 NPY 文件 (filenum {filenum_p_t})。跳过此帧。")
                continue

            original_features_x = create_24d_features(pos_x, vel_x, C_x, F_arr_x)
            normalized_features_x = normalizer_pass1.normalize_particle_features(original_features_x, particle_params)
            
            f_in.create_dataset(str(idx), data=normalized_features_x)
            
            # P(t+1) 对应的 filenum
            filenum_p_t_plus_1 = (idx + 1) * steps # t=0 -> 200 (P(1)); t=999 -> 200000 (P(1000))

            try:
                pos_y = np.load(f"{npy_dir}/F_x_{filenum_p_t_plus_1}.npy")
                vel_y = np.load(f"{npy_dir}/F_v_{filenum_p_t_plus_1}.npy")
                C_y = np.load(f"{npy_dir}/F_C_{filenum_p_t_plus_1}.npy")
                F_arr_y = np.load(f"{npy_dir}/F_F_{filenum_p_t_plus_1}.npy")
            except FileNotFoundError:
                print(f"警告：找不到 Y=P(t={idx+1}) 的 NPY 文件 (filenum {filenum_p_t_plus_1})。跳过此帧。")
                if str(idx) in f_in:
                    del f_in[str(idx)]
                continue

            raw_features_y = create_24d_features(pos_y, vel_y, C_y, F_arr_y)
            
            f_out.create_dataset(str(idx), data=raw_features_y)
            
    print("=== 阶段二完成。 ===")
    print(f"Input (X)  [归一化 P(t)]: {input_h5_path}")
    print(f"Output (Y) [原始值 P(t+1)]: {output_h5_path}")
    print("请确保在您的训练代码中加载 'normalization_params.npy' 来反归一化网络预测 (P_pred)。")

if __name__ == "__main__":
    main()