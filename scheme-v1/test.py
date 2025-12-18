import torch
import os
import numpy as np
from PTModules import ParticleTransformer
from tqdm import tqdm
import h5py
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

SAVE_PATH = '/home/wqz/fluid/experiment/demo1/scheme-v1/saved'
DATA_PATH = '/home/wqz/fluid/data'
DEMO_IDX = 1           
TEST_LEN = 1000        
DIM_IN = 24            
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EVAL_DIR = os.path.join(SAVE_PATH, 'eval_results_fluid_v1')
PLY_DIR = os.path.join(EVAL_DIR, 'ply_frames')
os.makedirs(PLY_DIR, exist_ok=True)

def apply_boundary_condition(pos, vel, bound_min, bound_max, damping=0.0):
    """
    处理正方形容器的边界条件 (Free-slip with optional damping)
    参数:
        pos: [B, N, 3] 粒子位置
        vel: [B, N, 3] 粒子速度
        bound_min: float 边界下限 (e.g., 0.05)
        bound_max: float 边界上限 (e.g., 0.95)
        damping: float 墙面摩擦系数 (0.0 = 完全光滑/Free-slip, 1.0 = 完全粘滞/No-slip)
    """
    mask_low = pos < bound_min
    mask_high = pos > bound_max
    pos = torch.clamp(pos, min=bound_min, max=bound_max)
    collision_mask = mask_low | mask_high
    if damping == 0.0:
        vel[collision_mask] = 0.0

    return pos, vel

def denormalize_prediction(norm_out, params_dict):
    """归一化 -> 物理坐标"""
    phys_out = torch.empty_like(norm_out)
    # 位置反归一化: [-1, 1] -> [0, pos_scale]
    phys_out[..., 0:3] = (norm_out[..., 0:3] + 1.0) / 2.0 * params_dict['pos']
    phys_out[..., 3:6] = norm_out[..., 3:6] * params_dict['vel']
    phys_out[..., 6:15] = norm_out[..., 6:15] * params_dict['C']
    
    norm_F_flat = norm_out[..., 15:24]
    if params_dict.get('F_use_delta', True):
        I_flat = torch.eye(3, device=norm_out.device).view(1, 1, 9)
        phys_out[..., 15:24] = norm_F_flat * params_dict['F'] + I_flat
    else:
        phys_out[..., 15:24] = norm_F_flat * params_dict['F']
    return phys_out

def normalize_physical_data(phys_data, params_dict):
    """物理坐标 -> 归一化"""
    norm_data = torch.empty_like(phys_data)
    # 位置归一化: [0, pos_scale] -> [-1, 1]
    norm_data[..., 0:3] = (phys_data[..., 0:3] / params_dict['pos']) * 2.0 - 1.0
    norm_data[..., 3:6] = phys_data[..., 3:6] / params_dict['vel']
    norm_data[..., 6:15] = phys_data[..., 6:15] / params_dict['C']
    
    phys_F_flat = phys_data[..., 15:24]
    if params_dict.get('F_use_delta', True):
        I_flat = torch.eye(3, device=phys_data.device).view(1, 1, 9)
        norm_data[..., 15:24] = (phys_F_flat - I_flat) / params_dict['F']
    else:
        norm_data[..., 15:24] = phys_F_flat / params_dict['F']
    return norm_data

def compute_rmse(pred, true):
    mse = torch.mean((pred - true) ** 2)
    return torch.sqrt(mse).item()

def write_ply_binary(filename, positions):
    positions = positions.astype(np.float32)
    num_points = positions.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(filename, 'wb') as f:
        f.write(header.encode('utf-8'))
        positions.tofile(f)

def main():
    print(f"🚀 开始纯流体测试 (FP32)...")
    torch.set_float32_matmul_precision('high')
    
    # 1. 加载参数
    print("📥 Loading params...")
    params_path = os.path.join(DATA_PATH, f'demo{DEMO_IDX}', 'normalization_params.npy')
    norm_params_np = np.load(params_path, allow_pickle=True).item()
    norm_params_torch = {
        'pos': torch.tensor(norm_params_np['pos'], dtype=torch.float32, device=DEVICE),
        'vel': torch.tensor(norm_params_np['vel'], dtype=torch.float32, device=DEVICE),
        'C': torch.tensor(norm_params_np['C'], dtype=torch.float32, device=DEVICE),
        'F': torch.tensor(norm_params_np['F'], dtype=torch.float32, device=DEVICE),
        'F_use_delta': norm_params_np.get('F_use_delta', True)
    }

    # 2. 加载模型
    model = ParticleTransformer(dim_in=DIM_IN).to(DEVICE)
    checkpoint_path = os.path.join(SAVE_PATH, "best_model.pt")
    print(f"🧠 Loading Model: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 3. 数据读取
    input_file = os.path.join(DATA_PATH, f'demo{DEMO_IDX}', 'input.h5')
    output_file = os.path.join(DATA_PATH, f'demo{DEMO_IDX}', 'output.h5')
    
    f_in = h5py.File(input_file, 'r', rdcc_nbytes=1024*1024*200)
    f_out = h5py.File(output_file, 'r', rdcc_nbytes=1024*1024*200)

    # 初始化状态
    curr_state_norm = torch.tensor(f_in['0'][:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # 动态获取粒子数，不再 hardcode
    N_total = curr_state_norm.shape[1] 
    mask = torch.ones((1, N_total), dtype=torch.float32, device=DEVICE)
    
    print(f"💧 检测到流体粒子数: {N_total}")

    I_batch = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0) # [1, 1, 3, 3]
    
    # 边界参数
    grid_margin = 3.0 / 128.0 
    epsilon = 1e-5
    BOUND_MIN = grid_margin + epsilon
    BOUND_MAX = 1.0 - (grid_margin + epsilon)

    rmse_pos_list = []
    rmse_vel_list = []
    
    with torch.no_grad():
        for t in tqdm(range(TEST_LEN), desc="Rollout"):
            
            pred_next_norm = model(curr_state_norm, mask)
            
            # 物理空间转换
            pred_next_phys = denormalize_prediction(pred_next_norm, norm_params_torch)
            
            # 纯流体后处理
            
            # 强制 F 重置 
            F_flat = pred_next_phys[..., 15:24]
            F_matrix = F_flat.view(-1, N_total, 3, 3)
            
            # 计算雅可比行列式 J
            J = torch.det(F_matrix) # [B, N]
            
            F_reset = I_batch.repeat(F_matrix.shape[0], N_total, 1, 1).clone()
            F_reset[:, :, 0, 0] = J

            pred_next_phys[..., 15:24] = F_reset.view(-1, N_total, 9)
            

            pos = pred_next_phys[..., 0:3]
            vel = pred_next_phys[..., 3:6]
            
            pos_new, vel_new = apply_boundary_condition(
                pos, vel, 
                bound_min=BOUND_MIN, 
                bound_max=BOUND_MAX,
                damping=0.0 
            )

            pred_next_phys[..., 0:3] = pos_new
            pred_next_phys[..., 3:6] = vel_new
            
            # 保存 PLY (用于 Houdini 可视化) ---
            # 仅保存位置信息
            pos_np = pred_next_phys[0, :, 0:3].cpu().numpy()
            save_name = os.path.join(PLY_DIR, f"frame_{t:04d}.ply")
            write_ply_binary(save_name, pos_np)

            # 计算误差
            gt_key = str(t)
            if gt_key in f_out:
                gt_next_phys = torch.tensor(f_out[gt_key][:], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                r_pos = compute_rmse(pred_next_phys[..., 0:3], gt_next_phys[..., 0:3])
                r_vel = compute_rmse(pred_next_phys[..., 3:6], gt_next_phys[..., 3:6])
                rmse_pos_list.append(r_pos)
                rmse_vel_list.append(r_vel)
            
            # 准备下一帧输入
            curr_state_norm = normalize_physical_data(pred_next_phys, norm_params_torch)

    f_in.close()
    f_out.close()
    
    steps = np.arange(len(rmse_pos_list))
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, rmse_pos_list, label='Position RMSE', color='blue', linewidth=1)
    plt.title(f'Fluid-Only Position RMSE (Mean: {np.mean(rmse_pos_list):.4f})')
    plt.xlabel('Step')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, rmse_vel_list, label='Velocity RMSE', color='orange', linewidth=1)
    plt.title('Velocity RMSE')
    plt.xlabel('Step')
    plt.grid(True, alpha=0.3)
    
    chart_path = os.path.join(EVAL_DIR, 'rmse_report_fluid.png')
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    
    print(f"\n✅ 纯流体测试完成！")
    print(f"   - 帧数据保存至: {PLY_DIR}")
    print(f"   - 误差曲线保存至: {chart_path}")

if __name__ == "__main__":
    main()