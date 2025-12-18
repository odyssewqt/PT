import torch
import os
import numpy as np
import time  
import h5py
from PTModules import ParticleTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
DATA_PATH = '/home/wqz/fluid/data'
SAVE_PATH = '/home/wqz/fluid/experiment/demo1/scheme-v2/saved'
DEMO_IDX = 1
DIM_IN = 24

WARM_UP_STEPS = 50   # 预热步数 
TEST_STEPS = 1000     # 正式测试步数 

def denormalize_prediction(norm_out, params_dict):
    """简化的反归一化 (仅保留计算逻辑)"""
    phys_out = torch.empty_like(norm_out)
    phys_out[..., 0:3] = (norm_out[..., 0:3] + 1.0) / 2.0 * params_dict['pos']
    phys_out[..., 3:6] = norm_out[..., 3:6] * params_dict['vel']
    phys_out[..., 6:15] = norm_out[..., 6:15] * params_dict['C']
    norm_F_flat = norm_out[..., 15:24]
    
    # 假设 F_use_delta=True
    I_flat = torch.eye(3, device=norm_out.device).view(1, 1, 9)
    phys_out[..., 15:24] = norm_F_flat * params_dict['F'] + I_flat
    return phys_out

def normalize_physical_data(phys_data, params_dict):
    """简化的归一化"""
    norm_data = torch.empty_like(phys_data)
    norm_data[..., 0:3] = (phys_data[..., 0:3] / params_dict['pos']) * 2.0 - 1.0
    norm_data[..., 3:6] = phys_data[..., 3:6] / params_dict['vel']
    norm_data[..., 6:15] = phys_data[..., 6:15] / params_dict['C']
    
    phys_F_flat = phys_data[..., 15:24]
    I_flat = torch.eye(3, device=phys_data.device).view(1, 1, 9)
    norm_data[..., 15:24] = (phys_F_flat - I_flat) / params_dict['F']
    return norm_data

def apply_boundary_condition(pos, vel):
    """简化的边界处理"""
    bound_min = 3.0/128.0 + 1e-5
    bound_max = 1.0 - (3.0/128.0 + 1e-5)
    pos = torch.clamp(pos, min=bound_min, max=bound_max)
    
    return pos, vel

def main():
    print(f"🏎️  开始速度基准测试 (Device: {DEVICE})...")
    
    params_path = os.path.join(DATA_PATH, f'demo{DEMO_IDX}', 'normalization_params.npy')
    norm_params_np = np.load(params_path, allow_pickle=True).item()
    norm_params_torch = {
        'pos': torch.tensor(norm_params_np['pos'], dtype=torch.float32, device=DEVICE),
        'vel': torch.tensor(norm_params_np['vel'], dtype=torch.float32, device=DEVICE),
        'C': torch.tensor(norm_params_np['C'], dtype=torch.float32, device=DEVICE),
        'F': torch.tensor(norm_params_np['F'], dtype=torch.float32, device=DEVICE)
    }

    input_file = os.path.join(DATA_PATH, f'demo{DEMO_IDX}', 'input.h5')
    f_in = h5py.File(input_file, 'r')
    curr_state_norm = torch.tensor(f_in['0'][:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    N_total = curr_state_norm.shape[1]
    mask = torch.ones((1, N_total), dtype=torch.float32, device=DEVICE)
    f_in.close()
    
    print(f"💧 测试粒子数: {N_total}")

    model = ParticleTransformer(dim_in=DIM_IN).to(DEVICE)
    checkpoint_path = os.path.join(SAVE_PATH, "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # 预热 (Warm-up)
    print(f"🔥 正在预热 GPU ({WARM_UP_STEPS} steps)...")
    with torch.no_grad():
        for _ in range(WARM_UP_STEPS):
            _ = model(curr_state_norm, mask)
    torch.cuda.synchronize() # 确保预热完成
    print("✅ 预热完成。")

    print(f"⏱️  开始测试 ({TEST_STEPS} steps)...")
    
    timings = []
    
    # 模拟必要的辅助变量
    I_batch = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for i in range(TEST_STEPS):
            # --- 计时起点 ---
            torch.cuda.synchronize()
            t_start = time.time()
            
            pred_next_norm = model(curr_state_norm, mask)
            
            pred_next_phys = denormalize_prediction(pred_next_norm, norm_params_torch)
            
            F_flat = pred_next_phys[..., 15:24]
            F_reset = I_batch.expand(1, N_total, 3, 3).reshape(1, N_total, 9)
            pred_next_phys[..., 15:24] = F_reset
            
            pos = pred_next_phys[..., 0:3]
            vel = pred_next_phys[..., 3:6]
            pos_new, vel_new = apply_boundary_condition(pos, vel)
            pred_next_phys[..., 0:3] = pos_new
            pred_next_phys[..., 3:6] = vel_new

            curr_state_norm = normalize_physical_data(pred_next_phys, norm_params_torch)

            # --- 计时终点 ---
            torch.cuda.synchronize()
            t_end = time.time()
            
            timings.append(t_end - t_start)
            
            if (i+1) % 50 == 0:
                print(f"   Step {i+1}/{TEST_STEPS}: {timings[-1]*1000:.2f} ms")

    # 统计结果
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    fps = 1.0 / avg_time
    
    print("\n" + "="*40)
    print(f"📊 性能测试报告 (Particle Transformer)")
    print(f"   - 粒子数量: {N_total}")
    print(f"   - 平均耗时: {avg_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"   - 推理速度: {fps:.2f} FPS (Frames Per Second)")
    print("="*40)

if __name__ == "__main__":
    main()