import taichi as ti
import torch.nn as nn
import os
ti.init(arch=ti.cuda, debug=False, default_fp=ti.f32)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PTModules import ParticleTransformer
import torch.optim as optim
from tqdm import tqdm

fluids = [
    37908*3
]

# 反归一化函数
def denormalize_prediction(norm_out, params_dict):
    assert norm_out.shape[-1] == 24, "反归一化需要 24 维的输入"
    phys_out = torch.zeros_like(norm_out)
    #phys_out[..., 0:3] = norm_out[..., 0:3] * params_dict['pos']
    phys_out[..., 0:3] = (norm_out[..., 0:3] + 1.0) / 2.0 * params_dict['pos']
    phys_out[..., 3:6] = norm_out[..., 3:6] * params_dict['vel']
    phys_out[..., 6:15] = norm_out[..., 6:15] * params_dict['C']
    norm_F_flat = norm_out[..., 15:24]
    if params_dict.get('F_use_delta', True):
        I_flat = torch.eye(3, device=norm_out.device).reshape(1, 1, 9).expand(
            norm_out.shape[0], norm_out.shape[1], -1
        )
        phys_out[..., 15:24] = norm_F_flat * params_dict['F'] + I_flat
    else:
        phys_out[..., 15:24] = norm_F_flat * params_dict['F']
    return phys_out

def normalize_physical_data(phys_data, params_dict):
    """
    将物理空间的真值 y 转换为归一化空间，以便与网络输出直接对比。
    phys_data: (B, N, 24)
    """
    norm_data = torch.zeros_like(phys_data)
    #norm_data[..., 0:3] = phys_data[..., 0:3] / params_dict['pos']
    norm_data[..., 0:3] = (phys_data[..., 0:3] / params_dict['pos']) * 2.0 - 1.0
    norm_data[..., 3:6] = phys_data[..., 3:6] / params_dict['vel']
    norm_data[..., 6:15] = phys_data[..., 6:15] / params_dict['C']
    
    phys_F_flat = phys_data[..., 15:24]
    
    if params_dict.get('F_use_delta', True):
        I_flat = torch.eye(3, device=phys_data.device).reshape(1, 1, 9).expand(
            phys_data.shape[0], phys_data.shape[1], -1
        )
        norm_data[..., 15:24] = (phys_F_flat - I_flat) / params_dict['F']
    else:
        norm_data[..., 15:24] = phys_F_flat / params_dict['F']
    return norm_data

class ParticleDataset(Dataset):
    def __init__(self, data_path, file_path, demo_idx=1):
        self.data_path = data_path
        self.file_path = file_path
        self.demo_idx = demo_idx

        idx_path = fr"{self.data_path}/demo{self.demo_idx}/file_index.h5"
        try:
            with h5py.File(idx_path, 'r') as f_idx:
                self.idx_arr = f_idx['index'][:]
        except FileNotFoundError:
            print(f"警告: 找不到 file_index.h5 at {idx_path}")
            print("将使用 0-999 的顺序索引。")
            self.idx_arr = [str(i).encode('utf-8') for i in range(1000)]

    def __len__(self):
        return 1000
    
    def getFileIdx(self, global_index):
        assert global_index < 1000, "数据集没那么多！"
        return global_index // 1000 + 1

    def __getitem__(self, idx):
        # idx 是 0 到 999 的整数
        file_idx = self.getFileIdx(idx)
        
        with h5py.File(fr"{self.data_path}/demo{file_idx}/input.h5", 'r') as f_in:
            x = torch.tensor(f_in[str(idx)][:], dtype=torch.float32)
        with h5py.File(fr"{self.data_path}/demo{file_idx}/output.h5", 'r') as f_out:
            y1_p2p = torch.tensor(f_out[str(idx)][:], dtype=torch.float32)

        return x, y1_p2p

def collate_fn(batch):
    """
    [修改] 自定义 collate_fn
    - 填充 x (P(t))
    - 填充 y1 (P(t+1))
    - 移除 y2
    """
    # 获取序列的最大长度
    max_seq_len = max(x.shape[0] for x, _ in batch)
    
    padded_inputs_x = []
    padded_targets_y1 = [] 
    masks = []
    
    for x, y1_p2p in batch:
        
        # 1. 填充输入序列 X = P(t)
        padded_x = torch.cat([
            x, 
            torch.zeros(max_seq_len - x.shape[0], x.shape[1])
        ], dim=0)
        
        # 2. 填充目标序列 Y = P(t+1)
        padded_y1 = torch.cat([
            y1_p2p,
            torch.zeros(max_seq_len - y1_p2p.shape[0], y1_p2p.shape[1])
        ], dim=0)

        # 3. 创建 mask
        mask = torch.cat([
            torch.ones(x.shape[0]),
            torch.zeros(max_seq_len - x.shape[0])
        ], dim=0)

        padded_inputs_x.append(padded_x)
        padded_targets_y1.append(padded_y1) 
        masks.append(mask)
    
    # 转换为批量 tensor
    padded_inputs_x = torch.stack(padded_inputs_x, dim=0)
    padded_targets_y1 = torch.stack(padded_targets_y1, dim=0) 
    masks = torch.stack(masks, dim=0)
    
    # 返回 (x, y1, y3, mask)
    return padded_inputs_x, padded_targets_y1, masks

dim = 3
n_grid = 128
dt = 5e-5
epsilon = 1e-14

n_particles = 37908+37908+37908
dx = 1 / n_grid
inv_dx = float(n_grid)
bound = 3

p_rho = 1.0
p_vol = dx ** dim
p_mass = p_rho * p_vol
gravity = 9.8
E = 1.2e3
nu = 0.3
la = E * nu / ((1 + nu) * (1 - 2 * nu))
MU_0_PHY = E / (2 * (1 + nu)) 

# --- 预测 (Prediction) 粒子字段 ---
F_x_pre = ti.Vector.field(3, dtype=ti.f32, shape=(n_particles), needs_grad=True)
F_v_pre = ti.Vector.field(3, dtype=ti.f32, shape=(n_particles), needs_grad=True)
F_F_pre = ti.Matrix.field(3, 3, ti.f32, n_particles, needs_grad=True)
F_C_pre = ti.Matrix.field(3, 3, ti.f32, n_particles, needs_grad=True)
F_R_pre = ti.Matrix.field(3, 3, ti.f32, n_particles, needs_grad=True)
F_mu_pre = ti.field(dtype=ti.f32, shape=(n_particles), needs_grad=True)

# --- 真值 (Ground Truth) 粒子字段 ---
F_x_tru = ti.Vector.field(3, dtype=ti.f32, shape=(n_particles), needs_grad=False)
F_v_tru = ti.Vector.field(3, dtype=ti.f32, shape=(n_particles), needs_grad=False)
F_F_tru = ti.Matrix.field(3, 3, ti.f32, n_particles, needs_grad=False)
F_C_tru = ti.Matrix.field(3, 3, ti.f32, n_particles, needs_grad=False)
F_R_tru = ti.Matrix.field(3, 3, ti.f32, n_particles, needs_grad=False)

# --- 网格 (Grid) 字段 (pre 和 tru) ---
F_grid_v_pre = ti.Vector.field(dim, float, (n_grid,) * dim, needs_grad=True)
F_grid_m_pre = ti.field(float, (n_grid,) * dim, needs_grad=True)
F_grid_v_tru = ti.Vector.field(dim, float, (n_grid,) * dim, needs_grad=False)
F_grid_m_tru = ti.field(float, (n_grid,) * dim, needs_grad=False)

neighbour = (3,) * dim

# --- 损失 (Loss) 字段---
loss1 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss1_m = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss1_v = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

total_loss_m_unavg = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
total_loss_v_unavg = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
total_active_cells_m = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
total_weight_sum_v = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def clear_fields():
    for p in range(n_particles):
        F_x_pre[p] = [0, 0, 0]; F_v_pre[p] = [0, 0, 0]
        F_C_pre[p].fill(0); F_F_pre[p].fill(0); F_R_pre[p].fill(0)
        F_mu_pre[p] = 0.0
        F_x_pre.grad[p] = [0, 0, 0]; F_v_pre.grad[p] = [0, 0, 0]
        F_C_pre.grad[p].fill(0); F_F_pre.grad[p].fill(0); F_R_pre.grad[p].fill(0)
        F_mu_pre.grad[p] = 0.0
        
        F_x_tru[p] = [0, 0, 0]; F_v_tru[p] = [0, 0, 0]
        F_C_tru[p].fill(0); F_F_tru[p].fill(0); F_R_tru[p].fill(0)
    for I in ti.grouped(F_grid_m_pre):
        F_grid_v_pre[I] = [0, 0, 0]; F_grid_v_tru[I] = [0, 0, 0]
        F_grid_m_pre[I] = 0; F_grid_m_tru[I] = 0
        F_grid_v_pre.grad[I] = [0, 0, 0]; 
        F_grid_m_pre.grad[I] = 0; 
    
    total_loss_m_unavg[None] = 0.0; total_loss_v_unavg[None] = 0.0
    total_active_cells_m[None] = 0.0; total_weight_sum_v[None] = 0.0
    total_loss_m_unavg.grad[None] = 0.0; total_loss_v_unavg.grad[None] = 0.0
    total_active_cells_m.grad[None] = 0.0; total_weight_sum_v.grad[None] = 0.0

@ti.kernel
def copy(
    x_pred: ti.types.ndarray(),   # 预测值 (N, 24)
    x_true: ti.types.ndarray(),   # 真值 (N, 24)
    n: ti.i32,                    # 粒子数
    R_pred: ti.types.ndarray(),   # 预测的 R (N, 3, 3)
    R_true: ti.types.ndarray()  # 真值的 R (N, 3, 3)
):
    for i in range(n):
        idx = 0
        for j in ti.static(range(3)): F_x_pre[i][j] = x_pred[i, idx]; idx += 1
        for j in ti.static(range(3)): F_v_pre[i][j] = x_pred[i, idx]; idx += 1
        for j, k in ti.static(ti.ndrange(3, 3)): F_C_pre[i][j, k] = x_pred[i, idx]; idx += 1
        for j, k in ti.static(ti.ndrange(3, 3)): F_F_pre[i][j, k] = x_pred[i, idx]; idx += 1
        
        F_mu_pre[i] = 0.0
        
        for j, k in ti.static(ti.ndrange(3, 3)): F_R_pre[i][j, k] = R_pred[i, j, k]

    for i in range(n):
        idx = 0
        for j in ti.static(range(3)): F_x_tru[i][j] = x_true[i, idx]; idx += 1
        for j in ti.static(range(3)): F_v_tru[i][j] = x_true[i, idx]; idx += 1
        for j, k in ti.static(ti.ndrange(3, 3)): F_C_tru[i][j, k] = x_true[i, idx]; idx += 1
        for j, k in ti.static(ti.ndrange(3, 3)): F_F_tru[i][j, k] = x_true[i, idx]; idx += 1

        for j, k in ti.static(ti.ndrange(3, 3)): F_R_tru[i][j, k] = R_true[i, j, k]

# --- P2G (Prediction) ---
@ti.kernel
def p2g_fluid_pre(n_fluid: ti.i32):
    for p in range(n_fluid):
        #x_clipped = ti.max(F_x_pre[p], 0.00195313)
        x_clipped = F_x_pre[p]
        x_clamped = ti.max(x_clipped, 0.5 * dx)
        x_clipped = ti.min(x_clamped, 1.0 - 0.5 * dx)
        
        base = ti.cast(x_clipped * inv_dx - 0.5, ti.i32)
        fx = x_clipped * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        mu = 0.0 
        J = abs(F_F_pre[p].determinant())
        new_F = ti.Matrix([[J, 0, 0], [0, 1, 0], [0, 0, 1]], ti.f32)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * (2 * mu * (new_F - F_R_pre[p]) @ new_F.transpose() + ti.Matrix.diag(dim, la * J * (J - 1)))
        affine = stress + p_mass * F_C_pre[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (ti.cast(ti.Vector(offset), ti.f32) - fx) * dx
            weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
            F_grid_v_pre[base + offset] += weight * (p_mass * F_v_pre[p] + affine @ dpos)
            F_grid_m_pre[base + offset] += weight * p_mass

# --- P2G (Ground Truth) ---
@ti.kernel
def p2g_fluid_tru(n_fluid: ti.i32):
    for p in range(n_fluid):
        #x_clipped = ti.max(F_x_tru[p], 0.00195313)
        x_clipped = F_x_tru[p]
        x_clamped = ti.max(x_clipped, 0.5 * dx)
        x_clipped = ti.min(x_clamped, 1.0 - 0.5 * dx)
        base = ti.cast(x_clipped * inv_dx - 0.5, ti.i32)
        fx = x_clipped * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        mu = 0.0 
        J = abs(F_F_tru[p].determinant())
        new_F = ti.Matrix([[J, 0, 0], [0, 1, 0], [0, 0, 1]], ti.f32)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * (2 * mu * (new_F - F_R_tru[p]) @ new_F.transpose() + ti.Matrix.diag(dim, la * J * (J - 1)))
        affine = stress + p_mass * F_C_tru[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (ti.cast(ti.Vector(offset), ti.f32) - fx) * dx
            weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
            F_grid_v_tru[base + offset] += weight * (p_mass * F_v_tru[p] + affine @ dpos)
            F_grid_m_tru[base + offset] += weight * p_mass

# --- Loss Kernels ---
@ti.kernel
def compute_loss_part1():
    for I in ti.grouped(F_grid_m_pre):
        if (F_grid_m_pre[I] > epsilon) or (F_grid_m_tru[I] > epsilon):
            diff_mass = F_grid_m_pre[I] - F_grid_m_tru[I]

            loss11 = diff_mass ** 2 
            total_loss_m_unavg[None] += loss11
            total_active_cells_m[None] += 1.0

        mv_pre = F_grid_v_pre[I] 
        mv_tru = F_grid_v_tru[I]
        if F_grid_m_tru[I] > epsilon or F_grid_m_pre[I] > epsilon:
            diff_mom_sqr = (mv_pre - mv_tru).norm_sqr()
            loss_v_stable = diff_mom_sqr 
            total_loss_v_unavg[None] += loss_v_stable
            total_weight_sum_v[None] += 1.0 
@ti.kernel
def compute_loss_part2():
    denom_m = ti.max(total_active_cells_m[None], 1.0)
    denom_v = ti.max(total_weight_sum_v[None], epsilon)
    loss1_m[None] = 1e11 * total_loss_m_unavg[None] / denom_m
    loss1_v[None] = 1e11 * total_loss_v_unavg[None] / denom_v
    loss1[None] = loss1_m[None] + loss1_v[None]

class ParticleLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_physical_pred, x_physical_true, masks, R_pred, R_true):
        
        B = x_physical_pred.shape[0]
        N = x_physical_pred.shape[1]
        device = x_physical_pred.device
        
        grads_pred = torch.zeros((B, N, 24), device=device)
        r_grads_pred = torch.zeros((B, N, 3, 3), device=device)
        
        losses = torch.zeros(B, device=device)
        losses_v = torch.zeros(B, device=device)
        losses_m = torch.zeros(B, device=device)

        requires_grad = torch.is_grad_enabled()

        for b in range(B):
            n = int(masks[b].sum().item())
            clear_fields()
            copy(
                x_physical_pred[b], 
                x_physical_true[b],
                n, 
                R_pred[b].contiguous(),
                R_true[b].contiguous()
            )
     
            loss1[None] = 0.0
            loss1_m[None] = 0.0
            loss1_v[None] = 0.0

            if requires_grad:
                with ti.ad.Tape(loss1):
                    p2g_fluid_pre(n)
                    p2g_fluid_tru(n)
                    compute_loss_part1()
                    compute_loss_part2()

                x_grad_pre = F_x_pre.grad.to_torch()
                v_grad_pre = F_v_pre.grad.to_torch()
                F_grad_pre = F_F_pre.grad.to_torch().view(-1, 9)
                C_grad_pre = F_C_pre.grad.to_torch().view(-1, 9)

                grads_pred[b] = torch.cat((x_grad_pre[:N, :], v_grad_pre[:N, :], C_grad_pre[:N, :], F_grad_pre[:N, :]), 1)
                r_grads_pred[b] = F_R_pre.grad.to_torch()[:N, :, :]

            else:
                p2g_fluid_pre(n)
                p2g_fluid_tru(n)
                compute_loss_part1()
                compute_loss_part2()

            losses[b] = loss1[None]
            losses_v[b] = loss1_v[None]
            losses_m[b] = loss1_m[None]

        if requires_grad:
            ctx.save_for_backward(grads_pred, r_grads_pred)

        return losses.mean(), losses_m.mean(), losses_v.mean()
    
    @staticmethod
    def backward(ctx, grad_output1, grad_output2, grad_output3):
        if not ctx.saved_tensors:
            return None, None, None, None, None
            
        grads_pred, r_grads_pred = ctx.saved_tensors
        
        return grad_output1 * grads_pred, \
               None, \
               None, \
               grad_output1 * r_grads_pred, \
               None
def masked_huber_loss(pred, true, mask, delta=1.0):
    """
    MPMNet 式的 Huber Loss 实现
    """
    diff = torch.abs(pred - true)
    loss_sq = 0.5 * diff ** 2
    loss_lin = delta * diff - 0.5 * delta ** 2
    loss_elementwise = torch.where(diff <= delta, loss_sq, loss_lin)

    mask_expanded = mask.unsqueeze(-1)

    loss_masked = loss_elementwise * mask_expanded

    num_valid_elements = mask_expanded.sum() * pred.shape[-1]
    return loss_masked.sum() / num_valid_elements.clamp(min=1)

def particle_loss(pred, true, mask):
    loss_huber = masked_huber_loss(pred, true, mask, delta=0.1)
    return loss_huber

def compute_boundary_loss(pos, n_grid=128, bound=3):
    """
    计算边界惩罚损失。
    pos: (B, N, 3) 物理坐标
    n_grid: 网格分辨率
    bound: 边界厚度 (你的代码里是 3)
    """
    dx = 1.0 / n_grid
    margin = bound * dx  # 3 * (1/128) ≈ 0.0234
    
    min_limit = margin
    max_limit = 1.0 - margin
    
    diff_min = torch.relu(min_limit - pos)
    
    diff_max = torch.relu(pos - max_limit)
    
    loss = torch.mean(diff_min**2 + diff_max**2)
    
    return loss

def compute_rotation_matrix(F_flat):
    """
    使用 Newton-Schulz 迭代法从形变梯度 F 中提取旋转矩阵 R。
    输入 F_flat: (B, N, 9)
    输出 R:      (B, N, 3, 3)
    """
    B, N, _ = F_flat.shape
    F = F_flat.view(-1, 3, 3)
    
    norm = torch.norm(F, p='fro', dim=(1, 2), keepdim=True).clamp(min=1e-6)
    X = F / norm 
    I = torch.eye(3, device=F.device).unsqueeze(0)
    
    for _ in range(5):
        XTX = torch.matmul(X.transpose(1, 2), X)
        X = 0.5 * torch.matmul(X, 3.0 * I - XTX)
        
    R = X
    return R.view(B, N, 3, 3)

class ParticleLossModule(nn.Module):
    def forward(self, x_physical_pred, x_physical_true, mask):
     
        F_pred = x_physical_pred[..., 15:24] # (B, N, 9)
        R_pred = compute_rotation_matrix(F_pred) # (B, N, 3, 3)

        F_true = x_physical_true[..., 15:24] # (B, N, 9)
        R_true = compute_rotation_matrix(F_true) # (B, N, 3, 3)

        return ParticleLoss.apply(x_physical_pred, x_physical_true, mask, R_pred, R_true)

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }, path)

data_path = "/home/wqz/fluid/data"
file_path = "/home/wqz/fluid/data"
log_dir = "/home/wqz/fluid/experiment/demo1/scheme-v2/logs"
save_path = '/home/wqz/fluid/experiment/demo1/scheme-v2/saved'
batch_size = 1
num_epochs = 150
lr = 1e-4 
dim_in = 24
w_phys = 1.0
w_data = 50.0
w_bound = 100.0
checkpoint_file = os.path.join(save_path, 'resume_checkpoint.pt')
best_model_file = os.path.join(save_path, 'best_model.pt')
last_model_file = os.path.join(save_path, 'last_model.pt')

os.makedirs(save_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=log_dir)

full_dataset = ParticleDataset(data_path, file_path, 1)

# 加载归一化参数
print("🚚 Loading normalization parameters...")
try:
    params_path = os.path.join(data_path, f'demo{full_dataset.demo_idx}', 'normalization_params.npy')
    norm_params_np = np.load(params_path, allow_pickle=True).item()
    norm_params_torch = {
        'pos': torch.tensor(norm_params_np['pos'], dtype=torch.float32, device=device),
        'vel': torch.tensor(norm_params_np['vel'], dtype=torch.float32, device=device),
        'C': torch.tensor(norm_params_np['C'], dtype=torch.float32, device=device),
        'F': torch.tensor(norm_params_np['F'], dtype=torch.float32, device=device),
        'F_use_delta': norm_params_np.get('F_use_delta', True)
    }
    print(f"✅ Normalization params loaded and moved to device: {device}")
except FileNotFoundError:
    print(f"❌ 错误: 找不到归一化文件: {params_path}")
    exit()

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

model = ParticleTransformer(dim_in=dim_in).to(device)
loss_fn = ParticleLossModule()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

start_epoch = 0
best_val_loss = float('inf')

# 加载 Checkpoint
if os.path.exists(checkpoint_file):
    print("🔄 Found checkpoint, loading...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"✅ Resumed from epoch {start_epoch}, best_val_loss = {best_val_loss:.6f}")
else:
    print("🆕 No checkpoint found, training from scratch.")

# --- 训练循环 ---
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_train_loss = 0
    total_train_loss_m = 0
    total_train_loss_v = 0
    total_train_loss_data = 0
    #total_train_loss_bound = 0

    for x, y1_padded, mask in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
        
        x = x.to(device)
        y1_padded = y1_padded.to(device)
        mask = mask.to(device)
        noise = torch.randn_like(x) 
        noise[..., 0:3] *= 5e-3
        noise[..., 3:6] *= 2e-3
        noise[..., 6:24] *= 5e-4        
        noise = noise * mask.unsqueeze(-1)
        x_input = x + noise

        out_normalized = model(x_input, mask)
        
        y1_normalized = normalize_physical_data(y1_padded, norm_params_torch)
        out_physical = denormalize_prediction(out_normalized, norm_params_torch)

        p_total_loss, p_m_loss, p_v_loss = loss_fn(out_physical, y1_padded, mask)
        data_loss = particle_loss(out_normalized, y1_normalized, mask)

        pred_pos = out_physical[..., 0:3]

        loss = w_phys * p_total_loss + w_data * data_loss
        data_loss = w_data * data_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        total_train_loss_m += p_m_loss.item()
        total_train_loss_v += p_v_loss.item()
        total_train_loss_data += data_loss.item()
        #total_train_loss_bound += loss_bound.item()

        print("Current Loss: ", total_train_loss, 
        "; p total loss: ", p_total_loss.item(), 
        "; p m loss: ", p_m_loss.item(), 
        "; p v loss: ", p_v_loss.item(),
        "; data loss: ", data_loss.item())
        #"; bound loss: ", loss_bound.item())

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_loss_m = total_train_loss_m / len(train_loader)
    avg_train_loss_v = total_train_loss_v / len(train_loader)
    avg_train_loss_data = total_train_loss_data / len(train_loader)
    #avg_train_loss_bound = total_train_loss_bound / len(train_loader)

    # --- 验证阶段 ---
    model.eval()
    total_val_loss = 0
    total_val_loss_m = 0
    total_val_loss_v = 0
    total_val_loss_data = 0
    #total_val_loss_bound = 0
    with torch.no_grad():
        for x, y1_padded, mask in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
            
            x = x.to(device)
            y1_padded = y1_padded.to(device)
            mask = mask.to(device)
            out_normalized = model(x, mask)
            
            y1_normalized = normalize_physical_data(y1_padded, norm_params_torch)
            out_physical = denormalize_prediction(out_normalized, norm_params_torch)
            val_loss, val_loss_m, val_loss_v = loss_fn(out_physical, y1_padded, mask)
            pred_pos = out_physical[..., 0:3]
            #loss_bound = compute_boundary_loss(pred_pos, n_grid=128, bound=3)

            data_loss = particle_loss(out_normalized, y1_normalized, mask)

            #loss_bound = loss_bound * w_bound
            val_loss = val_loss * w_phys
            data_loss = data_loss * w_data
            val_loss += data_loss
            #val_loss += loss_bound

            total_val_loss += val_loss.item()
            total_val_loss_m += val_loss_m.item()
            total_val_loss_v += val_loss_v.item()
            total_val_loss_data += data_loss.item()
            #total_val_loss_bound += loss_bound.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_loss_m = total_val_loss_m / len(val_loader)
    avg_val_loss_v = total_val_loss_v / len(val_loader)
    avg_val_loss_data = total_val_loss_data / len(val_loader)
    #avg_val_loss_bound = total_val_loss_bound / len(val_loader)

    scheduler.step()
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Train_m", avg_train_loss_m, epoch)
    writer.add_scalar("Loss/Train_v", avg_train_loss_v, epoch)
    writer.add_scalar("Loss/Train_data", avg_train_loss_data, epoch)
    #writer.add_scalar("Loss/Train_bound", avg_train_loss_bound, epoch)

    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Loss/Val_m", avg_val_loss_m, epoch)
    writer.add_scalar("Loss/Val_v", avg_val_loss_v, epoch)
    writer.add_scalar("Loss/Val_data", avg_val_loss_data, epoch)
    #writer.add_scalar("Loss/Val_bound", avg_val_loss_bound, epoch)
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Train Loss m: {avg_train_loss_m:.6f} | Train Loss v: {avg_train_loss_v:.6f} | Train Loss data: {avg_train_loss_data:.6f}")
    print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {avg_val_loss:.6f} | Val Loss m: {avg_val_loss_m:.6f} | Val Loss v: {avg_val_loss_v:.6f} | Val Loss data: {avg_val_loss_data:.6f}")


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
        print(f"📌 Model saved at epoch {epoch+1} with val loss {best_val_loss:.6f}")

    save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_file)

torch.save(model.state_dict(), os.path.join(save_path, "last_model.pt"))
writer.close()
print("✅ Training complete. Final model saved.")