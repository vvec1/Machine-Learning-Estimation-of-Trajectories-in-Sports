import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from functions import preprocess, create_LA_horizonta_LA_vertical
from functions import cut_QTM_min_velocity_start_posZ, create_zeroed_QTM
from loader import load_data
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
from pathlib import Path


# Configuration & Utilities

@dataclass
class Config:
    """Central configuration for easy hyperparameter tuning."""
    seq_len: int = 20
    input_dim: int = 16  # acc(3)+gyro(3)+quat(4)+pos(3)+vel(3)
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 50
    model_type: str = "LSTM"  # Options: "LSTM", "GRU", "RNN"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    alignment_len: int = 30
    nhead: int = 4  # For SelfAttention
    dim_feedforward: int = 128  # For SelfAttention
    feature_groups: Tuple[str, ...] = ("acc", "gyro", "quat", "pos", "vel")
    smoothness_lambda: float = 0.05


def resolve_feature_groups(feature_flags: Dict[str, bool]) -> Tuple[str, ...]:
    groups = [k for k, v in feature_flags.items() if v]
    return tuple(groups)


def compute_input_dim(feature_groups: Tuple[str, ...]) -> int:
    dims = {
        "acc": 3,
        "gyro": 3,
        "quat": 4,
        "pos": 3,
        "vel": 3,
    }
    return sum(dims[g] for g in feature_groups if g in dims)

class MathUtils:
    """Physics and Geometry helper functions."""
   
    @staticmethod
    def calculate_imu_path(df, dt=0.005):
        imu_acc = df[['LAx3', 'LAy3', 'LAz3']].values
        imu_quat = df[['IMUQx', 'IMUQy', 'IMUQz', 'IMUQw']].values
       
        if len(imu_acc) == 0: return [], []
       
        rotations = R.from_quat(imu_quat)
        acc_world_all = rotations.apply(imu_acc) * 9.81
       
        imu_path = [(0.0, 0.0, 0.0)]
        imu_velocities = [np.zeros(3)]
        velocity = np.zeros(3)
        position_m = np.zeros(3)
       
        for i in range(len(acc_world_all)):
            acc_t = acc_world_all[i]
            acc_t_plus_1 = acc_world_all[i+1] if i < len(acc_world_all) - 1 else acc_t
           
            # Velocity Verlet integration
            position_m = position_m + velocity * dt + 0.5 * acc_t * (dt**2)
            velocity = velocity + 0.5 * (acc_t + acc_t_plus_1) * dt  ##preveri !!!
           
            imu_path.append(tuple(position_m * 100.0))
            imu_velocities.append(velocity)
           
        return imu_path, imu_velocities
    @staticmethod
    def align_paths(imu_path, qtm_path, length=30):
        """Kabsch alignment."""
        imu_points = np.array(imu_path)
        qtm_points = np.array(qtm_path)
        min_len = min(len(imu_points), len(qtm_points), length)
       
        if min_len < 2:
            return imu_points, np.eye(3), np.zeros(3)
        imu_subset = imu_points[:min_len]
        qtm_subset = qtm_points[:min_len]
       
        imu_shifted = imu_subset - imu_subset[0]
        qtm_shifted = qtm_subset - qtm_subset[0]
       
        H = imu_shifted.T @ qtm_shifted
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
       
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
           
        t = qtm_points[0] - R_mat @ imu_points[0]
        imu_aligned = (R_mat @ imu_points.T).T + t
       
        # Sync: remove first point added in integration
        return imu_aligned[1:], R_mat, t
   
    @staticmethod
    def align_to_axis(imu_path, qtm_path, length):
        """
        Aligns both IMU and QTM paths independently so that their initial
        motion direction points along the +X axis [1, 0, 0].
        Returns also rotation matrices for reusing with velocities.
        """
        def compute_direction(path, length):
            idx = min(length, len(path) - 1)
            v = path[idx] - path[0]
            n = np.linalg.norm(v)
            return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        def compute_rotation_to_x(vec):
            target = np.array([1.0, 0.0, 0.0])
            v = vec / np.linalg.norm(vec)
            # Already aligned
            if np.allclose(v, target, atol=1e-6):
                return R.identity()
            axis = np.cross(v, target)
            axis_norm = np.linalg.norm(axis)
            # Opposite direction → rotate 180° around Z
            if axis_norm < 1e-9:
                return R.from_rotvec(np.pi * np.array([0, 0, 1]))
            axis /= axis_norm
            angle = np.arccos(np.dot(v, target))
            return R.from_rotvec(angle * axis)
        # --- Compute directions ---
        dir_imu = compute_direction(imu_path, length)
        dir_qtm = compute_direction(qtm_path, length)
        # --- Compute rotations ---
        rot_imu = compute_rotation_to_x(dir_imu)
        rot_qtm = compute_rotation_to_x(dir_qtm)
        # --- Convert to matrices for later use ---
        R_imu_mat = rot_imu.as_matrix()
        R_qtm_mat = rot_qtm.as_matrix()
        # --- Apply rotations to the paths ---
        aligned_imu = rot_imu.apply(imu_path - imu_path[0])
        aligned_qtm = rot_qtm.apply(qtm_path - qtm_path[0])
        return aligned_imu, aligned_qtm, R_imu_mat, R_qtm_mat
    

    @staticmethod
    def align_to_axis_xy(imu_path, qtm_path, length):
        """
        Align IMU and QTM paths by rotating ONLY in the XY plane (around Z),
        so that the projected initial motion direction points along +X.
        
        Returns aligned paths and rotation matrices.
        """

        def compute_direction_xy(path, length):
            idx = min(length, len(path) - 1)
            v = path[idx] - path[0]

            # Project to XY plane
            v_xy = np.array([v[0], v[1], 0.0])
            n = np.linalg.norm(v_xy)

            if n < 1e-9:
                return np.array([1.0, 0.0, 0.0])  # default +X
            return v_xy / n

        def compute_rotation_to_x_xy(vec_xy):
            """
            Compute yaw-only rotation that aligns vec_xy with +X.
            """
            # angle between vec_xy and +X in XY plane
            angle = np.arctan2(vec_xy[1], vec_xy[0])

            # Rotate by -angle around Z
            return R.from_euler('z', -angle)

        # --- Compute XY directions ---
        dir_imu_xy = compute_direction_xy(imu_path, length)
        dir_qtm_xy = compute_direction_xy(qtm_path, length)

        # --- Compute Z-axis-only rotations ---
        rot_imu = compute_rotation_to_x_xy(dir_imu_xy)
        rot_qtm = compute_rotation_to_x_xy(dir_qtm_xy)

        # --- Rotation matrices ---
        R_imu_mat = rot_imu.as_matrix()
        R_qtm_mat = rot_qtm.as_matrix()

        # --- Apply rotations (after removing origin offset) ---
        aligned_imu = rot_imu.apply(imu_path - imu_path[0])
        aligned_qtm = rot_qtm.apply(qtm_path - qtm_path[0])

        return aligned_imu, aligned_qtm, R_imu_mat, R_qtm_mat


# ==============================================================================
#  Data Handling

class DataHandler:
    def __init__(self, config: Config):
        self.cfg = config
        self.scaler = StandardScaler()
    def process_raw_dfs(self, df_list, path_lignment = 'QTM', sample_ids=None):
        """Applies initial cleaning and physics calculation."""
        processed_data = []
        dropped_count = 0
        dropped_high_rmse = 0
        temp_samples = []
        for idx, df in enumerate(df_list):
            current_id = sample_ids[idx] if sample_ids is not None else idx
            
            # Manually excluded samples (global IDs)
            if current_id in [395,789]:
                dropped_count += 1
                continue
            
            if len(df) < 10:
                dropped_count += 1
                continue
            
            # 0. Check timestep consistency (drop if average timestep != 0.0050)
            if 'LV Ts' in df.columns and len(df) > 1:
                timestamps = df['LV Ts'].values
                dt_array = np.diff(timestamps) / 1000  # Convert ms to seconds
                avg_dt = np.mean(dt_array)
                if abs(avg_dt - 0.0050) > 0.00005:  # 4 decimal places accuracy
                    dropped_count += 1
                    continue  # Skip this sample

            # 1. Check if QTM signal is zero (with numerical tolerance)
            qtm_data = np.array([df['QTM X1'].values / 10, df['QTM Y1'].values / 10, df['QTM Z1'].values / 10])
            if np.all(np.abs(qtm_data) < 1e-6):  # numerical tolerance for zero
                dropped_count += 1
                continue
            
           
            # 2. Physics Integration
            imu_path, imu_velocities = MathUtils.calculate_imu_path(df)
           
            # 2. Get Ground Truth
            qtmx, qtmy, qtmz = df['QTM X1']/10, df['QTM Y1']/10, df['QTM Z1']/10
            qtm_path = list(zip(qtmx, qtmy, qtmz))
           
            if not imu_path or not qtm_path: continue
            # 3. Alignment
            if path_lignment == 'QTM':
                imu_aligned, R_mat, t = MathUtils.align_paths(imu_path, qtm_path, self.cfg.alignment_len)
            elif path_lignment == 'OutsideAxis':
                imu_aligned, qtm_path, R_mat, _ = MathUtils.align_to_axis(np.array(imu_path), np.array(qtm_path), self.cfg.alignment_len)
                qtm_path = qtm_path.tolist()
            elif path_lignment == 'OutsideAxisXY':
                imu_aligned, qtm_path, R_mat, _ = MathUtils.align_to_axis_xy(np.array(imu_path), np.array(qtm_path), self.cfg.alignment_len)
                qtm_path = qtm_path.tolist()


            else:
                raise ValueError("Unsupported path alignment method.")
            # 4. Rotate Velocities
            imu_vel_np = np.array(imu_velocities)
            imu_vel_aligned = (R_mat @ imu_vel_np.T).T[1:] # Sync length
           
            # 5. Extract Features
            imu_acc = df[['LAx3', 'LAy3', 'LAz3']].values
            imu_quat = df[['IMUQx', 'IMUQy', 'IMUQz', 'IMUQw']].values
            imu_gyro = df[['Gx3', 'Gy3', 'Gz3']].values
           
            # 6. Truncate to shortest length
            min_len = min(len(imu_aligned), len(imu_acc), len(imu_quat),
                          len(imu_gyro), len(qtm_path))
            if min_len < 10: continue

            imu_pos_trim = np.array(imu_aligned[:min_len])
            qtm_pos_trim = np.array(qtm_path[:min_len])

            base_rmse = np.sqrt(np.mean((imu_pos_trim - qtm_pos_trim) ** 2))
           
            sample_dict = {
                'imu_pos': imu_pos_trim,
                'imu_acc': imu_acc[:min_len],
                'imu_quat': imu_quat[:min_len],
                'imu_gyro': imu_gyro[:min_len],
                'imu_vel': imu_vel_aligned[:min_len],
                'qtm_pos': qtm_pos_trim
            }
            # Add sample ID if provided
            if sample_ids is not None:
                sample_dict['sample_id'] = sample_ids[idx]
            temp_samples.append({'rmse': base_rmse, 'sample': sample_dict, 'sample_id': current_id})
        
        return temp_samples
    
    def filter_high_rmse(self, temp_samples, percentile=0.20):
        """Filter out top percentile of highest RMSE samples."""
        if len(temp_samples) == 0:
            return []
        
        rmses = np.array([s['rmse'] for s in temp_samples])
        k = max(1, int(np.ceil(percentile * len(temp_samples))))
        cutoff = np.partition(rmses, -k)[-k]
        
        processed_data = []
        dropped_count = 0
        for entry in temp_samples:
            if entry['rmse'] >= cutoff:
                dropped_count += 1
            else:
                processed_data.append(entry['sample'])
        
        return processed_data
    
    def create_sequences(self, processed_data, goal = "correction_acc" , is_training=True, return_groups=False):
        """Creates sliding windows for LSTM.

        When return_groups=True, sequences are preserved per original sample so
        temporal smoothness can be computed within each trajectory.
        """
        X_seq, y_seq = [], []
        grouped_X, grouped_y = [], []

        for sample in processed_data:
            imu_pos = sample['imu_pos']
            qtm_pos = sample['qtm_pos']
           
            # Goal: Delta of the error (incremental correction)
            correction = qtm_pos - imu_pos            
            delta_corr = np.diff(correction, axis=0)
            #Goal: directly predict step of path
            delta_path = np.diff(qtm_pos, axis=0)
           
            feature_blocks = []
            if "acc" in self.cfg.feature_groups:
                feature_blocks.append(sample['imu_acc'])
            if "gyro" in self.cfg.feature_groups:
                feature_blocks.append(sample['imu_gyro'])
            if "quat" in self.cfg.feature_groups:
                feature_blocks.append(sample['imu_quat'])
            if "pos" in self.cfg.feature_groups:
                feature_blocks.append(sample['imu_pos'])
            if "vel" in self.cfg.feature_groups:
                feature_blocks.append(sample['imu_vel'])

            if not feature_blocks:
                continue

            # Shape: (N-1, F)
            features = np.hstack(feature_blocks)[:-1]
            sample_X, sample_y = [], []
            for i in range(len(features) - self.cfg.seq_len):
                sample_X.append(features[i : i + self.cfg.seq_len])
                if goal == "correction_acc":
                    sample_y.append(delta_corr[i + self.cfg.seq_len - 1])
                elif goal == "path_acc":
                    sample_y.append(delta_path[i + self.cfg.seq_len - 1])

            if sample_X:
                sample_X_arr = np.array(sample_X)
                sample_y_arr = np.array(sample_y)
                grouped_X.append(sample_X_arr)
                grouped_y.append(sample_y_arr)
                X_seq.extend(sample_X)
                y_seq.extend(sample_y)

        if return_groups:
            if is_training and grouped_X:
                all_train_windows = np.concatenate(grouped_X, axis=0)
                _, _, F = all_train_windows.shape
                self.scaler.fit(all_train_windows.reshape(-1, F))

            transformed_X = []
            for sample_X in grouped_X:
                N, S, F = sample_X.shape
                transformed_X.append(self.scaler.transform(sample_X.reshape(-1, F)).reshape(N, S, F))

            return transformed_X, grouped_y

        X_arr = np.array(X_seq)
        y_arr = np.array(y_seq)

        if is_training and len(X_arr) > 0:
            # Fit scaler on flattened training data
            N, S, F = X_arr.shape
            self.scaler.fit(X_arr.reshape(-1, F))

        if len(X_arr) > 0:
            # Apply scaler
            N, S, F = X_arr.shape
            X_arr = self.scaler.transform(X_arr.reshape(-1, F)).reshape(N, S, F)

        return X_arr, y_arr

# ==============================================================================
# Models

class BaseCorrectionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
   
    def forward(self, x):
        out = self.extract_features(x)
        return self.fc(out)

class LSTMModel(BaseCorrectionModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.rnn = nn.LSTM(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)
       
    def extract_features(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]

class GRUModel(BaseCorrectionModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.rnn = nn.GRU(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)
       
    def extract_features(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]
   
class BiLSTMModel(BaseCorrectionModel):
    def __init__(self, config: Config):
        nn.Module.__init__(self)
       
        self.rnn = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(config.hidden_dim * 2, config.output_dim)
       
    def extract_features(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]


class StatefulLSTMModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.device = config.device
        
        self.rnn = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2 if config.num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        self.hidden = None
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        self.hidden = (h0, c0)
        return self.hidden
    
    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
    
    def reset_hidden(self):
        self.hidden = None
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        if hidden is not None:
            h = hidden
        elif self.hidden is not None:
            h = self.hidden
        else:
            h = self.init_hidden(batch_size)
        
        out, new_hidden = self.rnn(x, h)
        self.hidden = new_hidden
        out = out[:, -1, :]
        output = self.fc(out)
        
        return output, new_hidden
    
    def forward_sequence(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class StatefulGRUModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.device = config.device
        
        self.rnn = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2 if config.num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        self.hidden = None
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        self.hidden = h0
        return self.hidden
    
    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = self.hidden.detach()
    
    def reset_hidden(self):
        self.hidden = None
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        if hidden is not None:
            h = hidden
        elif self.hidden is not None:
            h = self.hidden
        else:
            h = self.init_hidden(batch_size)
        
        out, new_hidden = self.rnn(x, h)
        self.hidden = new_hidden
        out = out[:, -1, :]
        output = self.fc(out)
        
        return output, new_hidden
    
    def forward_sequence(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x

class SelfAttentionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Linear(config.input_dim, config.hidden_dim)
        self.pe = PositionalEncoding(config.hidden_dim, max_len=config.seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

def get_model(config: Config):
    """Factory function to instantiate models based on config."""
    if config.model_type == "LSTM":
        return LSTMModel(config)
    elif config.model_type == "GRU":
        return GRUModel(config)
    elif config.model_type == "BiLSTM":  
        return BiLSTMModel(config)
    elif config.model_type == "StatefulLSTM":
        return StatefulLSTMModel(config)
    elif config.model_type == "StatefulGRU":
        return StatefulGRUModel(config)
    elif config.model_type == "SelfAttention":
        return SelfAttentionModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def is_stateful_model(model):
    """Check if model is a stateful RNN model."""
    return isinstance(model, (StatefulLSTMModel, StatefulGRUModel))

# ==============================================================================
# Trainer

class Trainer:
    def __init__(self, model, config: Config, scaler):
        self.model = model.to(config.device)
        self.config = config
        self.scaler = scaler
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.is_stateful = is_stateful_model(model)
    
    def train(self, X_train, y_train, verbose=True):
        if len(X_train) == 0: return

        grouped_training = isinstance(X_train, list)

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0
            sample_count = 0

            if grouped_training:
                sample_pairs = list(zip(X_train, y_train))
                for xb_np, yb_np in sample_pairs:
                    xb = torch.tensor(xb_np, dtype=torch.float32, device=self.config.device)
                    yb = torch.tensor(yb_np, dtype=torch.float32, device=self.config.device)

                    self.optimizer.zero_grad()

                    if self.is_stateful:
                        self.model.reset_hidden()

                    if self.is_stateful:
                        pred = self.model.forward_sequence(xb)
                    else:
                        pred = self.model(xb)

                    loss = self.loss_fn(pred, yb)

                    if pred.ndim > 1 and pred.size(0) > 1:
                        loss += self.config.smoothness_lambda * torch.mean((pred[1:] - pred[:-1]) ** 2)

                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                    sample_count += xb.size(0)
            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32).to(self.config.device),
                    torch.tensor(y_train, dtype=torch.float32).to(self.config.device)
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

                for xb, yb in loader:
                    self.optimizer.zero_grad()

                    if self.is_stateful:
                        pred = self.model.forward_sequence(xb)
                    else:
                        pred = self.model(xb)

                    loss = self.loss_fn(pred, yb)

                    # Smoothness penalty
                    if pred.ndim > 1 and pred.size(0) > 1:
                        loss += self.config.smoothness_lambda * torch.mean((pred[1:] - pred[:-1]) ** 2)

                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                sample_count = len(dataset)
           
            if verbose and (epoch+1) % 10 == 0:
                denom = max(1, sample_count)
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss/denom:.6f}")

    def predict_trajectory(self, raw_sample_dict, goal="correction_acc"):
        """Applies incremental correction to a full trajectory."""
        # Build features using only the feature groups specified in config
        feature_blocks = []
        if "acc" in self.config.feature_groups:
            feature_blocks.append(raw_sample_dict['imu_acc'])
        if "gyro" in self.config.feature_groups:
            feature_blocks.append(raw_sample_dict['imu_gyro'])
        if "quat" in self.config.feature_groups:
            feature_blocks.append(raw_sample_dict['imu_quat'])
        if "pos" in self.config.feature_groups:
            feature_blocks.append(raw_sample_dict['imu_pos'])
        if "vel" in self.config.feature_groups:
            feature_blocks.append(raw_sample_dict['imu_vel'])
        
        features_raw = np.hstack(feature_blocks)
       
        # Normalize
        features_scaled = self.scaler.transform(features_raw)
       
        self.model.eval()
        corrected_path = []
        total_model_out = np.zeros(3)
        seq_len = self.config.seq_len        
       
        # Pre-allocate buffer for efficiency
        window_buffer = np.zeros((seq_len, features_scaled.shape[1]), dtype=np.float32)
        
        # For stateful models, initialize hidden state at the start of trajectory
        if self.is_stateful:
            self.model.init_hidden(batch_size=1)
       
        for i in range(len(features_scaled)):
            if goal == "correction_acc":
                current_pos = raw_sample_dict['imu_pos'][i]            
                corrected_path.append(current_pos + total_model_out)
            elif goal == "path_acc":
                corrected_path.append(total_model_out.copy())
            else:
                raise ValueError("Unsupported goal for prediction.")
           
            with torch.no_grad():
                if self.is_stateful:
                    current_features = features_scaled[i:i+1].astype(np.float32)
                    feature_tensor = torch.tensor(current_features[None], dtype=torch.float32, device=self.config.device)
                    delta, _ = self.model(feature_tensor)
                    delta = delta.cpu().numpy()[0]
                else:
                    if i < seq_len:
                        window_buffer[-i-1:] = features_scaled[:i+1]
                    else:
                        window_buffer = features_scaled[i-seq_len+1:i+1].astype(np.float32)
                    window_tensor = torch.tensor(window_buffer[None], dtype=torch.float32, device=self.config.device)
                    delta = self.model(window_tensor).cpu().numpy()[0]
           
            total_model_out += delta
        
        # Reset hidden state after trajectory prediction
        if self.is_stateful:
            self.model.reset_hidden()
           
        return np.array(corrected_path)


# ==============================================================================
# 5. Visualization

from matplotlib.widgets import Slider

def visualize_interactive(trainer, test_samples, model_id, goal):
    if not test_samples:
        return

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # --- 3D trajectory subplot ---
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    plt.subplots_adjust(bottom=0.2)

    qtm_line, = ax3d.plot([], [], [], 'gray', label='Ground Truth')
    imu_line, = ax3d.plot([], [], [], 'blue', label='Raw IMU')
    corr_line, = ax3d.plot([], [], [], 'red', label='Corrected')
    ax3d.legend()

    # --- Time-series subplot (QTM vs time) ---
    ax_ts = fig.add_subplot(gs[1, 1])
    qtm_x_line, = ax_ts.plot([], [], label='QTM X')
    qtm_y_line, = ax_ts.plot([], [], label='QTM Y')
    qtm_z_line, = ax_ts.plot([], [], label='QTM Z')

    ax_ts.set_xlabel("Time step")
    ax_ts.set_ylabel("Position")
    ax_ts.legend()
    ax_ts.grid(True)

    def update_plot(idx):
        sample = test_samples[idx]
        corrected = trainer.predict_trajectory(sample, goal=goal)

        qtm = sample['qtm_pos'][:50]
        raw = sample['imu_pos'][:50]
        corrected = corrected[:50]

        # --- Update 3D trajectories ---
        qtm_line.set_data_3d(qtm[:, 0], qtm[:, 1], qtm[:, 2])
        imu_line.set_data_3d(raw[:, 0], raw[:, 1], raw[:, 2])
        corr_line.set_data_3d(corrected[:, 0], corrected[:, 1], corrected[:, 2])

        ax3d.set_xlim(-50, 50)
        ax3d.set_ylim(-50, 50)
        ax3d.set_zlim(-50, 50)
        
        # Display sample ID if available
        sample_id = sample.get('sample_id', idx)
        
        # Calculate RMSE metrics
        base_rmse = np.sqrt(np.mean((raw - qtm)**2))
        corr_rmse = np.sqrt(np.mean((corrected - qtm)**2))
        improvement = base_rmse - corr_rmse
        
        # Create title with RMSE information
        title_text = f"Model: {model_id} - Sample ID: {sample_id}\n"
        title_text += f"RMSE - Original: {base_rmse:.4f} cm | Corrected: {corr_rmse:.4f} cm | Improvement: {improvement:.4f} cm"
        ax3d.set_title(title_text)

        # --- Update time-series ---
        t = np.arange(len(qtm))
        qtm_x_line.set_data(t, qtm[:, 0])
        qtm_y_line.set_data(t, qtm[:, 1])
        qtm_z_line.set_data(t, qtm[:, 2])

        ax_ts.set_xlim(0, len(qtm) - 1)
        ax_ts.relim()
        ax_ts.autoscale_view(scaley=True)

        fig.canvas.draw_idle()

    # --- Slider ---
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
    slider = Slider(ax_slider, 'Sample', 0, len(test_samples) - 1,
                    valinit=0, valstep=1)
    slider.on_changed(lambda val: update_plot(int(val)))

    update_plot(0)
    plt.show()



def load_and_preprocess_data():
    """Load and preprocess data once for all trials."""
    print("Loading data...")
    _, person_ids, loaded_df_list, _, _ = load_data()
   
    # Pre-cleaning
    empty_idx = [i for i, df in enumerate(loaded_df_list) if df.empty]
    for i in reversed(empty_idx):
        loaded_df_list.pop(i)
        person_ids.pop(i)
    
    # Check for frozen QTM
    print("Checking for frozen QTM samples...")
    frozen_idx = []
    for i, df in enumerate(loaded_df_list):
        if len(df) < 2:
            continue
        qtm_positions = np.column_stack([df['QTM X1'].values, df['QTM Y1'].values, df['QTM Z1'].values])
        diffs = np.diff(qtm_positions, axis=0)
        is_same = np.all(np.abs(diffs) < 1e-9, axis=1)
        max_consecutive = 0
        current_consecutive = 0
        for same in is_same:
            if same:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        if max_consecutive > 10:
            frozen_idx.append(i)
    
    for i in reversed(frozen_idx):
        loaded_df_list.pop(i)
        person_ids.pop(i)
    print(f"Dropped {len(frozen_idx)} frozen QTM samples, {len(loaded_df_list)} remaining")
    
    # Basic feature engineering
    for i, df in enumerate(loaded_df_list):
        df = create_LA_horizonta_LA_vertical(df)
        df = preprocess(df)
        df = cut_QTM_min_velocity_start_posZ(df, first_samples=50)    
        loaded_df_list[i] = df
    create_zeroed_QTM(loaded_df_list, per_throw=True)

    print(f"Samples after all removals: {len(loaded_df_list)}")
    
    return person_ids, loaded_df_list


def split_train_val_test(all_processed_data, person_ids, mode, target_person=None,
                         test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/val/test with a held-out test set."""
    if len(all_processed_data) < 3:
        return [], [], []

    if mode == 'global':
        train_val_raw, test_raw = train_test_split(
            all_processed_data, test_size=test_size, random_state=random_state
        )
        train_raw, val_raw = train_test_split(
            train_val_raw, test_size=val_size, random_state=random_state
        )
        return train_raw, val_raw, test_raw

    if mode == 'global_unseen':
        unique_persons = np.unique(person_ids)
        if len(unique_persons) < 3:
            return [], [], []
        train_val_persons, test_persons = train_test_split(
            unique_persons, test_size=test_size, random_state=random_state
        )
        train_persons, val_persons = train_test_split(
            train_val_persons, test_size=val_size, random_state=random_state
        )
        sample_lookup = {s['sample_id']: s for s in all_processed_data if 'sample_id' in s}
        valid_sample_ids = set(sample_lookup.keys())
        train_indices = [i for i, pid in enumerate(person_ids) if pid in train_persons]
        val_indices = [i for i, pid in enumerate(person_ids) if pid in val_persons]
        test_indices = [i for i, pid in enumerate(person_ids) if pid in test_persons]
        train_raw = [sample_lookup[i] for i in train_indices if i in valid_sample_ids]
        val_raw = [sample_lookup[i] for i in val_indices if i in valid_sample_ids]
        test_raw = [sample_lookup[i] for i in test_indices if i in valid_sample_ids]
        return train_raw, val_raw, test_raw

    if mode == 'global_persplit':
        # Per-person train/val/test split, then combine into global sets
        sample_lookup = {s['sample_id']: s for s in all_processed_data if 'sample_id' in s}
        valid_sample_ids = set(sample_lookup.keys())
        unique_persons = np.unique(person_ids)
        
        all_train, all_val, all_test = [], [], []
        for pid in unique_persons:
            person_indices = [i for i, p in enumerate(person_ids) if p == pid]
            person_samples = [sample_lookup[i] for i in person_indices if i in valid_sample_ids]
            if len(person_samples) < 3:
                all_train.extend(person_samples)
                continue
            train_val, test = train_test_split(
                person_samples, test_size=test_size, random_state=random_state
            )
            all_test.extend(test)
            if len(train_val) < 2:
                all_train.extend(train_val)
                continue
            train, val = train_test_split(
                train_val, test_size=val_size, random_state=random_state
            )
            all_train.extend(train)
            all_val.extend(val)
        return all_train, all_val, all_test

    if mode == 'personal':
        sample_lookup = {s['sample_id']: s for s in all_processed_data if 'sample_id' in s}
        valid_sample_ids = set(sample_lookup.keys())
        person_indices = [i for i, pid in enumerate(person_ids) if pid == target_person]
        person_samples = [sample_lookup[i] for i in person_indices if i in valid_sample_ids]
        if len(person_samples) < 3:
            return [], [], []
        train_val_raw, test_raw = train_test_split(
            person_samples, test_size=test_size, random_state=random_state
        )
        train_raw, val_raw = train_test_split(
            train_val_raw, test_size=val_size, random_state=random_state
        )
        return train_raw, val_raw, test_raw

    train_val_raw, test_raw = train_test_split(
        all_processed_data, test_size=test_size, random_state=random_state
    )
    train_raw, val_raw = train_test_split(
        train_val_raw, test_size=val_size, random_state=random_state
    )
    return train_raw, val_raw, test_raw


def create_objective(person_ids, loaded_df_list, mode='global', target_person=None):
    """Creates an objective function for Optuna optimization."""
    
    def objective(trial):
        # Suggest hyperparameters
        model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU', 'BiLSTM', 'StatefulLSTM', 'StatefulGRU', 'SelfAttention'])
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        seq_len = trial.suggest_int('seq_len', 10, 50, step=5)
        alignment_len = trial.suggest_int('alignment_len', 10, 50, step=5)
        epochs = trial.suggest_categorical('epochs', [30, 50, 75, 100])
        smoothness_lambda = trial.suggest_float('smoothness_lambda', 0.0, 0.2)

        feature_flags = {
            "acc": trial.suggest_categorical('use_acc', [True, False]),
            "gyro": trial.suggest_categorical('use_gyro', [True, False]),
            "quat": trial.suggest_categorical('use_quat', [True, False]),
            "pos": trial.suggest_categorical('use_pos', [True, False]),
            "vel": trial.suggest_categorical('use_vel', [True, False]),
        }
        if not any(feature_flags.values()):
            feature_flags["acc"] = True
        
        ALIGNMENT = 'OutsideAxisXY'  # Fixed alignment
        GOAL = trial.suggest_categorical('GOAL', ['correction_acc', 'path_acc'])
        
        # SelfAttention specific params
        nhead = 4
        dim_feedforward = 128
        if model_type == 'SelfAttention':
            # hidden_dim must be divisible by nhead
            nhead = trial.suggest_categorical('nhead', [2, 4, 8])
            # Ensure hidden_dim is divisible by nhead
            while hidden_dim % nhead != 0:
                hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
        
        feature_groups = resolve_feature_groups(feature_flags)
        input_dim = compute_input_dim(feature_groups)

        config = Config(
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seq_len=seq_len,
            input_dim=input_dim,
            alignment_len=alignment_len,
            epochs=epochs,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            feature_groups=feature_groups,
            smoothness_lambda=smoothness_lambda
        )
        
        try:
            # Process data with current alignment settings
            data_handler = DataHandler(config)
            all_sample_ids = list(range(len(loaded_df_list)))
            all_temp_samples = data_handler.process_raw_dfs(loaded_df_list, path_lignment=ALIGNMENT, sample_ids=all_sample_ids)
            
            # Filter top 20% highest RMSE
            all_processed_data = data_handler.filter_high_rmse(all_temp_samples, percentile=0.20)
            
            if len(all_processed_data) < 10:
                return float('inf')
            
            # Split data: optimize on validation, keep test held out
            train_raw, val_raw, _ = split_train_val_test(
                all_processed_data,
                person_ids,
                mode,
                target_person=target_person,
                test_size=0.2,
                val_size=0.2,
                random_state=42
            )

            if len(train_raw) == 0 or len(val_raw) == 0:
                return float('inf')
            
            # Create sequences
            X_train, y_train = data_handler.create_sequences(train_raw, goal=GOAL, is_training=True, return_groups=True)
            X_val, y_val = data_handler.create_sequences(val_raw, goal=GOAL, is_training=False, return_groups=True)

            if len(X_train) == 0 or len(X_val) == 0:
                return float('inf')
            
            # Train model
            model = get_model(config)
            trainer = Trainer(model, config, data_handler.scaler)
            trainer.train(X_train, y_train, verbose=False)
            
            # Evaluate
            corr_errs = []
            for sample in val_raw:
                corrected = trainer.predict_trajectory(sample, goal=GOAL)
                qtm = sample['qtm_pos']
                corr_rmse = np.sqrt(np.mean((corrected - qtm)**2))
                corr_errs.append(corr_rmse)
            
            avg_corr_rmse = np.mean(corr_errs)
            
            # Report intermediate value for pruning
            trial.report(avg_corr_rmse, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return avg_corr_rmse
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')
    
    return objective


def run_optimization(person_ids, loaded_df_list, mode, n_trials, output_dir, target_person=None):
    """Run optimization for a specific mode and return results."""
    mode_suffix = mode if target_person is None else f"{mode}_person{target_person}"
    study_name = f'path_options_automl_{mode_suffix}'
    
    print(f"\n{'='*60}")
    print(f"STARTING OPTIMIZATION: {mode_suffix.upper()}")
    print(f"{'='*60}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    )
    
    # Create objective function
    objective = create_objective(person_ids, loaded_df_list, mode=mode, target_person=target_person)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\nBest trial for {mode_suffix}:")
    print(f"  Value (Avg Corrected RMSE): {study.best_trial.value:.4f} cm")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results to JSON
    results = {
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'mode': mode_suffix
    }
    results_path = output_dir / f'optuna_results_{mode_suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Save visualization
    try:
        fig = plot_optimization_history(study)
        fig.write_html(str(output_dir / f"optimization_history_{mode_suffix}.html"))
        
        fig2 = plot_param_importances(study)
        fig2.write_html(str(output_dir / f"param_importances_{mode_suffix}.html"))
    except Exception as e:
        print(f"Could not save visualizations: {e}")
    
    return study, mode_suffix


def train_and_save_best_model(study, mode_suffix, person_ids, loaded_df_list, output_dir, target_person=None):
    """Train and save the best model from a study."""
    print(f"\n--- Training Final Model for {mode_suffix} ---")
    
    best_params = study.best_trial.params
    
    # Handle SelfAttention nhead
    nhead = best_params.get('nhead', 4)
    dim_feedforward = best_params.get('dim_feedforward', 128)
    
    feature_flags = {
        "acc": best_params.get('use_acc', True),
        "gyro": best_params.get('use_gyro', True),
        "quat": best_params.get('use_quat', True),
        "pos": best_params.get('use_pos', True),
        "vel": best_params.get('use_vel', True),
    }
    if not any(feature_flags.values()):
        feature_flags["acc"] = True

    feature_groups = resolve_feature_groups(feature_flags)
    input_dim = compute_input_dim(feature_groups)

    final_config = Config(
        model_type=best_params['model_type'],
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        seq_len=best_params['seq_len'],
        input_dim=input_dim,
        alignment_len=best_params['alignment_len'],
        epochs=best_params['epochs'],
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        feature_groups=feature_groups,
        smoothness_lambda=best_params.get('smoothness_lambda', 0.05)
    )
    
    ALIGNMENT = 'OutsideAxisXY'
    GOAL = best_params['GOAL']
    
    # Reprocess with best config
    data_handler = DataHandler(final_config)
    all_sample_ids = list(range(len(loaded_df_list)))
    all_temp_samples = data_handler.process_raw_dfs(loaded_df_list, path_lignment=ALIGNMENT, sample_ids=all_sample_ids)
    all_processed_data = data_handler.filter_high_rmse(all_temp_samples, percentile=0.20)
    
    # Split based on mode (train/val/test)
    mode = mode_suffix.split('_person')[0] if '_person' in mode_suffix else mode_suffix
    train_raw, val_raw, test_raw = split_train_val_test(
        all_processed_data,
        person_ids,
        mode,
        target_person=target_person,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    train_full = train_raw + val_raw

    X_train, y_train = data_handler.create_sequences(train_full, goal=GOAL, is_training=True, return_groups=True)
    X_test, y_test = data_handler.create_sequences(test_raw, goal=GOAL, is_training=False, return_groups=True)
    
    print(f"Train Sequences: {len(X_train)} | Test Sequences: {len(X_test)}")
    
    # Train final model
    final_model = get_model(final_config)
    final_trainer = Trainer(final_model, final_config, data_handler.scaler)
    final_trainer.train(X_train, y_train, verbose=True)
    
    # Final evaluation
    base_errs, corr_errs = [], []
    for sample in test_raw:
        corrected = final_trainer.predict_trajectory(sample, goal=GOAL)
        qtm = sample['qtm_pos']
        raw = sample['imu_pos']
       
        base_rmse = np.sqrt(np.mean((raw - qtm)**2))
        corr_rmse = np.sqrt(np.mean((corrected - qtm)**2))
        base_errs.append(base_rmse)
        corr_errs.append(corr_rmse)
       
    print(f"Avg Baseline RMSE:  {np.mean(base_errs):.4f} cm")
    print(f"Avg Corrected RMSE: {np.mean(corr_errs):.4f} cm")
    print(f"Improvement:        {np.mean(base_errs) - np.mean(corr_errs):.4f} cm")
    
    # Save final model
    model_path = output_dir / f'best_model_{mode_suffix}.pt'
    torch.save(final_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return final_trainer, test_raw, GOAL


def main():
    # Configuration for optimization
    N_TRIALS = 1000  

    output_dir = Path("output V18")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GLOBAL MODEL OPTUNA OPTIMIZATION")
    print("Train/test split done per-person, then combined")
    print("="*60)
    
    # Load data once
    person_ids, loaded_df_list = load_and_preprocess_data()
    
    # Get all unique person IDs
    unique_persons = np.unique(person_ids)
    print(f"\nFound {len(unique_persons)} unique persons | Total samples: {len(loaded_df_list)}")
    print("-" * 40)
    for pid in unique_persons:
        count = sum(1 for p in person_ids if p == pid)
        print(f"  Person {pid}: {count} samples")
    print("-" * 40)
    
    # Run single global optimization with per-person splits
    study, mode_suffix = run_optimization(
        person_ids, loaded_df_list, 
        mode='global', 
        n_trials=N_TRIALS,
        output_dir=output_dir,
        target_person=None
    )
    
    trainer, test_raw, goal = train_and_save_best_model(
        study, mode_suffix, 
        person_ids, loaded_df_list,
        output_dir=output_dir,
        target_person=None
    )
    
    # Print summary 
    print("\n" + "="*60)
    print("GLOBAL MODEL OPTIMIZATION COMPLETE - SUMMARY")
    print("="*60)
    print(f"  Best RMSE: {study.best_trial.value:.4f} cm")
    print(f"  Model: {study.best_trial.params['model_type']}")
    
    # Visualize
    if len(test_raw) > 0:
        print(f"\nLaunching visualization for {mode_suffix}...")
        visualize_interactive(
            trainer, 
            test_raw, 
            f"Best Model ({mode_suffix})", 
            goal=goal
        )


if __name__ == "__main__":
    main()
