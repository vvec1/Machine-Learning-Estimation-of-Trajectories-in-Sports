"""Sequential AutoML studies with architecture selection among six models.

Tune/refit a random-split global model, unseen-person cross-validation models, a global
model using the exact pooled personal splits, and one model per eligible person.
Each study optimizes architecture along with its applicable hyperparameters.
Every split also tunes/refits a direct-position BiLSTM baseline. SQLite studies,
fANOVA importance exports and complete test trajectories are saved for both methods.
The matched global and personal models are evaluated on identical sample IDs.
Normal and unseen global results use their own held-out test sets. Each study defaults to
100 trials; use --trials and --epochs to reduce the budget. No plots block runs.
Alignment length and the source's curated RMSE cohort are frozen before tuning.
BiLSTM uses the final hidden states of both directions within each input window.

Stateful LSTM/GRU models process each timestep once and backpropagate through
an entire trajectory before updating weights. State resets between trajectories.
seq_len applies only to window-based models. This benchmark trains grouped
trajectories, so batch_size is unused and is not optimized. Full trajectories
require more training memory.
"""
#final
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
    batch_size: int = 128  # Only for the optional ungrouped Trainer path.
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

def padded_feature_window(features_scaled, timestep, seq_len):
    """End a window at timestep; pad missing history with normalized zeros."""
    start = max(0, timestep - seq_len + 1)
    history = features_scaled[start:timestep + 1]
    window = np.zeros((seq_len, features_scaled.shape[1]), dtype=np.float32)
    window[-len(history):] = history
    return window


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
            imu_vel_aligned = (R_mat @ imu_vel_np.T).T
            # Axis alignment retains position at t=0, so retain velocity at t=0.
            # The legacy QTM alignment explicitly removes the initial position.
            if path_lignment == 'QTM':
                imu_vel_aligned = imu_vel_aligned[1:]
           
            # 5. Extract Features
            imu_acc = df[['LAx3', 'LAy3', 'LAz3']].values
            imu_quat = df[['IMUQx', 'IMUQy', 'IMUQz', 'IMUQw']].values
            imu_gyro = df[['Gx3', 'Gy3', 'Gz3']].values
           
            # 6. Truncate to shortest length
            min_len = min(len(imu_aligned), len(imu_acc), len(imu_quat),
                          len(imu_gyro), len(imu_vel_aligned), len(qtm_path))
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
        """Create full trajectories or windows including every padded prefix.

        When return_groups=True, sequences are preserved per original sample so
        temporal smoothness can be computed within each trajectory.
        Direct-position regression targets qtm_pos at every input timestep.
        Fit scaling on real training rows once each; pad only after scaling.
        """
        if goal not in ("correction_acc", "path_acc", "direct_position"):
            raise ValueError(f"Unsupported training goal: {goal}")
        stateful = self.cfg.model_type in ("StatefulLSTM", "StatefulGRU")
        if self.cfg.seq_len < 1:
            raise ValueError("seq_len must be positive.")
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
            features = np.hstack(feature_blocks)
            if goal != 'direct_position':
                features = features[:-1]
            if len(features) > 0:
                grouped_X.append(features)
                grouped_y.append(qtm_pos if goal == 'direct_position' else
                                 delta_corr if goal == "correction_acc" else delta_path)

        if is_training and grouped_X:
            self.scaler.fit(np.concatenate(grouped_X, axis=0))
        scaled_groups = [self.scaler.transform(x) for x in grouped_X]
        if stateful:
            return scaled_groups, grouped_y

        window_groups = [np.stack([
            padded_feature_window(x, i, self.cfg.seq_len) for i in range(len(x))
        ]) for x in scaled_groups]
        if return_groups:
            return window_groups, grouped_y
        if not window_groups:
            return (np.empty((0, self.cfg.seq_len, self.cfg.input_dim), dtype=np.float32),
                    np.empty((0, self.cfg.output_dim), dtype=np.float32))
        return np.concatenate(window_groups), np.concatenate(grouped_y)

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
        _, (hidden, _) = self.rnn(x)
        # Full forward and backward summaries from the final layer.
        return torch.cat((hidden[-2], hidden[-1]), dim=1)


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
    
    def forward(self, x, hidden=None, return_sequence=False):
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
        output = self.fc(out if return_sequence else out[:, -1, :])
        
        return output, new_hidden
    
    def forward_sequence(self, x):
        """Return every timestep while preserving state and its gradient graph."""
        output, _ = self.forward(x, return_sequence=True)
        return output


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
    
    def forward(self, x, hidden=None, return_sequence=False):
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
        output = self.fc(out if return_sequence else out[:, -1, :])
        
        return output, new_hidden
    
    def forward_sequence(self, x):
        """Return every timestep while preserving state and its gradient graph."""
        output, _ = self.forward(x, return_sequence=True)
        return output
    

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
        if self.is_stateful:
            return self._train_stateful(X_train, y_train, verbose=verbose)

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

    def _train_stateful(self, X_train, y_train, verbose=True):
        """Full BPTT: one chronological trajectory and optimizer step at a time."""
        if not isinstance(X_train, list) or not isinstance(y_train, list):
            raise ValueError("Stateful training requires lists of complete trajectories.")
        if len(X_train) != len(y_train):
            raise ValueError("Each stateful trajectory must have matching targets.")
        for x, y in zip(X_train, y_train):
            if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y)
                    or x.shape[1] != self.config.input_dim
                    or y.shape[1] != self.config.output_dim):
                raise ValueError("Expected aligned (T, input_dim) and (T, output_dim) arrays.")

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss, timestep_count = 0.0, 0
            for x, y in zip(X_train, y_train):
                if len(x) == 0:
                    continue
                xb = torch.as_tensor(x, dtype=torch.float32, device=self.config.device).unsqueeze(0)
                yb = torch.as_tensor(y, dtype=torch.float32, device=self.config.device)
                self.model.reset_hidden()
                self.optimizer.zero_grad()
                try:
                    # No overlapping windows, state detachment, or weight updates
                    # inside a trajectory: gradients span every timestep.
                    pred = self.model.forward_sequence(xb).squeeze(0)
                    loss = self.loss_fn(pred, yb)
                    if len(pred) > 1:
                        loss = loss + self.config.smoothness_lambda * torch.mean(
                            (pred[1:] - pred[:-1]) ** 2
                        )
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * len(x)
                    timestep_count += len(x)
                finally:
                    self.model.reset_hidden()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss/max(1, timestep_count):.6f}")

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
        if goal == 'direct_position':
            # Direct regression at t_i, without integration or origin forcing.
            with torch.no_grad():
                windows = np.stack([padded_feature_window(features_scaled, i, self.config.seq_len)
                                    for i in range(len(features_scaled))])
                return self.model(torch.as_tensor(windows, dtype=torch.float32,
                                                  device=self.config.device)).cpu().numpy()
        corrected_path = []
        total_model_out = np.zeros(3)
        seq_len = self.config.seq_len        
       
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
           
            # The final position needs no outgoing delta (nor a training target).
            if i == len(features_scaled) - 1:
                break
            with torch.no_grad():
                if self.is_stateful:
                    current_features = features_scaled[i:i+1].astype(np.float32)
                    feature_tensor = torch.tensor(current_features[None], dtype=torch.float32, device=self.config.device)
                    delta, _ = self.model(feature_tensor)
                    delta = delta.cpu().numpy()[0]
                else:
                    window_buffer = padded_feature_window(features_scaled, i, seq_len)
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
    print_sample_counts_per_person(
        "Initial samples per person before any exclusions",
        count_samples_per_person(person_ids)
    )
   
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


def print_sample_counts_per_person(title, counts):
    total_samples = sum(counts.values())

    print(f"\n{title}:")
    print(f"  Total samples: {total_samples}")
    print("-" * 40)
    for pid in sorted(counts):
        print(f"  Person {pid}: {counts[pid]} samples")
    print("-" * 40)


def count_samples_per_person(person_ids):
    counts = {pid: 0 for pid in np.unique(person_ids)}
    for pid in person_ids:
        counts[pid] += 1
    return counts


def count_processed_samples_per_person(processed_data, person_ids):
    """Count surviving samples by person using sample IDs kept after exclusions."""
    counts = {pid: 0 for pid in np.unique(person_ids)}
    for sample in processed_data:
        sample_id = sample.get('sample_id')
        if sample_id is None or sample_id >= len(person_ids):
            continue
        counts[person_ids[sample_id]] += 1
    return counts


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
        # Only assign athletes with surviving samples to the person-level split.
        unique_persons = np.unique([person_ids[s['sample_id']] for s in all_processed_data])
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


# Sequential mixed-architecture studies. Preprocessing and split IDs are frozen once.
import argparse
import gc
import random
from dataclasses import asdict

MODEL_TYPES = ('LSTM', 'GRU', 'BiLSTM', 'StatefulLSTM', 'StatefulGRU', 'SelfAttention')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_unseen_folds(samples, person_ids, seed=42, n_splits=5):
    """Disjoint athlete test folds; inner validation uses only remaining athletes."""
    if n_splits < 3:
        raise ValueError('Use at least three unseen folds for train/validation/test athletes.')
    persons = np.unique([person_ids[s['sample_id']] for s in samples])
    if len(persons) < 3:
        raise ValueError('Unseen cross-validation requires at least three surviving athletes.')
    shuffled = np.random.default_rng(seed).permutation(persons)
    folds = {}
    for index, test_persons in enumerate(np.array_split(shuffled, min(n_splits, len(persons)))):
        remaining = [p for p in persons if p not in test_persons]
        train_persons, val_persons = train_test_split(
            remaining, test_size=0.2, random_state=seed + index)
        folds[f'global_unseen_fold{index + 1:02d}'] = tuple(
            [s for s in samples if person_ids[s['sample_id']] in group]
            for group in (train_persons, val_persons, test_persons))
    expected = [s['sample_id'] for s in samples]
    test_ids = [s['sample_id'] for parts in folds.values() for s in parts[2]]
    if len(set(expected)) != len(expected) or sorted(test_ids) != sorted(expected):
        raise ValueError('Unseen test folds must cover every surviving sample exactly once.')
    return folds


def build_benchmark_splits(samples, person_ids, seed=42, unseen_folds=5):
    """Matched global sets are exact unions of the eligible personal sets."""
    personal = {}
    skipped = []
    for pid in np.unique(person_ids):
        parts = split_train_val_test(samples, person_ids, 'personal',
                                     target_person=pid, random_state=seed)
        if all(parts):
            personal[str(pid)] = parts
        else:
            skipped.append(str(pid))
    if not personal:
        raise ValueError('No person has enough samples for train/validation/test.')
    pooled = tuple([s for parts in personal.values() for s in parts[i]] for i in range(3))
    normal = split_train_val_test(samples, person_ids, 'global', random_state=seed)
    splits = {'global_normal': normal}
    splits.update(build_unseen_folds(samples, person_ids, seed, unseen_folds))
    splits['global_matched'] = pooled
    splits.update({f'personal_{pid}': parts for pid, parts in personal.items()})
    for name, parts in splits.items():
        ids = [set(s['sample_id'] for s in part) for part in parts]
        if not all(ids) or any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f'Empty or overlapping split: {name}')
        if name.startswith('global_unseen_fold'):
            persons = [{person_ids[s['sample_id']] for s in part} for part in parts]
            if any(persons[i] & persons[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError('Athlete overlap in global_unseen.')
    return splits, skipped


def suggest_config(trial, args, direct_baseline=False):
    model_type = trial.suggest_categorical('model_type', ['BiLSTM'] if direct_baseline else list(args.models))
    groups = tuple(g for g in ('acc', 'gyro', 'quat', 'pos', 'vel')
                   if trial.suggest_categorical(f'use_{g}', [True, False])) or ('acc',)
    return Config(
        model_type=model_type,
        hidden_dim=trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        num_layers=trial.suggest_int('num_layers', 1, 4),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        seq_len=(20 if model_type.startswith('Stateful') else
                 trial.suggest_int('seq_len', 10, 50, step=5)),
        alignment_len=args.alignment_len,
        epochs=(args.epochs if args.epochs is not None else
                trial.suggest_categorical('epochs', [30, 50, 75, 100])),
        feature_groups=groups, input_dim=compute_input_dim(groups),
        smoothness_lambda=(0.0 if direct_baseline else
                           trial.suggest_float('smoothness_lambda', 0.0, 0.2)),
        nhead=(trial.suggest_categorical('nhead', [2, 4, 8])
               if model_type == 'SelfAttention' else 4),
        dim_feedforward=(trial.suggest_categorical('dim_feedforward', [64, 128, 256])
                         if model_type == 'SelfAttention' else 128),
        device=args.device,
    )


def fit_benchmark_model(config, train_samples, goal, seed, verbose=False):
    seed_everything(seed)
    handler = DataHandler(config)
    x, y = handler.create_sequences(train_samples, goal=goal,
                                     is_training=True, return_groups=True)
    if not x:
        raise ValueError('No training trajectories with at least one target delta.')
    trainer = Trainer(get_model(config), config, handler.scaler)
    trainer.train(x, y, verbose=verbose)
    return trainer


def evaluate_samples(trainer, samples, goal, person_ids, model_type, mode, artifact_dir=None):
    rows = []
    time_frames = []
    if artifact_dir is not None:
        trajectory_dir = Path(artifact_dir) / 'trajectories'
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        prediction = trainer.predict_trajectory(sample, goal=goal)
        rmse = float(np.sqrt(np.mean((prediction - sample['qtm_pos']) ** 2)))
        if not np.isfinite(rmse):
            raise ValueError('Non-finite prediction RMSE.')
        if artifact_dir is not None:
            error = prediction - sample['qtm_pos']
            baseline_error = sample['imu_pos'] - sample['qtm_pos']
            time_s = np.arange(len(prediction)) * 0.005
            np.savez_compressed(trajectory_dir / f"sample_{sample['sample_id']}.npz",
                                prediction_cm=prediction, reference_cm=sample['qtm_pos'],
                                baseline_cm=sample['imu_pos'], time_s=time_s,
                                error_cm=error, baseline_error_cm=baseline_error,
                                sample_id=int(sample['sample_id']),
                                person_id=str(person_ids[sample['sample_id']]),
                                model_type=model_type, mode=mode, goal=goal)
            time_frames.append(pd.DataFrame(dict(
                sample_id=int(sample['sample_id']), person_id=str(person_ids[sample['sample_id']]),
                timestep=np.arange(len(prediction)), time_s=time_s,
                coordinate_rmse_cm=np.sqrt(np.mean(error ** 2, axis=1)),
                euclidean_error_cm=np.linalg.norm(error, axis=1),
                baseline_coordinate_rmse_cm=np.sqrt(np.mean(baseline_error ** 2, axis=1)),
                baseline_euclidean_error_cm=np.linalg.norm(baseline_error, axis=1))))
        rows.append(dict(model_type=model_type, mode=mode,
                         sample_id=int(sample['sample_id']),
                         person_id=str(person_ids[sample['sample_id']]),
                         baseline_rmse_cm=float(np.sqrt(np.mean(
                             (sample['imu_pos'] - sample['qtm_pos']) ** 2))),
                         corrected_rmse_cm=rmse))
    if time_frames:
        errors = pd.concat(time_frames, ignore_index=True)
        errors.to_csv(Path(artifact_dir) / 'time_errors.csv', index=False)
        profile = errors.groupby('time_s').agg(
            samples=('sample_id', 'count'),
            predicted_mean_cm=('coordinate_rmse_cm', 'mean'),
            predicted_std_cm=('coordinate_rmse_cm', 'std'),
            baseline_mean_cm=('baseline_coordinate_rmse_cm', 'mean'),
            baseline_std_cm=('baseline_coordinate_rmse_cm', 'std')).reset_index()
        profile.to_csv(Path(artifact_dir) / 'error_over_time.csv', index=False)
        fig, ax = plt.subplots()
        for label, key in [('Predicted', 'predicted'), ('IMU integration', 'baseline')]:
            ax.plot(profile['time_s'], profile[f'{key}_mean_cm'], label=label)
        ax.set(xlabel='Time since trimmed start (s)', ylabel='Mean per-step coordinate RMSE (cm)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(artifact_dir) / 'error_over_time.png', dpi=160)
        plt.close(fig)
        # Deterministic example: first test sample; no selection by prediction quality.
        with np.load(trajectory_dir / f"sample_{samples[0]['sample_id']}.npz") as example:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for key, label in [('prediction_cm', 'Predicted'), ('reference_cm', 'QTM'), ('baseline_cm', 'IMU')]:
                path = example[key]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], label=label)
            ax.set(xlabel='X (cm)', ylabel='Y (cm)', zlabel='Z (cm)',
                   title=f"Test sample {samples[0]['sample_id']}")
            ax.legend()
            fig.savefig(Path(artifact_dir) / 'example_trajectory.png', dpi=160)
            plt.close(fig)
    return rows


def export_importances(study, output_dir, seed):
    """Explicit fANOVA on common parameters and separately within architectures."""
    valid = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
             and t.value is not None and np.isfinite(t.value)]
    scopes = {'all_models': valid}
    for model_type in sorted({t.params['model_type'] for t in valid}):
        scopes[f'architecture_{model_type}'] = [t for t in valid if t.params['model_type'] == model_type]
    report, rows = {}, []
    for scope, trials in scopes.items():
        entry = dict(n_trials=len(trials), evaluator='fANOVA', seed=seed,
                     status='unavailable', importances={})
        report[scope] = entry
        if len(trials) < 2:
            entry['reason'] = 'Fewer than two finite completed trials.'
            continue
        common = set.intersection(*(set(t.params) for t in trials))
        varying = sorted(p for p in common if len({t.params[p] for t in trials}) > 1)
        entry['parameters'] = varying
        entry['constant_parameters'] = sorted(common - set(varying))
        entry['conditional_parameters_not_in_all_trials'] = sorted(
            set.union(*(set(t.params) for t in trials)) - common)
        if not varying or len({t.value for t in trials}) < 2:
            entry['reason'] = 'No varying parameters or objective values.'
            continue
        try:
            subset = optuna.create_study(direction='minimize')
            subset.add_trials(trials)
            values = optuna.importance.get_param_importances(
                subset, params=varying,
                evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed))
            entry.update(status='ok', importances=values)
            rows.extend(dict(scope=scope, parameter=p, importance=v, n_trials=len(trials))
                        for p, v in values.items())
        except Exception as exc:
            entry['reason'] = str(exc)
    (output_dir / 'parameter_importances.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(rows, columns=['scope', 'parameter', 'importance', 'n_trials']).to_csv(
        output_dir / 'parameter_importances.csv', index=False)


def train_one_benchmark(mode, parts, person_ids, args, output_dir, direct_baseline=False):
    train, val, test = parts
    method = 'direct_position_bilstm' if direct_baseline else 'proposed'
    run_dir = output_dir / method / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{(output_dir / 'optuna_studies.sqlite3').resolve().as_posix()}")
    study = optuna.create_study(direction='minimize', study_name=f'{method}__{mode}',
                               storage=storage,
                               sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.set_user_attr('method', method)
    study.set_user_attr('split_sample_ids', {key: [int(s['sample_id']) for s in part]
                        for key, part in zip(('train', 'validation', 'test'), parts)})

    def objective(trial):
        config = suggest_config(trial, args, direct_baseline=direct_baseline)
        goal = trial.suggest_categorical('GOAL', ['direct_position'] if direct_baseline else
                                        ['correction_acc', 'path_acc'])
        trial.set_user_attr('config', asdict(config))
        trainer = None
        try:
            trainer = fit_benchmark_model(config, train, goal, args.seed)
            rows = evaluate_samples(trainer, val, goal, person_ids, config.model_type, mode)
            return float(np.mean([r['corrected_rmse_cm'] for r in rows]))
        finally:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Architectures compete within one validation-selected study per mode/person.
    try:
        study.optimize(objective, n_trials=args.trials, n_jobs=1,
                       catch=(RuntimeError, ValueError), show_progress_bar=True)
    finally:
        try:
            study.trials_dataframe().to_csv(run_dir / 'trials.csv', index=False)
            export_importances(study, run_dir, args.seed)
            completed_trials = study.trials
        finally:
            storage.remove_session()
            storage.engine.dispose()
    successful = [t for t in completed_trials if t.value is not None and np.isfinite(t.value)
                  and t.state == optuna.trial.TrialState.COMPLETE]
    if not successful:
        raise RuntimeError(f'No successful trials for {mode}.')
    best = min(successful, key=lambda trial: trial.value)
    n_trials = len(completed_trials)
    config = Config(**best.user_attrs['config'])
    goal = best.params['GOAL']
    # As in the source, refit on train + validation after selecting parameters.
    trainer = fit_benchmark_model(config, train + val, goal, args.seed, verbose=True)
    rows = evaluate_samples(trainer, test, goal, person_ids, config.model_type, mode, run_dir)
    torch.save(trainer.model.state_dict(), run_dir / 'model.pt')
    np.savez(run_dir / 'scaler.npz', mean=trainer.scaler.mean_,
             scale=trainer.scaler.scale_, var=trainer.scaler.var_,
             n_samples_seen=trainer.scaler.n_samples_seen_)
    metadata = dict(config=asdict(config), goal=goal, alignment='OutsideAxisXY',
                    method=method, study_name=study.study_name,
                    candidate_models=['BiLSTM'] if direct_baseline else list(args.models), n_trials=n_trials,
                    architecture_selection='minimum mean validation trajectory RMSE',
                    seed=args.seed, best_validation_rmse_cm=best.value,
                    best_params=best.params, refit_on_train_and_validation=True,
                    test_mean_rmse_cm=float(np.mean([r['corrected_rmse_cm'] for r in rows])),
                    splits={key: [int(s['sample_id']) for s in part]
                            for key, part in zip(('train', 'validation', 'test'), parts)})
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(run_dir / 'test_samples.csv', index=False)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def save_comparisons(rows, output_dir, expected_unseen_ids=None, require_complete=False):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame.to_csv(output_dir / 'all_test_samples.csv', index=False)
    frame.groupby(['model_type', 'mode'], sort=False).agg(
        samples=('sample_id', 'count'), baseline_rmse_cm=('baseline_rmse_cm', 'mean'),
        corrected_rmse_cm=('corrected_rmse_cm', 'mean')
    ).reset_index().to_csv(output_dir / 'summary.csv', index=False)
    unseen = frame[frame['mode'].str.startswith('global_unseen_fold')].copy()
    if unseen['sample_id'].duplicated().any():
        raise ValueError('Duplicate unseen out-of-fold predictions.')
    observed = set(unseen['sample_id'])
    expected = set(expected_unseen_ids) if expected_unseen_ids is not None else None
    complete = expected is not None and observed == expected
    if expected is not None and not observed <= expected:
        raise ValueError('Unexpected unseen sample IDs.')
    unseen.to_csv(output_dir / 'global_unseen_oof.csv', index=False)
    if not unseen.empty:
        per_person = unseen.groupby('person_id').agg(
            samples=('sample_id', 'count'),
            baseline_rmse_cm=('baseline_rmse_cm', 'mean'),
            corrected_rmse_cm=('corrected_rmse_cm', 'mean')).reset_index()
        per_person.to_csv(output_dir / 'global_unseen_per_person.csv', index=False)
        pd.DataFrame([dict(
            complete=complete, expected_samples=len(expected) if expected is not None else None,
            samples=len(unseen), athletes=len(per_person), folds=unseen['mode'].nunique(),
            baseline_rmse_cm=unseen['baseline_rmse_cm'].mean(),
            corrected_rmse_cm=unseen['corrected_rmse_cm'].mean(),
            athlete_mean_baseline_rmse_cm=per_person['baseline_rmse_cm'].mean(),
            athlete_mean_corrected_rmse_cm=per_person['corrected_rmse_cm'].mean(),
        )]).to_csv(output_dir / 'global_unseen_summary.csv', index=False)
    if require_complete and not complete:
        raise ValueError('Unseen out-of-fold evaluation is incomplete.')
    global_rows = frame[frame['mode'] == 'global_matched']
    personal_rows = frame[frame['mode'].str.startswith('personal_')]
    # Winners may have different architectures; pair by throw, retain both names.
    paired = global_rows.merge(personal_rows, on=['sample_id', 'person_id'],
                               suffixes=('_global', '_personal'), validate='one_to_one')
    paired['personal_minus_global_rmse_cm'] = (paired['corrected_rmse_cm_personal'] -
                                              paired['corrected_rmse_cm_global'])
    paired.to_csv(output_dir / 'paired_global_vs_personal.csv', index=False)
    # global_normal has its own test set; never score it on personal test samples
    # that may have been used for its training or hyperparameter selection.


def save_baseline_comparison(proposed_rows, direct_rows, output_dir):
    """Pair independently validation-selected models on exactly the same throws."""
    if not proposed_rows or not direct_rows:
        return
    paired = pd.DataFrame(proposed_rows).merge(
        pd.DataFrame(direct_rows), on=['mode', 'sample_id', 'person_id'],
        suffixes=('_proposed', '_direct'), validate='one_to_one')
    if not np.allclose(paired['baseline_rmse_cm_proposed'], paired['baseline_rmse_cm_direct']):
        raise ValueError('Direct/proposed baseline trajectories differ.')
    paired['proposed_minus_direct_rmse_cm'] = (paired['corrected_rmse_cm_proposed'] -
                                               paired['corrected_rmse_cm_direct'])
    paired.to_csv(output_dir / 'paired_proposed_vs_direct.csv', index=False)
    paired['evaluation'] = paired['mode'].where(
        ~paired['mode'].str.startswith('global_unseen_fold'), 'global_unseen_oof')
    paired.groupby('evaluation').agg(
        samples=('sample_id', 'count'),
        imu_rmse_cm=('baseline_rmse_cm_proposed', 'mean'),
        proposed_rmse_cm=('corrected_rmse_cm_proposed', 'mean'),
        direct_rmse_cm=('corrected_rmse_cm_direct', 'mean'),
        proposed_minus_direct_rmse_cm=('proposed_minus_direct_rmse_cm', 'mean'),
    ).reset_index().to_csv(output_dir / 'proposed_vs_direct_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='AutoML architecture selection for global, unseen and personal models')
    parser.add_argument('--trials', type=int, default=100, help='Total Optuna trials per split/person, shared across architectures')
    parser.add_argument('--epochs', type=int, default=None, help='Override the epoch search, e.g. 1 for a smoke run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unseen-folds', type=int, default=5,
                        help='Disjoint athlete test folds (at least 3; capped at athlete count)')
    parser.add_argument('--alignment-len', type=int, default=30,
                        help='Fixed across all studies so preprocessing and sample sets stay identical')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='output AUTOML all models fully stateful')
    parser.add_argument('--models', nargs='+', choices=MODEL_TYPES, default=list(MODEL_TYPES),
                        help='Candidate architectures searched within every study')
    args = parser.parse_args()
    if (args.trials < 1 or (args.epochs is not None and args.epochs < 1)
            or args.alignment_len < 1 or args.unseen_folds < 3):
        parser.error('Trials, epochs, alignment length must be positive; unseen folds at least 3.')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'split_manifest.json').exists():
        parser.error('Output already contains a benchmark; choose a new --output-dir.')
    person_ids, dfs = load_and_preprocess_data()
    handler = DataHandler(Config(alignment_len=args.alignment_len))
    entries = handler.process_raw_dfs(dfs, path_lignment='OutsideAxisXY',
                                      sample_ids=list(range(len(dfs))))
    # Preserve the source's top-20% baseline-RMSE exclusion, applied only once.
    # This is a curated cohort, not an evaluation on all unfiltered throws.
    samples = handler.filter_high_rmse(entries, percentile=0.20)
    splits, skipped = build_benchmark_splits(samples, person_ids, args.seed, args.unseen_folds)
    expected_unseen_ids = [int(s['sample_id']) for s in samples]
    manifest = dict(seed=args.seed, alignment_len=args.alignment_len,
                    candidate_models=list(args.models), trials_per_study=args.trials,
                    architecture_selection='Optuna categorical model_type per study',
                    unseen_protocol='disjoint shuffled athlete folds; independent inner holdout optimization',
                    unseen_folds_requested=args.unseen_folds,
                    unseen_folds_actual=sum(name.startswith('global_unseen_fold') for name in splits),
                    unseen_test_coverage='every surviving sample exactly once',
                    direct_baseline='BiLSTM direct position, MSE only, separately optimized with same trial budget',
                    excluded_persons_from_matched_comparison=skipped,
                    sample_id_note='Indices into the cleaned load_and_preprocess_data output',
                    cohort_filter='Source top-20% baseline RMSE exclusion, frozen before studies',
                    persons={str(s['sample_id']): str(person_ids[s['sample_id']]) for s in samples},
                    splits={name: {key: [int(s['sample_id']) for s in part]
                                   for key, part in zip(('train', 'validation', 'test'), parts)}
                            for name, parts in splits.items()})
    (output_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, direct_rows, failures = [], [], []
    for mode, parts in splits.items():
        for direct_baseline in (False, True):
            method = 'direct_position_bilstm' if direct_baseline else 'proposed'
            method_rows = direct_rows if direct_baseline else rows
            print(f'\n=== {method}: {mode} ===', flush=True)
            try:
                method_rows.extend(train_one_benchmark(
                    mode, parts, person_ids, args, output_dir, direct_baseline=direct_baseline))
                save_comparisons(method_rows, output_dir / method, expected_unseen_ids)
                save_baseline_comparison(rows, direct_rows, output_dir)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failures.append(dict(method=method, mode=mode, error=str(exc)))
                (output_dir / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} runs failed; see {output_dir / "failures.json"}')
    save_comparisons(rows, output_dir / 'proposed', expected_unseen_ids, require_complete=True)
    save_comparisons(direct_rows, output_dir / 'direct_position_bilstm', expected_unseen_ids,
                     require_complete=True)
    save_baseline_comparison(rows, direct_rows, output_dir)
    print(f'Completed. Results and matched comparisons: {output_dir}')


if __name__ == '__main__':
    main()
