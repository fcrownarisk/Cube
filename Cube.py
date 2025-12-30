import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn

class CubeAI3D:
    """3D Cube AI with 16×16×16 sequence processing"""
    
    def __init__(self, embedding_dim: int = 256):
        self.cube_size = (16, 16, 16)  # 4096 positions total
        self.embedding_dim = embedding_dim
        
        # 3D positional encodings
        self.positional_encodings = self._create_3d_positional_encodings()
        
        # 3D convolutional layers for spatial processing
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
        )
        
        # Attention mechanism across cube dimensions
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Transformer layers for sequence processing
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)
    
    def _create_3d_positional_encodings(self) -> torch.Tensor:
        """Create sinusoidal positional encodings for 3D coordinates"""
        encodings = []
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    # Combine positional information
                    pos_x = self._sine_position(x, 10000)
                    pos_y = self._sine_position(y, 10000)
                    pos_z = self._sine_position(z, 10000)
                    encoding = torch.cat([pos_x, pos_y, pos_z])
                    encodings.append(encoding)
        return torch.stack(encodings).reshape(16, 16, 16, -1)
    
    def _sine_position(self, position: int, max_period: int = 10000) -> torch.Tensor:
        """Generate sinusoidal positional encoding"""
        d_model = self.embedding_dim // 3
        position = torch.tensor(position, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(max_period) / d_model))
        
        pe = torch.zeros(d_model)
        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        return pe
    
    def process_cube_sequence(self, input_cube: torch.Tensor) -> torch.Tensor:
        """Process 16×16×16 cube sequence"""
        batch_size = input_cube.shape[0]
        
        # Reshape to 3D convolutional format
        cube_3d = input_cube.view(batch_size, 1, 16, 16, 16)
        
        # Apply 3D convolutions
        spatial_features = self.conv_layers(cube_3d)
        
        # Flatten for sequence processing
        sequence = spatial_features.flatten(start_dim=2).permute(0, 2, 1)
        
        # Add positional encodings
        pos_enc = self.positional_encodings.flatten(start_dim=0, end_dim=2)
        sequence = sequence + pos_enc.unsqueeze(0)
        
        # Apply transformer
        output = self.transformer(sequence)
        
        return output
    
    def generate_cube_pattern(self, seed: int = 42) -> np.ndarray:
        """Generate a mathematical pattern in the 4096-position cube"""
        np.random.seed(seed)
        
        # Create a 16×16×16 cube with mathematical patterns
        cube = np.zeros((16, 16, 16), dtype=np.float32)
        
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    # Create interference pattern
                    pattern = (
                        np.sin(x * 0.5) * np.cos(y * 0.3) * np.sin(z * 0.4) +
                        np.sin(x * y * 0.02) * np.cos(z * 0.1) +
                        np.exp(-0.1 * ((x-8)**2 + (y-8)**2 + (z-8)**2))
                    )
                    cube[x, y, z] = pattern
        
        return cube

class CubeSequenceProcessor:
    """Process sequences across the cube dimensions"""
    
    def __init__(self):
        self.cube_ai = CubeAI3D()
    
    def extract_hyperdimensional_features(self, cube_data: np.ndarray) -> dict:
        """Extract features from cube in multiple dimensions"""
        features = {
            'x_sequences': [],
            'y_sequences': [],
            'z_sequences': [],
            'diagonal_sequences': [],
            'spiral_sequences': []
        }
        
        # Extract sequences along each axis
        for i in range(16):
            features['x_sequences'].append(cube_data[i, :, :])  # Fixed x
            features['y_sequences'].append(cube_data[:, i, :])  # Fixed y
            features['z_sequences'].append(cube_data[:, :, i])  # Fixed z
        
        # Extract diagonal sequences
        for offset in range(-15, 16):
            diag_seq = []
            for i in range(16):
                j = i + offset
                k = 15 - i
                if 0 <= j < 16 and 0 <= k < 16:
                    diag_seq.append(cube_data[i, j, k])
            if diag_seq:
                features['diagonal_sequences'].append(diag_seq)
        
        return features

# Example usage
if __name__ == "__main__":
    # Initialize cube AI
    cube_processor = CubeSequenceProcessor()
    
    # Generate cube pattern
    cube_pattern = cube_processor.cube_ai.generate_cube_pattern()
    print(f"Cube shape: {cube_pattern.shape}")
    print(f"Total positions: {cube_pattern.size}")
    
    # Extract features
    features = cube_processor.extract_hyperdimensional_features(cube_pattern)
    print(f"Extracted {len(features['x_sequences'])} x-sequences")
    print(f"Extracted {len(features['diagonal_sequences'])} diagonal sequences")