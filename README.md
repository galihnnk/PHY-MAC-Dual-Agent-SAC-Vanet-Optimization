# Dual-Agent SAC for VANET Parameter Optimization

A sophisticated dual-agent reinforcement learning system using Soft Actor-Critic (SAC) for real-time optimization of VANET (Vehicular Ad-hoc Network) communication parameters. The system intelligently controls transmission power, beacon rates, and modulation coding schemes to maximize network performance while minimizing interference.

## Overview

This dual-agent RL system addresses the complex challenge of VANET parameter optimization through a novel centralized learning architecture. By separating MAC (Medium Access Control) and PHY (Physical) layer responsibilities across two specialized agents, the system achieves superior performance compared to conventional single-agent approaches, particularly in high-density vehicular scenarios.

## Key Features

### Advanced RL Architecture
- **Dual-Agent Design**: Specialized MAC agent (beacon rate + MCS) and PHY agent (transmission power)
- **Centralized Training**: Prevents gradient conflicts while enabling collective fleet intelligence
- **Shared Neural Networks**: All vehicles benefit immediately when any vehicle learns
- **Density-Aware Adaptation**: Dynamic parameter adjustment based on neighbor density

### Production-Ready Features
- **Auto Model Loading**: Seamless deployment with pre-trained models in production mode
- **Request-Based Operation**: No episode limits - responds until simulation stops
- **Real-Time Optimization**: Live parameter adjustment during network operation
- **Thread-Safe Operation**: Concurrent vehicle handling with gradient conflict prevention

### Comprehensive Analysis
- **Excel Reporting**: Multi-sheet performance analysis with 100+ metrics
- **TensorBoard Integration**: Real-time training visualization and monitoring
- **Density Analysis**: Detailed behavior analysis across network density scenarios
- **Model Persistence**: Automatic saving with timestamp and performance tracking

### Intelligent Adaptation
- **Density-Aware Rewards**: Different optimization strategies for low/medium/high density
- **Exploration Control**: Progressive exploration decay with diversity maintenance
- **Safety Bounds**: Configurable parameter limits with fail-safe defaults
- **Performance Validation**: Statistical analysis and correlation detection

## Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Core dependencies
pip install torch numpy pandas openpyxl
pip install scipy networkx matplotlib

# Optional: TensorBoard for visualization
pip install tensorboard
```

### Quick Start

1. **Configure the system** (edit script configuration):
```python
# Set operation mode
SCRIPT_MODE = "training"  # or "production"

# Configure connection
system_config.host = '127.0.0.1'
system_config.port = 5005
```

2. **Training Mode** (Learn new models):
```python
# Edit the script
SCRIPT_MODE = "training"

# Run training
python dual_agent_sac.py
```

3. **Production Mode** (Use trained models):
```python
# Edit the script  
SCRIPT_MODE = "production"

# Run with trained models
python dual_agent_sac.py
```

## Configuration Guide

### Core Settings

| Parameter | Description | Options |
|-----------|-------------|---------|
| SCRIPT_MODE | Operation mode | "training", "production" |
| system_config.host | Server IP address | Any valid IP |
| system_config.port | Server port | Any available port |
| ENABLE_TENSORBOARD | TensorBoard logging | True/False |
| ENABLE_EXCEL_REPORTING | Excel analysis reports | True/False |

### Training Configuration

```python
# Choose your training configuration
training_config = AggressiveTrainingConfig()    # Maximum performance
# training_config = ConservativeTrainingConfig() # Stable convergence
# training_config = TrainingConfigOptimum()      # Balanced approach

# Key parameters
training_config.buffer_size = 150000      # Experience replay buffer
training_config.batch_size = 512          # Training batch size
training_config.lr = 1.5e-4              # Learning rate
training_config.gamma = 0.995             # Discount factor
```

### System Boundaries

```python
# Parameter ranges
system_config.power_min = 1       # Minimum transmission power (dBm)
system_config.power_max = 30      # Maximum transmission power (dBm)
system_config.beacon_rate_min = 1 # Minimum beacon rate (Hz)
system_config.beacon_rate_max = 20 # Maximum beacon rate (Hz)
system_config.mcs_min = 0         # Minimum MCS level
system_config.mcs_max = 9         # Maximum MCS level
```

## Architecture Details

### Dual-Agent Design

```
┌─────────────────┐    ┌─────────────────┐
│   MAC Agent     │    │   PHY Agent     │
│                 │    │                 │
│ • Beacon Rate   │    │ • Power Control │
│ • MCS Selection │    │ • SINR Optimize │
│ • CBR Control   │    │ • Coverage      │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 │
    ┌─────────────────────────┐
    │  Centralized Training  │
    │     Manager            │
    │                        │
    │ • Gradient Conflict    │
    │   Prevention           │
    │ • Collective Learning  │
    │ • Experience Sharing   │
    └─────────────────────────┘
```

### Density-Aware Adaptation

| Density Level | MAC Agent Strategy | PHY Agent Strategy |
|---------------|-------------------|-------------------|
| **High (>7 neighbors)** | Lower beacon rate, robust MCS | Reduce power, minimize interference |
| **Medium (3-7 neighbors)** | Balanced approach | Adaptive power control |
| **Low (<3 neighbors)** | Higher beacon rate, efficient MCS | Increase power for coverage |

## Output Files

### Training Reports
- **Excel Analysis**: `rl_performance_report_YYYYMMDD_HHMMSS.xlsx`
  - Summary Statistics
  - Performance Analysis  
  - Configuration Details

### Model Persistence
- **Shared Models**: `saved_models/shared_models_REASON_TIMESTAMP/`
  - `shared_mac_agent.pth` - MAC agent neural network
  - `shared_phy_agent.pth` - PHY agent neural network
  - `vehicle_*_params.json` - Individual vehicle states

### Real-time Monitoring
- **Logs**: `dual_agent_logs/dual_agent_rl_system.log`
- **TensorBoard**: `dual_agent_logs/vehicle_*/` (if enabled)

## Performance Metrics

### Key Performance Indicators
- **CBR (Channel Busy Ratio)**: Target maintenance around 0.65
- **SINR (Signal-to-Interference-plus-Noise Ratio)**: Adaptive targets (8-22 dB)
- **Network Throughput**: Maximized through intelligent parameter selection
- **Exploration Factor**: Progressive decay from 2.5 to 0.15
- **Training Convergence**: Actor/critic loss trends and stability

### Density Analysis Metrics
- **High Density Performance**: Interference minimization effectiveness
- **Low Density Performance**: Coverage maximization success
- **Transition Adaptation**: Response speed to density changes
- **Collective Learning**: Fleet-wide improvement rates

## Usage Examples

### Basic Training Session
```python
# Configure for basic training
SCRIPT_MODE = "training"
training_config = ConservativeTrainingConfig()
ENABLE_TENSORBOARD = True
ENABLE_EXCEL_REPORTING = True

# Run the system
python dual_agent_sac.py
```

### Production Deployment
```python
# Configure for production use
SCRIPT_MODE = "production"
# System will auto-load latest trained models

# Deploy the system
python dual_agent_sac.py
```

### High-Performance Configuration
```python
# Maximum performance setup
training_config = AggressiveTrainingConfig()
training_config.buffer_size = 200000
training_config.batch_size = 1024
training_config.lr = 1e-4
```

## Troubleshooting

### Common Issues

**Training Not Converging**
```
[TRAINING] Actor Loss: nan, Critic Loss: nan
```
*Solution*: Reduce learning rate, check for NaN in input data, use ConservativeTrainingConfig

**Model Loading Failed**
```
[ERROR] Model directory doesn't exist: saved_models
```
*Solution*: Run training mode first to generate models, or provide pre-trained models

**Connection Refused**
```
[ERROR] Connection refused: 127.0.0.1:5005
```
*Solution*: Ensure RL server is running and port is available

**Memory Issues**
```
RuntimeError: CUDA out of memory
```
*Solution*: Reduce batch_size, use CPU mode (force_cpu=True), or reduce buffer_size

### Performance Optimization

**Slow Training**
- Reduce buffer_size for faster sampling
- Increase batch_size for more stable gradients
- Use AggressiveTrainingConfig for faster convergence

**Poor Density Adaptation**
- Increase max_neighbors for better density scaling
- Adjust density-aware reward weights (w1, w2, w3)
- Monitor density transition metrics in Excel reports

## Research Applications

- **VANET Optimization**: Real-time parameter tuning for vehicular networks
- **Multi-Agent RL**: Studying cooperative vs competitive agent interactions
- **Density-Aware Systems**: Adaptive algorithms for varying network conditions
- **Communication Engineering**: PHY/MAC layer cross-optimization
- **Production Deployment**: Trained model deployment in real networks

## Technical Specifications

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB for models and logs
- **Network**: TCP socket communication capability

### Supported Platforms
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

### Dependencies
- PyTorch 1.12+
- NumPy 1.21+
- Pandas 1.3+
- OpenPyXL 3.0+

## License

This project is licensed under the MIT License.

## Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **SAC Algorithm**: Soft Actor-Critic original research and implementations
- **VANET Research Community**: For domain knowledge and validation requirements

---
