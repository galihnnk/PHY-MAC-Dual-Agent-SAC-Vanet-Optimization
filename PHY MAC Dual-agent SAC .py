import socket
import threading
import numpy as np
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import sys
import os
import pandas as pd
import time
from datetime import datetime
import torch.serialization
import traceback
import logging
import shutil
import signal
import argparse
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import glob

# ==============================
# SCRIPT MODE CONFIGURATION (CHANGE HERE INSTEAD OF COMMAND LINE)
# ==============================
SCRIPT_MODE = "training"  # Change this to "production" or "training"
# SCRIPT_MODE = "production"  # Uncomment this line for production mode

# ==============================
# TensorBoard Import Protection
# ==============================
try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.distributions import Categorical, Normal
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    print("Warning: TensorBoard not available. Install tensorboard for logging support.")

# ==============================
# Enhanced Configuration Management
# ==============================

@dataclass
class TrainingConfig:
    """Centralized configuration for training parameters"""
    buffer_size: int = 50000
    batch_size: int = 64
    gamma: float = 0.95
    tau: float = 0.01
    alpha: float = 0.3
    lr: float = 1e-3
    hidden_units: int = 128
    max_neighbors: int = 10
    
    # Prioritized replay parameters
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.001
    per_eps: float = 1e-6
    
    # Enhanced exploration parameters
    initial_exploration_factor: float = 5.0
    exploration_decay: float = 0.9999
    min_exploration: float = 0.8
    exploration_noise_scale: float = 1.2
    action_noise_scale: float = 0.8
    
    # Enhanced entropy settings
    initial_log_std: float = 1.0
    log_std_min: float = -8
    log_std_max: float = 4
    
    # Targets
    cbr_target_base: float = 0.65
    sinr_target_base: float = 18.0
    
    # Reward weights
    w1: float = 3.0
    w2: float = 1.2
    w3: float = 4.0
    w4: float = 1.0
    w5: float = 2.0
    beta: float = 20
    
class TrainingConfigOptimum:
    """Optimized centralized configuration for superior dual-agent performance"""
    # Core training parameters - matched and improved from conventional SAC
    buffer_size: int = 120000  # Increased from 50000 (larger than conventional's 100000)
    batch_size: int = 256      # Increased from 64 (matched conventional SAC)
    gamma: float = 0.99        # Increased from 0.95 (matched conventional SAC)
    tau: float = 0.003         # Decreased from 0.01 (slower than conventional's 0.005 for stability)
    alpha: float = 0.2         # Decreased from 0.3 (lower entropy for better convergence)
    lr: float = 2e-4           # Decreased from 1e-3 (more stable than conventional's 3e-4)
    hidden_units: int = 384    # Increased from 128 (larger than conventional's 256)
    max_neighbors: int = 15    # Increased from 10 (better density handling)
    
    # Enhanced prioritized replay parameters
    per_alpha: float = 0.7     # Increased from 0.6 (more prioritization)
    per_beta: float = 0.5      # Increased from 0.4 (stronger importance sampling)
    per_beta_increment: float = 0.0005  # Decreased from 0.001 (slower beta increase)
    per_eps: float = 1e-8      # Decreased from 1e-6 (more numerical stability)
    
    # Optimized exploration parameters for better PDR
    initial_exploration_factor: float = 2.5  # Decreased from 5.0 (less aggressive)
    exploration_decay: float = 0.9998       # Slightly faster than 0.9999
    min_exploration: float = 0.3             # Decreased from 0.8 (better exploitation)
    exploration_noise_scale: float = 0.6     # Decreased from 1.2 (less noise)
    action_noise_scale: float = 0.4          # Decreased from 0.8 (more focused)
    
    # Conservative entropy settings for stability
    initial_log_std: float = 0.5    # Decreased from 1.0 (less initial randomness)
    log_std_min: float = -10        # Decreased from -8 (tighter bounds)
    log_std_max: float = 2          # Decreased from 4 (tighter bounds)
    
    # Targets - keeping CBR as requested
    cbr_target_base: float = 0.65   # Kept as standard
    sinr_target_base: float = 20.0  # Increased from 18.0 (higher quality target)
    
    # Optimized reward weights for better PDR performance
    w1: float = 5.0   # Increased from 3.0 (stronger CBR optimization)
    w2: float = 0.8   # Decreased from 1.2 (less power penalty)
    w3: float = 6.0   # Increased from 4.0 (stronger SINR optimization)
    w4: float = 0.6   # Decreased from 1.0 (less MCS penalty)
    w5: float = 1.5   # Decreased from 2.0 (less neighbor penalty)
    beta: float = 25  # Increased from 20 (sharper CBR response)

# Alternative aggressive configuration for maximum performance
class AggressiveTrainingConfig:
    """Alternative aggressive configuration for maximum dual-agent advantage"""
    # Aggressive learning parameters
    buffer_size: int = 150000       # Even larger buffer
    batch_size: int = 512           # Larger batches for stability
    gamma: float = 0.995            # Higher future reward emphasis
    tau: float = 0.002              # Very slow target updates
    alpha: float = 0.15             # Low entropy for exploitation
    lr: float = 1.5e-4              # Conservative learning rate
    hidden_units: int = 512         # Large network capacity
    max_neighbors: int = 20         # Handle high density
    
    # Prioritized replay for critical experiences
    per_alpha: float = 0.8          # Strong prioritization
    per_beta: float = 0.6           # Strong importance sampling
    per_beta_increment: float = 0.0003  # Very slow beta increase
    per_eps: float = 1e-10          # Maximum numerical precision
    
    # Focused exploration for convergence
    initial_exploration_factor: float = 1.8  # Minimal initial exploration
    exploration_decay: float = 0.9995       # Gradual decay
    min_exploration: float = 0.15            # Strong exploitation
    exploration_noise_scale: float = 0.3     # Minimal noise
    action_noise_scale: float = 0.2          # Very focused actions
    
    # Tight entropy control
    initial_log_std: float = 0.2
    log_std_min: float = -12
    log_std_max: float = 1
    
    # Optimized targets
    cbr_target_base: float = 0.65
    sinr_target_base: float = 22.0  # High quality target
    
    # PDR-optimized reward weights
    w1: float = 8.0   # Very strong CBR focus
    w2: float = 0.5   # Minimal power penalty
    w3: float = 7.0   # Strong SINR focus
    w4: float = 0.3   # Minimal MCS penalty
    w5: float = 1.0   # Minimal neighbor penalty
    beta: float = 30  # Very sharp CBR response

# Conservative configuration for stable convergence
class ConservativeTrainingConfig:
    """Conservative configuration for guaranteed stable performance"""
    # Conservative but effective parameters
    buffer_size: int = 100000
    batch_size: int = 128           # Moderate batch size
    gamma: float = 0.98
    tau: float = 0.005              # Conventional SAC tau
    alpha: float = 0.25
    lr: float = 3e-4                # Match conventional SAC
    hidden_units: int = 256         # Match conventional SAC
    max_neighbors: int = 12
    
    # Moderate prioritized replay
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.001
    per_eps: float = 1e-6
    
    # Balanced exploration
    initial_exploration_factor: float = 2.0
    exploration_decay: float = 0.9997
    min_exploration: float = 0.4
    exploration_noise_scale: float = 0.8
    action_noise_scale: float = 0.6
    
    # Standard entropy settings
    initial_log_std: float = 0.7
    log_std_min: float = -8
    log_std_max: float = 2
    
    # Standard targets
    cbr_target_base: float = 0.65
    sinr_target_base: float = 18.0
    
    # Balanced reward weights
    w1: float = 4.0
    w2: float = 1.0
    w3: float = 5.0
    w4: float = 0.8
    w5: float = 1.8
    beta: float = 22

@dataclass
class SystemConfig:
    """System-wide configuration"""
    power_min: int = 1
    power_max: int = 30
    beacon_rate_min: int = 1
    beacon_rate_max: int = 20
    mcs_min: int = 0
    mcs_max: int = 9
    
    host: str = '127.0.0.1'
    port: int = 5005
    log_dir: str = 'dual_agent_logs'
    model_save_dir: str = "saved_models"
    model_save_interval: int = 500  # Save every 500 requests instead of episodes
    report_interval: int = 1000     # Generate report every 1000 requests
    max_log_length: int = 200
    
    # Auto-save configuration (kept but changed to request-based)
    auto_save_timeout: int = 10  # minutes
    periodic_save_interval: int = 5  # minutes

# Global configurations
training_config = AggressiveTrainingConfig()
system_config = SystemConfig()

# Control flags
ENABLE_TENSORBOARD = TENSORBOARD_AVAILABLE
ENABLE_EXCEL_REPORTING = True

# Ensure directories exist
os.makedirs(system_config.log_dir, exist_ok=True)
if SCRIPT_MODE == "training" and os.path.exists(system_config.model_save_dir):
    shutil.rmtree(system_config.model_save_dir)
    os.makedirs(system_config.model_save_dir, exist_ok=True)
elif not os.path.exists(system_config.model_save_dir):
    os.makedirs(system_config.model_save_dir, exist_ok=True)

# ==============================
# Enhanced Logging Setup
# ==============================

def setup_logging():
    """Setup enhanced logging with Unicode support for Windows"""
    
    # Create logs directory if it doesn't exist
    log_dir = 'dual_agent_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging with UTF-8 encoding for file handler
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'dual_agent_rl_system.log'), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create console handler with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        format=log_format
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def log_message(message: str, level: str = "INFO"):
    """Backward compatibility logging function with Unicode protection"""
    try:
        # Clean the message of problematic Unicode characters
        clean_message = message.encode('ascii', 'replace').decode('ascii')
        getattr(logger, level.lower())(clean_message)
    except Exception as e:
        # Fallback to basic print if logging fails
        print(f"[{level}] {message}")


# ==============================
# Model Loading Utilities (NEW)
# ==============================

def find_latest_model_directory():
    """Find the latest saved model directory"""
    try:
        model_base_dir = system_config.model_save_dir
        if not os.path.exists(model_base_dir):
            logger.warning(f"Model directory doesn't exist: {model_base_dir}")
            return None
        
        # Find all model directories with the expected structure
        model_dirs = []
        for item in os.listdir(model_base_dir):
            item_path = os.path.join(model_base_dir, item)
            if os.path.isdir(item_path):
                # Look for directories with model files
                mac_model = os.path.join(item_path, "shared_mac_agent.pth")
                phy_model = os.path.join(item_path, "shared_phy_agent.pth")
                if os.path.exists(mac_model) and os.path.exists(phy_model):
                    model_dirs.append(item_path)
        
        if not model_dirs:
            logger.warning("No valid model directories found")
            return None
        
        # Sort by creation time and return the latest
        model_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        latest = model_dirs[0]
        logger.info(f"Found latest model directory: {latest}")
        return latest
        
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return None

def load_shared_models_from_directory(shared_mac_agent, shared_phy_agent, model_dir: str):
    """Load shared models from specified directory"""
    try:
        logger.info(f"Loading shared models from: {model_dir}")
        
        # Check if model files exist
        mac_model_path = os.path.join(model_dir, "shared_mac_agent.pth")
        phy_model_path = os.path.join(model_dir, "shared_phy_agent.pth")
        
        if not os.path.exists(mac_model_path):
            logger.error(f"MAC model not found: {mac_model_path}")
            return False
            
        if not os.path.exists(phy_model_path):
            logger.error(f"PHY model not found: {phy_model_path}")
            return False
        
        # Load MAC agent
        logger.info("Loading MAC agent state...")
        mac_checkpoint = torch.load(mac_model_path, map_location=device())
        shared_mac_agent.actor.load_state_dict(mac_checkpoint['actor_state_dict'])
        shared_mac_agent.critic1.load_state_dict(mac_checkpoint['critic1_state_dict'])
        shared_mac_agent.critic2.load_state_dict(mac_checkpoint['critic2_state_dict'])
        shared_mac_agent.feature_extractor.load_state_dict(mac_checkpoint['feature_extractor_state_dict'])
        shared_mac_agent.log_alpha = mac_checkpoint['log_alpha']
        shared_mac_agent.exploration_factor = mac_checkpoint['exploration_factor']
        shared_mac_agent.training_step = mac_checkpoint['training_step']
        
        # Update target networks for MAC agent
        shared_mac_agent._hard_update(shared_mac_agent.target_critic1, shared_mac_agent.critic1)
        shared_mac_agent._hard_update(shared_mac_agent.target_critic2, shared_mac_agent.critic2)
        
        # Load PHY agent
        logger.info("Loading PHY agent state...")
        phy_checkpoint = torch.load(phy_model_path, map_location=device())
        shared_phy_agent.actor.load_state_dict(phy_checkpoint['actor_state_dict'])
        shared_phy_agent.critic1.load_state_dict(phy_checkpoint['critic1_state_dict'])
        shared_phy_agent.critic2.load_state_dict(phy_checkpoint['critic2_state_dict'])
        shared_phy_agent.feature_extractor.load_state_dict(phy_checkpoint['feature_extractor_state_dict'])
        shared_phy_agent.log_alpha = phy_checkpoint['log_alpha']
        shared_phy_agent.exploration_factor = phy_checkpoint['exploration_factor']
        shared_phy_agent.training_step = phy_checkpoint['training_step']
        
        # Update target networks for PHY agent
        shared_phy_agent._hard_update(shared_phy_agent.target_critic1, shared_phy_agent.critic1)
        shared_phy_agent._hard_update(shared_phy_agent.target_critic2, shared_phy_agent.critic2)
        
        logger.info(" Successfully loaded shared models for all vehicles")
        logger.info(f"  MAC training steps: {shared_mac_agent.training_step}")
        logger.info(f"  PHY training steps: {shared_phy_agent.training_step}")
        logger.info(f"  MAC exploration factor: {shared_mac_agent.exploration_factor:.4f}")
        logger.info(f"  PHY exploration factor: {shared_phy_agent.exploration_factor:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading shared models: {e}")
        return False

# ==============================
# Auto-Save System (Modified for Request-Based)
# ==============================

class AutoSaveManager:
    """Manages automatic saving based on inactivity and periodic intervals"""
    
    def __init__(self, timeout_minutes: int):
        self.timeout_minutes = timeout_minutes
        self.last_activity = time.time()
        self.periodic_save_interval = system_config.periodic_save_interval * 60
        self.last_periodic_save = time.time()
        self.running = True
        self.save_callback = None
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_activity, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Auto-save manager started:")
        logger.info(f"  - Idle timeout: {timeout_minutes} minutes")
        logger.info(f"  - Periodic save interval: {system_config.periodic_save_interval} minutes")
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def set_save_callback(self, callback):
        """Set the callback function for saving"""
        self.save_callback = callback
    
    def _monitor_activity(self):
        """Monitor activity and trigger saves when needed"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for idle timeout
                idle_time = current_time - self.last_activity
                if idle_time >= (self.timeout_minutes * 60):
                    logger.info(f"System idle for {idle_time/60:.1f} minutes, triggering auto-save...")
                    if self.save_callback:
                        self.save_callback("idle_timeout")
                    self.last_activity = current_time
                
                # Check for periodic save
                time_since_periodic = current_time - self.last_periodic_save
                if time_since_periodic >= self.periodic_save_interval:
                    logger.info(f"Periodic save triggered after {time_since_periodic/60:.1f} minutes")
                    if self.save_callback:
                        self.save_callback("periodic")
                    self.last_periodic_save = current_time
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Auto-save monitor error: {e}")
                time.sleep(60)
    
    def stop(self):
        """Stop the auto-save manager"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# ==============================
# FIXED: Simplified Excel Reporting System
# ==============================

class SimplifiedExcelReporter:
    """Simplified Excel reporter for whole simulation statistics"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simplified data storage - aggregate statistics only
        self.request_count = 0
        self.start_time = time.time()
        self.vehicle_count = 0
        self.unique_vehicles = set()
        
        # Performance aggregates
        self.total_rewards = []
        self.total_exploration_factors = []
        self.power_changes = []
        self.beacon_changes = []
        self.mcs_changes = []
        
        # Environment aggregates
        self.cbr_values = []
        self.snr_values = []
        self.neighbor_counts = []
        
        # Training aggregates
        self.training_updates = 0
        self.actor_losses = []
        self.critic_losses = []
        
        logger.info(f"Simplified Excel reporter initialized, output directory: {output_dir}")
    
    def add_batch_data(self, batch_data: Dict, batch_responses: Dict):
        """Add data from a batch request - SIMPLIFIED"""
        self.request_count += 1
        
        for vehicle_id, vehicle_data in batch_data.items():
            self.unique_vehicles.add(vehicle_id)
            self.vehicle_count += 1
            
            # Environment data
            cbr = float(vehicle_data.get('CBR', 0))
            snr = float(vehicle_data.get('SNR', 0))
            neighbors = float(vehicle_data.get('neighbors', 0))
            
            self.cbr_values.append(cbr)
            self.snr_values.append(snr)
            self.neighbor_counts.append(neighbors)
            
            # Training data (if available)
            if vehicle_id in batch_responses.get('vehicles', {}):
                response = batch_responses['vehicles'][vehicle_id]
                if 'training' in response:
                    training_data = response['training']
                    self.total_rewards.append(training_data.get('reward', 0))
                    self.total_exploration_factors.append(training_data.get('exploration_factor', 1.0))
                    self.power_changes.append(abs(training_data.get('power_delta', 0)))
                    self.beacon_changes.append(abs(training_data.get('beacon_delta', 0)))
                    self.mcs_changes.append(abs(training_data.get('mcs_delta', 0)))
    
    def add_training_update(self, actor_loss: float, critic_loss: float):
        """Add training update metrics"""
        self.training_updates += 1
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
    
    def generate_final_report(self, filename_prefix: str = "rl_performance_report") -> str:
        """Generate SIMPLIFIED final Excel report - single document with multiple sheets"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.xlsx"
            filepath = os.path.join(self.output_dir, filename)
            
            elapsed_time = time.time() - self.start_time
            
            # SHEET 1: Summary Statistics (Main metrics)
            summary_stats = {
                'Metric': [
                    'Total Requests Processed',
                    'Total Vehicle Interactions', 
                    'Unique Vehicles',
                    'Simulation Duration (minutes)',
                    'Requests per Minute',
                    'Average Reward',
                    'Reward Std Dev',
                    'Min Reward',
                    'Max Reward',
                    'Final Exploration Factor',
                    'Total Training Updates',
                    'Average Actor Loss',
                    'Average Critic Loss',
                    'Average CBR',
                    'CBR Std Dev',
                    'Average SNR', 
                    'SNR Std Dev',
                    'Average Neighbors',
                    'Max Neighbors',
                    'Average Power Change',
                    'Average Beacon Change',
                    'Average MCS Change'
                ],
                'Value': [
                    self.request_count,
                    self.vehicle_count,
                    len(self.unique_vehicles),
                    elapsed_time / 60,
                    (self.request_count / (elapsed_time / 60)) if elapsed_time > 0 else 0,
                    np.mean(self.total_rewards) if self.total_rewards else 0,
                    np.std(self.total_rewards) if self.total_rewards else 0,
                    np.min(self.total_rewards) if self.total_rewards else 0,
                    np.max(self.total_rewards) if self.total_rewards else 0,
                    self.total_exploration_factors[-1] if self.total_exploration_factors else 0,
                    self.training_updates,
                    np.mean(self.actor_losses) if self.actor_losses else 0,
                    np.mean(self.critic_losses) if self.critic_losses else 0,
                    np.mean(self.cbr_values) if self.cbr_values else 0,
                    np.std(self.cbr_values) if self.cbr_values else 0,
                    np.mean(self.snr_values) if self.snr_values else 0,
                    np.std(self.snr_values) if self.snr_values else 0,
                    np.mean(self.neighbor_counts) if self.neighbor_counts else 0,
                    np.max(self.neighbor_counts) if self.neighbor_counts else 0,
                    np.mean(self.power_changes) if self.power_changes else 0,
                    np.mean(self.beacon_changes) if self.beacon_changes else 0,
                    np.mean(self.mcs_changes) if self.mcs_changes else 0
                ]
            }
            
            # SHEET 2: Detailed Performance Analysis
            performance_analysis = {
                'Category': ['Reward', 'Reward', 'Reward', 'Reward', 'Exploration', 'Exploration', 
                           'Environment', 'Environment', 'Environment', 'Environment', 'Environment', 'Environment',
                           'Actions', 'Actions', 'Actions', 'Training', 'Training', 'Training'],
                'Metric': ['Mean', 'Std', 'Min', 'Max', 'Initial Factor', 'Final Factor',
                          'Mean CBR', 'Std CBR', 'Mean SNR', 'Std SNR', 'Mean Neighbors', 'Max Neighbors',
                          'Mean Power Change', 'Mean Beacon Change', 'Mean MCS Change', 
                          'Total Updates', 'Mean Actor Loss', 'Mean Critic Loss'],
                'Value': [
                    np.mean(self.total_rewards) if self.total_rewards else 0,
                    np.std(self.total_rewards) if self.total_rewards else 0,
                    np.min(self.total_rewards) if self.total_rewards else 0,
                    np.max(self.total_rewards) if self.total_rewards else 0,
                    self.total_exploration_factors[0] if self.total_exploration_factors else 0,
                    self.total_exploration_factors[-1] if self.total_exploration_factors else 0,
                    np.mean(self.cbr_values) if self.cbr_values else 0,
                    np.std(self.cbr_values) if self.cbr_values else 0,
                    np.mean(self.snr_values) if self.snr_values else 0,
                    np.std(self.snr_values) if self.snr_values else 0,
                    np.mean(self.neighbor_counts) if self.neighbor_counts else 0,
                    np.max(self.neighbor_counts) if self.neighbor_counts else 0,
                    np.mean(self.power_changes) if self.power_changes else 0,
                    np.mean(self.beacon_changes) if self.beacon_changes else 0,
                    np.mean(self.mcs_changes) if self.mcs_changes else 0,
                    self.training_updates,
                    np.mean(self.actor_losses) if self.actor_losses else 0,
                    np.mean(self.critic_losses) if self.critic_losses else 0
                ]
            }
            
            # SHEET 3: Configuration
            config_data = {
                'Parameter': [
                    'Buffer Size', 'Batch Size', 'Learning Rate', 'Gamma', 'Tau', 'Hidden Units',
                    'Power Min', 'Power Max', 'Beacon Min', 'Beacon Max', 'MCS Min', 'MCS Max',
                    'Initial Exploration', 'Min Exploration', 'Exploration Decay',
                    'CBR Target', 'SINR Target'
                ],
                'Value': [
                    training_config.buffer_size, training_config.batch_size, training_config.lr,
                    training_config.gamma, training_config.tau, training_config.hidden_units,
                    system_config.power_min, system_config.power_max,
                    system_config.beacon_rate_min, system_config.beacon_rate_max,
                    system_config.mcs_min, system_config.mcs_max,
                    training_config.initial_exploration_factor, training_config.min_exploration,
                    training_config.exploration_decay, training_config.cbr_target_base,
                    training_config.sinr_target_base
                ]
            }
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Performance analysis sheet
                performance_df = pd.DataFrame(performance_analysis)
                performance_df.to_excel(writer, sheet_name='Performance_Analysis', index=False)
                
                # Configuration sheet
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            logger.info(f"SIMPLIFIED Excel report generated: {filepath}")
            logger.info(f"  - Summary_Statistics: Key metrics for whole simulation")
            logger.info(f"  - Performance_Analysis: Detailed breakdown by category")
            logger.info(f"  - Configuration: System settings")
            logger.info(f"  - Total data points: {self.vehicle_count}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating simplified Excel report: {e}")
            return ""
    
    def clear_data(self):
        """Clear stored data"""
        self.total_rewards.clear()
        self.total_exploration_factors.clear()
        self.power_changes.clear()
        self.beacon_changes.clear()
        self.mcs_changes.clear()
        self.cbr_values.clear()
        self.snr_values.clear()
        self.neighbor_counts.clear()
        self.actor_losses.clear()
        self.critic_losses.clear()

# ==============================
# Utility Functions (Kept All)
# ==============================

def safe_tensor_conversion(data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Safely convert data to tensor with proper error handling"""
    try:
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        return torch.tensor(data, dtype=dtype)
    except Exception as e:
        logger.error(f"Tensor conversion failed: {e}")
        return torch.zeros(1, dtype=dtype)

def safe_normalize_state(state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Safely normalize state with dimension checks"""
    try:
        if state.shape[-1] != mean.shape[-1]:
            logger.warning(f"Dimension mismatch in normalization: state {state.shape} vs mean {mean.shape}")
            return state
        
        return torch.clamp((state - mean) / (std + 1e-8), -5.0, 5.0)
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return state

def device(force_cpu: bool = True) -> str:
    """Get device for computations"""
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

def set_seed(seed: int = 0):
    """Set random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_adaptive_targets(neighbor_count: int) -> Dict[str, float]:
    """Enhanced adaptive targets with more aggressive density adaptation"""
    density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
    
    return {
        # More aggressive adaptations
        'CBR_TARGET': training_config.cbr_target_base * (1 - 0.5 * density_factor),  # Was 0.2
        'SINR_TARGET': max(6.0, training_config.sinr_target_base - 8.0 * density_factor),  # Was 3.0
        
        # New targets for MAC agent
        'BEACON_TARGET': system_config.beacon_rate_max * (1 - 0.7 * density_factor),
        'MCS_TARGET': system_config.mcs_max * (1 - 0.6 * density_factor),
        
        # New targets for PHY agent
        'POWER_TARGET': system_config.power_max * (1 - 0.5 * density_factor)
    }


def get_reward_weights(cbr: float, sinr: float) -> Dict[str, float]:
    """Enhanced base reward weights with new parameters"""
    w = {
        'W1': training_config.w1,      # CBR weight
        'W2': training_config.w2,      # Power weight  
        'W3': training_config.w3,      # SINR weight
        'W4': training_config.w4,      # Legacy MCS weight
        'W5': training_config.w5,      # Neighbor weight
        'BETA': training_config.beta,  # CBR sharpness
        
        # New weights for density adaptation
        'W_BEACON': 1.0,      # Beacon control weight (will be adjusted by density)
        'W_MCS': 1.0,         # MCS control weight (will be adjusted by density)
        'W_INTERFERENCE': 1.0, # Interference penalty weight  
        'W_COVERAGE': 0.8,    # Coverage bonus weight (PHY only)
    }
    
    # Dynamic adjustment based on current performance
    if cbr < 0.2:
        w['W1'] *= 1.5
    if sinr < 10.0:
        w['W3'] *= 1.5
        
    return w

# ==============================
# Network Architectures (Kept All)
# ==============================

class ExploratorySharedFeatureExtractor(nn.Module):
    """Feature extractor optimized for exploration"""
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        # Fixed normalization values
        self.register_buffer('state_mean', torch.tensor([0.5, 15.0, 5.0]))
        self.register_buffer('state_std', torch.tensor([0.2, 5.0, 3.0]))
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, training_config.hidden_units),
            nn.ReLU(),
            nn.Linear(training_config.hidden_units, training_config.hidden_units),
            nn.ReLU()
        )
        
        self._initialize_weights_for_exploration()
        
    def _initialize_weights_for_exploration(self):
        """Initialize weights to encourage exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=2.0)
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.shape[-1] == 3:
                x_norm = safe_normalize_state(x, self.state_mean, self.state_std)
            else:
                x_norm = x
            
            return self.net(x_norm)
            
        except Exception as e:
            logger.error(f"Feature extractor forward pass failed: {e}")
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            return torch.zeros(batch_size, training_config.hidden_units)

class HighlyExplorativeActor(nn.Module):
    """Actor network designed for maximum exploration"""
    
    def __init__(self, feature_dim: int, action_dim: int, action_bounds: List[Tuple[float, float]]):
        super().__init__()
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_dim, training_config.hidden_units),
            nn.ReLU(),
            nn.Linear(training_config.hidden_units, training_config.hidden_units // 2),
            nn.ReLU()
        )
        
        self.action_mean = nn.Linear(training_config.hidden_units // 2, action_dim)
        
        self.action_log_std = nn.Parameter(
            torch.full((action_dim,), training_config.initial_log_std)
        )
        
        self._initialize_weights_for_exploration()
    
    def _initialize_weights_for_exploration(self):
        """Initialize weights to encourage exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=0.2, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.2, 0.2)
        
        nn.init.uniform_(self.action_mean.weight.data, -0.1, 0.1)
        nn.init.zeros_(self.action_mean.bias.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action means"""
        try:
            shared = self.shared_layers(x)
            means = self.action_mean(shared)
            
            bounded_means = torch.tanh(means)
            
            for i, (low, high) in enumerate(self.action_bounds):
                scale = (high - low) / 2.0
                offset = (high + low) / 2.0
                bounded_means[:, i] = bounded_means[:, i] * scale + offset
            
            return bounded_means
        except Exception as e:
            logger.error(f"Actor forward pass failed: {e}")
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            return torch.zeros(batch_size, self.action_dim)
    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions with enhanced exploration"""
        try:
            shared = self.shared_layers(x)
            means = self.action_mean(shared)
            
            log_std = torch.clamp(self.action_log_std, 
                                training_config.log_std_min, 
                                training_config.log_std_max)
            std = torch.exp(log_std)
            
            if TENSORBOARD_AVAILABLE:
                normal = Normal(means, std)
                x_t = normal.rsample()
            else:
                x_t = means + torch.randn_like(means) * std
            
            y_t = torch.tanh(x_t)
            
            # Calculate log probability
            if TENSORBOARD_AVAILABLE:
                log_prob = normal.log_prob(x_t)
            else:
                log_prob = -0.5 * (((x_t - means) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
            
            log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            
            actions = y_t.clone()
            for i, (low, high) in enumerate(self.action_bounds):
                scale = (high - low) / 2.0
                offset = (high + low) / 2.0
                actions[:, i] = actions[:, i] * scale + offset
            
            return actions, log_prob
        except Exception as e:
            logger.error(f"Actor sampling failed: {e}")
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            actions = torch.zeros(batch_size, self.action_dim)
            log_prob = torch.zeros(batch_size, 1)
            return actions, log_prob

class ImprovedCritic(nn.Module):
    """Enhanced critic network"""
    
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, training_config.hidden_units),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, training_config.hidden_units // 2),
            nn.ReLU()
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(training_config.hidden_units + training_config.hidden_units // 2, 
                     training_config.hidden_units),
            nn.ReLU(),
            nn.Linear(training_config.hidden_units, training_config.hidden_units // 2),
            nn.ReLU(),
            nn.Linear(training_config.hidden_units // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        try:
            feature_out = self.feature_net(features)
            action_out = self.action_net(actions)
            combined = torch.cat([feature_out, action_out], dim=1)
            return self.combined_net(combined)
        except Exception as e:
            logger.error(f"Critic forward pass failed: {e}")
            batch_size = features.shape[0] if len(features.shape) > 1 else 1
            return torch.zeros(batch_size, 1)

# ==============================
# Prioritized Replay Buffer (Kept All - Fixed)
# ==============================

class SumTree:
    def __init__(self, size: int):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size
        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self) -> float:
        return self.nodes[0]

    def update(self, data_idx: int, value: float):
        idx = data_idx + self.size - 1
        change = value - self.nodes[idx]
        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value: float, data: Any):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum: float) -> Tuple[int, float, Any]:
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, state_size: int, action_size: int, buffer_size: int, 
                 eps: float = None, alpha: float = None, beta: float = None):
        self.tree = SumTree(size=buffer_size)
        
        self.eps = eps or training_config.per_eps
        self.alpha = alpha or training_config.per_alpha
        self.beta = beta or training_config.per_beta
        self.max_priority = self.eps
        
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)
        
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition: Tuple[torch.Tensor, ...]):
        state, action, reward, next_state, done = transition
        
        self.tree.add(self.max_priority, self.count)
        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, List[int]]:
        assert self.real_size >= batch_size
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        
        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()
        
        batch = (
            self.state[sample_idxs].to(device()),
            self.action[sample_idxs].to(device()),
            self.reward[sample_idxs].to(device()),
            self.next_state[sample_idxs].to(device()),
            self.done[sample_idxs].to(device())
        )
        
        return batch, weights.to(device()), tree_idxs

    def update_priorities(self, data_idxs: List[int], priorities: Union[torch.Tensor, np.ndarray]):
        """Fixed priority update to handle tensor arrays properly"""
        try:
            if isinstance(priorities, torch.Tensor):
                if priorities.dim() > 1:
                    priorities = priorities.flatten()
                priorities = priorities.detach().cpu().numpy()
            
            if hasattr(priorities, 'flatten'):
                priorities = priorities.flatten()
            
            for data_idx, priority in zip(data_idxs, priorities):
                priority = float(priority)
                priority = (priority + self.eps) ** self.alpha
                self.tree.update(data_idx, priority)
                self.max_priority = max(self.max_priority, priority)
        except Exception as e:
            logger.error(f"Priority update failed: {e}")
            for data_idx in data_idxs:
                self.tree.update(data_idx, self.max_priority)

    def __len__(self) -> int:
        return self.real_size

# ==============================
# SAC Agent Implementation (FIXED for Centralized Learning)
# ==============================

class CentralizedTrainingManager:
    """Manages centralized training to avoid gradient conflicts"""
    
    def __init__(self, mac_agent, phy_agent):  # FIXED: Removed type hints to avoid forward reference
        self.mac_agent = mac_agent
        self.phy_agent = phy_agent
        
        # Batch collection for centralized training
        self.mac_experiences = []
        self.phy_experiences = []
        
        # Training control
        self.train_interval = 5
        self.batch_count = 0
        
        # Thread lock for safe updates
        import threading
        self.training_lock = threading.Lock()
        
    def add_experience(self, state: List[float], actions: Dict[str, Any], reward: float, next_state: List[float]):
        """Add experience for centralized batch training with detailed logging"""
        try:
            state_tensor = safe_tensor_conversion(state)
            next_state_tensor = safe_tensor_conversion(next_state)
            
            # MAC experience
            mac_actions = torch.FloatTensor([actions['power_delta'], actions['beacon_delta']])
            mac_transition = (
                state_tensor,
                mac_actions,
                torch.FloatTensor([reward]),
                next_state_tensor,
                torch.FloatTensor([False])
            )
            self.mac_experiences.append(mac_transition)
            
            # PHY experience  
            phy_action = torch.FloatTensor([actions['mcs_delta']])
            phy_transition = (
                state_tensor,
                phy_action,
                torch.FloatTensor([reward]),
                next_state_tensor,
                torch.FloatTensor([False])
            )
            self.phy_experiences.append(phy_transition)
            
            # Add to replay buffers
            self.mac_agent.replay_buffer.add(mac_transition)
            self.phy_agent.replay_buffer.add(phy_transition)
            
            self.batch_count += 1
            
            # Log experience details every 10 experiences or when training triggers
            if self.batch_count % 10 == 0 or self.batch_count % self.train_interval == 0:
                logger.info(f"[EXPERIENCE] Batch {self.batch_count}: State=[{state[0]:.3f}, {state[1]:.1f}, {int(state[2])}], "
                           f"Actions=[{actions['power_delta']:.2f}, {actions['beacon_delta']:.2f}, {actions['mcs_delta']:.2f}], "
                           f"Reward={reward:.3f}")
            
            # Trigger centralized training
            if self.batch_count % self.train_interval == 0:
                logger.info(f"[TRIGGER] Centralized training triggered after {self.train_interval} experiences")
                self._perform_centralized_training()
                
        except Exception as e:
            logger.error(f"Error adding experience to centralized training: {e}")
    
    def _perform_centralized_training(self):
        """Perform centralized training with thread safety and detailed logging"""
        with self.training_lock:
            try:
                logger.info(f"[CENTRALIZED TRAINING] Starting training cycle {self.batch_count}")
                logger.info(f"  MAC Buffer size: {len(self.mac_agent.replay_buffer)}")
                logger.info(f"  PHY Buffer size: {len(self.phy_agent.replay_buffer)}")
                
                training_results = {'mac_losses': {}, 'phy_losses': {}}
                
                # Train MAC agent
                if len(self.mac_agent.replay_buffer) >= training_config.batch_size:
                    logger.info(f"[MAC TRAINING] Sampling {training_config.batch_size} experiences...")
                    batch, weights, indices = self.mac_agent.replay_buffer.sample(training_config.batch_size)
                    mac_losses = self.mac_agent.update(batch, weights)
                    training_results['mac_losses'] = mac_losses
                    
                    logger.info(f"[MAC TRAINING] Results:")
                    logger.info(f"  Actor Loss: {mac_losses.get('actor_loss', 0):.4f}")
                    logger.info(f"  Critic1 Loss: {mac_losses.get('critic1_loss', 0):.4f}")
                    logger.info(f"  Critic2 Loss: {mac_losses.get('critic2_loss', 0):.4f}")
                    logger.info(f"  Alpha: {mac_losses.get('alpha', 0):.4f}")
                    logger.info(f"  Exploration Factor: {mac_losses.get('exploration_factor', 0):.4f}")
                    
                    if 'td_errors' in mac_losses:
                        self.mac_agent.replay_buffer.update_priorities(indices, mac_losses['td_errors'])
                        logger.info(f"[MAC TRAINING] Updated {len(indices)} priority values")
                
                # Train PHY agent  
                if len(self.phy_agent.replay_buffer) >= training_config.batch_size:
                    logger.info(f"[PHY TRAINING] Sampling {training_config.batch_size} experiences...")
                    batch, weights, indices = self.phy_agent.replay_buffer.sample(training_config.batch_size)
                    phy_losses = self.phy_agent.update(batch, weights)
                    training_results['phy_losses'] = phy_losses
                    
                    logger.info(f"[PHY TRAINING] Results:")
                    logger.info(f"  Actor Loss: {phy_losses.get('actor_loss', 0):.4f}")
                    logger.info(f"  Critic1 Loss: {phy_losses.get('critic1_loss', 0):.4f}")
                    logger.info(f"  Critic2 Loss: {phy_losses.get('critic2_loss', 0):.4f}")
                    logger.info(f"  Alpha: {phy_losses.get('alpha', 0):.4f}")
                    logger.info(f"  Exploration Factor: {phy_losses.get('exploration_factor', 0):.4f}")
                    
                    if 'td_errors' in phy_losses:
                        self.phy_agent.replay_buffer.update_priorities(indices, phy_losses['td_errors'])
                        logger.info(f"[PHY TRAINING] Updated {len(indices)} priority values")
                
                # Clear batch experiences
                self.mac_experiences.clear()
                self.phy_experiences.clear()
                
                logger.info(f"[CENTRALIZED TRAINING] Training cycle completed - ALL vehicles now benefit from updated shared models")
                logger.info(f"="*60)
                
                return training_results
                
            except Exception as e:
                logger.error(f"[CENTRALIZED TRAINING] Training failed: {e}")
                return {'mac_losses': {}, 'phy_losses': {}}

class UltraExplorativeSACAgent:
    """SAC agent with maximum exploration capabilities"""
    
    def __init__(self, action_dim: int, action_bounds: List[Tuple[float, float]], 
                 training_mode: bool = True):
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.training_mode = training_mode
        
        self.feature_extractor = ExploratorySharedFeatureExtractor(state_dim=3)
        
        feature_dim = training_config.hidden_units
        self.actor = HighlyExplorativeActor(feature_dim, action_dim, action_bounds)
        self.critic1 = ImprovedCritic(feature_dim, action_dim)
        self.critic2 = ImprovedCritic(feature_dim, action_dim)
        
        self.target_critic1 = ImprovedCritic(feature_dim, action_dim)
        self.target_critic2 = ImprovedCritic(feature_dim, action_dim)
        
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)
        
        enhanced_lr = training_config.lr * 1.5
        all_actor_params = list(self.actor.parameters()) + list(self.feature_extractor.parameters())
        self.actor_optimizer = optim.Adam(all_actor_params, lr=enhanced_lr)
        self.critic1_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.feature_extractor.parameters()), 
            lr=enhanced_lr
        )
        self.critic2_optimizer = optim.Adam(
            list(self.critic2.parameters()) + list(self.feature_extractor.parameters()), 
            lr=enhanced_lr
        )
        
        self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item() * 0.8
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=enhanced_lr)
        
        self.replay_buffer = PrioritizedReplayBuffer(
            state_size=3,
            action_size=action_dim,
            buffer_size=training_config.buffer_size
        )
        
        self.training_step = 0
        self.exploration_factor = training_config.initial_exploration_factor
        self.action_noise_std = training_config.action_noise_scale
        self.recent_actions = deque(maxlen=100)
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced action selection with proper NaN protection"""
        try:
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                
                if state.shape[-1] != 3:
                    logger.error(f"Invalid state dimensions: {state.shape}")
                    batch_size = state.shape[0]
                    return torch.zeros(batch_size, self.action_dim), torch.zeros(batch_size, 1)
                
                features = self.feature_extractor(state)
                
                if deterministic:
                    action = self.actor(features)
                    log_prob = torch.zeros(action.shape[0], 1)
                else:
                    action, log_prob = self.actor.sample(features)
                    
                    if self.training_mode:
                        if self.exploration_factor > 1.0:
                            action = action * self.exploration_factor
                            action = torch.where(torch.isnan(action) | torch.isinf(action), 
                                               torch.zeros_like(action), action)
                            action = torch.clamp(action, -50.0, 50.0)
                        
                        noise_scale = self.exploration_factor * training_config.exploration_noise_scale
                        exploration_noise = torch.randn_like(action) * noise_scale
                        action = action + exploration_noise
                        
                        action = torch.where(torch.isnan(action) | torch.isinf(action), 
                                           torch.zeros_like(action), action)
                        
                        # FIXED: Prevent std() calculation error when not enough actions
                        if len(self.recent_actions) >= 2:
                            try:
                                action_std = torch.std(torch.cat(list(self.recent_actions)), dim=0)
                                diversity_noise = torch.randn_like(action) * self.action_noise_std * (1.0 / (action_std + 0.1))
                                action = action + diversity_noise
                            except Exception as e:
                                # Skip diversity noise if calculation fails
                                pass
                        
                        if random.random() < 0.1 * self.exploration_factor:
                            for i, (low, high) in enumerate(self.action_bounds):
                                if random.random() < 0.5:
                                    action[:, i] = torch.FloatTensor(action.shape[0]).uniform_(low, high)
                        
                        self.exploration_factor *= training_config.exploration_decay
                        self.exploration_factor = max(self.exploration_factor, training_config.min_exploration)
                        
                        self.recent_actions.append(action.clone())
                
                action = torch.where(torch.isnan(action) | torch.isinf(action), 
                                   torch.zeros_like(action), action)
                
                for i, (low, high) in enumerate(self.action_bounds):
                    action[:, i] = torch.clamp(action[:, i], low, high)
                
                return action, log_prob
                
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            batch_size = state.shape[0] if len(state.shape) > 1 else 1
            return torch.zeros(batch_size, self.action_dim), torch.zeros(batch_size, 1)
    
    def calculate_reward(self, cbr: float, current_power: float, neighbor_count: int) -> float:
        """Base reward calculation"""
        targets = get_adaptive_targets(neighbor_count)
        weights = get_reward_weights(cbr, targets['SINR_TARGET'])
        
        cbr_error = abs(cbr - targets['CBR_TARGET'])
        cbr_term = weights['W1'] * (1 - math.tanh(weights['BETA'] * cbr_error))
        
        power_norm = (current_power - system_config.power_min) / (system_config.power_max - system_config.power_min)
        power_term = weights['W2'] * (power_norm ** 2)
        
        neighbor_impact = weights['W5'] * math.log(1 + neighbor_count)
        
        return cbr_term - power_term - neighbor_impact
    
    def update(self, batch: Tuple[torch.Tensor, ...], 
               weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Enhanced update method"""
        try:
            states, actions, rewards, next_states, dones = batch
            
            features = self.feature_extractor(states)
            
            critic_loss1, critic_loss2, td_errors = self._update_critics(
                features, actions, rewards, next_states, dones, weights
            )
            
            with torch.no_grad():
                fresh_features = self.feature_extractor(states)
            
            actor_loss, alpha_loss = self._update_actor_and_alpha(fresh_features.detach())
            
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
            
            self.replay_buffer.beta = min(1.0, self.replay_buffer.beta + training_config.per_beta_increment)
            
            self.training_step += 1
            
            return {
                'actor_loss': actor_loss,
                'critic1_loss': critic_loss1,
                'critic2_loss': critic_loss2,
                'alpha_loss': alpha_loss,
                'alpha': self.log_alpha.exp().item(),
                'td_errors': td_errors,
                'exploration_factor': self.exploration_factor
            }
        except Exception as e:
            logger.error(f"Agent update failed: {e}")
            return {
                'actor_loss': 0.0,
                'critic1_loss': 0.0,
                'critic2_loss': 0.0,
                'alpha_loss': 0.0,
                'alpha': 0.1,
                'td_errors': torch.zeros(1),
                'exploration_factor': self.exploration_factor
            }
    
    def _update_critics(self, features: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[float, float, torch.Tensor]:
        """FIXED: Update critic networks without gradient conflicts"""
        try:
            # Calculate targets
            with torch.no_grad():
                next_features = self.feature_extractor(next_states)
                next_actions, next_log_probs = self.actor.sample(next_features)
                
                target_q1 = self.target_critic1(next_features, next_actions)
                target_q2 = self.target_critic2(next_features, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
                
                target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * training_config.gamma * target_q
            
            # Get current Q values
            current_q1 = self.critic1(features, actions)
            current_q2 = self.critic2(features, actions)
            
            # Calculate losses
            td_errors1 = current_q1 - target_q
            td_errors2 = current_q2 - target_q
            
            if weights is not None:
                critic1_loss = (td_errors1.pow(2) * weights.unsqueeze(-1)).mean()
                critic2_loss = (td_errors2.pow(2) * weights.unsqueeze(-1)).mean()
            else:
                critic1_loss = td_errors1.pow(2).mean()
                critic2_loss = td_errors2.pow(2).mean()
            
            # FIXED: Update critics separately to avoid gradient conflicts
            # Update Critic 1
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.feature_extractor.parameters()), 1.0)
            self.critic1_optimizer.step()
            
            # Update Critic 2 (recalculate to avoid gradient graph conflicts)
            current_q2_fresh = self.critic2(features.detach(), actions.detach())
            td_errors2_fresh = current_q2_fresh - target_q.detach()
            
            if weights is not None:
                critic2_loss_fresh = (td_errors2_fresh.pow(2) * weights.detach().unsqueeze(-1)).mean()
            else:
                critic2_loss_fresh = td_errors2_fresh.pow(2).mean()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss_fresh.backward()
            torch.nn.utils.clip_grad_norm_(list(self.critic2.parameters()) + list(self.feature_extractor.parameters()), 1.0)
            self.critic2_optimizer.step()
            
            # Return TD errors for priority updates
            td_errors = torch.min(td_errors1.abs(), td_errors2.abs()).detach()
            
            return critic1_loss.item(), critic2_loss_fresh.item(), td_errors
            
        except Exception as e:
            logger.error(f"FIXED Critic update failed: {e}")
            return 0.0, 0.0, torch.zeros(1)
    
    def _update_actor_and_alpha(self, features: torch.Tensor) -> Tuple[float, float]:
        """Update actor network and entropy coefficient"""
        try:
            features = features.detach()
            
            new_actions, log_probs = self.actor.sample(features)
            
            q1 = self.critic1(features, new_actions)
            q2 = self.critic2(features, new_actions)
            q = torch.min(q1, q2)
            
            actor_loss = (self.log_alpha.exp().detach() * log_probs - q).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            with torch.no_grad():
                fresh_actions, fresh_log_probs = self.actor.sample(features)
            
            alpha_loss = -(self.log_alpha * (fresh_log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            return actor_loss.item(), alpha_loss.item()
        except Exception as e:
            logger.error(f"Actor/alpha update failed: {e}")
            return 0.0, 0.0
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                training_config.tau * param.data + (1 - training_config.tau) * target_param.data
            )
    
    def _hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# Legacy agent classes
class MACAgent(UltraExplorativeSACAgent):
    """MAC agent - Controls beacon rate and MCS with density awareness"""
    def __init__(self, feature_extractor: nn.Module = None, training_mode: bool = True):
        # CORRECTED: 2 actions [beacon_delta, mcs_delta]
        super().__init__(2, [(-10.0, 10.0), (-7.5, 7.5)], training_mode)
    
    def get_density_aware_weights(self, cbr: float, sinr: float, neighbor_count: int) -> Dict[str, float]:
        """MAC-specific density-aware weight adjustment"""
        # Start with base weights
        base_weights = get_reward_weights(cbr, sinr)
        
        density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
        
        # MAC-specific density adaptations
        if density_factor > 0.7:  # High density scenario
            base_weights['W1'] *= 1.5    # STRONGER focus on CBR (MAC's primary job)
            base_weights['W_BEACON'] = 2.5  # STRONG beacon rate penalty
            base_weights['W_MCS'] = 2.0     # Prefer robust MCS
            base_weights['W5'] *= 1.4       # Cooperation incentive
            
        elif density_factor < 0.3:  # Low density scenario  
            base_weights['W1'] *= 0.9       # Relax CBR focus slightly
            base_weights['W_BEACON'] = 0.6  # Allow aggressive beacon rates
            base_weights['W_MCS'] = 1.8     # Encourage high MCS for efficiency
            base_weights['W5'] *= 0.5       # Minimal cooperation needed
            
        else:  # Medium density (3-8 neighbors)
            base_weights['W1'] *= 1.2       # Balanced CBR focus
            base_weights['W_BEACON'] = 1.3  # Moderate beacon control
            base_weights['W_MCS'] = 1.4     # Adaptive MCS selection
            base_weights['W5'] *= 0.8       # Some cooperation
        
        return base_weights
    
    def calculate_reward(self, cbr: float, sinr: float, current_beacon: float, 
                        current_mcs: int, neighbor_count: int) -> float:
        """Enhanced MAC reward with integrated density awareness"""
        try:
            # Get density-aware targets
            targets = get_adaptive_targets(neighbor_count)
            
            # Get MAC-specific density-aware weights
            weights = self.get_density_aware_weights(cbr, sinr, neighbor_count)
            
            density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
            
            # 1. CBR Control (Primary MAC responsibility)
            cbr_error = abs(cbr - targets['CBR_TARGET'])
            cbr_term = weights['W1'] * (1 - math.tanh(weights['BETA'] * cbr_error))
            
            # 2. Density-Aware Beacon Rate Control
            beacon_norm = (current_beacon - system_config.beacon_rate_min) / \
                         (system_config.beacon_rate_max - system_config.beacon_rate_min)
            
            # High density → stronger beacon penalty, Low density → beacon reward
            if density_factor > 0.6:
                beacon_term = -weights['W_BEACON'] * density_factor * (beacon_norm ** 2)
            else:
                # Low density rewards higher beacon rates for coverage
                beacon_term = weights['W_BEACON'] * (1 - density_factor) * beacon_norm
            
            # 3. Density-Aware MCS Control  
            mcs_norm = current_mcs / system_config.mcs_max
            
            if density_factor > 0.7:  # High density
                # Prefer lower MCS for robustness (reward = 1 - mcs_norm)
                mcs_term = weights['W_MCS'] * (1 - mcs_norm)
                logger.debug(f"MAC High density: prefer low MCS, mcs_norm={mcs_norm:.2f}, reward={mcs_term:.3f}")
                
            elif density_factor < 0.3:  # Low density
                # Prefer higher MCS for efficiency (reward = mcs_norm)  
                mcs_term = weights['W_MCS'] * mcs_norm
                logger.debug(f"MAC Low density: prefer high MCS, mcs_norm={mcs_norm:.2f}, reward={mcs_term:.3f}")
                
            else:  # Medium density - adaptive based on SINR
                if sinr > targets['SINR_TARGET']:
                    mcs_term = weights['W_MCS'] * mcs_norm  # Can afford higher MCS
                else:
                    mcs_term = weights['W_MCS'] * (1 - mcs_norm)  # Need robust MCS
                logger.debug(f"MAC Medium density: SINR-adaptive MCS, sinr={sinr:.1f}, target={targets['SINR_TARGET']:.1f}")
            
            # 4. Cooperation/Neighbor Impact
            neighbor_penalty = weights['W5'] * math.log(1 + neighbor_count)
            
            # Final MAC reward
            mac_reward = cbr_term + beacon_term + mcs_term - neighbor_penalty
            
            # Logging for debugging
            if random.random() < 0.1:  # Log 10% of calculations
                logger.debug(f"MAC Reward Breakdown (density={density_factor:.2f}):")
                logger.debug(f"  CBR: {cbr_term:.3f}, Beacon: {beacon_term:.3f}, MCS: {mcs_term:.3f}, Neighbor: -{neighbor_penalty:.3f}")
                logger.debug(f"  Total: {mac_reward:.3f}")
            
            return float(mac_reward)
            
        except Exception as e:
            logger.error(f"MAC reward calculation failed: {e}")
            return 0.0


class PHYAgent(UltraExplorativeSACAgent):
    """PHY agent - Controls power transmission with density awareness"""
    def __init__(self, feature_extractor: nn.Module = None, training_mode: bool = True):
        # CORRECTED: 1 action [power_delta]
        super().__init__(1, [(-15.0, 15.0)], training_mode)
    
    def get_density_aware_weights(self, cbr: float, sinr: float, neighbor_count: int) -> Dict[str, float]:
        """PHY-specific density-aware weight adjustment"""
        # Start with base weights
        base_weights = get_reward_weights(cbr, sinr)
        
        density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
        
        # PHY-specific density adaptations
        if density_factor > 0.7:  # High density scenario
            base_weights['W3'] *= 0.9       # Slightly relax SINR (interference tolerance)
            base_weights['W2'] *= 2.0       # VERY STRONG power penalty
            base_weights['W_INTERFERENCE'] = 3.0  # New: Strong interference penalty
            base_weights['W5'] *= 1.6       # Strong cooperation incentive
            
        elif density_factor < 0.3:  # Low density scenario
            base_weights['W3'] *= 1.4       # MORE focus on SINR quality
            base_weights['W2'] *= 0.6       # RELAX power penalty for coverage
            base_weights['W_INTERFERENCE'] = 0.5  # Minimal interference concern
            base_weights['W5'] *= 0.4       # Minimal cooperation needed
            
        else:  # Medium density
            base_weights['W3'] *= 1.1       # Slight SINR focus
            base_weights['W2'] *= 1.0       # Standard power penalty
            base_weights['W_INTERFERENCE'] = 1.5  # Moderate interference awareness
            base_weights['W5'] *= 0.7       # Some cooperation
        
        return base_weights
    
    def calculate_reward(self, cbr: float, sinr: float, current_power: float, 
                        neighbor_count: int) -> float:
        """Enhanced PHY reward with integrated density awareness"""
        try:
            # Get density-aware targets
            targets = get_adaptive_targets(neighbor_count)
            
            # Get PHY-specific density-aware weights  
            weights = self.get_density_aware_weights(cbr, sinr, neighbor_count)
            
            density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
            
            # 1. SINR Control (Primary PHY responsibility)
            sinr_diff = sinr - targets['SINR_TARGET']
            if abs(sinr_diff) < 2.0:
                sinr_term = 0  # Good enough SINR
            else:
                sinr_term = weights['W3'] * math.tanh(sinr_diff / 5.0)
            
            # 2. Density-Aware Power Control
            power_norm = (current_power - system_config.power_min) / \
                        (system_config.power_max - system_config.power_min)
            
            # Progressive power penalty based on density
            power_penalty_multiplier = 1.0 + (density_factor ** 2)  # Quadratic scaling
            power_penalty = weights['W2'] * power_penalty_multiplier * (power_norm ** 2)
            
            # 3. Interference Awareness (density-dependent)
            interference_penalty = weights['W_INTERFERENCE'] * power_norm * density_factor
            
            # 4. Coverage vs Interference Trade-off
            if density_factor < 0.3:  # Low density - reward coverage
                coverage_bonus = weights['W_COVERAGE'] * power_norm * (1 - density_factor)
            else:
                coverage_bonus = 0
            
            # 5. Cooperation Incentive
            cooperation_penalty = weights['W5'] * density_factor * power_norm
            
            # Final PHY reward
            phy_reward = sinr_term - power_penalty - interference_penalty + coverage_bonus - cooperation_penalty
            
            # Logging for debugging
            if random.random() < 0.1:  # Log 10% of calculations
                logger.debug(f"PHY Reward Breakdown (density={density_factor:.2f}):")
                logger.debug(f"  SINR: {sinr_term:.3f}, Power: -{power_penalty:.3f}, Interference: -{interference_penalty:.3f}")
                if coverage_bonus > 0:
                    logger.debug(f"  Coverage: +{coverage_bonus:.3f}")
                logger.debug(f"  Cooperation: -{cooperation_penalty:.3f}, Total: {phy_reward:.3f}")
            
            return float(phy_reward)
            
        except Exception as e:
            logger.error(f"PHY reward calculation failed: {e}")
            return 0.0


   

    def get_enhanced_adaptive_targets(neighbor_count: int) -> Dict[str, float]:
        """Enhanced adaptive targets for both MAC and PHY parameters"""
        density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
        
        return {
            # Existing targets
            'CBR_TARGET': training_config.cbr_target_base * (1 - 0.4 * density_factor),
            'SINR_TARGET': max(8.0, training_config.sinr_target_base - 6.0 * density_factor),
            
            # New MAC targets
            'BEACON_TARGET': system_config.beacon_rate_max * (1 - 0.6 * density_factor),
            'MCS_TARGET': system_config.mcs_max * (1 - 0.5 * density_factor),
            
            # New PHY targets  
            'POWER_TARGET': system_config.power_max * (1 - 0.4 * density_factor)
        }

# ==============================
# Enhanced Vehicle Node (Kept Most Functionality)
# ==============================

class MaxExplorativeVehicleNode:
    """Vehicle node with centralized training support"""
    
    def __init__(self, node_id: str, shared_mac_agent, shared_phy_agent,
                 centralized_trainer, training_mode: bool = True):
        self.node_id = node_id
        self.training_mode = training_mode
        
        # FIXED: Use shared agents and centralized trainer
        self.mac_agent = shared_mac_agent
        self.phy_agent = shared_phy_agent
        self.centralized_trainer = centralized_trainer  # NEW: Centralized training
        
        self.current_power = 20.0
        self.current_beacon_rate = 10.0
        self.current_mcs = 5
        
        self.last_state = None
        self.last_actions = None
        self.step_count = 0
        
        self.action_history = {
            'power_deltas': deque(maxlen=200),
            'beacon_deltas': deque(maxlen=200),
            'mcs_deltas': deque(maxlen=200)
        }
        
        # Setup logging and metrics
        if training_mode:
            if ENABLE_TENSORBOARD:
                log_dir = os.path.join(system_config.log_dir, f"vehicle_{node_id}")
                os.makedirs(log_dir, exist_ok=True)
                try:
                    self.writer = SummaryWriter(log_dir)
                except Exception as e:
                    logger.warning(f"TensorBoard writer creation failed for {node_id}: {e}")
                    self.writer = None
            else:
                self.writer = None
                
            self.metrics = {
                'cbr_violations': 0,
                'sinr_violations': 0,
                'power_changes': [],
                'beacon_changes': [],
                'mcs_changes': [],
                'episode_rewards': [],
                'exploration_metrics': []
            }
    
    def get_coordinated_actions(self, state: List[float], current_params: Dict[str, float]) -> Dict[str, Any]:
        """
        CORRECTED: Get coordinated actions with proper agent responsibilities and comprehensive density adaptation
        
        Architecture:
        - MAC Agent: Controls beacon_rate + MCS (2 actions)
        - PHY Agent: Controls power (1 action)
        """
        try:
            # ================================
            # 1. INPUT VALIDATION & PREPROCESSING
            # ================================
            if len(state) != 3:
                logger.error(f"Vehicle {self.node_id}: Invalid state length: {len(state)}, expected 3")
                return self._get_safe_default_actions()
            
            # Validate state values
            cbr, snr, neighbor_count = state
            if not (0 <= cbr <= 1.0):
                logger.warning(f"Vehicle {self.node_id}: CBR out of range: {cbr}")
                cbr = max(0.0, min(1.0, cbr))
            
            if snr < 0:
                logger.warning(f"Vehicle {self.node_id}: Negative SNR: {snr}")
                snr = max(0.0, snr)
                
            neighbor_count = max(0, int(neighbor_count))
            state = [cbr, snr, float(neighbor_count)]  # Ensure clean state
            
            # Update current parameters with validation
            self._update_current_params(current_params)
            
            # Convert state to tensor
            state_tensor = safe_tensor_conversion(state)
            if state_tensor.shape[-1] != 3:
                logger.error(f"Vehicle {self.node_id}: State tensor has wrong dimensions: {state_tensor.shape}")
                return self._get_safe_default_actions()
            
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Calculate density factor for logging and analysis
            density_factor = min(1.0, neighbor_count / training_config.max_neighbors)
            
            # ================================
            # 2. AGENT ACTION SELECTION (CORRECTED ARCHITECTURE)
            # ================================
            try:
                # MAC Agent: Controls beacon_rate and MCS (2 actions)
                mac_actions, mac_log_prob = self.mac_agent.select_action(
                    state_tensor.clone(), 
                    deterministic=not self.training_mode
                )
                
                # PHY Agent: Controls power (1 action)  
                phy_actions, phy_log_prob = self.phy_agent.select_action(
                    state_tensor.clone(),
                    deterministic=not self.training_mode
                )
                
            except Exception as e:
                logger.error(f"Vehicle {self.node_id}: Agent action selection failed: {e}")
                return self._get_safe_default_actions()
            
            # ================================
            # 3. CORRECTED ACTION MAPPING
            # ================================
            # Extract actions with NaN protection
            try:
                beacon_delta = float(mac_actions[0, 0].item())  # MAC action 0: beacon control
                mcs_delta = float(mac_actions[0, 1].item())     # MAC action 1: MCS control
                power_delta = float(phy_actions[0, 0].item())   # PHY action 0: power control
                
                # NaN protection
                if math.isnan(beacon_delta) or math.isinf(beacon_delta):
                    logger.warning(f"Vehicle {self.node_id}: Invalid beacon_delta: {beacon_delta}")
                    beacon_delta = 0.0
                if math.isnan(mcs_delta) or math.isinf(mcs_delta):
                    logger.warning(f"Vehicle {self.node_id}: Invalid mcs_delta: {mcs_delta}")
                    mcs_delta = 0.0
                if math.isnan(power_delta) or math.isinf(power_delta):
                    logger.warning(f"Vehicle {self.node_id}: Invalid power_delta: {power_delta}")
                    power_delta = 0.0
                    
            except Exception as e:
                logger.error(f"Vehicle {self.node_id}: Action extraction failed: {e}")
                beacon_delta = mcs_delta = power_delta = 0.0
            
            # ================================
            # 4. ENHANCED DIVERSITY EXPLORATION
            # ================================
            if self.training_mode:
                beacon_delta, mcs_delta, power_delta = self._apply_enhanced_diversity_exploration(
                    beacon_delta, mcs_delta, power_delta, density_factor
                )
            
            # ================================
            # 5. CALCULATE NEW PARAMETERS WITH DENSITY-AWARE BOUNDS
            # ================================
            new_params = self._calculate_new_parameters_with_density_bounds(
                beacon_delta, mcs_delta, power_delta, density_factor
            )
            
            # ================================
            # 6. DENSITY-AWARE REWARD CALCULATION
            # ================================
            # Calculate individual agent rewards using their density-aware methods
            mac_reward = self.mac_agent.calculate_reward(
                cbr=cbr,
                sinr=snr, 
                current_beacon=new_params['beacon_rate'],
                current_mcs=new_params['mcs'],
                neighbor_count=neighbor_count
            )
            
            phy_reward = self.phy_agent.calculate_reward(
                cbr=cbr,
                sinr=snr,
                current_power=new_params['power'],
                neighbor_count=neighbor_count
            )
            
            # Combined reward with MAC-heavy weighting (since MAC handles 2 parameters)
            joint_reward = 0.65 * mac_reward + 0.35 * phy_reward
            
            # Add exploration bonus in training mode
            if self.training_mode and self.mac_agent.exploration_factor > 1.0:
                exploration_bonus = 0.05 * (self.mac_agent.exploration_factor - 1.0)
                joint_reward += exploration_bonus
            
            # ================================
            # 7. EXPERIENCE STORAGE FOR CENTRALIZED TRAINING
            # ================================
            if self.last_state is not None and self.training_mode:
                self._store_experience_for_centralized_training(state, joint_reward)
            
            # ================================
            # 8. UPDATE METRICS AND HISTORY
            # ================================
            if self.training_mode:
                self._update_comprehensive_metrics(
                    state, new_params, joint_reward, mac_reward, phy_reward,
                    beacon_delta, mcs_delta, power_delta, density_factor
                )
            
            # ================================
            # 9. UPDATE INTERNAL STATE
            # ================================
            self.last_state = state.copy()
            self.last_actions = {
                'mac': mac_actions.clone(),
                'phy': phy_actions.clone(),
                'mac_log_prob': mac_log_prob,
                'phy_log_prob': phy_log_prob,
                'beacon_delta': beacon_delta,
                'mcs_delta': mcs_delta,
                'power_delta': power_delta
            }
            
            self.step_count += 1
            
            # ================================
            # 10. ENHANCED DENSITY LOGGING
            # ================================
            self._log_density_adaptation_analysis(
                state, new_params, density_factor, mac_reward, phy_reward, joint_reward,
                beacon_delta, mcs_delta, power_delta
            )
            
            # ================================
            # 11. PREPARE RESPONSE
            # ================================
            result = {
                # CORRECTED: Proper agent responsibilities
                'beacon_delta': beacon_delta,        # MAC Agent responsibility
                'mcs_delta': mcs_delta,             # MAC Agent responsibility  
                'power_delta': power_delta,         # PHY Agent responsibility
                
                # New parameter values
                'new_power': new_params['power'],
                'new_beacon_rate': new_params['beacon_rate'],
                'new_mcs': new_params['mcs'],
                
                # Reward information
                'joint_reward': joint_reward,
                'mac_reward': mac_reward,
                'phy_reward': phy_reward,
                
                # Analysis information
                'density_factor': density_factor,
                'neighbor_count': neighbor_count,
                'exploration_factor': self.mac_agent.exploration_factor,
                
                # Log probabilities for training
                'mac_log_prob': float(mac_log_prob.item()),
                'phy_log_prob': float(phy_log_prob.item()),
                
                # Debugging information
                'current_power': self.current_power,
                'current_beacon_rate': self.current_beacon_rate,
                'current_mcs': self.current_mcs
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Vehicle {self.node_id}: Critical error in coordinated actions: {e}")
            logger.error(f"Vehicle {self.node_id}: State: {state}, Current params: {current_params}")
            return self._get_safe_default_actions()
    
    def _apply_enhanced_diversity_exploration(self, beacon_delta: float, mcs_delta: float, 
                                            power_delta: float, density_factor: float) -> Tuple[float, float, float]:
        """Apply diversity exploration based on action history and density"""
        try:
            # Only apply if we have enough history
            if len(self.action_history['beacon_deltas']) > 10:
                
                # Calculate diversity metrics for recent actions
                recent_beacon = list(self.action_history['beacon_deltas'])[-20:]
                recent_mcs = list(self.action_history['mcs_deltas'])[-20:]
                recent_power = list(self.action_history['power_deltas'])[-20:]
                
                try:
                    beacon_std = np.std(recent_beacon) if len(recent_beacon) > 1 else 1.0
                    mcs_std = np.std(recent_mcs) if len(recent_mcs) > 1 else 1.0
                    power_std = np.std(recent_power) if len(recent_power) > 1 else 1.0
                    
                    # Ensure valid std values
                    beacon_std = max(0.1, beacon_std) if math.isfinite(beacon_std) else 1.0
                    mcs_std = max(0.1, mcs_std) if math.isfinite(mcs_std) else 1.0
                    power_std = max(0.1, power_std) if math.isfinite(power_std) else 1.0
                    
                    # Density-aware exploration thresholds
                    if density_factor > 0.7:  # High density - less exploration
                        beacon_threshold = 0.3
                        mcs_threshold = 0.2  
                        power_threshold = 0.4
                    else:  # Low/medium density - more exploration
                        beacon_threshold = 0.5
                        mcs_threshold = 0.4
                        power_threshold = 0.6
                    
                    # Add diversity noise if actions are too uniform
                    if beacon_std < beacon_threshold:
                        noise = torch.randn(1).item() * (2.0 - density_factor)
                        if torch.isfinite(torch.tensor(noise)):
                            beacon_delta += noise
                    
                    if mcs_std < mcs_threshold:
                        noise = torch.randn(1).item() * (1.5 - density_factor)
                        if torch.isfinite(torch.tensor(noise)):
                            mcs_delta += noise
                    
                    if power_std < power_threshold:
                        noise = torch.randn(1).item() * (2.5 - density_factor)
                        if torch.isfinite(torch.tensor(noise)):
                            power_delta += noise
                            
                except Exception as e:
                    logger.debug(f"Vehicle {self.node_id}: Diversity calculation failed: {e}")
            
            # Random exploration bursts (density-aware)
            exploration_prob = 0.05 * self.mac_agent.exploration_factor * (2.0 - density_factor)
            if random.random() < exploration_prob:
                if random.random() < 0.33:  # Beacon exploration
                    beacon_delta += random.uniform(-3.0, 3.0) * (2.0 - density_factor)
                elif random.random() < 0.66:  # MCS exploration  
                    mcs_delta += random.uniform(-2.0, 2.0) * (2.0 - density_factor)
                else:  # Power exploration
                    power_delta += random.uniform(-4.0, 4.0) * (2.0 - density_factor)
            
            return beacon_delta, mcs_delta, power_delta
            
        except Exception as e:
            logger.error(f"Vehicle {self.node_id}: Enhanced diversity exploration failed: {e}")
            return beacon_delta, mcs_delta, power_delta
    
    def _calculate_new_parameters_with_density_bounds(self, beacon_delta: float, mcs_delta: float, 
                                                    power_delta: float, density_factor: float) -> Dict[str, float]:
        """Calculate new parameters with density-aware bounds and enhanced safety"""
        try:
            # Density-aware action bounds (more conservative in high density)
            if density_factor > 0.7:  # High density - conservative bounds
                beacon_delta = max(-8.0, min(8.0, beacon_delta))
                mcs_delta = max(-4.0, min(4.0, mcs_delta))
                power_delta = max(-10.0, min(10.0, power_delta))
            elif density_factor < 0.3:  # Low density - aggressive bounds
                beacon_delta = max(-15.0, min(15.0, beacon_delta))  
                mcs_delta = max(-6.0, min(6.0, mcs_delta))
                power_delta = max(-20.0, min(20.0, power_delta))
            else:  # Medium density - standard bounds
                beacon_delta = max(-10.0, min(10.0, beacon_delta))
                mcs_delta = max(-5.0, min(5.0, mcs_delta))
                power_delta = max(-15.0, min(15.0, power_delta))
            
            # Calculate new parameters with system bounds
            new_power = max(system_config.power_min,
                           min(system_config.power_max,
                               self.current_power + power_delta))
            
            new_beacon = max(system_config.beacon_rate_min,
                            min(system_config.beacon_rate_max,
                                self.current_beacon_rate + beacon_delta))
            
            # MCS is integer, so round the delta
            mcs_delta_rounded = round(mcs_delta) if math.isfinite(mcs_delta) else 0
            new_mcs = max(system_config.mcs_min,
                         min(system_config.mcs_max,
                             self.current_mcs + mcs_delta_rounded))
            
            # Update action history for diversity tracking
            self.action_history['beacon_deltas'].append(beacon_delta)
            self.action_history['mcs_deltas'].append(mcs_delta)
            self.action_history['power_deltas'].append(power_delta)
            
            return {
                'power': float(new_power),
                'beacon_rate': float(new_beacon),
                'mcs': int(new_mcs)
            }
            
        except Exception as e:
            logger.error(f"Vehicle {self.node_id}: Parameter calculation failed: {e}")
            return {
                'power': float(self.current_power),
                'beacon_rate': float(self.current_beacon_rate),
                'mcs': int(self.current_mcs)
            }
    
    def _store_experience_for_centralized_training(self, current_state: List[float], reward: float):
        """Store experience using centralized training manager"""
        if self.last_state is None or self.last_actions is None:
            return
        
        try:
            # Create actions dictionary for centralized trainer
            actions_dict = {
                'beacon_delta': self.last_actions['beacon_delta'],
                'mcs_delta': self.last_actions['mcs_delta'], 
                'power_delta': self.last_actions['power_delta']
            }
            
            # Use centralized training manager
            self.centralized_trainer.add_experience(
                self.last_state, actions_dict, reward, current_state
            )
            
        except Exception as e:
            logger.error(f"Vehicle {self.node_id}: Experience storage failed: {e}")
    
    def _update_comprehensive_metrics(self, state: List[float], new_params: Dict[str, float], 
                                    joint_reward: float, mac_reward: float, phy_reward: float,
                                    beacon_delta: float, mcs_delta: float, power_delta: float,
                                    density_factor: float):
        """Update comprehensive metrics with density analysis"""
        try:
            cbr, snr, neighbor_count = state
            
            # Update violation counters
            targets = get_adaptive_targets(int(neighbor_count))
            if cbr < 0.6 or cbr > 0.7:
                self.metrics['cbr_violations'] += 1
            if snr < 10.0:
                self.metrics['sinr_violations'] += 1
            
            # Update action change metrics
            power_change = abs(new_params['power'] - self.current_power)
            beacon_change = abs(new_params['beacon_rate'] - self.current_beacon_rate)
            mcs_change = abs(new_params['mcs'] - self.current_mcs)
            
            self.metrics['power_changes'].append(power_change)
            self.metrics['beacon_changes'].append(beacon_change)
            self.metrics['mcs_changes'].append(mcs_change)
            self.metrics['episode_rewards'].append(joint_reward)
            
            # Density-specific metrics
            if not hasattr(self.metrics, 'density_metrics'):
                self.metrics['density_metrics'] = {
                    'high_density_rewards': [],
                    'medium_density_rewards': [],
                    'low_density_rewards': [],
                    'density_transitions': []
                }
            
            # Categorize reward by density
            if density_factor > 0.7:
                self.metrics['density_metrics']['high_density_rewards'].append(joint_reward)
            elif density_factor < 0.3:
                self.metrics['density_metrics']['low_density_rewards'].append(joint_reward)
            else:
                self.metrics['density_metrics']['medium_density_rewards'].append(joint_reward)
            
            # Track density transitions
            if hasattr(self, 'last_density_factor'):
                density_change = abs(density_factor - self.last_density_factor)
                if density_change > 0.2:  # Significant density change
                    self.metrics['density_metrics']['density_transitions'].append({
                        'step': self.step_count,
                        'from': self.last_density_factor,
                        'to': density_factor,
                        'reward_impact': joint_reward
                    })
            
            self.last_density_factor = density_factor
            
            # Enhanced exploration metrics
            if len(self.action_history['beacon_deltas']) > 10:
                exploration_diversity = {
                    'beacon_std': np.std(list(self.action_history['beacon_deltas'])[-50:]),
                    'mcs_std': np.std(list(self.action_history['mcs_deltas'])[-50:]),
                    'power_std': np.std(list(self.action_history['power_deltas'])[-50:]),
                    'action_magnitude': abs(beacon_delta) + abs(mcs_delta) + abs(power_delta),
                    'density_factor': density_factor,
                    'mac_reward': mac_reward,
                    'phy_reward': phy_reward
                }
                self.metrics['exploration_metrics'].append(exploration_diversity)
            
            # TensorBoard logging
            if self.writer:
                try:
                    self.writer.add_scalar('Rewards/Joint', joint_reward, self.step_count)
                    self.writer.add_scalar('Rewards/MAC', mac_reward, self.step_count)
                    self.writer.add_scalar('Rewards/PHY', phy_reward, self.step_count)
                    self.writer.add_scalar('Density/Factor', density_factor, self.step_count)
                    self.writer.add_scalar('Density/Neighbors', neighbor_count, self.step_count)
                    self.writer.add_scalar('Actions/Beacon_Delta', beacon_delta, self.step_count)
                    self.writer.add_scalar('Actions/MCS_Delta', mcs_delta, self.step_count)
                    self.writer.add_scalar('Actions/Power_Delta', power_delta, self.step_count)
                    self.writer.add_scalar('Exploration/Factor', self.mac_agent.exploration_factor, self.step_count)
                except Exception as e:
                    logger.warning(f"Vehicle {self.node_id}: TensorBoard logging failed: {e}")
            
        except Exception as e:
            logger.error(f"Vehicle {self.node_id}: Comprehensive metrics update failed: {e}")
    
    def _log_density_adaptation_analysis(self, state: List[float], new_params: Dict[str, float],
                                       density_factor: float, mac_reward: float, phy_reward: float, 
                                       joint_reward: float, beacon_delta: float, mcs_delta: float, 
                                       power_delta: float):
        """Enhanced logging for density adaptation analysis"""
        try:
            # Log every 25 steps for detailed analysis
            if self.step_count % 25 == 0:
                cbr, snr, neighbor_count = state
                
                # Determine density category
                if density_factor > 0.7:
                    density_category = "HIGH"
                elif density_factor < 0.3:
                    density_category = "LOW"
                else:
                    density_category = "MEDIUM"
                
                logger.info(f"[DENSITY ANALYSIS] Vehicle {self.node_id} Step {self.step_count}")
                logger.info(f"  Environment: CBR={cbr:.3f}, SNR={snr:.1f}dB, Neighbors={int(neighbor_count)} ({density_category})")
                logger.info(f"  Density Factor: {density_factor:.3f}")
                logger.info(f"  Agent Actions:")
                logger.info(f"    MAC: Beacon Δ={beacon_delta:+.2f}Hz, MCS Δ={mcs_delta:+.2f}")
                logger.info(f"    PHY: Power Δ={power_delta:+.2f}dBm")
                logger.info(f"  New Parameters:")
                logger.info(f"    Power: {self.current_power:.1f} → {new_params['power']:.1f}dBm")
                logger.info(f"    Beacon: {self.current_beacon_rate:.1f} → {new_params['beacon_rate']:.1f}Hz")
                logger.info(f"    MCS: {self.current_mcs} → {new_params['mcs']}")
                logger.info(f"  Rewards: MAC={mac_reward:.3f}, PHY={phy_reward:.3f}, Joint={joint_reward:.3f}")
                logger.info(f"  Exploration: {self.mac_agent.exploration_factor:.3f}")
                
                # Analyze adaptation behavior
                if density_category == "HIGH":
                    self._analyze_high_density_behavior(beacon_delta, mcs_delta, power_delta)
                elif density_category == "LOW":
                    self._analyze_low_density_behavior(beacon_delta, mcs_delta, power_delta)
                
        except Exception as e:
            logger.error(f"Vehicle {self.node_id}: Density analysis logging failed: {e}")
    
    def _analyze_high_density_behavior(self, beacon_delta: float, mcs_delta: float, power_delta: float):
        """Analyze if the agent is learning proper high-density behavior (ASCII-safe logging)"""
        try:
            expected_behaviors = []
            if power_delta < -1.0:
                expected_behaviors.append("OK: Reducing power (good for interference)")      # FIXED: ASCII-safe
            elif power_delta > 1.0:
                expected_behaviors.append("WARN: Increasing power (may cause interference)") # FIXED: ASCII-safe
            else:
                expected_behaviors.append("NEUTRAL: Maintaining power")
                
            if beacon_delta < -1.0:
                expected_behaviors.append("OK: Reducing beacon rate (good for congestion)")      # FIXED: ASCII-safe
            elif beacon_delta > 1.0:
                expected_behaviors.append("WARN: Increasing beacon rate (may cause congestion)") # FIXED: ASCII-safe
            else:
                expected_behaviors.append("NEUTRAL: Maintaining beacon rate")
                
            if mcs_delta < -1.0:
                expected_behaviors.append("OK: Choosing lower MCS (more robust)")      # FIXED: ASCII-safe
            elif mcs_delta > 1.0:
                expected_behaviors.append("WARN: Choosing higher MCS (less robust)")   # FIXED: ASCII-safe
            else:
                expected_behaviors.append("NEUTRAL: Maintaining MCS")
            
            logger.info(f"    High Density Analysis: {', '.join(expected_behaviors)}")
            
        except Exception as e:
            logger.error(f"High density analysis failed: {e}")
    
    def _analyze_low_density_behavior(self, beacon_delta: float, mcs_delta: float, power_delta: float):
        """Analyze if the agent is learning proper low-density behavior (ASCII-safe logging)"""
        try:
            expected_behaviors = []
            if power_delta > 1.0:
                expected_behaviors.append("OK: Increasing power (good for coverage)")         # FIXED: ASCII-safe
            elif power_delta < -1.0:
                expected_behaviors.append("WARN: Reducing power (wasting coverage opportunity)") # FIXED: ASCII-safe
            else:
                expected_behaviors.append("NEUTRAL: Maintaining power")
                
            if beacon_delta > 1.0:
                expected_behaviors.append("OK: Increasing beacon rate (good for connectivity)")      # FIXED: ASCII-safe
            elif beacon_delta < -1.0:
                expected_behaviors.append("WARN: Reducing beacon rate (wasting connectivity)")       # FIXED: ASCII-safe
            else:
                expected_behaviors.append("NEUTRAL: Maintaining beacon rate")
                
            if mcs_delta > 1.0:
                expected_behaviors.append("OK: Choosing higher MCS (good for efficiency)")     # FIXED: ASCII-safe
            elif mcs_delta < -1.0:
                expected_behaviors.append("WARN: Choosing lower MCS (wasting efficiency)")     # FIXED: ASCII-safe
            else:
                expected_behaviors.append("NEUTRAL: Maintaining MCS")
            
            logger.info(f"    Low Density Analysis: {', '.join(expected_behaviors)}")
            
        except Exception as e:
            logger.error(f"Low density analysis failed: {e}")
    
    def _get_safe_default_actions(self) -> Dict[str, Any]:
        """Return safe default actions in case of errors (CORRECTED)"""
        return {
            'beacon_delta': 0.0,        # MAC responsibility
            'mcs_delta': 0.0,          # MAC responsibility  
            'power_delta': 0.0,        # PHY responsibility
            'new_power': self.current_power,
            'new_beacon_rate': self.current_beacon_rate,
            'new_mcs': self.current_mcs,
            'joint_reward': 0.0,
            'mac_reward': 0.0,
            'phy_reward': 0.0,
            'density_factor': 0.0,
            'neighbor_count': 0,
            'exploration_factor': 1.0,
            'mac_log_prob': 0.0,
            'phy_log_prob': 0.0,
            'current_power': self.current_power,
            'current_beacon_rate': self.current_beacon_rate,
            'current_mcs': self.current_mcs
        }
    
    def _apply_diversity_exploration(self, mac_actions: torch.Tensor, 
                                   phy_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FIXED: Apply additional exploration based on action diversity with proper NaN protection"""
        try:
            if len(self.action_history['power_deltas']) > 10:  # Require at least 10 samples
                try:
                    power_std = np.std(list(self.action_history['power_deltas'])[-50:])
                    beacon_std = np.std(list(self.action_history['beacon_deltas'])[-50:])
                    mcs_std = np.std(list(self.action_history['mcs_deltas'])[-50:])
                    
                    power_std = max(0.1, power_std) if math.isfinite(power_std) else 1.0
                    beacon_std = max(0.1, beacon_std) if math.isfinite(beacon_std) else 1.0
                    mcs_std = max(0.1, mcs_std) if math.isfinite(mcs_std) else 1.0
                    
                    if power_std < 0.5:
                        noise = torch.randn(1) * 2.0
                        if torch.isfinite(noise):
                            mac_actions[0, 0] += noise
                    
                    if beacon_std < 0.3:
                        noise = torch.randn(1) * 1.0
                        if torch.isfinite(noise):
                            mac_actions[0, 1] += noise
                    
                    if mcs_std < 0.2:
                        noise = torch.randn(1) * 1.0
                        if torch.isfinite(noise):
                            phy_actions[0, 0] += noise
                            
                except Exception as e:
                    # Skip diversity exploration if calculation fails
                    logger.debug(f"Diversity exploration calculation failed: {e}")
            
            mac_actions = torch.where(torch.isnan(mac_actions) | torch.isinf(mac_actions), 
                                    torch.zeros_like(mac_actions), mac_actions)
            phy_actions = torch.where(torch.isnan(phy_actions) | torch.isinf(phy_actions), 
                                    torch.zeros_like(phy_actions), phy_actions)
            
            return mac_actions, phy_actions
        except Exception as e:
            logger.error(f"Diversity exploration failed: {e}")
            return mac_actions, phy_actions
    
    def _calculate_new_parameters_enhanced(self, mac_actions: torch.Tensor, 
                                         phy_actions: torch.Tensor) -> Dict[str, float]:
        """Calculate new parameters with enhanced bounds and NaN protection"""
        try:
            power_delta = mac_actions[0, 0].item()
            beacon_delta = mac_actions[0, 1].item()
            mcs_delta = phy_actions[0, 0].item()
            
            if math.isnan(power_delta) or math.isinf(power_delta):
                power_delta = 0.0
            if math.isnan(beacon_delta) or math.isinf(beacon_delta):
                beacon_delta = 0.0
            if math.isnan(mcs_delta) or math.isinf(mcs_delta):
                mcs_delta = 0.0
            
            # Enhanced bounds based on paper recommendations
            power_delta = max(-30.0, min(30.0, power_delta))  # Full power range coverage
            beacon_delta = max(-20.0, min(20.0, beacon_delta))  # Full beacon range coverage
            mcs_delta = max(-11.0, min(11.0, mcs_delta))  # Full MCS range coverage
            
            new_power = max(system_config.power_min,
                           min(system_config.power_max,
                               self.current_power + power_delta))
            
            new_beacon = max(system_config.beacon_rate_min,
                            min(system_config.beacon_rate_max,
                                self.current_beacon_rate + beacon_delta))
            
            mcs_delta_rounded = round(mcs_delta) if math.isfinite(mcs_delta) else 0
            new_mcs = max(system_config.mcs_min,
                         min(system_config.mcs_max,
                             self.current_mcs + mcs_delta_rounded))
            
            self.action_history['power_deltas'].append(power_delta)
            self.action_history['beacon_deltas'].append(beacon_delta)
            self.action_history['mcs_deltas'].append(mcs_delta)
            
            return {
                'power': float(new_power),
                'beacon_rate': float(new_beacon),
                'mcs': int(new_mcs)
            }
        except Exception as e:
            logger.error(f"Enhanced parameter calculation failed: {e}")
            return {
                'power': float(self.current_power),
                'beacon_rate': float(self.current_beacon_rate),
                'mcs': int(self.current_mcs)
            }
    
    def _update_enhanced_metrics(self, state: List[float], new_params: Dict[str, float], 
                               joint_reward: float, mac_actions: torch.Tensor, phy_actions: torch.Tensor):
        """Update metrics with exploration tracking"""
        try:
            targets = get_adaptive_targets(int(state[2]))
            if state[0] < 0.6 or state[0] > 0.7:
                self.metrics['cbr_violations'] += 1
            if state[1] < 10.0:
                self.metrics['sinr_violations'] += 1
                
            power_change = abs(new_params['power'] - self.current_power)
            beacon_change = abs(new_params['beacon_rate'] - self.current_beacon_rate)
            mcs_change = abs(new_params['mcs'] - self.current_mcs)
            
            self.metrics['power_changes'].append(power_change)
            self.metrics['beacon_changes'].append(beacon_change)
            self.metrics['mcs_changes'].append(mcs_change)
            self.metrics['episode_rewards'].append(joint_reward)
            
            if len(self.action_history['power_deltas']) > 10:
                exploration_diversity = {
                    'power_std': np.std(list(self.action_history['power_deltas'])[-50:]),
                    'beacon_std': np.std(list(self.action_history['beacon_deltas'])[-50:]),
                    'mcs_std': np.std(list(self.action_history['mcs_deltas'])[-50:]),
                    'power_range': np.ptp(list(self.action_history['power_deltas'])[-50:]),
                    'action_magnitude': float(torch.sqrt(mac_actions.pow(2).sum() + phy_actions.pow(2).sum()).item())
                }
                self.metrics['exploration_metrics'].append(exploration_diversity)
            
            if self.writer:
                try:
                    self.writer.add_scalar('Joint_Reward', joint_reward, self.step_count)
                    self.writer.add_scalar('Power_Change', power_change, self.step_count)
                    self.writer.add_scalar('Beacon_Change', beacon_change, self.step_count)
                    self.writer.add_scalar('MCS_Change', mcs_change, self.step_count)
                    self.writer.add_scalar('Exploration_Factor', self.mac_agent.exploration_factor, self.step_count)
                    
                    if self.metrics['exploration_metrics']:
                        latest_exp = self.metrics['exploration_metrics'][-1]
                        self.writer.add_scalar('Exploration/Power_Std', latest_exp['power_std'], self.step_count)
                        self.writer.add_scalar('Exploration/Action_Magnitude', latest_exp['action_magnitude'], self.step_count)
                except Exception as e:
                    logger.warning(f"TensorBoard logging failed: {e}")
                    
        except Exception as e:
            logger.error(f"Enhanced metrics update failed: {e}")
    
    def _calculate_joint_reward(self, state: List[float], new_params: Dict[str, float]) -> float:
        """Calculate joint reward with exploration bonus"""
        try:
            cbr, sinr, neighbor_count = state
            targets = get_adaptive_targets(int(neighbor_count))
            weights = get_reward_weights(cbr, sinr)
            
            cbr_error = abs(cbr - targets['CBR_TARGET'])
            cbr_reward = weights['W1'] * (1 - math.tanh(weights['BETA'] * cbr_error))
            
            sinr_diff = sinr - targets['SINR_TARGET']
            if abs(sinr_diff) < 2.0:
                sinr_reward = 0
            else:
                sinr_reward = weights['W3'] * math.tanh(sinr_diff / 5.0)
            
            power_norm = (new_params['power'] - system_config.power_min) / \
                        (system_config.power_max - system_config.power_min)
            power_penalty = (weights['W2'] + weights['W4']) * (power_norm ** 2)
            
            neighbor_penalty = weights['W5'] * math.log(1 + neighbor_count)
            
            joint_reward = 0.6 * (cbr_reward - power_penalty - neighbor_penalty) + \
                          0.4 * (sinr_reward - power_penalty - neighbor_penalty)
            
            if self.training_mode and self.mac_agent.exploration_factor > 1.0:
                exploration_bonus = 0.1 * self.mac_agent.exploration_factor
                joint_reward += exploration_bonus
            
            return float(joint_reward)
        except Exception as e:
            logger.error(f"Joint reward calculation failed: {e}")
            return 0.0
    
    def _update_current_params(self, params: Dict[str, float]):
        """Update current parameters with validation"""
        try:
            self.current_power = max(system_config.power_min, 
                                   min(system_config.power_max, 
                                       float(params.get('transmissionPower', self.current_power))))
            self.current_beacon_rate = max(system_config.beacon_rate_min,
                                         min(system_config.beacon_rate_max,
                                             float(params.get('beaconRate', self.current_beacon_rate))))
            self.current_mcs = max(system_config.mcs_min,
                                 min(system_config.mcs_max,
                                     int(params.get('MCS', self.current_mcs))))
        except (ValueError, TypeError) as e:
            logger.warning(f"Parameter update failed for {self.node_id}: {e}")
    
    def _get_safe_default_actions(self) -> Dict[str, Any]:
        """Return safe default actions in case of errors"""
        return {
            'power_delta': 0.0,
            'beacon_delta': 0.0,
            'mcs_delta': 0,
            'new_power': self.current_power,
            'new_beacon_rate': self.current_beacon_rate,
            'new_mcs': self.current_mcs,
            'joint_reward': 0.0,
            'mac_log_prob': 0.0,
            'phy_log_prob': 0.0,
            'exploration_factor': 1.0
        }
    
    def store_experience(self, state: List[float], actions: torch.Tensor, 
                        reward: float, next_state: List[float], done: bool = False):
        """DEPRECATED: Now handled by centralized training manager"""
        # This method is deprecated in favor of centralized training
        # Keep for backward compatibility but redirect to centralized trainer
        try:
            if not isinstance(actions, torch.Tensor):
                actions_tensor = safe_tensor_conversion(actions)
            else:
                actions_tensor = actions
                
            if actions_tensor.dim() > 1:
                actions_tensor = actions_tensor.squeeze()
            
            # Convert to actions dict for centralized trainer
            actions_dict = {
                'power_delta': float(actions_tensor[0].item()) if len(actions_tensor) > 0 else 0.0,
                'beacon_delta': float(actions_tensor[1].item()) if len(actions_tensor) > 1 else 0.0,
                'mcs_delta': float(actions_tensor[2].item()) if len(actions_tensor) > 2 else 0.0
            }
            
            self.centralized_trainer.add_experience(state, actions_dict, reward, next_state)
                
        except Exception as e:
            logger.error(f"Error in deprecated store_experience for {self.node_id}: {e}")

    def _store_experience_step(self, current_state: List[float], reward: float):
        """FIXED: Store experience using centralized training manager"""
        if self.last_state is None or self.last_actions is None:
            return
        
        try:
            # Use centralized training manager instead of direct agent updates
            actions_dict = {
                'power_delta': float(self.last_actions['mac'][0, 0].item()),
                'beacon_delta': float(self.last_actions['mac'][0, 1].item()),
                'mcs_delta': float(self.last_actions['phy'][0, 0].item())
            }
            
            self.centralized_trainer.add_experience(
                self.last_state, actions_dict, reward, current_state
            )
            
        except Exception as e:
            logger.error(f"FIXED Experience storage failed: {e}")

    def update_agents(self):
        """DEPRECATED: Now handled by centralized training manager"""
        # This method is now handled by the centralized training manager
        # Keep for backward compatibility but do nothing
        pass
    
    def _log_enhanced_metrics(self):
        """Enhanced metrics logging"""
        if not self.training_mode or not self.metrics['episode_rewards']:
            return
            
        try:
            avg_reward = np.mean(self.metrics['episode_rewards'][-50:])
            cbr_violation_rate = self.metrics['cbr_violations'] / len(self.metrics['episode_rewards'])
            action_smoothness = np.mean(self.metrics['power_changes'][-50:] + self.metrics['beacon_changes'][-50:])
            
            exploration_info = ""
            if self.metrics['exploration_metrics']:
                latest_exp = self.metrics['exploration_metrics'][-1]
                exploration_info = (f", Power Std: {latest_exp['power_std']:.3f}, "
                                  f"Action Mag: {latest_exp['action_magnitude']:.3f}")
            
            if self.step_count % 500 == 0:  # Reduced logging frequency
                log_message(
                    f"Vehicle {self.node_id} Step {self.step_count} - "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"CBR Violations: {cbr_violation_rate:.1%}, "
                    f"Action Smoothness: {action_smoothness:.3f}, "
                    f"Exploration Factor: {self.mac_agent.exploration_factor:.3f}"
                    f"{exploration_info}"
                )
            
            if self.writer:
                try:
                    self.writer.add_scalar('Metrics/Avg_Reward', avg_reward, self.step_count)
                    self.writer.add_scalar('Metrics/CBR_Violation_Rate', cbr_violation_rate, self.step_count)
                    self.writer.add_scalar('Metrics/Action_Smoothness', action_smoothness, self.step_count)
                except Exception as e:
                    logger.warning(f"TensorBoard metrics logging failed: {e}")
            
        except Exception as e:
            logger.error(f"Enhanced metrics logging failed: {e}")

    def apply_actions(self, actions: Dict[str, Any]) -> Dict[str, float]:
        """Apply actions and update internal state"""
        try:
            power_delta = actions.get('power_delta', 0.0)
            beacon_delta = actions.get('beacon_delta', 0.0)
            mcs_delta = actions.get('mcs_delta', 0)
            
            new_power = max(system_config.power_min, min(system_config.power_max, 
                           self.current_power + power_delta))
            new_beacon_rate = max(system_config.beacon_rate_min, min(system_config.beacon_rate_max, 
                                 self.current_beacon_rate + beacon_delta))
            new_mcs = max(system_config.mcs_min, min(system_config.mcs_max, 
                         self.current_mcs + mcs_delta))
            
            self.current_power = new_power
            self.current_beacon_rate = new_beacon_rate
            self.current_mcs = new_mcs
            
            return {
                'power': float(new_power),
                'beacon_rate': float(new_beacon_rate), 
                'mcs': int(new_mcs)
            }
            
        except Exception as e:
            logger.error(f"Error applying actions for {self.node_id}: {e}")
            return {
                'power': float(self.current_power),
                'beacon_rate': float(self.current_beacon_rate),
                'mcs': int(self.current_mcs)
            }

    def save_models(self, reason: str = "manual"):
        """FIXED: Save shared models (not per-vehicle) since all vehicles use the same neural networks"""
        if not self.training_mode:
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f"{system_config.model_save_dir}/shared_models_{reason}_{timestamp}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save shared MAC agent (used by all vehicles)
            torch.save({
                'actor_state_dict': self.mac_agent.actor.state_dict(),
                'critic1_state_dict': self.mac_agent.critic1.state_dict(),
                'critic2_state_dict': self.mac_agent.critic2.state_dict(),
                'feature_extractor_state_dict': self.mac_agent.feature_extractor.state_dict(),
                'log_alpha': self.mac_agent.log_alpha,
                'exploration_factor': self.mac_agent.exploration_factor,
                'training_step': self.mac_agent.training_step,
                'timestamp': timestamp,
            }, f"{model_dir}/shared_mac_agent.pth")
            
            # Save shared PHY agent (used by all vehicles)
            torch.save({
                'actor_state_dict': self.phy_agent.actor.state_dict(),
                'critic1_state_dict': self.phy_agent.critic1.state_dict(),
                'critic2_state_dict': self.phy_agent.critic2.state_dict(),
                'feature_extractor_state_dict': self.phy_agent.feature_extractor.state_dict(),
                'log_alpha': self.phy_agent.log_alpha,
                'exploration_factor': self.phy_agent.exploration_factor,
                'training_step': self.phy_agent.training_step,
                'timestamp': timestamp,
            }, f"{model_dir}/shared_phy_agent.pth")
            
            # Save vehicle-specific parameters (individual vehicle states)
            with open(f"{model_dir}/vehicle_{self.node_id}_params.json", 'w') as f:
                json.dump({
                    'vehicle_id': self.node_id,
                    'power': self.current_power,
                    'beacon_rate': self.current_beacon_rate,
                    'mcs': self.current_mcs,
                    'timestamp': timestamp,
                }, f)
            
            logger.info(f"Saved SHARED models accessible to all vehicles (triggered by {self.node_id})")
            return True
            
        except Exception as e:
            log_message(f"Failed to save shared models via vehicle {self.node_id}: {str(e)}", "ERROR")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.writer:
                self.writer.close()
        except Exception as e:
            logger.warning(f"Error during cleanup for {self.node_id}: {e}")

# ==============================
# FIXED: Simplified Server with AUTO MODEL LOADING (No Episode Management, Request-Based Only)
# ==============================

class SimplifiedDecentralizedRLServer:
    """FIXED: Simplified server with Unicode-safe logging"""
    
    def __init__(self, host: str, port: int, training_mode: bool = True, 
                 timeout_minutes: int = None):
        self.host = host
        self.port = port
        self.training_mode = training_mode
        self.running = True
        
        # FIXED: Request-based tracking instead of episode-based
        self.request_count = 0
        
        # Auto-save management (kept but modified)
        timeout = timeout_minutes or system_config.auto_save_timeout
        self.auto_save_manager = AutoSaveManager(timeout)
        self.auto_save_manager.set_save_callback(self._auto_save_callback)
        
        # FIXED: Simplified Excel reporting
        if ENABLE_EXCEL_REPORTING:
            self.excel_reporter = SimplifiedExcelReporter()
        else:
            self.excel_reporter = None
            
        # CENTRALIZED LEARNING: Create shared agents that all vehicles will use
        self.shared_feature_extractor = ExploratorySharedFeatureExtractor(state_dim=3)
        self.shared_mac_agent = MACAgent(None, training_mode)
        self.shared_phy_agent = PHYAgent(None, training_mode)
        
        # FIXED: Create centralized training manager
        self.centralized_trainer = CentralizedTrainingManager(
            self.shared_mac_agent, self.shared_phy_agent
        )
        
        logger.info("FIXED CENTRALIZED LEARNING ARCHITECTURE:")
        logger.info("  - ONE shared MAC agent neural network for all vehicles")
        logger.info("  - ONE shared PHY agent neural network for all vehicles") 
        logger.info("  - CENTRALIZED training manager prevents gradient conflicts")
        logger.info("  - When any vehicle learns, ALL vehicles immediately benefit")
        logger.info("  - Collective intelligence: vehicles learn from each other's experiences")
        
        # NEW: AUTO MODEL LOADING FOR PRODUCTION MODE
        if not training_mode:  # Production mode
            logger.info("= PRODUCTION MODE: Attempting to load latest trained model...")
            latest_model_dir = find_latest_model_directory()
            if latest_model_dir:
                success = load_shared_models_from_directory(
                    self.shared_mac_agent, self.shared_phy_agent, latest_model_dir
                )
                if success:
                    logger.info("SUCCESS: Production mode successfully loaded pre-trained models!")
                    logger.info(f"Model source: {latest_model_dir}")
                    logger.info("All vehicles will now use the trained neural networks")
                else:
                    logger.error("ERROR: Production mode failed to load models, using random weights")
                    logger.warning("This may result in poor performance")
            else:
                logger.warning("WARNING: Production mode - No trained models found!")
                logger.warning("Please ensure you have run training mode first")
                logger.warning("Starting with randomly initialized networks")
        else:
            logger.info("TRAINING MODE: Starting with fresh neural networks")  # FIXED: ASCII-safe
            logger.info("Models will be saved periodically and can be used in production mode")
        
        # Vehicle tracking
        self.vehicle_nodes: Dict[str, MaxExplorativeVehicleNode] = {}
        
        # FIXED: Request-based intervals instead of episode-based
        self.last_report_request = 0
        self.last_save_request = 0
        
        # Setup server socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"FIXED Simplified Server Configuration:")
        logger.info(f"  - Host: {host}:{port}")
        logger.info(f"  - Mode: {'TRAINING' if training_mode else 'PRODUCTION'}")
        logger.info(f"  - NO episode limits - responds until simulation stops")
        logger.info(f"  - Request-based intervals: Save every {system_config.model_save_interval}, Report every {system_config.report_interval}")
        logger.info(f"  - Auto-save timeout: {timeout} minutes")
        logger.info(f"  - TensorBoard: {'Enabled' if ENABLE_TENSORBOARD else 'Disabled'}")
        logger.info(f"  - Excel reporting: {'Enabled' if ENABLE_EXCEL_REPORTING else 'Disabled'}")
        logger.info(f"  - Auto model loading: {'Enabled' if not training_mode else 'Disabled (Training Mode)'}")
    
    def start(self):
        """Start the server"""
        logger.info(f"Starting FIXED Simplified Dual-Agent SAC Server...")
        
        try:
            while self.running:
                try:
                    self.server.settimeout(1.0)
                    conn, addr = self.server.accept()
                    logger.info(f"Connection established with {addr}")
                    
                    client_thread = threading.Thread(
                        target=self._handle_client_enhanced,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except socket.error as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self._shutdown()
    
    def _handle_client_enhanced(self, conn: socket.socket, addr: Tuple[str, int]):
        """Enhanced client handler WITHOUT episode management"""
        buffer = b""
        conn.settimeout(10.0)
        client_start_time = time.time()
        
        try:
            while self.running:
                try:
                    data = conn.recv(65536)
                    if not data:
                        logger.info(f"Client {addr} disconnected")
                        break
                    
                    buffer += data
                    
                    while len(buffer) >= 4:
                        msg_length = int.from_bytes(buffer[:4], byteorder='little', signed=False)
                        
                        if len(buffer) < 4 + msg_length:
                            break
                        
                        message_bytes = buffer[4:4 + msg_length]
                        buffer = buffer[4 + msg_length:]
                        
                        try:
                            message = json.loads(message_bytes.decode('utf-8', errors='strict'))
                            response = self._process_batch_fixed(message, addr)
                            
                            response_data = json.dumps(response, ensure_ascii=False).encode('utf-8')
                            response_header = len(response_data).to_bytes(4, byteorder='little')
                            conn.sendall(response_header + response_data)
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error from {addr}: {e}")
                            self._send_error_response(conn, "JSON decode error")
                        except Exception as e:
                            logger.error(f"Message processing error from {addr}: {e}")
                            self._send_error_response(conn, "Processing error")
                
                except socket.timeout:
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    logger.info(f"Client {addr} connection lost")
                    break
                except Exception as e:
                    logger.error(f"Client handling error for {addr}: {e}")
                    break
                    
        finally:
            try:
                conn.close()
            except:
                pass
            duration = time.time() - client_start_time
            logger.info(f"Client {addr} handler terminated. Duration: {duration:.2f}s")
    
    def _process_batch_fixed(self, batch_data: Dict[str, Any], addr: Tuple[str, int]) -> Dict[str, Any]:
        """FIXED: Request-based batch processing WITHOUT episode management - WITH DETAILED LOGGING"""
        # Update activity
        self.auto_save_manager.update_activity()
        
        # FIXED: Increment request count instead of episodes
        self.request_count += 1
        
        logger.info(f"="*80)
        logger.info(f"[REQUEST {self.request_count}] Processing batch from {addr}")
        logger.info(f"Vehicles in batch: {len(batch_data)}")
        logger.info(f"Vehicle IDs: {list(batch_data.keys())}")
        logger.info(f"="*80)
        
        batch_response = {
            "vehicles": {}, 
            "timestamp": time.time(),
            "request_count": self.request_count
        }
        
        # Track batch statistics
        batch_rewards = []
        batch_explorations = []
        
        for vehicle_id, vehicle_data in batch_data.items():
            try:
                state = [
                    float(vehicle_data.get('CBR', 0)),
                    float(vehicle_data.get('SNR', 0)),
                    float(vehicle_data.get('neighbors', 0))
                ]
                
                current_params = {
                    'transmissionPower': float(vehicle_data.get('transmissionPower', 20)),
                    'beaconRate': float(vehicle_data.get('beaconRate', 10)),
                    'MCS': int(vehicle_data.get('MCS', 0))
                }
                
                logger.info(f"[VEHICLE {vehicle_id}] Input Data:")
                logger.info(f"  State: CBR={state[0]:.3f}, SNR={state[1]:.1f}dB, Neighbors={int(state[2])}")
                logger.info(f"  Current: Power={current_params['transmissionPower']:.1f}dBm, Beacon={current_params['beaconRate']:.1f}Hz, MCS={current_params['MCS']}")
                
                # FIXED CENTRALIZED LEARNING: Initialize vehicle with shared agents for collective learning
                if vehicle_id not in self.vehicle_nodes:
                    self.vehicle_nodes[vehicle_id] = MaxExplorativeVehicleNode(
                        vehicle_id, self.shared_mac_agent, self.shared_phy_agent, 
                        self.centralized_trainer, self.training_mode
                    )
                    logger.info(f"[NEW VEHICLE] Created {vehicle_id} with shared neural networks - "
                               f"Fleet size: {len(self.vehicle_nodes)} vehicles")
                
                # Get actions from RL
                vehicle = self.vehicle_nodes[vehicle_id]
                actions = vehicle.get_coordinated_actions(state, current_params)
                new_params = vehicle.apply_actions(actions)
                
                logger.info(f"[VEHICLE {vehicle_id}] RL Actions:")
                logger.info(f"  MAC Actions: Power_delta={actions['power_delta']:.2f}dBm, Beacon_delta={actions['beacon_delta']:.2f}Hz")
                logger.info(f"  PHY Actions: MCS_delta={actions['mcs_delta']:.2f}")
                logger.info(f"  New Settings: Power={new_params['power']:.1f}dBm, Beacon={new_params['beacon_rate']:.1f}Hz, MCS={new_params['mcs']}")
                logger.info(f"  Reward: {actions['joint_reward']:.3f}, Exploration: {actions['exploration_factor']:.3f}")
                
                # Prepare response
                response_data = {
                    'transmissionPower': new_params['power'],
                    'beaconRate': new_params['beacon_rate'],
                    'MCS': new_params['mcs'],
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
                
                batch_response["vehicles"][vehicle_id] = response_data
                
                # Enhanced training (only in training mode)
                if self.training_mode:
                    next_state = self._simulate_next_state(state)
                    reward = vehicle._calculate_joint_reward(state, new_params)
                    
                    batch_response["vehicles"][vehicle_id]['training'] = {
                        'reward': float(reward),
                        'exploration_factor': actions.get('exploration_factor', 1.0),
                        'power_delta': actions['power_delta'],
                        'beacon_delta': actions['beacon_delta'],
                        'mcs_delta': actions['mcs_delta']
                    }
                    
                    actions_combined = torch.FloatTensor([
                        actions['power_delta'],
                        actions['beacon_delta'],
                        actions['mcs_delta']
                    ])
                    
                    vehicle.store_experience(state, actions_combined, reward, next_state, False)
                    
                    batch_rewards.append(reward)
                    batch_explorations.append(actions.get('exploration_factor', 1.0))
                    
                    logger.info(f"[TRAINING] Vehicle {vehicle_id}: Reward={reward:.3f}, Next_CBR={next_state[0]:.3f}")
                    
            except Exception as e:
                error_msg = f"Error processing vehicle {vehicle_id}: {str(e)}"
                logger.error(f"[ERROR] {error_msg}")
                batch_response["vehicles"][vehicle_id] = {
                    'status': 'error',
                    'error': error_msg,
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
        
        # Batch summary logging
        if batch_rewards:
            avg_reward = np.mean(batch_rewards)
            avg_exploration = np.mean(batch_explorations)
            logger.info(f"[BATCH SUMMARY] Request {self.request_count}:")
            logger.info(f"  Vehicles processed: {len(batch_data)}")
            logger.info(f"  Average reward: {avg_reward:.3f}")
            logger.info(f"  Average exploration: {avg_exploration:.3f}")
            logger.info(f"  Training updates in progress via centralized manager...")
        
        # FIXED: Add data to simplified Excel reporter
        if self.excel_reporter:
            self.excel_reporter.add_batch_data(batch_data, batch_response)
        
        # FIXED: Request-based periodic reporting and saving (only in training mode)
        if self.training_mode:
            if (self.request_count - self.last_report_request >= system_config.report_interval):
                self._generate_periodic_report()
                self.last_report_request = self.request_count
            
            if (self.request_count - self.last_save_request >= system_config.model_save_interval):
                self._save_all_models("periodic")
                self.last_save_request = self.request_count
        
        logger.info(f"[RESPONSE] Sending optimized parameters back to simulation for {len(batch_response['vehicles'])} vehicles")
        logger.info(f"="*80)
        
        return batch_response
    
    def _generate_periodic_report(self):
        """Generate periodic performance reports with detailed logging"""
        try:
            logger.info(f"[PERIODIC REPORT] Request {self.request_count}")
            logger.info(f"="*60)
            logger.info(f"Fleet Status:")
            logger.info(f"  Total vehicles in fleet: {len(self.vehicle_nodes)}")
            logger.info(f"  Active vehicle IDs: {list(self.vehicle_nodes.keys())}")
            
            # Shared model status
            logger.info(f"Shared Learning Status:")
            logger.info(f"  MAC Agent buffer: {len(self.shared_mac_agent.replay_buffer)} experiences")
            logger.info(f"  PHY Agent buffer: {len(self.shared_phy_agent.replay_buffer)} experiences")
            logger.info(f"  MAC Training steps: {self.shared_mac_agent.training_step}")
            logger.info(f"  PHY Training steps: {self.shared_phy_agent.training_step}")
            logger.info(f"  Current exploration factor: {self.shared_mac_agent.exploration_factor:.4f}")
            
            # Vehicle performance summary
            if self.vehicle_nodes:
                logger.info(f"Vehicle Performance Summary:")
                for vehicle_id, vehicle in list(self.vehicle_nodes.items())[:5]:  # Show first 5 vehicles
                    logger.info(f"  {vehicle_id}: Power={vehicle.current_power:.1f}dBm, "
                               f"Beacon={vehicle.current_beacon_rate:.1f}Hz, MCS={vehicle.current_mcs}, "
                               f"Steps={vehicle.step_count}")
                
                if len(self.vehicle_nodes) > 5:
                    logger.info(f"  ... and {len(self.vehicle_nodes)-5} more vehicles")
            
            # Excel reporter status
            if self.excel_reporter:
                logger.info(f"Excel Reporter Status:")
                logger.info(f"  Data points collected: {self.excel_reporter.vehicle_count}")
                logger.info(f"  Unique vehicles tracked: {len(self.excel_reporter.unique_vehicles)}")
                logger.info(f"  Training updates logged: {self.excel_reporter.training_updates}")
                
                if self.excel_reporter.total_rewards:
                    avg_reward = np.mean(self.excel_reporter.total_rewards[-100:])  # Last 100 rewards
                    logger.info(f"  Recent average reward: {avg_reward:.3f}")
            
            logger.info(f"="*60)
                        
        except Exception as e:
            logger.error(f"Error generating periodic report: {e}")
    
    def _auto_save_callback(self, reason: str):
        """Callback for auto-save manager"""
        try:
            logger.info(f"Auto-save triggered by {reason} at request {self.request_count}")
            
            # Save models (only in training mode)
            if self.training_mode:
                self._save_all_models(reason)
            
            # Generate Excel report
            if self.excel_reporter:
                filename = self.excel_reporter.generate_final_report(f"autosave_{reason}_req{self.request_count}")
                if filename:
                    logger.info(f"Auto-save Excel report: {filename}")
            
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
    
    def _save_all_models(self, reason: str):
        """FIXED: Save shared models once (not per vehicle) since all vehicles use same neural networks"""
        try:
            if not self.vehicle_nodes:
                logger.warning("No vehicles to save models for")
                return
            
            # Save shared models only once using any vehicle as a proxy
            # All vehicles share the same neural networks, so saving once is sufficient
            first_vehicle = list(self.vehicle_nodes.values())[0]
            success = first_vehicle.save_models(reason)
            
            if success:
                logger.info(f"Saved SHARED models for ALL vehicles ({reason}) - "
                           f"Total vehicles using shared model: {len(self.vehicle_nodes)}")
                
                # Save individual vehicle parameters (current power/beacon/mcs states)
                self._save_vehicle_parameters(reason)
            else:
                logger.error(f"Failed to save shared models ({reason})")
            
        except Exception as e:
            logger.error(f"Error saving shared models: {e}")
    
    def _save_vehicle_parameters(self, reason: str):
        """Save individual vehicle parameters (states) while models are shared"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_dir = f"{system_config.model_save_dir}/vehicle_parameters_{reason}_{timestamp}"
            os.makedirs(params_dir, exist_ok=True)
            
            all_vehicle_params = {}
            for vehicle_id, vehicle in self.vehicle_nodes.items():
                all_vehicle_params[vehicle_id] = {
                    'power': vehicle.current_power,
                    'beacon_rate': vehicle.current_beacon_rate,
                    'mcs': vehicle.current_mcs,
                    'request_count': self.request_count,
                    'step_count': vehicle.step_count
                }
            
            # Save all vehicle parameters in one file
            with open(f"{params_dir}/all_vehicle_parameters.json", 'w') as f:
                json.dump({
                    'request_count': self.request_count,
                    'timestamp': timestamp,
                    'reason': reason,
                    'vehicle_count': len(self.vehicle_nodes),
                    'vehicles': all_vehicle_params
                }, f, indent=2)
            
            logger.info(f"Saved parameters for {len(self.vehicle_nodes)} vehicles ({reason})")
            
        except Exception as e:
            logger.error(f"Error saving vehicle parameters: {e}")
    
    def _final_save_and_report(self):
        """Final save and report generation"""
        try:
            logger.info("Generating final reports and saving models...")
            
            # Save all models (only in training mode)
            if self.training_mode:
                self._save_all_models("final")
            
            # Generate final Excel report
            if self.excel_reporter:
                filename = self.excel_reporter.generate_final_report(f"final_report_req{self.request_count}")
                if filename:
                    logger.info(f"Final Excel report: {filename}")
            
            # Display final statistics
            if self.excel_reporter:
                elapsed_time = time.time() - self.excel_reporter.start_time
                logger.info("FINAL SIMULATION STATISTICS:")
                logger.info("="*60)
                logger.info(f"Total Requests: {self.request_count}")
                logger.info(f"Unique Vehicles: {len(self.excel_reporter.unique_vehicles)}")
                logger.info(f"Total Vehicle Interactions: {self.excel_reporter.vehicle_count}")
                logger.info(f"Duration: {elapsed_time/60:.1f} minutes")
                logger.info(f"Requests per Minute: {(self.request_count/(elapsed_time/60)):.1f}")
                
                if self.excel_reporter.total_rewards:
                    logger.info(f"Average Reward: {np.mean(self.excel_reporter.total_rewards):.3f}")
                    logger.info(f"Final Exploration: {self.excel_reporter.total_exploration_factors[-1]:.3f}")
                
                logger.info(f"Training Updates: {self.excel_reporter.training_updates}")
                logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error in final save and report: {e}")
    
    def _send_error_response(self, conn: socket.socket, error_msg: str):
        """Send error response to client"""
        try:
            error_response = json.dumps({
                'status': 'error',
                'error': error_msg,
                'timestamp': time.time()
            }).encode('utf-8')
            
            header = len(error_response).to_bytes(4, byteorder='little')
            conn.sendall(header + error_response)
        except:
            pass

    def _simulate_next_state(self, current_state: List[float]) -> List[float]:
        """Simple environment transition model for training"""
        try:
            cbr, snr, neighbors = current_state
            
            # Simulate CBR change
            new_cbr = cbr + random.uniform(-0.05, 0.05)
            new_cbr = max(0.0, min(1.0, new_cbr))
            
            # Simulate SNR change
            new_snr = snr + random.uniform(-2.0, 2.0)
            new_snr = max(0.0, new_snr)
            
            # Simulate neighbor count change
            neighbor_change = random.choice([-1, 0, 1])
            new_neighbors = max(0, neighbors + neighbor_change)
            
            return [new_cbr, new_snr, new_neighbors]
        except Exception as e:
            logger.error(f"Error in state simulation: {e}")
            return current_state

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self._shutdown()

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating server shutdown...")
        self.running = False
        
        try:
            # Stop auto-save manager
            self.auto_save_manager.stop()
            
            # Final save and report
            if self.training_mode:
                self._final_save_and_report()
            
            # Cleanup vehicle nodes
            for vehicle in self.vehicle_nodes.values():
                vehicle.cleanup()
            
            # Close server socket
            try:
                self.server.close()
            except:
                pass
            
            logger.info("Server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def stop(self):
        """External stop method"""
        self._shutdown()

# Legacy compatibility wrappers
class DecentralizedRLServer(SimplifiedDecentralizedRLServer):
    """Backward compatibility wrapper"""
    pass

class EnhancedDecentralizedRLServer(SimplifiedDecentralizedRLServer):
    """Backward compatibility wrapper"""
    pass

class VehicleNode(MaxExplorativeVehicleNode):
    """Backward compatibility wrapper"""
    pass

# Also support the old class name
class MaxExplorativeDecentralizedRLServer(SimplifiedDecentralizedRLServer):
    """Support old class name for backwards compatibility"""
    pass

# ==============================
# SIMPLIFIED Main Execution (No Command Line Arguments)
# ==============================

def print_startup_banner(training_mode: bool):
    """Print startup configuration banner with ASCII-safe characters"""
    logger.info("=" * 80)
    logger.info("COMPLETE REVISED Dual-Agent SAC with Auto Model Loading")
    logger.info("=" * 80)
    logger.info(f"SCRIPT MODE: {SCRIPT_MODE.upper()}")
    logger.info(f"Configuration:")
    logger.info(f"  Server: {system_config.host}:{system_config.port}")
    logger.info(f"  Mode: {'TRAINING' if training_mode else 'PRODUCTION'}")
    logger.info(f"  Auto-save timeout: {system_config.auto_save_timeout} minutes")
    logger.info(f"  Request-based intervals:")
    logger.info(f"    - Model save every {system_config.model_save_interval} requests")
    logger.info(f"    - Report every {system_config.report_interval} requests")
    
    logger.info(f"Features:")
    logger.info(f"  TensorBoard: {'Enabled' if ENABLE_TENSORBOARD else 'Disabled'}")
    logger.info(f"  Excel reporting: {'Enabled' if ENABLE_EXCEL_REPORTING else 'Disabled'}")
    
    logger.info(f"COMPLETE REVISION Features:")
    logger.info(f"  - NO command line arguments (mode set in script)")
    logger.info(f"  - AUTO model loading in production mode")
    logger.info(f"  - NO episode management (responds until simulation stops)")
    logger.info(f"  - SIMPLIFIED Excel report (single document, 3 sheets)")
    logger.info(f"  - CENTRALIZED model saving (one shared model, not per vehicle)")
    logger.info(f"  - FIXED gradient conflicts with centralized training manager")
    logger.info(f"  - Request-based intervals instead of episode-based")
    
    if training_mode:
        logger.info(f"TRAINING MODE:")
        logger.info(f"  - Fresh neural networks initialized")
        logger.info(f"  - Models will be saved automatically")
        logger.info(f"  - Training and exploration enabled")
    else:
        logger.info(f"PRODUCTION MODE:")
        logger.info(f"  - Auto-loading latest trained model")
        logger.info(f"  - No training or model saving")
        logger.info(f"  - Deterministic actions for deployment")
    
    logger.info(f"Action bounds:")
    logger.info(f"  Power action bounds: +/- 15.0 dBm")  # FIXED: ASCII-safe characters
    logger.info(f"  Beacon action bounds: +/- 10.0 Hz")   # FIXED: ASCII-safe characters
    logger.info(f"  MCS action bounds: +/- 7.5 levels")   # FIXED: ASCII-safe characters
    
    logger.info(f"Enhanced safety limits:")
    logger.info(f"  Power delta limits: +/- 30.0 dBm (full range coverage)")   # FIXED: ASCII-safe
    logger.info(f"  Beacon delta limits: +/- 20.0 Hz (full range coverage)")  # FIXED: ASCII-safe
    logger.info(f"  MCS delta limits: +/- 11.0 levels (full range coverage)") # FIXED: ASCII-safe
    
    logger.info(f"CENTRALIZED Learning Architecture:")
    logger.info(f"  Model Type: CENTRALIZED LEARNING (single shared model)")
    logger.info(f"  Training Manager: Prevents gradient computation conflicts")
    logger.info(f"  Knowledge Sharing: When one vehicle learns, ALL vehicles benefit immediately")
    logger.info(f"  Collective Intelligence: Fleet gets smarter as a whole")
    logger.info(f"  Model Storage: One set of neural networks saved (not per vehicle)")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        # Determine training mode from script configuration
        training_mode = (SCRIPT_MODE.lower() == "training")
        
        if SCRIPT_MODE.lower() not in ["training", "production"]:
            logger.error(f"Invalid SCRIPT_MODE: {SCRIPT_MODE}")
            logger.error("Please set SCRIPT_MODE to either 'training' or 'production' at the top of the script")
            sys.exit(1)
        
        # Set random seed for reproducibility
        set_seed(0)
        logger.info(f"Random seed set to: 0")
        
        # Print startup banner
        print_startup_banner(training_mode)
        
        # Create and start server
        server = SimplifiedDecentralizedRLServer(
            host=system_config.host,
            port=system_config.port,
            training_mode=training_mode,
            timeout_minutes=system_config.auto_save_timeout
        )
        
        logger.info("Starting COMPLETE REVISED Dual-Agent SAC server...")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("Server will respond to simulation requests until simulation naturally stops")
        
        if training_mode:
            logger.info("TRAINING MODE: Server will learn and save models automatically")
        else:
            logger.info("PRODUCTION MODE: Server will use pre-trained models for optimal performance")
        
        # Start server
        server.start()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")
