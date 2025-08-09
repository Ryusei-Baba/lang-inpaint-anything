import yaml
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

class ConfigLoader:
    """Class for unified management of configuration files and command line arguments"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                print(f"Configuration file loaded: {self.config_path}")
            else:
                print(f"Configuration file not found: {self.config_path}")
                print("Using default configuration")
                self.config = self._get_default_config()
        except Exception as e:
            print(f"Configuration file loading error: {e}")
            print("Using default configuration")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'models': {
                'sam': {
                    'model_type': 'vit_h',
                    'checkpoint_path': './Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth'
                },
                'lama': {
                    'config_path': './Inpaint-Anything/lama/configs/prediction/default.yaml',
                    'checkpoint_path': './Inpaint-Anything/pretrained_models/big-lama/big-lama'
                },
                'grounding_dino': {
                    'config_path': './models/grounding_dino/GroundingDINO_SwinT_OGC.py',
                    'checkpoint_path': './models/grounding_dino/groundingdino_swint_ogc.pth'
                }
            },
            'processing': {
                'box_threshold': 0.35,
                'text_threshold': 0.25,
                'dilate_kernel_size': 15,
                'combine_masks': False,
                'output_dir': './output_image',
                'save_intermediate': True,
                'interactive': False
            },
            'memory_optimization': {
                'use_memory_efficient_sam': True,
                'max_memory_mb': 3072,
                'monitor_gpu': True,
                'save_monitoring_stats': './output_image/monitoring_stats.json'
            },
            'device': {
                'type': 'auto'
            },
            'logging': {
                'level': 'INFO',
                'file_path': None
            },
            'batch_processing': {
                'batch_size': 1,
                'num_workers': 1
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path (e.g., "models.sam.model_type")
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set nested configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path
            value: Configuration value
        """
        keys = key_path.split('.')
        config = self.config
        
        # Create/get dictionaries for all keys except the last one
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        # Set value for the last key
        config[keys[-1]] = value
    
    def setup_argparse(self) -> argparse.ArgumentParser:
        """
        Set up argparse based on configuration file contents
        
        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="Text Remove Anything - Text-based Object Removal Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Configuration examples:
  python lang_inpaint_anything.py --input_img image.jpg --text_prompt "dog"
  python lang_inpaint_anything.py --config custom_config.yaml --input_img image.jpg --text_prompt "person"
  python lang_inpaint_anything.py --input_img image.jpg --text_prompt "car" --interactive
            """
        )
        
        # Configuration file specification
        parser.add_argument(
            "--config", type=str, default="config.yaml",
            help="Configuration file path (default: config.yaml)"
        )
        
        # Required arguments
        parser.add_argument(
            "--input_img", type=str, required=False,
            help="Input image path"
        )
        parser.add_argument(
            "--text_prompt", type=str, required=False,
            default=self.get("processing.text_prompt"),
            help="Text description of objects to remove"
        )
        
        # Input directory for batch processing
        parser.add_argument(
            "--input_dir", type=str,
            default=self.get("processing.input_dir"),
            help="Input image directory (for batch processing)"
        )
        
        # Optional arguments (default values can be set in config file)
        parser.add_argument(
            "--output_dir", type=str,
            default=self.get("processing.output_dir"),
            help=f"Output directory (default: {self.get('processing.output_dir')})"
        )
        
        # Detection parameters
        parser.add_argument(
            "--box_threshold", type=float,
            default=self.get("processing.box_threshold"),
            help=f"Bounding box detection confidence threshold (default: {self.get('processing.box_threshold')})"
        )
        parser.add_argument(
            "--text_threshold", type=float,
            default=self.get("processing.text_threshold"),
            help=f"Text matching confidence threshold (default: {self.get('processing.text_threshold')})"
        )
        
        # Mask processing
        parser.add_argument(
            "--dilate_kernel_size", type=int,
            default=self.get("processing.dilate_kernel_size"),
            help=f"Mask dilation kernel size (default: {self.get('processing.dilate_kernel_size')})"
        )
        
        # Model configuration
        parser.add_argument(
            "--sam_model_type", type=str,
            default=self.get("models.sam.model_type"),
            choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
            help=f"SAM model type (default: {self.get('models.sam.model_type')})"
        )
        parser.add_argument(
            "--sam_ckpt", type=str,
            default=self.get("models.sam.checkpoint_path"),
            help="SAM checkpoint path"
        )
        parser.add_argument(
            "--lama_config", type=str,
            default=self.get("models.lama.config_path"),
            help="LaMa configuration file path"
        )
        parser.add_argument(
            "--lama_ckpt", type=str,
            default=self.get("models.lama.checkpoint_path"),
            help="LaMa checkpoint path"
        )
        parser.add_argument(
            "--grounding_dino_config", type=str,
            default=self.get("models.grounding_dino.config_path"),
            help="GroundingDINO configuration file path"
        )
        parser.add_argument(
            "--grounding_dino_ckpt", type=str,
            default=self.get("models.grounding_dino.checkpoint_path"),
            help="GroundingDINO checkpoint path"
        )
        
        # Memory optimization
        parser.add_argument(
            "--use_memory_efficient_sam", action="store_true",
            default=self.get("memory_optimization.use_memory_efficient_sam"),
            help="Use memory-efficient SAM processing"
        )
        parser.add_argument(
            "--max_memory_mb", type=int,
            default=self.get("memory_optimization.max_memory_mb"),
            help=f"Maximum GPU memory usage (MB) (default: {self.get('memory_optimization.max_memory_mb')})"
        )
        parser.add_argument(
            "--monitor_gpu", action="store_true",
            default=self.get("memory_optimization.monitor_gpu"),
            help="Enable GPU monitoring"
        )
        parser.add_argument(
            "--save_monitoring_stats", type=str,
            default=self.get("memory_optimization.save_monitoring_stats"),
            help="Monitoring statistics save path"
        )
        
        # Optional flags
        parser.add_argument(
            "--interactive", action="store_true",
            default=self.get("processing.interactive"),
            help="Enable interactive mode"
        )
        parser.add_argument(
            "--combine_masks", action="store_true",
            default=self.get("processing.combine_masks"),
            help="Combine multiple masks"
        )
        parser.add_argument(
            "--save_intermediate", action="store_true",
            default=self.get("processing.save_intermediate"),
            help="Save intermediate results"
        )
        
        # Device configuration
        parser.add_argument(
            "--device", type=str,
            default=self.get("device.type"),
            choices=['cuda', 'cpu', 'auto'],
            help=f"Device to use (default: {self.get('device.type')})"
        )
        
        return parser
    
    def merge_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Merge command line arguments with configuration file contents
        Command line arguments take precedence
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Merged configuration
        """
        # Reload configuration if a different config file is specified
        if hasattr(args, 'config') and args.config != self.config_path:
            self.config_path = args.config
            self.load_config()
        
        # Process device configuration
        if args.device == 'auto':
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return args
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, 
                          allow_unicode=True, sort_keys=False)
        print(f"Configuration file saved: {output_path}")
    
    def print_config(self):
        """Display current configuration"""
        print("\n=== Current Configuration ===")
        print(yaml.dump(self.config, default_flow_style=False, 
                       allow_unicode=True, sort_keys=False))
    
    def validate_paths(self) -> bool:
        """
        Check if configured paths exist
        
        Returns:
            True if all paths exist
        """
        required_paths = {
            "SAM checkpoint": self.get("models.sam.checkpoint_path"),
            "LaMa config": self.get("models.lama.config_path"),
            "LaMa checkpoint": self.get("models.lama.checkpoint_path"),
            "GroundingDINO config": self.get("models.grounding_dino.config_path"),
            "GroundingDINO checkpoint": self.get("models.grounding_dino.checkpoint_path")
        }
        
        missing_paths = []
        for name, path in required_paths.items():
            if path and not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("\n⚠️  The following files were not found:")
            for missing in missing_paths:
                print(f"  - {missing}")
            print("\nPlease run install/download_models.sh to download the models")
            return False
        
        return True


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Configuration loader factory function
    
    Args:
        config_path: Configuration file path
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)