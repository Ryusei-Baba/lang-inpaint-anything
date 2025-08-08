import yaml
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

class ConfigLoader:
    """設定ファイルとコマンドライン引数を統合管理するクラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        設定ローダーを初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """設定ファイルを読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                print(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                print(f"設定ファイルが見つかりません: {self.config_path}")
                print("デフォルト設定を使用します")
                self.config = self._get_default_config()
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            print("デフォルト設定を使用します")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
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
        ドット記法でネストした設定値を取得
        
        Args:
            key_path: ドット区切りのキーパス（例: "models.sam.model_type"）
            default: デフォルト値
            
        Returns:
            設定値
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
        ドット記法でネストした設定値を設定
        
        Args:
            key_path: ドット区切りのキーパス
            value: 設定値
        """
        keys = key_path.split('.')
        config = self.config
        
        # 最後のキー以外は辞書を作成/取得
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        # 最後のキーに値を設定
        config[keys[-1]] = value
    
    def setup_argparse(self) -> argparse.ArgumentParser:
        """
        設定ファイルの内容に基づいてargparseを設定
        
        Returns:
            設定済みのArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="Text Remove Anything - テキストベースのオブジェクト除去ツール",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
設定例:
  python text_remove_anything.py --input_img image.jpg --text_prompt "dog"
  python text_remove_anything.py --config custom_config.yaml --input_img image.jpg --text_prompt "person"
  python text_remove_anything.py --input_img image.jpg --text_prompt "car" --interactive
            """
        )
        
        # 設定ファイル指定
        parser.add_argument(
            "--config", type=str, default="config.yaml",
            help="設定ファイルのパス (デフォルト: config.yaml)"
        )
        
        # 必須引数
        parser.add_argument(
            "--input_img", type=str, required=False,
            help="入力画像のパス"
        )
        parser.add_argument(
            "--text_prompt", type=str, required=False,
            default=self.get("processing.text_prompt"),
            help="除去したいオブジェクトのテキスト記述"
        )
        
        # バッチ処理用の入力ディレクトリ
        parser.add_argument(
            "--input_dir", type=str,
            default=self.get("processing.input_dir"),
            help="入力画像ディレクトリ（バッチ処理用）"
        )
        
        # オプション引数（設定ファイルでデフォルト値を設定可能）
        parser.add_argument(
            "--output_dir", type=str,
            default=self.get("processing.output_dir"),
            help=f"出力ディレクトリ (デフォルト: {self.get('processing.output_dir')})"
        )
        
        # 検出パラメータ
        parser.add_argument(
            "--box_threshold", type=float,
            default=self.get("processing.box_threshold"),
            help=f"境界ボックス検出の信頼度閾値 (デフォルト: {self.get('processing.box_threshold')})"
        )
        parser.add_argument(
            "--text_threshold", type=float,
            default=self.get("processing.text_threshold"),
            help=f"テキストマッチングの信頼度閾値 (デフォルト: {self.get('processing.text_threshold')})"
        )
        
        # マスク処理
        parser.add_argument(
            "--dilate_kernel_size", type=int,
            default=self.get("processing.dilate_kernel_size"),
            help=f"マスク拡張のカーネルサイズ (デフォルト: {self.get('processing.dilate_kernel_size')})"
        )
        
        # モデル設定
        parser.add_argument(
            "--sam_model_type", type=str,
            default=self.get("models.sam.model_type"),
            choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
            help=f"SAMモデルタイプ (デフォルト: {self.get('models.sam.model_type')})"
        )
        parser.add_argument(
            "--sam_ckpt", type=str,
            default=self.get("models.sam.checkpoint_path"),
            help="SAMチェックポイントのパス"
        )
        parser.add_argument(
            "--lama_config", type=str,
            default=self.get("models.lama.config_path"),
            help="LaMa設定ファイルのパス"
        )
        parser.add_argument(
            "--lama_ckpt", type=str,
            default=self.get("models.lama.checkpoint_path"),
            help="LaMaチェックポイントのパス"
        )
        parser.add_argument(
            "--grounding_dino_config", type=str,
            default=self.get("models.grounding_dino.config_path"),
            help="GroundingDINO設定ファイルのパス"
        )
        parser.add_argument(
            "--grounding_dino_ckpt", type=str,
            default=self.get("models.grounding_dino.checkpoint_path"),
            help="GroundingDINOチェックポイントのパス"
        )
        
        # メモリ最適化
        parser.add_argument(
            "--use_memory_efficient_sam", action="store_true",
            default=self.get("memory_optimization.use_memory_efficient_sam"),
            help="メモリ効率的SAM処理を使用"
        )
        parser.add_argument(
            "--max_memory_mb", type=int,
            default=self.get("memory_optimization.max_memory_mb"),
            help=f"最大GPU メモリ使用量(MB) (デフォルト: {self.get('memory_optimization.max_memory_mb')})"
        )
        parser.add_argument(
            "--monitor_gpu", action="store_true",
            default=self.get("memory_optimization.monitor_gpu"),
            help="GPU監視を有効化"
        )
        parser.add_argument(
            "--save_monitoring_stats", type=str,
            default=self.get("memory_optimization.save_monitoring_stats"),
            help="監視統計の保存パス"
        )
        
        # オプションフラグ
        parser.add_argument(
            "--interactive", action="store_true",
            default=self.get("processing.interactive"),
            help="インタラクティブモードを有効化"
        )
        parser.add_argument(
            "--combine_masks", action="store_true",
            default=self.get("processing.combine_masks"),
            help="複数マスクを結合"
        )
        parser.add_argument(
            "--save_intermediate", action="store_true",
            default=self.get("processing.save_intermediate"),
            help="中間結果を保存"
        )
        
        # デバイス設定
        parser.add_argument(
            "--device", type=str,
            default=self.get("device.type"),
            choices=['cuda', 'cpu', 'auto'],
            help=f"使用デバイス (デフォルト: {self.get('device.type')})"
        )
        
        return parser
    
    def merge_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        コマンドライン引数と設定ファイルの内容をマージ
        コマンドライン引数が優先される
        
        Args:
            args: パースされたコマンドライン引数
            
        Returns:
            マージされた設定
        """
        # 設定ファイルが指定されている場合は再読み込み
        if hasattr(args, 'config') and args.config != self.config_path:
            self.config_path = args.config
            self.load_config()
        
        # デバイス設定の処理
        if args.device == 'auto':
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return args
    
    def save_config(self, output_path: str):
        """
        現在の設定をファイルに保存
        
        Args:
            output_path: 保存先パス
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, 
                          allow_unicode=True, sort_keys=False)
        print(f"設定ファイルを保存しました: {output_path}")
    
    def print_config(self):
        """現在の設定を表示"""
        print("\n=== 現在の設定 ===")
        print(yaml.dump(self.config, default_flow_style=False, 
                       allow_unicode=True, sort_keys=False))
    
    def validate_paths(self) -> bool:
        """
        設定されたパスが存在するかチェック
        
        Returns:
            すべてのパスが存在する場合True
        """
        required_paths = {
            "SAMチェックポイント": self.get("models.sam.checkpoint_path"),
            "LaMa設定": self.get("models.lama.config_path"),
            "LaMaチェックポイント": self.get("models.lama.checkpoint_path"),
            "GroundingDINO設定": self.get("models.grounding_dino.config_path"),
            "GroundingDINOチェックポイント": self.get("models.grounding_dino.checkpoint_path")
        }
        
        missing_paths = []
        for name, path in required_paths.items():
            if path and not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("\n⚠️  以下のファイルが見つかりません:")
            for missing in missing_paths:
                print(f"  - {missing}")
            print("\ninstall/download_models.shを実行してモデルをダウンロードしてください")
            return False
        
        return True

def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    設定ローダーのファクトリ関数
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        ConfigLoaderインスタンス
    """
    return ConfigLoader(config_path)