#!/usr/bin/env python3
"""
テキストベースオブジェクト除去システム - バッチ処理対応版

GroundingDINO + SAM + LaMaを使用して、入力ディレクトリ内の全画像に対して
テキストプロンプトに基づくオブジェクト除去を実行します。

使用方法:
    python text_remove_anything.py --text_prompt "dog"
"""

# 環境変数設定（ライブラリインポート前に設定）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlowログを抑制
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN警告を抑制
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA同期エラーを防ぐ
os.environ['MPLBACKEND'] = 'Agg'  # matplotlib バックエンドをheadlessに設定

import argparse
import sys
import cv2
import torch
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンドに設定
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# パスの設定
sys.path.insert(0, str(Path(__file__).parent / "Inpaint-Anything"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# 必要なモジュールのインポート
try:
    from utils import load_img_to_array, save_array_to_img
    from lama_inpaint import inpaint_img_with_lama
    from text_grounding_detector import TextGroundingDetector
    from sam_processor import create_sam_processor
    from config_loader import ConfigLoader
    IMPORTS_OK = True
except ImportError as e:
    print(f"必要なモジュールのインポートエラー: {e}")
    print("Inpaint-Anythingサブモジュールが正しく設定されているか確認してください。")
    IMPORTS_OK = False


def setup_output_directories(base_dir: str, subdirs: dict) -> dict:
    """
    出力ディレクトリを作成
    
    Args:
        base_dir: ベースディレクトリ
        subdirs: サブディレクトリ設定
        
    Returns:
        作成されたディレクトリのパス辞書
    """
    output_dirs = {}
    base_path = Path(base_dir)
    
    for key, subdir_name in subdirs.items():
        dir_path = base_path / subdir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        output_dirs[key] = str(dir_path)
        
    return output_dirs


def get_input_images(input_dir: str, supported_formats: List[str]) -> List[str]:
    """
    入力ディレクトリから対応画像ファイルを取得
    
    Args:
        input_dir: 入力ディレクトリ
        supported_formats: 対応画像形式のリスト
        
    Returns:
        画像ファイルパスのリスト
    """
    image_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"入力ディレクトリが存在しません: {input_dir}")
        return []
    
    for format_ext in supported_formats:
        pattern = f"*{format_ext}"
        files = glob.glob(str(input_path / pattern))
        image_files.extend(files)
        
        # 大文字小文字の区別
        pattern = f"*{format_ext.upper()}"
        files = glob.glob(str(input_path / pattern))
        image_files.extend(files)
    
    # 重複削除とソート
    image_files = sorted(list(set(image_files)))
    print(f"検出された画像ファイル: {len(image_files)}枚")
    
    return image_files


def save_grounding_dino_result(image: np.ndarray, boxes: torch.Tensor, 
                             phrases: List[str], confidences: torch.Tensor,
                             output_path: str):
    """
    GroundingDINOの検出結果を保存
    
    Args:
        image: 入力画像
        boxes: バウンディングボックス
        phrases: 検出されたフレーズ
        confidences: 信頼度スコア
        output_path: 出力パス
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for box, phrase, confidence in zip(boxes, phrases, confidences):
        x1, y1, x2, y2 = box.cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        # バウンディングボックスを描画
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        
        # ラベルを描画
        plt.text(x1, y1 - 10, f"{phrase}: {confidence:.2f}", 
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.title(f"GroundingDINO Detection Results")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')  # すべての図を確実にクローズ


def process_single_image(image_path: str, text_prompt: str, 
                        config: dict, output_dirs: dict,
                        detector: TextGroundingDetector,
                        sam_processor) -> bool:
    """
    単一画像の処理（バッチ処理用）
    
    Args:
        image_path: 入力画像のパス
        text_prompt: テキストプロンプト
        config: 設定辞書
        output_dirs: 出力ディレクトリ辞書
        detector: GroundingDINOディテクター
        sam_processor: SAMプロセッサー
        
    Returns:
        処理成功時True、失敗時False
    """
    try:
        input_filename = Path(image_path).stem
        file_extension = Path(image_path).suffix
        
        print(f"\n処理中: {input_filename}{file_extension}")
        
        # 画像の読み込み
        image = load_img_to_array(image_path)
        
        # GroundingDINOで物体検出
        boxes, logits, phrases = detector.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=config['processing']['box_threshold'],
            text_threshold=config['processing']['text_threshold']
        )
        
        if len(boxes) == 0:
            print(f"  オブジェクトが検出されませんでした: {input_filename}")
            return False
        
        print(f"  検出されたオブジェクト数: {len(boxes)}")
        
        # GroundingDINO結果を保存
        if config['processing']['save_intermediate']:
            grounding_output_path = os.path.join(
                output_dirs['grounding_dino'], 
                f"{input_filename}{file_extension}"
            )
            save_grounding_dino_result(image, boxes, phrases, logits, grounding_output_path)
        
        # SAMでセグメンテーション
        masks = sam_processor.predict_masks(image, boxes)
        
        if not masks:
            print(f"  マスクの生成に失敗しました: {input_filename}")
            return False
        
        # マスクの膨張処理
        if config['processing']['dilate_kernel_size'] > 0:
            masks = sam_processor.dilate_masks(masks, config['processing']['dilate_kernel_size'])
        
        # マスクの結合
        if config['processing']['combine_masks'] and len(masks) > 1:
            final_mask = sam_processor.combine_masks(masks)
            masks = [final_mask]
        
        # SAMマスクを保存
        if config['processing']['save_intermediate']:
            for i, mask in enumerate(masks):
                mask_output_path = os.path.join(
                    output_dirs['sam'], 
                    f"{input_filename}_mask{i}{file_extension}"
                )
                mask_normalized = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                cv2.imwrite(mask_output_path, mask_normalized)
        
        # LaMaでインペインティング
        lama_config = config['models']['lama']
        best_result = None
        
        for i, mask in enumerate(masks):
            try:
                # メモリ最適化
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 画像リサイズ処理
                original_height, original_width = image.shape[:2]
                max_size = config['processing'].get('max_image_size', 2048)
                
                if original_height > max_size or original_width > max_size:
                    scale = config['processing'].get('memory_scale_factor', 0.5)
                    new_height = int(original_height * scale)
                    new_width = int(original_width * scale)
                    
                    resized_image = cv2.resize(image, (new_width, new_height))
                    resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height))
                    
                    inpainted_img = inpaint_img_with_lama(
                        resized_image, resized_mask,
                        lama_config['config_path'], lama_config['checkpoint_path'],
                        device=config['device']['type']
                    )
                    
                    inpainted_img = cv2.resize(inpainted_img, (original_width, original_height))
                else:
                    inpainted_img = inpaint_img_with_lama(
                        image, mask,
                        lama_config['config_path'], lama_config['checkpoint_path'],
                        device=config['device']['type']
                    )
                
                # LAMA結果を保存
                lama_output_path = os.path.join(
                    output_dirs['lama'], 
                    f"{input_filename}{file_extension}"
                )
                save_array_to_img(inpainted_img, lama_output_path)
                best_result = lama_output_path
                break  # 最初に成功したマスクの結果を使用
                
            except torch.cuda.OutOfMemoryError:
                print(f"  GPU メモリ不足、CPUで再試行: {input_filename}")
                torch.cuda.empty_cache()
                
                try:
                    inpainted_img = inpaint_img_with_lama(
                        image, mask,
                        lama_config['config_path'], lama_config['checkpoint_path'],
                        device="cpu"
                    )
                    
                    lama_output_path = os.path.join(
                        output_dirs['lama'], 
                        f"{input_filename}{file_extension}"
                    )
                    save_array_to_img(inpainted_img, lama_output_path)
                    best_result = lama_output_path
                    break
                    
                except Exception as e:
                    print(f"  CPU処理でもエラー: {e}")
                    continue
                    
            except Exception as e:
                print(f"  インペインティングエラー: {e}")
                continue
        
        if best_result:
            print(f"  処理完了: {input_filename}")
            return True
        else:
            print(f"  インペインティング失敗: {input_filename}")
            return False
            
    except Exception as e:
        print(f"  処理エラー ({input_filename}): {e}")
        return False


def main():
    """メイン関数 - バッチ処理対応"""
    if not IMPORTS_OK:
        print("必要なモジュールのインポートに失敗しました。")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="テキストベースオブジェクト除去システム - バッチ処理対応")
    parser.add_argument("--text_prompt", type=str, help="除去するオブジェクトのテキスト記述（省略時は設定ファイルのデフォルト値を使用）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--input_dir", type=str, help="入力ディレクトリ（設定ファイルの値を上書き）")
    parser.add_argument("--output_dir", type=str, help="出力ディレクトリ（設定ファイルの値を上書き）")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], help="使用デバイス（設定ファイルの値を上書き）")
    
    args = parser.parse_args()
    
    # 設定の読み込み
    print("=== テキストベースオブジェクト除去システム ===")
    print("設定ファイルを読み込み中...")
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # コマンドライン引数で設定を上書き
    text_prompt = args.text_prompt if args.text_prompt else config.get('general', {}).get('default_text_prompt')
    
    if not text_prompt:
        print("エラー: テキストプロンプトが指定されていません。--text_prompt引数を指定するか、config.yamlでdefault_text_promptを設定してください。")
        sys.exit(1)
    
    if args.input_dir:
        config['processing']['input_dir'] = args.input_dir
    if args.output_dir:
        config['processing']['output_dir'] = args.output_dir
    if args.device:
        config['device']['type'] = args.device
    
    # デバイスの自動選択
    if config['device']['type'] == "auto":
        config['device']['type'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 設定の表示
    print(f"入力ディレクトリ: {config['processing']['input_dir']}")
    print(f"テキストプロンプト: {text_prompt}")
    print(f"使用デバイス: {config['device']['type']}")
    print(f"出力ディレクトリ: {config['processing']['output_dir']}")
    
    # ログの設定
    log_level = config.get('logging', {}).get('level', 'WARNING')  # デフォルトをWARNINGに変更
    logging.basicConfig(level=getattr(logging, log_level), 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 特定のライブラリのログレベルを調整
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('saicinpainting').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    
    # 出力ディレクトリの作成
    output_subdirs = config.get('general', {}).get('output_subdirs', {
        'grounding_dino': 'GroundingDINO',
        'sam': 'SAM', 
        'lama': 'LAMA'
    })
    output_dirs = setup_output_directories(config['processing']['output_dir'], output_subdirs)
    
    # 入力画像ファイルの取得
    supported_formats = config.get('general', {}).get('supported_image_formats', 
                                                    ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
    image_files = get_input_images(config['processing']['input_dir'], supported_formats)
    
    if not image_files:
        print("処理対象の画像ファイルが見つかりませんでした。")
        sys.exit(1)
    
    # モデルの初期化
    print("\nモデルを初期化中...")
    
    # GroundingDINO
    grounding_config = config['models']['grounding_dino']
    detector = TextGroundingDetector(
        config_path=grounding_config['config_path'],
        checkpoint_path=grounding_config['checkpoint_path'],
        device=config['device']['type']
    )
    
    # SAM
    sam_config = config['models']['sam']
    sam_processor = create_sam_processor(
        model_type=sam_config['model_type'],
        checkpoint_path=sam_config['checkpoint_path'],
        device=config['device']['type']
    )
    
    print("モデル初期化完了")
    
    # バッチ処理の実行
    print(f"\n=== バッチ処理開始 ===")
    print(f"処理対象ファイル数: {len(image_files)}")
    
    success_count = 0
    failure_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        
        success = process_single_image(
            image_path=image_file,
            text_prompt=text_prompt,
            config=config,
            output_dirs=output_dirs,
            detector=detector,
            sam_processor=sam_processor
        )
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # リソースのクリーンアップ
    sam_processor.cleanup()
    
    # 結果の表示
    print(f"\n=== 処理完了 ===")
    print(f"成功: {success_count}枚")
    print(f"失敗: {failure_count}枚")
    print(f"総処理数: {len(image_files)}枚")
    
    # 出力先の表示
    print(f"\n出力結果:")
    for key, path in output_dirs.items():
        print(f"  {key.upper()}: {path}")
    
    if success_count > 0:
        print(f"\n✅ バッチ処理が完了しました!")
    else:
        print(f"\n❌ すべての処理に失敗しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()