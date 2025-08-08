#!/usr/bin/env python3
"""
テキストベースオブジェクト除去システム - メインファイル

GroundingDINO + SAM + LaMaを使用して、テキストプロンプトに基づいて画像からオブジェクトを除去します。

使用方法:
    python text_remove_anything.py --input_img "image.jpg" --text_prompt "dog"
"""

import argparse
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional

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


def interactive_selection(phrases: List[str], confidences: List[float]) -> List[int]:
    """
    インタラクティブモード: ユーザーが削除するオブジェクトを選択
    
    Args:
        phrases: 検出されたオブジェクトの名前リスト
        confidences: 信頼度スコアのリスト
        
    Returns:
        選択されたオブジェクトのインデックスのリスト
    """
    print("\n=== 検出されたオブジェクト ===")
    for i, (phrase, confidence) in enumerate(zip(phrases, confidences)):
        print(f"{i}: {phrase} (信頼度: {confidence:.3f})")
    
    while True:
        try:
            selection = input("\n削除するオブジェクトを選択してください (番号をカンマ区切り、または 'all' で全て): ").strip()
            
            if selection.lower() in ['all', 'すべて']:
                return list(range(len(phrases)))
            
            if not selection:
                print("何も入力されていません。もう一度お試しください。")
                continue
                
            selected_indices = [int(x.strip()) for x in selection.split(',')]
            
            # インデックスの妥当性チェック
            if all(0 <= idx < len(phrases) for idx in selected_indices):
                return selected_indices
            else:
                print(f"無効なインデックスです。0-{len(phrases)-1}の範囲で入力してください。")
                
        except ValueError:
            print("無効な入力です。数字をカンマ区切りで入力してください。")
        except KeyboardInterrupt:
            print("\n処理を中断しました。")
            return []


def process_single_image(image_path: str, 
                        text_prompt: str, 
                        config: dict, 
                        interactive: bool = False) -> Optional[str]:
    """
    単一画像の処理
    
    Args:
        image_path: 入力画像のパス
        text_prompt: テキストプロンプト
        config: 設定辞書
        interactive: インタラクティブモードかどうか
        
    Returns:
        出力画像のパス（成功時）、None（失敗時）
    """
    try:
        print(f"\n=== 画像処理開始: {image_path} ===")
        print(f"プロンプト: '{text_prompt}'")
        
        # 画像の読み込み
        if not os.path.exists(image_path):
            print(f"エラー: 画像ファイルが見つかりません: {image_path}")
            return None
            
        image = load_img_to_array(image_path)
        print(f"画像サイズ: {image.shape}")
        
        # GroundingDINOで物体検出
        print("物体検出中...")
        grounding_config = config['models']['grounding_dino']
        detector = TextGroundingDetector(
            config_path=grounding_config['config_path'],
            checkpoint_path=grounding_config['checkpoint_path']
        )
        
        boxes, logits, phrases = detector.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=config['processing']['box_threshold'],
            text_threshold=config['processing']['text_threshold']
        )
        
        if len(boxes) == 0:
            print("オブジェクトが検出されませんでした。")
            return None
            
        print(f"検出されたオブジェクト数: {len(boxes)}")
        
        # インタラクティブモードでの選択
        selected_indices = list(range(len(boxes)))
        if interactive:
            selected_indices = interactive_selection(phrases, logits.cpu().numpy())
            if not selected_indices:
                print("処理をスキップしました。")
                return None
        
        # 選択されたボックスのみを使用
        selected_boxes = boxes[selected_indices]
        selected_phrases = [phrases[i] for i in selected_indices]
        
        print(f"処理対象: {len(selected_boxes)}個のオブジェクト")
        for i, phrase in enumerate(selected_phrases):
            print(f"  - {phrase}")
        
        # SAMでセグメンテーション
        print("セグメンテーション中...")
        sam_config = config['models']['sam']
        sam_processor = create_sam_processor(
            model_type=sam_config['model_type'],
            checkpoint_path=sam_config['checkpoint_path'],
            device=config['device']['type']
        )
        
        masks = sam_processor.predict_masks(image, selected_boxes)
        
        if not masks:
            print("マスクの生成に失敗しました。")
            return None
            
        print(f"生成されたマスク数: {len(masks)}")
        
        # マスクの膨張処理
        if config['processing']['dilate_kernel_size'] > 0:
            print("マスクの膨張処理中...")
            masks = sam_processor.dilate_masks(masks, config['processing']['dilate_kernel_size'])
        
        # マスクの結合
        if config['processing']['combine_masks'] and len(masks) > 1:
            print("マスクの結合中...")
            final_mask = sam_processor.combine_masks(masks)
            masks = [final_mask]
        
        # 出力ディレクトリの作成
        input_filename = Path(image_path).stem
        output_base_dir = config['processing']['output_dir']
        output_dir = Path(output_base_dir) / f"{input_filename}_{text_prompt.replace(' ', '_').replace(',', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaMaでインペインティング
        print("インペインティング中...")
        lama_config = config['models']['lama']
        
        inpainted_results = []
        for i, mask in enumerate(masks):
            try:
                # GPUメモリをクリアしてからインペインティング
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 画像サイズを小さくしてメモリ使用量を削減
                original_height, original_width = image.shape[:2]
                max_size = config['processing'].get('max_image_size', 2048)
                if original_height > max_size or original_width > max_size:
                    # 設定されたスケールファクターでリサイズ
                    scale = config['processing'].get('memory_scale_factor', 0.5)
                    new_height = int(original_height * scale)
                    new_width = int(original_width * scale)
                    
                    print(f"メモリ節約のため画像をリサイズ: {original_width}x{original_height} → {new_width}x{new_height}")
                    
                    # 画像とマスクをリサイズ
                    resized_image = cv2.resize(image, (new_width, new_height))
                    resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height))
                    
                    # インペインティング実行
                    inpainted_img = inpaint_img_with_lama(
                        resized_image,
                        resized_mask,
                        lama_config['config_path'],
                        lama_config['checkpoint_path'],
                        device=config['device']['type']
                    )
                    
                    # 元のサイズに戻す
                    inpainted_img = cv2.resize(inpainted_img, (original_width, original_height))
                    
                else:
                    # 元のサイズで処理
                    inpainted_img = inpaint_img_with_lama(
                        image,
                        mask,
                        lama_config['config_path'],
                        lama_config['checkpoint_path'],
                        device=config['device']['type']
                    )
                
                # 結果の保存
                output_path = output_dir / f"inpainted_{i}.png"
                save_array_to_img(inpainted_img, str(output_path))
                inpainted_results.append(str(output_path))
                print(f"保存完了: {output_path}")
                
                # マスクも保存
                mask_path = output_dir / f"mask_{i}.png"
                mask_normalized = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                cv2.imwrite(str(mask_path), mask_normalized)
                
            except torch.cuda.OutOfMemoryError:
                print(f"GPU メモリ不足です。CPUで再試行します... (マスク {i})")
                torch.cuda.empty_cache()
                
                try:
                    # CPUで再試行
                    inpainted_img = inpaint_img_with_lama(
                        image,
                        mask,
                        lama_config['config_path'],
                        lama_config['checkpoint_path'],
                        device="cpu"
                    )
                    
                    # 結果の保存
                    output_path = output_dir / f"inpainted_{i}.png"
                    save_array_to_img(inpainted_img, str(output_path))
                    inpainted_results.append(str(output_path))
                    print(f"CPU処理完了: {output_path}")
                    
                    # マスクも保存
                    mask_path = output_dir / f"mask_{i}.png"
                    mask_normalized = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                    cv2.imwrite(str(mask_path), mask_normalized)
                    
                except Exception as e:
                    print(f"CPU処理でもエラーが発生: {e}")
                    continue
                    
            except Exception as e:
                print(f"インペインティング {i} でエラー: {e}")
                continue
        
        # リソースのクリーンアップ
        sam_processor.cleanup()
        
        if inpainted_results:
            print(f"\n=== 処理完了 ===")
            print(f"出力ディレクトリ: {output_dir}")
            print(f"生成されたファイル数: {len(inpainted_results)}")
            return inpainted_results[0]  # 最初の結果を返す
        else:
            print("インペインティングに失敗しました。")
            return None
            
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """メイン関数"""
    if not IMPORTS_OK:
        print("必要なモジュールのインポートに失敗しました。")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="テキストベースオブジェクト除去システム")
    parser.add_argument("--input_img", type=str, help="入力画像のパス（省略時は設定ファイルのデフォルト値を使用）")
    parser.add_argument("--text_prompt", type=str, help="除去するオブジェクトのテキスト記述（省略時は設定ファイルのデフォルト値を使用）")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルのパス")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--output_dir", type=str, help="出力ディレクトリ（設定ファイルの値を上書き）")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], help="使用デバイス（設定ファイルの値を上書き）")
    
    args = parser.parse_args()
    
    # 設定の読み込み
    print("設定ファイルを読み込み中...")
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # デフォルト値の設定（引数が指定されない場合）
    input_img = args.input_img if args.input_img else config.get('general', {}).get('default_input_img')
    text_prompt = args.text_prompt if args.text_prompt else config.get('general', {}).get('default_text_prompt')
    
    if not input_img:
        print("エラー: 入力画像が指定されていません。--input_img引数を指定するか、config.yamlでdefault_input_imgを設定してください。")
        sys.exit(1)
    
    if not text_prompt:
        print("エラー: テキストプロンプトが指定されていません。--text_prompt引数を指定するか、config.yamlでdefault_text_promptを設定してください。")
        sys.exit(1)
    
    # コマンドライン引数で設定を上書き
    if args.output_dir:
        config['processing']['output_dir'] = args.output_dir
    if args.device:
        config['device']['type'] = args.device
    
    # デバイスの自動選択
    if config['device']['type'] == "auto":
        config['device']['type'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"入力画像: {input_img}")
    print(f"テキストプロンプト: {text_prompt}")
    print(f"使用デバイス: {config['device']['type']}")
    print(f"出力ディレクトリ: {config['processing']['output_dir']}")
    
    # 画像処理の実行
    result = process_single_image(
        image_path=input_img,
        text_prompt=text_prompt,
        config=config,
        interactive=args.interactive
    )
    
    if result:
        print(f"\n✅ 処理が正常に完了しました!")
        print(f"結果: {result}")
    else:
        print(f"\n❌ 処理に失敗しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()