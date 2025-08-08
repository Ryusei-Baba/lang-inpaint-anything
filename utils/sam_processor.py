"""
シンプルなSAM処理モジュール
GroundingDINOで検出されたバウンディングボックスからSAMでマスクを生成
"""

import torch
import numpy as np
import cv2
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Optional

# Inpaint-Anything のパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "Inpaint-Anything"))

try:
    from sam_segment import predict_masks_with_sam
    from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"SAM関連のインポートエラー: {e}")
    SAM_AVAILABLE = False


class SAMProcessor:
    """シンプルなSAMセグメンテーション処理クラス"""
    
    def __init__(self, 
                 model_type: str = "vit_h",
                 checkpoint_path: str = None,
                 device: str = "cuda"):
        """
        SAMプロセッサーを初期化
        
        Args:
            model_type: SAMモデルタイプ (vit_h, vit_l, vit_b)
            checkpoint_path: SAMチェックポイントファイルのパス
            device: 使用デバイス (cuda/cpu/auto)
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM関連のモジュールがインポートできません。Inpaint-Anythingの設定を確認してください。")
        
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # デバイスの設定
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"SAM Processor initialized: {model_type} on {self.device}")
    
    def boxes_to_points(self, boxes: torch.Tensor) -> Tuple[List[List[float]], List[int]]:
        """
        バウンディングボックスをSAM用のポイントプロンプトに変換
        
        Args:
            boxes: バウンディングボックス [N, 4] (x1, y1, x2, y2)
            
        Returns:
            points: ポイント座標のリスト [[x, y], ...]
            labels: ポイントラベルのリスト [1, 1, ...]
        """
        points = []
        labels = []
        
        for box in boxes:
            # ボックスの中心点を取得
            x1, y1, x2, y2 = box.cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            points.append([float(center_x), float(center_y)])
            labels.append(1)  # 前景ポイント
        
        return points, labels
    
    def predict_masks(self, 
                     image: np.ndarray,
                     boxes: torch.Tensor) -> List[np.ndarray]:
        """
        バウンディングボックスからマスクを予測
        
        Args:
            image: 入力画像 (H, W, C) RGB形式
            boxes: バウンディングボックス [N, 4]
            
        Returns:
            masks: 予測されたマスクのリスト
        """
        if len(boxes) == 0:
            return []
        
        try:
            # ボックスをポイントプロンプトに変換
            points, labels = self.boxes_to_points(boxes)
            
            print(f"SAMでマスク予測中... ポイント数: {len(points)}")
            
            # SAMでマスク予測を実行
            masks, scores, logits = predict_masks_with_sam(
                img=image,
                point_coords=points,
                point_labels=labels,
                model_type=self.model_type,
                ckpt_p=self.checkpoint_path,
                device=self.device,
            )
            
            # 結果を整理 - 最も良いマスクだけを選択
            if masks is not None and scores is not None:
                if isinstance(masks, np.ndarray):
                    # 複数のマスクの場合、最高スコアのものを選択
                    if len(masks.shape) == 3:
                        best_mask_idx = np.argmax(scores)
                        best_mask = masks[best_mask_idx]
                        print(f"最適なマスクを選択しました（スコア: {scores[best_mask_idx]:.3f}）")
                        return [best_mask]
                    # 単一のマスクの場合
                    elif len(masks.shape) == 2:
                        return [masks]
                elif isinstance(masks, list):
                    # リストの場合も最高スコアを選択
                    if len(masks) > 1 and scores is not None:
                        best_mask_idx = np.argmax(scores)
                        print(f"最適なマスクを選択しました（スコア: {scores[best_mask_idx]:.3f}）")
                        return [masks[best_mask_idx]]
                    return [masks[0]] if masks else []
            
            print("SAMマスク予測が失敗しました")
            return []
            
        except torch.cuda.OutOfMemoryError:
            print("GPU メモリ不足です。CPUで再試行します...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # CPUで再試行
            masks, scores, logits = predict_masks_with_sam(
                img=image,
                point_coords=points,
                point_labels=labels,
                model_type=self.model_type,
                ckpt_p=self.checkpoint_path,
                device="cpu",
            )
            
            if masks is not None and scores is not None:
                if isinstance(masks, np.ndarray):
                    # 複数のマスクの場合、最高スコアのものを選択
                    if len(masks.shape) == 3:
                        best_mask_idx = np.argmax(scores)
                        best_mask = masks[best_mask_idx]
                        print(f"CPU処理: 最適なマスクを選択しました（スコア: {scores[best_mask_idx]:.3f}）")
                        return [best_mask]
                    # 単一のマスクの場合
                    elif len(masks.shape) == 2:
                        return [masks]
                elif isinstance(masks, list):
                    # リストの場合も最高スコアを選択
                    if len(masks) > 1 and scores is not None:
                        best_mask_idx = np.argmax(scores)
                        print(f"CPU処理: 最適なマスクを選択しました（スコア: {scores[best_mask_idx]:.3f}）")
                        return [masks[best_mask_idx]]
                    return [masks[0]] if masks else []
            
            return []
            
        except Exception as e:
            print(f"SAMマスク予測中にエラーが発生しました: {e}")
            return []
    
    def dilate_masks(self, masks: List[np.ndarray], kernel_size: int = 15) -> List[np.ndarray]:
        """
        マスクを膨張処理して拡張
        
        Args:
            masks: マスクのリスト
            kernel_size: 膨張処理のカーネルサイズ
            
        Returns:
            dilated_masks: 膨張処理後のマスクのリスト
        """
        dilated_masks = []
        
        for mask in masks:
            try:
                # dilate_mask関数を使用
                dilated_mask = dilate_mask(mask, kernel_size)
                dilated_masks.append(dilated_mask)
            except Exception as e:
                print(f"マスク膨張処理中にエラー: {e}")
                # エラーの場合は元のマスクをそのまま使用
                dilated_masks.append(mask)
        
        return dilated_masks
    
    def combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        複数のマスクを1つに結合
        
        Args:
            masks: マスクのリスト
            
        Returns:
            combined_mask: 結合されたマスク
        """
        if len(masks) == 0:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        # 最初のマスクから開始
        combined_mask = masks[0].copy()
        
        # 他のマスクを順次結合
        for mask in masks[1:]:
            combined_mask = np.logical_or(combined_mask, mask)
        
        return combined_mask.astype(np.uint8)
    
    def cleanup(self):
        """リソースをクリーンアップ"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def create_sam_processor(model_type: str = "vit_h",
                        checkpoint_path: str = None,
                        device: str = "cuda") -> SAMProcessor:
    """
    SAMプロセッサーのファクトリ関数
    
    Args:
        model_type: SAMモデルタイプ
        checkpoint_path: チェックポイントファイルのパス
        device: 使用デバイス
        
    Returns:
        SAMProcessor インスタンス
    """
    return SAMProcessor(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device
    )