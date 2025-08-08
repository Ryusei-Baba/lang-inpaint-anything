# テキストベースオブジェクト除去システム

自然言語で指定したオブジェクトを画像から除去するAIツール

## 結果例

<div align="center">

| 元画像 | 除去結果 |
|--------|----------|
| ![元画像](doc/original.jpg) | ![結果](doc/result.png) |

プロンプト: `"red pylon"` → 赤いパイロンが自動的に除去され自然に修復

</div>

## インストール

```bash
# 依存関係のインストール
pip install -r install/requirements.txt
pip install groundingdino-py

# モデルファイルのダウンロード
chmod +x install/download_models.sh
./install/download_models.sh

# サブモジュールの初期化
git submodule update --init --recursive
```

## 使い方

```bash
# 基本的な使用法
python3 text_remove_anything.py

# カスタム画像・プロンプト
python3 text_remove_anything.py --input_img "image.jpg" --text_prompt "dog"
```

## システム構成

- **GroundingDINO**: テキストでオブジェクト検出
- **SAM**: 高精度セグメンテーション  
- **LaMa**: 自然な画像修復

## 設定 (config.yaml)

```yaml
# デフォルト値
general:
  default_input_img: "./input_image/IMG_8160.jpg"
  default_text_prompt: "red pylon"

# 検出パラメータ
processing:
  box_threshold: 0.35    # 物体検出の信頼度
  dilate_kernel_size: 20 # マスク拡張サイズ
  max_image_size: 2048   # メモリ節約の画像リサイズ上限

device:
  type: "auto"           # cuda/cpu/auto
```

## プロジェクト構造

```
├── text_remove_anything.py  # メインスクリプト
├── config.yaml             # 設定ファイル
├── utils/                  # ユーティリティ
├── doc/                    # ドキュメント・画像
└── output_image/           # 結果出力
```

## システム要件

- Python 3.8+
- CUDA対応GPU（推奨、CPUでも動作）
- 8GB以上のRAM

## 参考

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM](https://github.com/facebookresearch/segment-anything)  
- [LaMa](https://github.com/advimman/lama)