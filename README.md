# PAMIQ Curiosity Experiments

VRChat環境における好奇心駆動型強化学習の実験リポジトリ

## 概要

このリポジトリは、VRChat環境（やその類似環境）でエージェントが好奇心（内発的動機）によって自律的に探索行動を学習する実験を行うためのコードベース。

### 実験の目的

- さまざまな好奇心メカニズムを比較評価するために行う
- まとめて実装することで、今後の実験を楽にする

## 実験環境のセットアップ

### 必要な環境

- x86_64のCPU
- Linux環境
- NVIDIA GPU（必須）
- デスクトップ環境（VRChatを表示するため）

### セットアップ手順

1. **VRChatの準備**

   [pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat)のREADMEに従って、ホストマシンでVRChat環境をセットアップする。

2. **必要なソフトウェアのインストール**

   - git
   - make
   - docker

3. **実験環境の構築**

   ```bash
   # このリポジトリをクローン
   git clone https://github.com/MLShukai/pamiq-curiosity-exp.git
   cd pamiq-curiosity-exp

   # Dockerイメージをビルド（GPUサポート付き）
   make docker-build

   # VRChatとOBSを起動しておく（重要！）

   # Dockerコンテナを起動
   # LOG_DIRは実験ログを保存する場所を指定（デフォルトは./logs）
   make docker-up LOG_DIR=/path/to/log_dir

   # コンテナに接続
   make docker-attach
   ```

   VSCodeを使っている場合は、Container ToolsやDev Container拡張機能でアタッチすることも可能。

## 実験の実行方法

### 基本的な実行コマンド

コンテナ内で以下のコマンドを実行して実験を開始する：

```bash
# 実験を開始（experimentには実験設定ファイル名を指定）
python src/train.py experiment=<実験設定名>

# タブ補完で利用可能な実験設定を確認できる
python src/train.py experiment=[TAB]
```

### 実験パラメータの調整

Hydraを使用しているため、コマンドラインから簡単にパラメータを変更できる：

```bash
# 学習率を変更して実行
python src/train.py experiment=<実験設定名> trainers.policy.partial_optimizer.lr=3e-4
```

### 長時間実験のための設定

実験はtmuxを使用することを推奨：

```bash
# tmuxセッションを作成
tmux new -s my_experiment

# 実験を開始
python src/train.py experiment=<実験設定名>

# Ctrl+B → D でセッションから離脱（実験は継続）

# 後で再接続する場合
tmux a -t my_experiment

# 実行中のセッション一覧
tmux ls
```

### 実験の進行状況を確認

Aimで実験のメトリクスをリアルタイムで確認できる：

```bash
# ホストマシンの別ターミナルで実行
make aim

# ブラウザで http://localhost:43800 を開く
# 実験一覧から現在実行中の実験を選択してメトリクスを確認
```

## コードの開発・修正方法

### 開発時の注意点

- Python 3.12以降の機能を使用（match文など）
- 型アノテーションは必須（`list[str]`の形式で記述）
- Docstringは英語でGoogle Styleで記述

### プロジェクトの構造

```
src/
├── train.py              # 実験のエントリーポイント
├── configs/              # Hydra設定ファイル
│   ├── experiment/       # 実験設定（これを指定して実行）
│   ├── data/             # データバッファの設定
│   ├── models/           # モデルのアーキテクチャ設定
│   ├── trainers/         # 学習アルゴリズムの設定
│   └── interaction/      # エージェント・環境の設定
└── exp/                  # ソースコード本体
    ├── agents/           # エージェント実装
    ├── models/           # ニューラルネットワーク
    ├── trainers/         # 学習ループ
    └── envs/             # 環境実装
```

### 新しい実験を追加する場合

1. **既存の実験設定を参考に新しい設定ファイルを作成**

2. **必要に応じてコンポーネントを修正**

   - 新しい好奇心メカニズムが必要な場合は`src/exp/agents/curiosity/`に実装
   - モデル構造を変更する場合は`src/configs/models/`に設定を追加

3. **テストを書いて動作確認**

   ```bash
   # 新しいコードのテストを実行
   uv run pytest tests/exp/agents/test_my_new_agent.py -v
   ```

### 実験中のデバッグ

コンテナ内で開発する際の便利なコマンド：

```bash
# コードの自動フォーマット
make format

# 型チェック（エラーがないか確認）
make type

# テストを実行
make test

# 上記すべてを一度に実行
make run
```

### Claudeを使った開発支援

コンテナ内では`claude`コマンドが使用可能：

```bash
# Claudeを起動
claude
```

## 依存関係の管理

```bash
# 新しいパッケージを追加
uv add <package_name>

# 開発用パッケージを追加
uv add --dev <package_name>

# 依存関係を最新に更新
uv sync

# 仮想環境を作り直す（問題が起きた時）
make venv
```

## その他の情報

- 実験ログは指定した`LOG_DIR`に保存される
- チェックポイントは定期的に保存され、実験の再開が可能
- `extensions.json`に記載のVSCode拡張機能をインストールすると開発が便利になる
