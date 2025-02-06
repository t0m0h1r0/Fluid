#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime

from src.simulation.runner import SimulationRunner
from src.visualization.visualizer import Visualizer
from src.data_io.data_manager import DataManager

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='二相流体シミュレーション')
    
    # 必須の引数
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='設定ファイルのパス'
    )
    
    # オプション引数
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='出力ディレクトリ（未指定時は設定ファイルの値を使用）'
    )
    parser.add_argument(
        '--visualize-interval',
        type=float,
        default=None,
        help='可視化の間隔（秒）（未指定時は設定ファイルの値を使用）'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=float,
        default=None,
        help='チェックポイントの保存間隔（秒）（未指定時は設定ファイルの値を使用）'
    )
    parser.add_argument(
        '--max-time',
        type=float,
        default=None,
        help='最大計算時間（秒）（未指定時は設定ファイルの値を使用）'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='可視化を無効化'
    )
    
    return parser.parse_args()

def validate_config_file(config_path: str) -> dict:
    """設定ファイルの検証と読み込み"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの形式が不正です: {str(e)}")
    
    # 必須項目のチェック
    required_keys = ['physical', 'domain', 'initial_condition']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"設定ファイルに必要な項目が不足しています: {missing_keys}")
    
    return config

def setup_output_directory(base_dir: str) -> Path:
    """出力ディレクトリの設定"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    
    # ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリの作成
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    return output_dir

def save_initial_visualization(runner: SimulationRunner, output_dir: Path):
    """初期状態の可視化を保存"""
    print("初期状態を可視化しています...")
    
    # 可視化の作成と保存
    vis = Visualizer(str(output_dir))
    fields = runner.fields
    stats = runner.stats_analyzer.get_current_stats()
    
    # 2D断面図
    vis.create_visualization(
        fields,
        time=0.0,
        step=0,
        stats=stats,
        filename="initial_state"
    )
    
    # 3D表示
    vis.create_3d_visualization(
        fields,
        time=0.0,
        step=0,
        stats=stats,
        filename="initial_state_3d"
    )

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    args = parse_arguments()
    
    try:
        # 設定ファイルの読み込みと検証
        config = validate_config_file(args.config)
        
        # コマンドライン引数で設定を上書き
        if args.output_dir:
            config['output']['directory'] = args.output_dir
        if args.visualize_interval:
            config['output']['visualize_interval'] = args.visualize_interval
        if args.checkpoint_interval:
            config['output']['checkpoint_interval'] = args.checkpoint_interval
        if args.max_time:
            config['time']['max_time'] = args.max_time
        
        # 出力ディレクトリの設定
        output_dir = setup_output_directory(config['output']['directory'])
        
        # 設定のバックアップ
        with open(output_dir / "config_backup.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"出力ディレクトリ: {output_dir}")
        
        # シミュレーションの初期化
        runner = SimulationRunner(args.config)
        
        # 初期状態の可視化
        if not args.no_visualization:
            save_initial_visualization(runner, output_dir)
        
        print("シミュレーションを開始します...")
        
        # シミュレーションの実行
        runner.run()
        
        print("\nシミュレーションが完了しました。")
        print(f"計算時間: {runner.state.get_elapsed_time():.2f}秒")
        
        # 最終状態の統計情報を表示
        final_stats = runner.stats_analyzer.get_current_stats()
        print("\n最終状態の統計情報:")
        for key, value in final_stats.items():
            print(f"  {key}: {value:.6g}")
        
    except KeyboardInterrupt:
        print("\nシミュレーションが中断されました。")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
