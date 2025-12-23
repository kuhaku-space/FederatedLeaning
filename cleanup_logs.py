import json
import shutil
import sys
from pathlib import Path

import pandas as pd

# ログ保存場所
LOG_DIR = Path("./logs")


def get_experiment_status(exp_dir: Path):
    """
    実験フォルダの状態を判定する
    Return: (is_complete, status_message, current_round, total_rounds)
    """
    config_path = exp_dir / "config.json"
    log_path = exp_dir / "log.csv"

    # 1. Configがない = 起動直後に落ちたゴミデータ
    if not config_path.exists():
        return False, "Missing config.json", 0, 0

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            target_rounds = config.get("rounds", 0)
    except Exception as e:
        return False, f"Broken config: {e}", 0, 0

    # 2. Logがない = 学習が始まる前に落ちた
    if not log_path.exists():
        return False, "Missing log.csv", 0, target_rounds

    try:
        df = pd.read_csv(log_path)
        if df.empty or "Round" not in df.columns:
            return False, "Empty or invalid log.csv", 0, target_rounds

        # 最終ラウンドを取得
        last_round = df["Round"].max()

        # 3. 完了判定 (最終ラウンドが設定値に達しているか)
        # ※ 1-based indexing と仮定して比較
        if last_round >= target_rounds:
            return True, "Complete", last_round, target_rounds
        else:
            return False, "Incomplete", last_round, target_rounds

    except Exception as e:
        return False, f"Error reading log: {e}", 0, target_rounds


def main():
    if not LOG_DIR.exists():
        print(f"Directory {LOG_DIR} does not exist.")
        return

    print(f"Checking experiments in {LOG_DIR}...\n")

    experiments = sorted([d for d in LOG_DIR.iterdir() if d.is_dir()])
    deletion_candidates = []

    # ヘッダー表示
    print(f"{'Experiment ID':<25} | {'Status':<25} | {'Progress':<10}")
    print("-" * 65)

    for exp_dir in experiments:
        is_complete, msg, curr, total = get_experiment_status(exp_dir)

        status_str = f"{curr}/{total}" if total > 0 else "-"

        if is_complete:
            # 完了しているものは緑色（対応端末のみ）またはそのまま表示
            print(f"{exp_dir.name:<25} | {'✅ ' + msg:<25} | {status_str:<10}")
        else:
            # 未完了のものはリストに追加
            print(f"{exp_dir.name:<25} | {'❌ ' + msg:<25} | {status_str:<10}")
            deletion_candidates.append(exp_dir)

    print("-" * 65)

    if not deletion_candidates:
        print("\n✨ No incomplete experiments found. Everything is clean!")
        return

    # --- 削除確認 ---
    print(f"\nFound {len(deletion_candidates)} incomplete experiments.")
    print("WARNING: These directories will be permanently deleted.")

    choice = input("Do you want to delete them? [y/N]: ").strip().lower()

    if choice == "y":
        print("\nDeleting...")
        for d in deletion_candidates:
            try:
                shutil.rmtree(d)
                print(f"Deleted: {d.name}")
            except Exception as e:
                print(f"Failed to delete {d.name}: {e}")
        print("\nCleanup finished!")
    else:
        print("\nOperation cancelled. No files were deleted.")


if __name__ == "__main__":
    main()
