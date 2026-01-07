import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


# --- ロガーヘルパー ---
class ExperimentLogger:
    def __init__(self, config: Config):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.log_base_dir) / self.run_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.save_dir / "config.json"
        self.log_csv = self.save_dir / "log.csv"
        self.allocation_csv = self.save_dir / "client_allocation.csv"

        # Config保存
        with self.config_file.open("w") as f:
            json.dump(asdict(config), f, indent=4)

        logger.info(f"Experiment initialized. Logs at: {self.save_dir}")

    def save_round_log(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        center_counts: dict[int, int],
        elapsed_time: float,
    ):
        record = {
            "Round": round_num,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Accuracy": accuracy,
            "Time": elapsed_time,
        }
        for k, v in center_counts.items():
            record[f"Center_{k}_Clients"] = v

        df = pd.DataFrame([record])
        if not self.log_csv.exists():
            df.to_csv(self.log_csv, index=False)
        else:
            df.to_csv(self.log_csv, mode="a", header=False, index=False)

    def save_allocations(self, allocations: list[dict[str, int]]):
        """
        allocations: [{"Round": 1, "writerA": 0, ...}]
        """
        df = pd.DataFrame(allocations)

        # 初回作成時と追記時で処理を分ける
        if not self.allocation_csv.exists():
            # 列順を整形 (Round, writer1, writer2...)
            cols = ["Round"] + sorted([c for c in df.columns if c != "Round"])
            df = df[cols]
            df.to_csv(self.allocation_csv, index=False)
        else:
            # 既存のヘッダーに合わせて追記
            existing_cols = pd.read_csv(self.allocation_csv, nrows=0).columns.tolist()
            # 足りない列があればNaNで埋め、余計な列は無視あるいは追加（今回は固定Writer前提なので無視）
            df = df.reindex(columns=existing_cols)
            df.to_csv(self.allocation_csv, mode="a", header=False, index=False)
