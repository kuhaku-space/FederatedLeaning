import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from client import Client
from config import Config
from dataset import NistDataManager
from logger import ExperimentLogger
from server import Server

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- メイン実行 ---
def main():
    config = Config.from_args()
    logger.info(f"Using device: {config.device}")
    logger.info(
        f"Partitioning: {config.noniid_type} "
        f"(alpha={config.alpha if config.noniid_type == 'dirichlet' else 'N/A'})"
    )

    # 再現性のためのシード固定
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    if "cuda" in config.device:
        # GPUの場合: デバイス名を表示
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU Name: {gpu_name}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # 1. ログ管理初期化
    exp_logger = ExperimentLogger(config)

    # 2. データ準備
    data_manager = NistDataManager(config)
    datasets = data_manager.prepare_data()

    if not datasets:
        logger.error("No datasets created. Exiting.")
        return

    # 3. Client / Server 初期化
    clients = [Client(ds, config) for ds in datasets]
    server = Server(config)

    # 各ユーザーのデータ数分布を出力
    all_counts = [len(c.train_dataset) + len(c.test_dataset) for c in clients]
    if all_counts:
        logger.info("=== Client Data Distribution Summary ===")
        logger.info(f"Total Clients: {len(clients)}")
        logger.info(f"Total Samples: {sum(all_counts)}")
        logger.info(
            f"Stats: Mean={np.mean(all_counts):.1f}, Std={np.std(all_counts):.1f}, Min={np.min(all_counts)}, Max={np.max(all_counts)}"
        )
        # 上位5名の内訳を表示
        logger.info("Sample Clients (Writer ID: Total Samples):")
        for c in clients[:5]:
            total = len(c.train_dataset) + len(c.test_dataset)
            logger.info(f"  - {c.writer_id}: {total}")
        logger.info("========================================")

    logger.info(
        f"Start Training with {len(clients)} writers. (Centers: {config.num_centers})"
    )

    # 4. 学習ループ
    num_sampled_clients = max(int(config.frac * len(clients)), 1)

    for r in range(config.rounds):
        start_time = time.time()

        round_updates = []
        round_losses = []
        current_allocations = {"Round": r + 1}

        # クライアントのランダムサンプリング
        sampled_indices = np.random.choice(
            range(len(clients)), num_sampled_clients, replace=False
        )

        # 選択されたクライアントで学習
        for idx in tqdm(
            sampled_indices,
            desc=f"Round {r + 1}/{config.rounds}",
            leave=False,
            ncols=80,
        ):
            client = clients[idx]
            probs, weights, loss = client.participate(server.models)

            round_updates.append((probs, weights, loss))
            round_losses.append(loss)
            # 最大確率のセンターを割り当てとして記録
            best_center = probs.index(max(probs))
            current_allocations[client.writer_id] = best_center

        # 集約
        server.aggregate(round_updates)

        # 評価
        acc, val_loss = server.evaluate(clients)
        avg_train_loss = float(np.mean(round_losses)) if round_losses else 0.0

        # クラスタ所属数集計（最大確率のセンターでカウント）
        center_counts = {i: 0 for i in range(config.num_centers)}
        for probs, _, _ in round_updates:
            best_center = probs.index(max(probs))
            center_counts[best_center] += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        counts_str = ", ".join([f"C{k}: {v}" for k, v in center_counts.items()])
        logger.info(
            f"Round {r + 1:02d} | Time: {elapsed_time:.2f}s | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2f}% | [{counts_str}]"
        )

        # ログ保存
        exp_logger.save_round_log(
            r + 1, avg_train_loss, val_loss, acc, center_counts, elapsed_time
        )
        exp_logger.save_allocations([current_allocations])

    logger.info("Training Finished.")


if __name__ == "__main__":
    main()
