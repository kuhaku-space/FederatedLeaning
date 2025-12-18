import argparse
import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 設定管理 (Dataclass) ---
@dataclass
class Config:
    # 基本設定
    num_centers: int = 2
    rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.05

    # データ設定
    max_writers: int = 100
    max_imgs_per_writer: int = 100
    data_root: str = "./data/nist/extracted"

    # システム設定
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    log_base_dir: str = "./logs"
    num_classes: int = 62

    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(
            description="NIST Federated Learning Simulation"
        )
        defaults = cls()

        parser.add_argument(
            "--num-centers",
            type=int,
            default=defaults.num_centers,
            help="Number of centers",
        )
        parser.add_argument(
            "--rounds", type=int, default=defaults.rounds, help="Number of rounds"
        )
        parser.add_argument(
            "--local-epochs",
            type=int,
            default=defaults.local_epochs,
            help="Local epochs",
        )
        parser.add_argument(
            "--batch-size", type=int, default=defaults.batch_size, help="Batch size"
        )
        parser.add_argument(
            "--lr", type=float, default=defaults.lr, help="Learning rate"
        )
        parser.add_argument(
            "--max-writers", type=int, default=defaults.max_writers, help="Max writers"
        )
        parser.add_argument(
            "--max-imgs-per-writer",
            type=int,
            default=defaults.max_imgs_per_writer,
            help="Max images per writer",
        )
        parser.add_argument(
            "--data-root", type=str, default=defaults.data_root, help="Data root path"
        )
        parser.add_argument(
            "--device", type=str, default=defaults.device, help="Device (cpu/cuda)"
        )
        parser.add_argument(
            "--log-base-dir",
            type=str,
            default=defaults.log_base_dir,
            help="Log directory",
        )

        args = parser.parse_args()
        return cls(**vars(args))


# --- Dataset 定義 ---
class NistWriterDataset(Dataset):
    def __init__(
        self,
        writer_id: str,
        images: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.writer_id = writer_id
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.images[index]
        label = self.labels[index]
        try:
            with Image.open(path) as img:
                img = img.convert("L")
                if self.transform:
                    img = self.transform(img)

            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            return img, label
        except (UnidentifiedImageError, OSError) as e:
            # 破損画像の場合は警告を出してゼロ埋めデータを返す（学習を止めないため）
            # 本来はDataset作成時に除外するのがベスト
            logger.warning(f"Error loading image {path}: {e}")
            return torch.zeros((1, 28, 28)), label


# --- データ管理クラス ---
class NistDataManager:
    def __init__(self, config: Config):
        self.cfg = config
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def _generate_hex_label_map(self) -> Dict[str, int]:
        mapping = {}
        # 0-9
        for i in range(10):
            mapping[f"{0x30 + i:02x}"] = i
        # A-Z
        for i in range(26):
            mapping[f"{0x41 + i:02x}"] = 10 + i
        # a-z
        for i in range(26):
            mapping[f"{0x61 + i:02x}"] = 36 + i
        return mapping

    def prepare_data(self) -> List[NistWriterDataset]:
        data_root = Path(self.cfg.data_root)
        class_dir = data_root / "by_class"

        logger.info("--- Data Preparation Start ---")
        if not class_dir.exists():
            logger.error(f"Directory not found: {class_dir}")
            return []

        writers_data: Dict[str, Dict[str, List[Any]]] = {}
        hex_to_label = self._generate_hex_label_map()

        # .mit ファイルの解析
        for hex_code, label in tqdm(hex_to_label.items(), desc="Parsing classes"):
            # ディレクトリ検索（大文字小文字対応）
            target_dir = class_dir / hex_code
            if not target_dir.exists():
                target_dir = class_dir / hex_code.upper()
            if not target_dir.exists():
                continue

            self._parse_mit_files_in_dir(target_dir, label, writers_data)

        logger.info(f"Found {len(writers_data)} unique writers.")
        return self._create_datasets(writers_data)

    def _parse_mit_files_in_dir(self, target_dir: Path, label: int, writers_data: Dict):
        for mit_file in target_dir.rglob("*.mit"):
            mit_stem = mit_file.stem
            try:
                with mit_file.open("r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                # ヘッダー行(0行目)スキップ、各行解析
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue

                    filename = parts[0]
                    orig_path = parts[1]
                    writer_id = self._extract_writer_id(orig_path)

                    if not writer_id:
                        continue

                    # 画像パスの解決
                    candidate_sub = mit_file.parent / mit_stem / filename
                    candidate_flat = mit_file.parent / filename

                    target_path = None
                    if candidate_sub.exists():
                        target_path = candidate_sub
                    elif candidate_flat.exists():
                        target_path = candidate_flat

                    if target_path:
                        if writer_id not in writers_data:
                            writers_data[writer_id] = {"images": [], "labels": []}
                        writers_data[writer_id]["images"].append(str(target_path))
                        writers_data[writer_id]["labels"].append(label)

            except Exception as e:
                # ファイル破損等はスキップ
                pass

    @staticmethod
    def _extract_writer_id(orig_path: str) -> Optional[str]:
        for comp in orig_path.split("/"):
            # "f"で始まり、数字が含まれるディレクトリ名をWriterIDとみなす
            if comp.startswith("f") and any(c.isdigit() for c in comp):
                return comp
        return None

    def _create_datasets(self, writers_data: Dict) -> List[NistWriterDataset]:
        writers_datasets = []

        # データ数が多い順にソート
        sorted_writers = sorted(
            writers_data.items(), key=lambda x: len(x[1]["images"]), reverse=True
        )

        count = 0
        for w_id, data in sorted_writers:
            images = data["images"]
            labels = data["labels"]

            # データ数間引き
            if len(images) > self.cfg.max_imgs_per_writer:
                indices = np.random.permutation(len(images))[
                    : self.cfg.max_imgs_per_writer
                ]
                images = [images[i] for i in indices]
                labels = [labels[i] for i in indices]

            # 最低枚数チェック
            if len(images) > 10:
                ds = NistWriterDataset(w_id, images, labels, self.transform)
                writers_datasets.append(ds)
                count += 1
                if count >= self.cfg.max_writers:
                    break

        logger.info(f"Prepared {len(writers_datasets)} writer datasets.")
        return writers_datasets


# --- モデル定義 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# --- Client 定義 ---
class Client:
    def __init__(self, dataset: NistWriterDataset, config: Config):
        self.dataset = dataset
        self.cfg = config
        self.device = torch.device(config.device)
        self.dataloader = DataLoader(
            dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True
        )
        self.criterion = nn.CrossEntropyLoss()

    def participate(
        self, global_models: List[nn.Module]
    ) -> Tuple[int, Dict[str, torch.Tensor], float]:
        """
        1. 複数のGlobal Modelの中からLossが最も低いモデルを選択
        2. 選択したモデルをローカルデータで追加学習
        3. 重み差分（あるいは更新後の重み）とLossを返す
        """
        best_idx, model = self._select_best_model(global_models)

        # 学習
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr)

        epoch_loss = 0.0
        batch_count = 0

        for _ in range(self.cfg.local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        return best_idx, model.state_dict(), avg_loss

    def _select_best_model(
        self, global_models: List[nn.Module]
    ) -> Tuple[int, nn.Module]:
        # チェック用バッチの取得
        try:
            x_check, y_check = next(iter(self.dataloader))
        except StopIteration:
            # データが無い場合のフォールバック（通常ありえないが念のため）
            return 0, copy.deepcopy(global_models[0])

        x_check, y_check = x_check.to(self.device), y_check.to(self.device)

        best_idx = 0
        best_loss = float("inf")

        with torch.no_grad():
            for i, model in enumerate(global_models):
                model.eval()
                output = model(x_check)
                loss = self.criterion(output, y_check).item()
                if loss < best_loss:
                    best_loss = loss
                    best_idx = i

        # Deepcopyして返す
        return best_idx, copy.deepcopy(global_models[best_idx])


# --- Server 定義 ---
class Server:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.models = [
            SimpleCNN(config.num_classes).to(self.device)
            for _ in range(config.num_centers)
        ]
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, updates: List[Tuple[int, Dict[str, torch.Tensor], float]]):
        if not updates:
            return

        # センターごとのアップデートリストを作成
        center_updates: Dict[int, List[Dict[str, torch.Tensor]]] = {
            i: [] for i in range(len(self.models))
        }
        for idx, weights, _ in updates:
            center_updates[idx].append(weights)

        # FedAvg
        for i, model in enumerate(self.models):
            updates_for_model = center_updates[i]
            if not updates_for_model:
                continue

            avg_weights = copy.deepcopy(updates_for_model[0])
            for key in avg_weights.keys():
                for j in range(1, len(updates_for_model)):
                    avg_weights[key] += updates_for_model[j][key]
                # 平均計算: float除算
                avg_weights[key] = torch.div(avg_weights[key], len(updates_for_model))

            model.load_state_dict(avg_weights)

    def evaluate(self, clients: List[Client]) -> Tuple[float, float]:
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        # クライアントごとにテスト（各クライアントは最適なモデルを選択して評価）
        for client in clients:
            test_loader = DataLoader(
                client.dataset, batch_size=self.cfg.batch_size, shuffle=False
            )

            # クライアントに最適なモデルを選択（本来はValidation setを使うべきだが、簡易的に1バッチで選択）
            # ここでは Client クラスのロジックを再利用せず、サーバー側で検証ロジックを持つ形にする
            try:
                x_chk, y_chk = next(iter(test_loader))
            except StopIteration:
                continue

            x_chk, y_chk = x_chk.to(self.device), y_chk.to(self.device)
            best_model = self.models[0]
            best_val_loss = float("inf")

            with torch.no_grad():
                for model in self.models:
                    model.eval()
                    loss = self.criterion(model(x_chk), y_chk).item()
                    if loss < best_val_loss:
                        best_val_loss = loss
                        best_model = model

            # 選択されたモデルで全データを評価
            best_model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = best_model(x)
                    loss_val = self.criterion(out, y).item()
                    total_loss += loss_val * x.size(0)

                    _, predicted = torch.max(out, 1)
                    total_correct += (predicted == y).sum().item()
                    total_samples += y.size(0)

        accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return accuracy, avg_loss


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
        center_counts: Dict[int, int],
    ):
        record = {
            "Round": round_num,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Accuracy": accuracy,
        }
        for k, v in center_counts.items():
            record[f"Center_{k}_Clients"] = v

        df = pd.DataFrame([record])
        if not self.log_csv.exists():
            df.to_csv(self.log_csv, index=False)
        else:
            df.to_csv(self.log_csv, mode="a", header=False, index=False)

    def save_allocations(self, allocations: List[Dict]):
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


# --- メイン実行 ---
def main():
    config = Config.from_args()
    logger.info(f"Using device: {config.device}")

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

    logger.info(
        f"Start Training with {len(clients)} writers. (Centers: {config.num_centers})"
    )

    # 4. 学習ループ
    for r in range(config.rounds):
        round_updates = []
        round_losses = []

        # 今ラウンドのクライアント所属記録用辞書
        current_allocations = {"Round": r + 1}

        # クライアント学習
        for client in tqdm(clients, desc=f"Round {r + 1}/{config.rounds}", leave=False):
            center_idx, weights, loss = client.participate(server.models)

            round_updates.append((center_idx, weights, loss))
            round_losses.append(loss)
            current_allocations[client.dataset.writer_id] = center_idx

        # 集約
        server.aggregate(round_updates)

        # 評価
        acc, val_loss = server.evaluate(clients)
        avg_train_loss = np.mean(round_losses) if round_losses else 0.0

        # クラスタ所属数集計
        center_counts = {i: 0 for i in range(config.num_centers)}
        for idx, _, _ in round_updates:
            center_counts[idx] += 1

        counts_str = ", ".join([f"C{k}: {v}" for k, v in center_counts.items()])
        logger.info(
            f"Round {r + 1:02d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2f}% | [{counts_str}]"
        )

        # ログ保存
        exp_logger.save_round_log(r + 1, avg_train_loss, val_loss, acc, center_counts)
        exp_logger.save_allocations([current_allocations])

    logger.info("Training Finished.")


if __name__ == "__main__":
    main()
