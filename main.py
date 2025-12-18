import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --- 引数解析 ---
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIST Federated Learning Simulation")

    # 基本設定
    parser.add_argument(
        "--num-centers", type=int, default=2, help="Number of centers (servers)"
    )
    parser.add_argument(
        "--rounds", type=int, default=50, help="Number of communication rounds"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=5, help="Local training epochs per round"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")

    # データ設定
    parser.add_argument(
        "--max-writers", type=int, default=100, help="Max writers to use"
    )
    parser.add_argument(
        "--max-imgs-per-writer", type=int, default=100, help="Max images per writer"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nist/extracted",
        help="Path to data root",
    )

    # システム設定
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--log-base-dir", type=str, default="./logs", help="Root directory for all logs"
    )

    return parser.parse_args()


NUM_CLASSES: int = 62


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
        except Exception:
            return torch.zeros((1, 28, 28)), 0


# --- データ読み込み ---
def prepare_data_final(args: argparse.Namespace) -> List[NistWriterDataset]:
    data_root = Path(args.data_root)
    class_dir = data_root / "by_class"

    print("--- Data Preparation Start ---")
    if not class_dir.exists():
        print(f"[Error] Directory not found: {class_dir}")
        return []

    writers_data: Dict[str, Dict[str, List[Any]]] = {}
    hex_to_label: Dict[str, int] = {}

    for i in range(10):
        hex_to_label[f"{0x30 + i:02x}"] = i
    for i in range(26):
        hex_to_label[f"{0x41 + i:02x}"] = 10 + i
    for i in range(26):
        hex_to_label[f"{0x61 + i:02x}"] = 36 + i

    for hex_code, label in tqdm(hex_to_label.items(), desc="Parsing .mit files"):
        target_dir = class_dir / hex_code
        if not target_dir.exists():
            target_dir = class_dir / hex_code.upper()
        if not target_dir.exists():
            continue

        for mit_file in target_dir.rglob("*.mit"):
            mit_stem = mit_file.stem
            try:
                with mit_file.open("r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    filename = parts[0]
                    orig_path = parts[1]

                    writer_id = "unknown"
                    for comp in orig_path.split("/"):
                        if comp.startswith("f") and any(c.isdigit() for c in comp):
                            writer_id = comp
                            break
                    if writer_id == "unknown":
                        continue

                    candidate = mit_file.parent / mit_stem / filename
                    target = (
                        candidate
                        if candidate.exists()
                        else (mit_file.parent / filename)
                    )

                    if target.exists():
                        if writer_id not in writers_data:
                            writers_data[writer_id] = {"images": [], "labels": []}
                        writers_data[writer_id]["images"].append(str(target))
                        writers_data[writer_id]["labels"].append(label)
            except Exception:
                pass

    print(f"Found {len(writers_data)} unique writers.")

    writers_datasets: List[NistWriterDataset] = []
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    sorted_writers = sorted(
        writers_data.items(), key=lambda x: len(x[1]["images"]), reverse=True
    )

    count = 0
    for w_id, data in sorted_writers:
        images: List[str] = data["images"]
        labels: List[int] = data["labels"]

        if len(images) > args.max_imgs_per_writer:
            indices = np.random.permutation(len(images))[: args.max_imgs_per_writer]
            images = [images[i] for i in indices]
            labels = [labels[i] for i in indices]

        if len(images) > 10:
            ds = NistWriterDataset(w_id, images, labels, transform)
            writers_datasets.append(ds)
            count += 1
            if count >= args.max_writers:
                break

    print(f"Prepared {len(writers_datasets)} writer datasets.")
    return writers_datasets


# --- モデル定義 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 62):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# --- クライアント定義 ---
class Client:
    def __init__(self, dataset: NistWriterDataset, args: argparse.Namespace):
        self.dataset = dataset
        self.args = args
        self.dataloader = DataLoader(
            dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True
        )
        self.device = torch.device(args.device)

    def train(
        self, global_models: List[nn.Module]
    ) -> Tuple[int, Dict[str, torch.Tensor], float]:
        best_idx: int = 0
        best_loss: float = float("inf")
        criterion = nn.CrossEntropyLoss()

        try:
            x_check, y_check = next(iter(self.dataloader))
        except StopIteration:
            return 0, global_models[0].state_dict(), 0.0

        x_check, y_check = x_check.to(self.device), y_check.to(self.device)

        with torch.no_grad():
            for i, model in enumerate(global_models):
                model.eval()
                output = model(x_check)
                loss = criterion(output, y_check).item()
                if loss < best_loss:
                    best_loss = loss
                    best_idx = i

        model = copy.deepcopy(global_models[best_idx])
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr)

        epoch_loss: float = 0.0
        batches: int = 0

        for _ in range(self.args.local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss_val = criterion(out, y)
                loss_val.backward()
                optimizer.step()
                epoch_loss += loss_val.item()
                batches += 1

        avg_loss = epoch_loss / batches if batches > 0 else 0.0
        return best_idx, model.state_dict(), avg_loss


# --- 集約・評価関数 ---
def aggregate(
    global_models: List[nn.Module],
    updates: List[Tuple[int, Dict[str, torch.Tensor], float]],
) -> None:
    if not updates:
        return

    center_updates: Dict[int, List[Dict[str, torch.Tensor]]] = {
        i: [] for i in range(len(global_models))
    }
    for idx, w, _ in updates:
        center_updates[idx].append(w)

    for i in range(len(global_models)):
        if center_updates[i]:
            base = copy.deepcopy(center_updates[i][0])
            for key in base:
                for j in range(1, len(center_updates[i])):
                    base[key] += center_updates[i][j][key]
                base[key] = torch.div(base[key], len(center_updates[i]))
            global_models[i].load_state_dict(base)


def evaluate_global_accuracy(
    global_models: List[nn.Module], clients: List[Client], args: argparse.Namespace
) -> Tuple[float, float]:
    correct: int = 0
    total: int = 0
    total_loss: float = 0.0
    criterion = nn.CrossEntropyLoss()
    device = torch.device(args.device)

    for client in clients:
        test_loader = DataLoader(
            client.dataset, batch_size=args.batch_size, shuffle=False
        )
        try:
            x_check, y_check = next(iter(test_loader))
        except StopIteration:
            continue

        x_check, y_check = x_check.to(device), y_check.to(device)
        best_idx: int = 0
        best_val_loss: float = float("inf")

        with torch.no_grad():
            for i, model in enumerate(global_models):
                model.eval()
                loss = criterion(model(x_check), y_check).item()
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_idx = i

        target_model = global_models[best_idx]
        target_model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = target_model(x)
                loss_val = criterion(out, y)
                total_loss += loss_val.item() * x.size(0)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss


# --- メイン処理 ---
def main() -> None:
    args = get_args()

    # 1. 実験IDとディレクトリ作成
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_base = Path(args.log_base_dir)
    save_dir = log_base / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    config_file = save_dir / "config.json"
    csv_file = save_dir / "log.csv"
    # 追加: 所属記録用CSV
    allocation_file = save_dir / "client_allocation.csv"

    print(f"Device: {args.device}")
    print(f"Experiment Directory: {save_dir}")
    print("-" * 30)

    # 2. 設定保存
    with config_file.open("w") as f:
        json.dump(vars(args), f, indent=4)

    # 3. データ準備
    datasets = prepare_data_final(args)
    if not datasets:
        print("No datasets created.")
        return

    clients = [Client(d, args) for d in datasets]
    models = [
        SimpleCNN(NUM_CLASSES).to(torch.device(args.device))
        for _ in range(args.num_centers)
    ]

    print(
        f"\nStart Training with {len(clients)} writers. (Centers: {args.num_centers})"
    )

    # 4. 学習ループ
    for r in range(args.rounds):
        updates: List[Tuple[int, Dict[str, torch.Tensor], float]] = []
        train_losses: List[float] = []

        # ユーザー所属記録用 (1行分のデータ)
        round_allocations = {"Round": r + 1}

        for c in tqdm(clients, desc=f"Round {r + 1}/{args.rounds}", leave=False):
            center_idx, weights, loss = c.train(global_models=models)
            updates.append((center_idx, weights, loss))
            train_losses.append(loss)

            # どのユーザーがどのセンターを選んだか記録
            round_allocations[c.dataset.writer_id] = center_idx

        # クラスタ人数の集計
        center_counts = {i: 0 for i in range(args.num_centers)}
        for idx, _, _ in updates:
            center_counts[idx] += 1

        aggregate(models, updates)

        acc, val_loss = evaluate_global_accuracy(models, clients, args)
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

        # 画面表示
        counts_str = ", ".join([f"C{k}: {v}" for k, v in center_counts.items()])
        print(
            f"Round {r + 1:02d}/{args.rounds} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2f}% | Clients: [{counts_str}]"
        )

        # 5. 学習ログ保存 (log.csv)
        record = {
            "Round": r + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": val_loss,
            "Accuracy": acc,
        }
        for k, v in center_counts.items():
            record[f"Center_{k}_Clients"] = v

        df_log = pd.DataFrame([record])
        if not csv_file.exists():
            df_log.to_csv(csv_file, index=False)
        else:
            df_log.to_csv(csv_file, mode="a", header=False, index=False)

        # 6. ユーザー所属ログ保存 (client_allocation.csv)
        # カラム順序を整える: Round, writer1, writer2, ...
        # 初回だけカラム順序を確定させれば、あとはその順で保存される
        df_alloc = pd.DataFrame([round_allocations])

        # Writer IDのカラム順を固定したい場合（任意ですが見やすいため）
        if r == 0:
            cols = ["Round"] + sorted(
                [k for k in round_allocations.keys() if k != "Round"]
            )
            df_alloc = df_alloc[cols]
            df_alloc.to_csv(allocation_file, index=False)
        else:
            # 2回目以降は、初回と同じカラム順で追記する（Writerが増減しない前提）
            # pandasのto_csv(mode='a')は列順を保証しないことがあるので、既存ファイルの列順に合わせる処理を入れるのが安全だが、
            # 今回はWriter固定なので辞書のキー順が崩れなければ概ね大丈夫。
            # 安全のため読み込んで列順を合わせるのがベストだが、重くなるので単純追記にする。
            # ただし、DataFrame作成時に辞書のキー順が変わる可能性があるので、列指定推奨。

            # 既存ファイルのヘッダーを読んで列順を取得
            header = pd.read_csv(allocation_file, nrows=0).columns.tolist()
            df_alloc = df_alloc[header]
            df_alloc.to_csv(allocation_file, mode="a", header=False, index=False)

    print("\nTraining Finished.")
    print(f"Logs saved in: {save_dir}")


if __name__ == "__main__":
    main()
