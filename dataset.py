import logging
from pathlib import Path
from typing import Any, Optional, override

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


# --- Dataset 定義 ---
class NistWriterDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        writer_id: str,
        images: list[str],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
        device: str = "cpu",
    ):
        self.writer_id = writer_id

        loaded_images = []
        loaded_labels = []

        for path, label in zip(images, labels):
            try:
                with Image.open(path) as img:
                    img = img.convert("L")
                    if transform:
                        img = transform(img)

                    if not isinstance(img, torch.Tensor):
                        img = transforms.ToTensor()(img)

                    loaded_images.append(img)
                    loaded_labels.append(label)

            except (UnidentifiedImageError, OSError) as e:
                logger.warning(f"Error loading image {path}: {e}")
                continue

        if not loaded_images:
            logger.warning(f"Writer {writer_id} has no valid images.")
            # 型チェッカーのために明示的に型注釈を行う
            self.data: torch.Tensor = torch.empty(0)
            self.targets: torch.Tensor = torch.empty(0)
            return

        # データをGPUメモリへ転送
        # ここで明示的に型注釈 (: torch.Tensor) をつけることで Unknown を回避
        self.data: torch.Tensor = torch.stack(loaded_images).to(device)
        self.targets: torch.Tensor = torch.tensor(loaded_labels, dtype=torch.long).to(
            device
        )

    def __len__(self) -> int:
        return len(self.data)

    # 修正箇所: 戻り値の型ヒントを Tensor, Tensor に変更
    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


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

    def _generate_hex_label_map(self) -> dict[str, int]:
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

    def prepare_data(self) -> list[NistWriterDataset]:
        data_root = Path(self.cfg.data_root)
        class_dir = data_root / "by_class"

        logger.info("--- Data Preparation Start ---")
        if not class_dir.exists():
            logger.error(f"Directory not found: {class_dir}")
            return []

        writers_data: dict[str, dict[str, list[Any]]] = {}
        hex_to_label = self._generate_hex_label_map()

        # .mit ファイルの解析
        for hex_code, label in tqdm(
            hex_to_label.items(), desc="Parsing classes", ncols=80
        ):
            # ディレクトリ検索（大文字小文字対応）
            target_dir = class_dir / hex_code
            if not target_dir.exists():
                target_dir = class_dir / hex_code.upper()
            if not target_dir.exists():
                continue

            self._parse_mit_files_in_dir(target_dir, label, writers_data)

        logger.info(f"Found {len(writers_data)} unique writers.")
        if self.cfg.noniid_type == "dirichlet":
            return self._create_dirichlet_datasets(writers_data)
        else:
            return self._create_datasets(writers_data)

    def _parse_mit_files_in_dir(
        self,
        target_dir: Path,
        label: int,
        writers_data: dict[str, dict[str, list[Any]]],
    ):
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

            except Exception:
                # ファイル破損等はスキップ
                pass

    @staticmethod
    def _extract_writer_id(orig_path: str) -> Optional[str]:
        for comp in orig_path.split("/"):
            # "f"で始まり、数字が含まれるディレクトリ名をWriterIDとみなす
            if comp.startswith("f") and any(c.isdigit() for c in comp):
                return comp
        return None

    def _create_datasets(
        self, writers_data: dict[str, dict[str, list[Any]]]
    ) -> list[NistWriterDataset]:
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
                ds = NistWriterDataset(
                    w_id, images, labels, self.transform, self.cfg.device
                )
                writers_datasets.append(ds)
                count += 1
                if count >= self.cfg.max_writers:
                    break

        logger.info(f"Prepared {len(writers_datasets)} writer datasets.")
        return writers_datasets

    def _create_dirichlet_datasets(
        self, writers_data: dict[str, dict[str, list[Any]]]
    ) -> list[NistWriterDataset]:
        """
        Dirichlet分布を用いたNon-IID分割。
        各クライアントには特定の1人のライターのデータのみを割り当て、
        その中でのクラス分布をDirichlet分布に基づいてサンプリング（間引き）することで不均衡を作ります。
        """
        # データ数が多い順にライターを選別
        sorted_writers = sorted(
            writers_data.items(), key=lambda x: len(x[1]["images"]), reverse=True
        )
        selected_writers = sorted_writers[: self.cfg.max_writers]
        num_clients = len(selected_writers)
        alpha = self.cfg.alpha
        num_classes = self.cfg.num_classes

        # クラスごとに、全クライアント（ライター）への配分比率をサンプリング
        # proportions[k][i] は クラスk を クライアントi にどれだけ割り当てるかの比率
        proportions = []
        for k in range(num_classes):
            proportions.append(np.random.dirichlet([alpha] * num_clients))

        datasets = []
        for i, (w_id, data) in enumerate(selected_writers):
            images = np.array(data["images"])
            labels = np.array(data["labels"])

            final_indices = []
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                if len(idx_k) == 0:
                    continue

                # このライターが持つクラスkのうち、Dirichlet比率に基づいて採用する数を決定
                # p * num_clients の期待値は1。alphaが小さいと特定のクライアントに集中する。
                p = proportions[k][i]
                target_num = int(len(idx_k) * p * num_clients)
                num_to_keep = min(len(idx_k), target_num)

                # 完全に0にならないよう、比率がある程度あれば最低1つは残す
                if p > (1.0 / (num_clients * 5)) and num_to_keep == 0:
                    num_to_keep = 1

                if num_to_keep > 0:
                    sel = np.random.choice(idx_k, num_to_keep, replace=False)
                    final_indices.extend(sel)

            if len(final_indices) < 5:
                continue

            # 最大枚数制限
            if len(final_indices) > self.cfg.max_imgs_per_writer:
                final_indices = np.random.choice(
                    final_indices, self.cfg.max_imgs_per_writer, replace=False
                )

            ds = NistWriterDataset(
                w_id,
                images[final_indices].tolist(),
                labels[final_indices].tolist(),
                self.transform,
                self.cfg.device,
            )
            datasets.append(ds)
        logger.info(
            f"Prepared {len(datasets)} Dirichlet-partitioned datasets (One writer per client)."
        )
        return datasets
