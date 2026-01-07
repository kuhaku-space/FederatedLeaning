import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import NistWriterDataset


# --- Client 定義 ---
class Client:
    def __init__(self, dataset: NistWriterDataset, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.writer_id = dataset.writer_id

        # --- データの分割 (80% Train, 20% Test) ---
        total_len = len(dataset)
        train_len = int(total_len * 0.8)
        test_len = total_len - train_len

        if train_len == 0:
            train_len = total_len
            test_len = 0

        # seed固定
        self.train_dataset, self.test_dataset = random_split(
            dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42)
        )

        # DataLoaderの作成
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=min(config.batch_size, len(self.train_dataset)),
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory if torch.cuda.is_available() else False,
        )

        if test_len > 0:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory if torch.cuda.is_available() else False,
            )
        else:
            self.test_loader = None

    def participate(
        self, global_models: list[nn.Module]
    ) -> tuple[list[float], dict[str, torch.Tensor], float]:
        """
        E-step: ソフト割り当て確率を計算し、最良モデルで学習

        Returns:
            tuple: (確率リスト, 更新後の重み, 平均損失)
        """
        probs, _best_idx, model = self._compute_center_probabilities(global_models)

        # 学習
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr)

        epoch_loss = 0.0
        batch_count = 0

        for _ in range(self.cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        return probs, model.state_dict(), avg_loss

    def _compute_center_probabilities(
        self, global_models: list[nn.Module]
    ) -> tuple[list[float], int, nn.Module]:
        """
        E-step: 全センターに対するソフト割り当て確率を計算

        論文のアルゴリズムに基づき、softmax(-loss/τ) で確率を計算

        Note:
            公式実装 (https://github.com/caifederated/multi-center-fed-learning) では
            クライアントのモデル重みに対するK-Meansクラスタリングが使用されていますが、
            本実装では論文(ArXiv:2005.01026)の記述通り、損失値に基づくEMアルゴリズムを採用しています。
        """
        # 学習データの最初のバッチを使って判断する
        try:
            x_check, y_check = next(iter(self.train_loader))
        except StopIteration:
            # データがない場合は均等確率
            num_centers = len(global_models)
            uniform_probs = [1.0 / num_centers] * num_centers
            return uniform_probs, 0, copy.deepcopy(global_models[0])

        x_check, y_check = x_check.to(self.device), y_check.to(self.device)

        # 各センターの損失を計算
        losses: list[float] = []
        with torch.no_grad():
            for model in global_models:
                model.eval()
                output = model(x_check)
                loss = self.criterion(output, y_check).item()
                losses.append(loss)

        # Softmax on negative losses (lower loss = higher probability)
        # p_k = exp(-loss_k / τ) / Σ_j exp(-loss_j / τ)
        temperature = self.cfg.em_temperature
        neg_losses = [-loss_val / temperature for loss_val in losses]
        max_nl = max(neg_losses)  # for numerical stability
        exp_vals = [math.exp(nl - max_nl) for nl in neg_losses]
        total = sum(exp_vals)
        probs = [ev / total for ev in exp_vals]

        # 最良のセンターを選択（学習には最も確率が高いモデルを使用）
        best_idx = probs.index(max(probs))

        return probs, best_idx, copy.deepcopy(global_models[best_idx])
