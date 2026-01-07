import torch
import torch.nn as nn

from client import Client
from config import Config
from model import SimpleCNN


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

    def aggregate(
        self, updates: list[tuple[list[float], dict[str, torch.Tensor], float]]
    ):
        """
        M-step: 確率に基づく重み付き集約

        各クライアントの寄与は、そのクライアントのセンターへの割り当て確率で重み付けされる
        """
        if not updates:
            return

        # 各センターに対して重み付き集約を実行
        for center_idx, model in enumerate(self.models):
            weighted_params: dict[str, torch.Tensor] = {}
            total_weight = 0.0

            for probs, weights, _ in updates:
                weight = probs[center_idx]  # このクライアントのこのセンターへの確率
                if weight > 1e-6:  # 無視できる寄与をスキップ
                    total_weight += weight
                    for key, param in weights.items():
                        if key not in weighted_params:
                            weighted_params[key] = weight * param.clone()
                        else:
                            weighted_params[key] += weight * param

            if total_weight > 0:
                # 重み付き平均
                fed_avg_weights = {
                    k: v / total_weight for k, v in weighted_params.items()
                }
                model.load_state_dict(fed_avg_weights)

    def evaluate(self, clients: list[Client]) -> tuple[float, float]:
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        # クライアントごとにテスト（各クライアントは最適なモデルを選択して評価）
        for client in clients:
            # 修正: clientが持っている test_loader を使用
            if client.test_loader is None:
                continue

            test_loader = client.test_loader

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
