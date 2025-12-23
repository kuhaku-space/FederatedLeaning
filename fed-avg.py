import copy
from typing import Dict, List, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ==========================================
# 1. モデル定義 (論文 Source: 192, 193)
# ==========================================


class MNIST_2NN(nn.Module):
    """
    単純な多層パーセプトロン (2NN)
    2 hidden layers with 200 units each using ReLu activations
    """

    def __init__(self) -> None:
        super(MNIST_2NN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class MNIST_CNN(nn.Module):
    """
    CNN
    Two 5x5 convolution layers (32, 64 channels), 2x2 max pooling,
    fully connected layer with 512 units and ReLu, softmax output.
    """

    def __init__(self) -> None:
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


# ==========================================
# 2. データ分割ヘルパー (論文 Source: 195)
# ==========================================


def get_mnist() -> Tuple[datasets.MNIST, datasets.MNIST]:
    """MNISTデータのダウンロードと変換"""
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset_train = datasets.MNIST(
        "./data/mnist/", train=True, download=True, transform=trans
    )
    dataset_test = datasets.MNIST(
        "./data/mnist/", train=False, download=True, transform=trans
    )
    return dataset_train, dataset_test


def mnist_iid(dataset: datasets.MNIST, num_users: int) -> Dict[int, Set[int]]:
    """
    IID: データをシャッフルして各ユーザーに均等に分配
    """
    num_items = int(len(dataset) / num_users)
    dict_users: Dict[int, Set[int]] = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset: datasets.MNIST, num_users: int) -> Dict[int, np.ndarray]:
    """
    Non-IID: ラベルでソートし、200個のシャードに分割。
    各ユーザーに2つのシャードを割り当てる（結果として各ユーザーは少数の数字クラスしか持たない）。
    Source: 195
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users: Dict[int, np.ndarray] = {
        i: np.array([], dtype="int64") for i in range(num_users)
    }
    idxs = np.arange(num_shards * num_imgs)
    labels: np.ndarray = dataset.targets.numpy()

    # ラベルでソート
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 各ユーザーに2つのシャードを割り当て
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


# ==========================================
# 3. Local Update (Client Side) (論文 Source: 184)
# ==========================================


class Args:
    """ハイパーパラメータ管理"""

    def __init__(self) -> None:
        self.epochs: int = 50
        self.num_users: int = 100
        self.frac: float = 0.1
        self.local_ep: int = 5
        self.local_bs: int = 10
        self.bs: int = 128
        self.lr: float = 0.01
        self.momentum: float = 0.5
        self.model: str = "mlp"
        self.dataset: str = "mnist"
        self.iid: bool = False
        self.gpu: int = 0 if torch.cuda.is_available() else -1
        self.device: torch.device = torch.device(
            "cuda:{}".format(self.gpu)
            if torch.cuda.is_available() and self.gpu != -1
            else "cpu"
        )


class LocalUpdate(object):
    def __init__(
        self, args: Args, dataset: datasets.MNIST, idxs: Union[Set[int], np.ndarray]
    ) -> None:
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # Subsetはindicesとしてsequenceを受け取るためlist化
        self.ldr_train: DataLoader = DataLoader(
            Subset(dataset, list(idxs)), batch_size=self.args.local_bs, shuffle=True
        )

    def train(self, net: nn.Module) -> Tuple[Dict[str, torch.Tensor], float]:
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss: List[float] = []
        for iter in range(self.args.local_ep):
            batch_loss: List[float] = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = (
                    images.to(self.args.device),
                    labels.to(self.args.device),
                )
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# ==========================================
# 4. Federated Averaging (Server Side) (論文 Source: 184)
# ==========================================


def FedAvg(w: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    重みの平均化 (Aggregation)
    w: list of client state_dicts
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test_img(
    net_g: nn.Module, datatest: datasets.MNIST, args: Args
) -> Tuple[float, float]:
    """グローバルモデルの評価"""
    net_g.eval()
    test_loss: float = 0
    correct: float = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)

    with torch.no_grad():  # 推論時は勾配計算不要
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction="sum").item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

    test_loss /= len(data_loader.dataset)  # type: ignore
    accuracy = 100.00 * correct / len(data_loader.dataset)  # type: ignore
    return accuracy, test_loss


# ==========================================
# 5. メイン実行スクリプト
# ==========================================


def main() -> None:
    args = Args()

    # データのロード
    dataset_train, dataset_test = get_mnist()

    # データの分割
    dict_users: Union[Dict[int, Set[int]], Dict[int, np.ndarray]]
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users = mnist_noniid(dataset_train, args.num_users)

    # モデルの初期化
    net_glob: nn.Module
    if args.model == "cnn":
        net_glob = MNIST_CNN().to(args.device)
    else:
        net_glob = MNIST_2NN().to(args.device)

    print(net_glob)
    net_glob.train()

    # グローバルモデルの重みをコピー
    w_glob = net_glob.state_dict()

    # 学習履歴
    loss_train: List[float] = []
    acc_test_history: List[float] = []

    # 通信ラウンドのループ
    for iter in range(args.epochs):
        w_locals: List[Dict[str, torch.Tensor]] = []
        loss_locals: List[float] = []

        # クライアントの選択 (m = max(C * K, 1))
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # 選択された各クライアントでローカル学習
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # 重みの集約 (FedAvg)
        w_glob = FedAvg(w_locals)

        # グローバルモデルを更新
        net_glob.load_state_dict(w_glob)

        # ロスの記録
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # 評価
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)

        print(
            f"Round {iter:3d}, Average Loss: {loss_avg:.3f}, Test Accuracy: {acc_test:.2f}%"
        )

    # 結果のプロット
    plt.figure()
    plt.plot(range(len(acc_test_history)), acc_test_history)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig("fedavg_result.png")
    print("Saved plot to fedavg_result.png")


if __name__ == "__main__":
    main()
