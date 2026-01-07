import argparse
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # 基本設定
    num_centers: int = 2
    rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.05
    frac: float = 0.1  # クライアントのサンプリング割合
    seed: int = 42  # 再現性のためのシード

    # データ設定
    max_writers: int = 100
    max_imgs_per_writer: int = 100
    data_root: str = "./data/nist/extracted"
    noniid_type: str = "writer"  # "writer" or "dirichlet"
    alpha: float = 0.5  # Dirichlet concentration parameter

    # EM algorithm設定 (Multi-Center FL paper)
    em_temperature: float = (
        1.0  # Temperature for soft assignment (lower = harder assignment)
    )

    # 高速化設定
    num_workers: int = 0  # GPUなら0推奨
    pin_memory: bool = False  # データが既にGPUにある場合はFalse

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
            "--frac",
            type=float,
            default=defaults.frac,
            help="Fraction of clients to sample per round",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=defaults.seed,
            help="Random seed for reproducibility",
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
        # Note: Official FEMNIST uses NIST SD19 (by_class and by_write).
        # This implementation uses 'by_class' and extracts writer info from .mit files,
        # which yields the same natural writer partitioning.
        parser.add_argument(
            "--noniid-type",
            type=str,
            default=defaults.noniid_type,
            choices=["writer", "dirichlet"],
            help="Type of non-IID partitioning (writer: natural, dirichlet: synthetic)",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=defaults.alpha,
            help="Dirichlet concentration parameter (smaller alpha = more non-IID)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=defaults.num_workers,
            help="Number of workers",
        )

        # --- pin_memory の自動判定ロジック ---
        # 1. 相互排他グループを作成（オンとオフを同時に指定できないようにする）
        group = parser.add_mutually_exclusive_group()

        # 2. default=None にして、「指定なし」の状態を作れるようにする
        group.add_argument(
            "--pin-memory",
            action="store_true",
            default=None,
            help="Force enable pin_memory",
        )
        group.add_argument(
            "--no-pin-memory",
            action="store_false",
            dest="pin_memory",
            help="Force disable pin_memory",
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

        # 3. 指定がなければ device に応じて自動決定
        if args.pin_memory is None:
            # device名に "cuda" が含まれていれば、データがGPUにあるためPin Memoryは不要(エラーになる)
            args.pin_memory = False
        elif "cuda" in args.device and args.pin_memory:
            # 明示的に有効にされた場合でも、CUDAなら無効化しないとエラーになるが、
            # ユーザーの意図を尊重して警告を出すか、強制無効化するか。
            # ここでは安全のため強制無効化する
            print("Warning: pin_memory forced to False because data is on GPU.")
            args.pin_memory = False

        return cls(**vars(args))
