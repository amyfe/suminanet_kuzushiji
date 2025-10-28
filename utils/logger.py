import time
from pathlib import Path
from typing import Sequence, Union, Optional

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt


class SimpleLogger:
    """Simple console logger that measures epoch time."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None

    def start_epoch(self, epoch: int) -> None:
        print(f"\n--- Epoch {epoch+1} ---")
        self.start_time = time.time()

    def end_epoch(self, epoch: int, loss: float) -> None:
        if self.start_time is None:
            print(f"Epoch {epoch+1} finished | Loss: {loss:.4f}")
            return
        elapsed = time.time() - self.start_time
        print(f"Epoch {epoch+1} finished | Loss: {loss:.4f} | Time: {elapsed:.1f}s")


class TrainLogger:
    """Stores epoch-level metrics and can save a simple plot.

    save_plot accepts either a string path or a pathlib.Path and will create
    the parent directory if it does not exist. If no data was logged, the
    function writes a small empty plot and returns gracefully.
    """

    def __init__(self) -> None:
        self.epoch_losses: list[float] = []
        self.epoch_acc: list[float] = []

    def log(self, loss: float, acc: float) -> None:
        self.epoch_losses.append(loss)
        self.epoch_acc.append(acc)

    def save_plot(self, out_path: Union[str, Path] = "training_curve.png") -> None:
        out_path = Path(out_path)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        # Guard against empty data
        losses: Sequence[float] = self.epoch_losses or [0.0]
        acc: Sequence[float] = self.epoch_acc or [0.0]

        try:
            fig, ax1 = plt.subplots()
            ax1.plot(losses, label="Loss", color="tab:red")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color="tab:red")
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.plot(acc, label="Accuracy", color="tab:blue")
            ax2.set_ylabel("Accuracy", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")

            fig.tight_layout()
            plt.title("Training Progress")
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved training curve to {out_path}")
        except Exception as e:
            print(f"Failed to save training plot to {out_path}: {e}")