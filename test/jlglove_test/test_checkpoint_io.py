import pickle
from pathlib import Path

import pytest
import torch

from jlglove.rep import CustomTorchCheckpointIO


def _create_marker(marker_path: str) -> dict[str, object]:
    Path(marker_path).touch()
    return {}


class _ExecutableCheckpoint:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return _create_marker, (str(self.marker_path),)


def test_checkpoint_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.Adagrad([parameter])
    checkpoint_io = CustomTorchCheckpointIO()
    checkpoint_io.save_checkpoint(
        {
            "epoch": 2,
            "state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer_states": [optimizer.state_dict()],
        },
        checkpoint_path,
    )

    checkpoint = checkpoint_io.load_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 2
    torch.testing.assert_close(
        checkpoint["state_dict"]["weight"],
        torch.tensor([1.0, 2.0]),
    )
    assert len(checkpoint["optimizer_states"]) == 1


def test_load_checkpoint_rejects_executable_pickle(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    marker_path = tmp_path / "payload-executed"
    torch.save(_ExecutableCheckpoint(marker_path), checkpoint_path)

    with pytest.raises(pickle.UnpicklingError):
        CustomTorchCheckpointIO().load_checkpoint(checkpoint_path)

    assert not marker_path.exists()
