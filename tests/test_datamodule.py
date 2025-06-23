import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from malaria.data import MalariaDataModule


def _prepare_dummy_dataset(root: Path) -> str:
    """Create a tiny dataset with one image per class for train/val."""
    example_dir = Path(__file__).resolve().parents[1] / "examples"
    images = {
        "healthy": example_dir / "normal_cell.png",
        "infected": example_dir / "infected_cell.png",
    }
    for split in ("train", "val"):
        for cls, img_path in images.items():
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            # copy image
            shutil.copy(img_path, cls_dir / f"{cls}.png")
    return str(root)


def test_dataloaders(tmp_path):
    data_dir = _prepare_dummy_dataset(tmp_path)
    dm = MalariaDataModule(data_dir=data_dir, batch_size=2, num_workers=0, img_size=64)
    dm.setup()
    loader = dm.train_dataloader()
    images, labels = next(iter(loader))
    assert images.shape == (2, 3, 64, 64)
    assert labels.shape == (2,)
    assert dm.num_classes() == 2
