import os
import torch
from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel
from malaria.utils import save_submission

DATASET_DIR = "dataset"
TEST_IMG = os.path.join(DATASET_DIR, "test_images")
TRAIN_DATA = os.path.join(DATASET_DIR, "train_data.csv")
TRAIN_IMG = os.path.join(DATASET_DIR, "train_images")
OUTPUTS_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUTS_DIR, "malaria_cnn.pth")
SUBMISSION_PATH = os.path.join(OUTPUTS_DIR, "submission.csv")
BATCH_SIZE = 32

if __name__ == "__main__":
    # Data
    data_module = MalariaDataModule(
        train_csv=TRAIN_DATA,
        train_img_dir=TRAIN_IMG,
        test_img_dir=TEST_IMG,
        batch_size=BATCH_SIZE
    )
    test_loader = data_module.test_dataloader()

    # Model
    model = MalariaLitModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference
    img_names = []
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            images, names = batch
            images = images.to(device)
            logits, _ = model(images)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            img_names.extend(names)
    save_submission(img_names, preds, save_path=SUBMISSION_PATH)
    print(f"Test predictions saved to {SUBMISSION_PATH}")
