import os
import torch
from malaria.data import MalariaDataModule
from malaria.cnn.model import MalariaLitModel
from malaria.utils import save_submission

# Input data paths
DATA_DIR = "data"
RESULTS_DIR = "results/malaria_cnn"
MODEL_PATH = "lightning_logs/malaria_cnn/version_0/checkpoints/last.ckpt"
MODEL_VERSION = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))
SUBMISSION_PATH = os.path.join(RESULTS_DIR, f"submission_{MODEL_VERSION}.csv")

# Hyperparameters
BATCH_SIZE = 32

if __name__ == "__main__":
    # Data
    data_module = MalariaDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
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
            logits, _ = model.forward(images)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            img_names.extend(names)
    save_submission(img_names, preds, save_path=SUBMISSION_PATH)
    print(f"Test predictions saved to {SUBMISSION_PATH}")
