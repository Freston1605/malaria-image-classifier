import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from malaria.cnn.model import MalariaLitModel
from malaria.yolo.model import YOLOLitModel


@st.cache_resource
def load_model(ckpt_path: str, model_type: str):
    """Load a Lightning model from a checkpoint."""
    if model_type == "cnn":
        model = MalariaLitModel.load_from_checkpoint(ckpt_path, map_location="cpu")
    else:
        model = YOLOLitModel.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def main():
    st.title("Malaria Cell Classifier")
    st.write(
        "Upload a cell image and the model will predict whether it is healthy or infected."
    )

    ckpt_path = st.text_input("Checkpoint path", "lightning_logs/.../checkpoints/last.ckpt")
    model_type = st.selectbox("Model type", ["cnn", "yolo"])

    model = None
    if ckpt_path and Path(ckpt_path).is_file():
        model = load_model(ckpt_path, model_type)
    else:
        st.warning("Provide a valid checkpoint path to load the model.")

    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if file and model:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)[0]
            pred = torch.argmax(output, dim=1).item()
            label = "infected" if pred == 1 else "healthy"
        st.image(image, caption=f"Prediction: {label}")


if __name__ == "__main__":
    main()
