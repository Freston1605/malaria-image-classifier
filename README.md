# Malaria Cell Classification (Deep Learning Homework)

This project is a deep learning solution for classifying red blood cell images as healthy or infected with malaria, using a Convolutional Neural Network (CNN) implemented in PyTorch and PyTorch Lightning. It was developed as part of a Neural Networks & Deep Learning in Science course at CMU.

## Project Overview
- **Goal:** Automatically classify cell images as healthy or malaria-infected.
- **Approach:** Train a CNN on labeled cell images, evaluate on test images, and visualize learned features.
- **Technologies:** PyTorch, PyTorch Lightning, scikit-learn, matplotlib, pandas.

## Dataset
- **Training data:** `dataset/train_data.csv` (CSV with `img_name,label`), images in `dataset/train_images/`.
- **Test data:** images in `dataset/test_images/`.
- **Sample submission:** `dataset/sample_submission.csv`.

## Model Architecture
- 3 convolutional blocks (Conv2d + ReLU + MaxPool)
- Adaptive average pooling
- Fully connected layers with dropout
- Binary output (healthy/infected)

## Results
- Training metrics and t-SNE visualizations are shown below:

### Training Metrics
![Training Metrics](training_metrics.png)

### t-SNE Embedding Visualization
![t-SNE Visualization](tsne_20250223_121403.png)

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset in the `dataset/` folder as described above.
3. Run the main script:
   ```bash
   python malaria_cnn.py
   ```

## File Structure
- `malaria_cnn.py`: Main script (refactor to modules for best practice)
- `malaria/`: Python package with model, data, and utility modules
- `requirements.txt`: Python dependencies
- `training_metrics.png`: Training loss/accuracy plot
- `tsne_20250223_121403.png`: t-SNE embedding visualization
- `malaria_cnn.pth`: Trained model weights
- `submission.csv`: Test predictions

## License
MIT License

## Author
Franco (update with your full name or GitHub handle)
