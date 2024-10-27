# Card Classification Project

This project is an image classification model to classify playing cards using PyTorch. It includes scripts for training, evaluating, and visualizing the model’s predictions.

### Project Overview

This project is designed to classify images of playing cards built with PyTorch. The project includes the following major components:
- Training the model using images of playing cards.
- Evaluating the model on a test set.
- Visualizing model predictions on sample images.


### Setting Up the Environment

1. **Create and activate the environment:**
    ```bash
    conda create -n yourenv python=3.11.10
    conda activate yourenv
    ```

2. **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install additional packages using Conda:**
    ```bash
    conda install -c conda-forge ipywidgets
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    ```
