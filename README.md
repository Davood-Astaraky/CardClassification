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

### Additional Notes:
- Make sure to have `conda` and `pip` installed and properly configured before running these commands.
- You can specify versions of packages in `requirements.txt` if specific versions are needed for the project.
