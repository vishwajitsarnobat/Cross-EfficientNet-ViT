# Cross-Efficient-ViT Deepfake Detection

This project provides a complete, end-to-end pipeline for training a hybrid EfficientNet-ViT model to detect deepfaked media (videos and images). It includes scripts for preprocessing, training, evaluation, and a powerful prediction tool.


## ✨ Features

-   **Generalized Pipeline:** Works seamlessly for both videos and images, controlled by a simple `MODE` switch in the config.
-   **Automated Preprocessing:** Intelligently runs face detection and cropping, skipping any work that has already been completed.
-   **Data Auditing Tool:** Includes a script to generate a report on your dataset, showing exactly which files failed face detection.
-   **Robust Training:** Includes validation, early stopping, per-epoch confusion matrices, and generates loss/accuracy plots.
-   **Powerful Evaluation:** A single, unified script (`evaluate.py`) can analyze a single file for a detailed report or evaluate an entire test set.
-   **Modern & Fast Tooling:** Uses `uv` for a lightning-fast, reproducible setup.

## 🚀 Setup

### Prerequisites

-   Python 3.9+
-   Git
-   NVIDIA GPU with CUDA installed (e.g., CUDA 12.1)

### Installation

1.  **Get `uv` (if you don't have it):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository and enter it:**
    ```bash
    git clone <repository-url>
    cd Cross-Efficient-ViT
    ```

3.  **Activate the environment and install everything with `uv sync`:**
    ```bash
    # Create and activate a virtual environment
    uv venv
    source .venv/bin/activate

    # This single command installs all dependencies from the lockfile
    uv sync
    ```
    That's it! Everything is now installed and ready to go.

## ⚙️ Configuration

Before running, you **must** configure your project by editing `config.py`.

1.  **Set the `MODE`:** Choose `"image"` or `"video"` depending on your dataset.
    ```python
    MODE = "image"
    ```
2.  **Set Dataset Paths:** Update the lists with the absolute paths to your dataset folders.
    ```python
    # Example for image mode
    real_images_paths = ["/path/to/your/real_images_folder"]
    fake_images_paths = ["/path/to/your/fake_images_folder"]
    ```
3.  **Toggle Preprocessing:** For the first run, ensure this is `True`.
    ```python
    run_preprocessing = True
    ```

## 🛠️ Recommended Workflow

The `uv run <script_name>` command executes scripts within your virtual environment.

### 1. Preprocess the Data

The first step is always to run the preprocessing pipeline. This is done automatically by the training script.

```bash
# This command will first run face detection and crop extraction
uv run train.py
```
**Note:** The training may fail if your dataset is too small after this step. This is expected. The primary goal of this first run is to generate the preprocessing artifacts.

### 2. Audit Your Dataset (Crucial Step)

After the first run, some of your files may have failed face detection. The audit script tells you exactly which ones.

```bash
uv run audit_preprocessing.py
```
-   **Output:**
    -   A `preprocessing_audit_report.csv` file in your `results/<mode>/` directory.
    -   Open this CSV. The files with `faces_found = 0` are your problematic data.
    -   **Action:** Review these files. Remove them from your dataset if they are invalid (no faces, corrupted, etc.). A clean dataset is the key to a good model.

### 3. Train the Model

After cleaning your dataset, re-run the training script. It will intelligently skip the already-processed files and only process the new ones (if any).

```bash
uv run train.py
```
-   **Output:**
    -   Model checkpoints (`best_model.pth`) in the `models/<mode>/` directory.
    -   Training plots and reports in the `results/<mode>/` directory.

### 4. Evaluate and Predict

The `evaluate.py` script serves two purposes.

**A) Evaluate on a Test Set:**
Point it to a directory containing `real` and `fake` subfolders (the paths should be set in `config.py`).
```bash
# The script will use the test paths defined in your config.py
uv run evaluate.py /path/to/your/test_dataset_root_folder/
```
-   **Output:** A final classification report and confusion matrix printed to the console and saved to the results folder.

**B) Predict on a Single File:**
Point it to a single video or image file.
```bash
# Basic usage for an image or video
uv run evaluate.py /path/to/your/media_file.mp4
```
-   **Output:**
    1.  **Console:** A clean verdict and confidence score.
    2.  **Reports:** A visual report (for videos) and a detailed JSON file saved in `results/<mode>/`.