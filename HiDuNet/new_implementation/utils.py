

from pathlib import Path
import csv
import torch

def get_root_dir(project_name: str = "text-steganalysis") -> Path:
    """
    Get the root directory of the project by searching for the folder with the given project name.
    Works in scripts, IDEs and Jupyter notebooks.

    Arguments:
        project_name: Name of the root project folder.
    Return:
        Path to the project root directory.
    """
    try:
        # Try to get the directory of the current file
        current_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., Jupyter notebook)
        current_dir = Path.cwd().resolve()

    # Walk up the directory tree until we find the project root
    for parent in [current_dir] + list(current_dir.parents):
        if parent.name == project_name:
            return parent

    # If not found, raise an error
    raise FileNotFoundError(f"Project root folder '{project_name}' not found from {current_dir}")

def get_device() -> str:
    """
    Get the available device for PyTorch (Nvidia GPU if available, otherwise Apple Silicon
    if available, otherwise CPU)
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def txt_to_tsv(natural_txt_path: Path, stego_txt_path: Path, output_tsv_path: Path = None):
    """
    Convert a txt file of natural texts and a txt file of stego texts into a single tsv file
    Args:
        natural_txt_path: Path to the txt file of natural texts
        stego_txt_path: Path to the txt file of stego texts
        output_tsv_path: Path to the output tsv file. If None, will be saved in the same directory as natural_txt_path
    """
    if output_tsv_path is None:
        output_tsv_path = natural_txt_path.parent / "data.tsv"

    with open(natural_txt_path, 'r', encoding='utf-8') as f:
        natural_texts = f.readlines()

    with open(stego_txt_path, 'r', encoding='utf-8') as f:
        stego_texts = f.readlines()

    with open(output_tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['text', 'label'])  # Header
        for text in natural_texts:
            writer.writerow([text.strip(), '0'])  # Natural texts labeled as '0'
        for text in stego_texts:
            writer.writerow([text.strip(), '1'])  # Stego texts labeled as '1'