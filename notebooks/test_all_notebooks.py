"""Command line script to run all Jupyter notebooks in the current directory and check for errors.

Example:

    .. code-block:: bash
        conda activate da-tools
        cd notebooks
        python test_all_notebooks.py

This will run every Jupyter notebook in the current directory and throw an error if any notebook fails to execute any cell.
The script will skip files that are not Jupyter notebooks (i.e., do not end with `.ipynb`).
"""

import logging
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook_for_errors(path: str, timeout: int = 600, as_version: int = nbformat.NO_CONVERT) -> bool:
    """Executes a Jupyter notebook checking for errors.

    Args:
        path (str): Path to the Jupyter notebook.
        timeout (int): Timeout for notebook execution in seconds.
        as_version (int): The version of the notebook format to use. defaults to NO_CONVERT.

    Returns:
        bool: True if the notebook executed without errors, False otherwise.
    """
    logging.info(f"Executing notebook: {path}")
    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=as_version)

        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(path)}})

        logging.info(f"Notebook '{path}' executed successfully without errors.")
        return True
    except Exception as e:
        logging.error(f"Error executing notebook '{path}': {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = os.getcwd()
    notebooks = os.listdir(path)
    for notebook in notebooks:
        if notebook.endswith(".ipynb"):
            notebook_path = os.path.join(path, notebook)
            success = run_notebook_for_errors(notebook_path)
        else:
            logging.warning(f"Skipping non-notebook file: {notebook}")
