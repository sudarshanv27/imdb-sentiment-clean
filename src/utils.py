from pathlib import Path
import pickle
import joblib


def save_pickle(data, file_path):
    """
    Save Python object using pickle.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    print(f"Saved successfully: {file_path}")


def load_pickle(file_path):
    """
    Load Python object from pickle file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "rb") as file:
        data = pickle.load(file)

    return data


def save_model(model, file_path):
    """
    Save machine learning model using joblib.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, file_path)

    print(f"Model saved successfully: {file_path}")


def load_model(file_path):
    """
    Load machine learning model using joblib.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Model not found: {file_path}"
        )

    model = joblib.load(file_path)

    return model


def check_file_exists(file_path):
    """
    Check whether a file exists.
    """

    file_path = Path(file_path)

    return file_path.exists()


def print_section(title):
    """
    Print a formatted section title.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)