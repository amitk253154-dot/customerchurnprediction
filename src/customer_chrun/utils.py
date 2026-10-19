import os
import pickle
import sys

from src.customer_chrun.exception import CustomException


def save_object(file_path, obj):
    """Save any Python object (for example a fitted preprocessor or model) with pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load an object saved with save_object."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)