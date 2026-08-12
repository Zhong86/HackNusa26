"""
Load/save the trained model artifact. Kept separate so classifier.py
doesn't care about file formats.
"""
import joblib

def load_model(path: str):
    return joblib.load(path)

def save_model(model, path: str):
    joblib.dump(model, path)