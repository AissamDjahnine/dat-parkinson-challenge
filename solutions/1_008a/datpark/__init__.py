"""Shared code for the DaT Parkinson's Challenge.

This package is imported by BOTH the training scripts and the submission's
main.py, so that inference reproduces training preprocessing exactly. Keep it
dependency-light (numpy / scipy / nibabel only) and free of any training-time
state.
"""

from datpark.preprocess import PreprocessConfig, preprocess, preprocess_array

__all__ = ["PreprocessConfig", "preprocess", "preprocess_array"]
