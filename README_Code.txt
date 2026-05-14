Spectral Reconstruction Code
============================

This package contains the English-cleaned Python script for spectral reconstruction based on gate-resolved response curves.

Main script:
- spectral_reconstruction_code_english.py

Main functions:
- Load and normalize calibration response curves.
- Generate single-peak, multi-peak, RGB-based, or TXT-based input spectra.
- Generate or load measured photocurrent signals.
- Reconstruct spectra using Lasso and ElasticNet methods.
- Save reconstructed spectra, simulated signals, residuals, and related diagnostic outputs.

Required Python packages:
- numpy
- matplotlib
- scipy
- scikit-learn
- pillow

Input file expected by default:
- downsample_5nm_transposed.txt

The script was edited to remove Chinese comments, prompts, print messages, titles, and labels while preserving the computational workflow.
