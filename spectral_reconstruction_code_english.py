# *********************************************************************
# 3D Reconstruction of  Spectra with Multi-Peak Calibration
# This script reconstructs spectra using multi-peak calibration.
# January 4, 2026
# Author: Linglan Guo
# This script uses a single-peak penalty to significantly improve reconstruction accuracy.
# This script uses Lasso regression and ElasticNet for spectral reconstruction.
# This script includes a deconvolution function named deconvolve_spectrum.
# Further optimization is mainly needed for multi-peak penalty design and multi-peak weight optimization.
# This script dynamically adjusts reconstruction coefficients according to the FWHM.
# This script can export and save result files.
# The NNLS part has been removed; this version uses only Lasso and ElasticNet.
# mode 1         Enter specific wavelengths (single peak); noise simulation has been added to study the influence of noise on reconstruction.
# mode 2         Set wavelength range and interval (single peak)
# mode 3         Enter multi-peak parameters (combined peak)
# mode 4         Directly load measured_signal data
# mode 5         Load
# mode 6         Load spectral information and construct photocurrent signals for spectral reconstruction
# *********************************************************************


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import splev, splrep
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNetCV
from scipy import fftpack
from scipy.signal import wiener
from sklearn.metrics import r2_score
import os
import datetime


def deconvolve_response_curves(response_curves, wavelengths, fwhm, reg_param=0.01):
    """
    Deconvolve the response curves with left-right symmetry processing.
    """
    n_detectors, n_wavelengths = response_curves.shape
    deconvolved_curves = np.zeros_like(response_curves)

    # 1. Generate a symmetric Gaussian kernel
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = wavelengths - np.mean(wavelengths)
    gaussian_kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    # 2. Perform deconvolution for each detector
    for i in range(n_detectors):
        response = response_curves[i, :]

        # Wiener deconvolution
        response_fft = fftpack.fft(response)
        kernel_fft = fftpack.fft(gaussian_kernel)

        # Calculate the power spectral density
        signal_power = np.abs(response_fft) ** 2
        kernel_power = np.abs(kernel_fft) ** 2

        # Wiener filter
        wiener_filter = np.conj(kernel_fft) / (kernel_power + reg_param)

        # Apply the filter and perform inverse Fourier transform
        deconvolved = np.real(fftpack.ifft(response_fft * wiener_filter))

        # 3. Symmetry processing
        mid_point = len(deconvolved) // 2
        if len(deconvolved) % 2 == 0:  # even length
            left_half = deconvolved[:mid_point]
            right_half = deconvolved[mid_point:]
            # Ensure equal length
            if len(right_half) > len(left_half):
                right_half = right_half[:-1]
            elif len(left_half) > len(right_half):
                left_half = left_half[:-1]
            # Swap the left and right parts
            deconvolved = np.concatenate([right_half, left_half])
        else:  # odd length
            left_half = deconvolved[:mid_point]
            center = deconvolved[mid_point]
            right_half = deconvolved[mid_point:]
            # # Ensure equal length
            # if len(right_half) > len(left_half):
            #     right_half = right_half[:-1]
            # elif len(left_half) > len(right_half):
            #     left_half = left_half[:-1]
            # Swap the left and right parts while retaining the center point
            deconvolved = np.concatenate([right_half, left_half])
            print(len(left_half), len(right_half), len(deconvolved))

        # Store the result
        deconvolved_curves[i, :] = deconvolved

    return deconvolved_curves


def validate_deconvolution(original, deconvolved, wavelengths, fwhm):
    """Validate the deconvolution result"""
    print("\nSymmetry validation of the deconvolution result:")
    mid_point = len(wavelengths) // 2
    print(f"Center wavelength: {wavelengths[mid_point]:.1f} nm")

    # Ensure the left and right halves have equal length
    left_half = deconvolved[:, :mid_point]
    right_half = deconvolved[:, -mid_point:]  # Take the same length from the right side

    # Check symmetry
    symmetry_error = np.mean(np.abs(left_half - np.flip(right_half, axis=1)))
    print(f"Symmetry error: {symmetry_error:.6f}")

    # Validate reconstruction
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = wavelengths - np.mean(wavelengths)
    gaussian_kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    reconvolved = np.zeros_like(deconvolved)
    for i in range(deconvolved.shape[0]):
        reconvolved[i, :] = np.convolve(deconvolved[i, :], gaussian_kernel, mode='same')

    error = np.mean((original - reconvolved) ** 2)
    return reconvolved, error


# **************Parameter settings*************
NumWavelengths = 561  # Number of wavelengths for reconstruction
NumWavelengthsInterp = NumWavelengths * 10  # Number of interpolated wavelengths for plotting
NumGaussianBasis = 561  # Number of Gaussian basis functions
MinLambda = 200  # Minimum wavelength (nm)
MaxLambda = 3000  # Maximum wavelength (nm)
FWHMset = 15  # FWHM of the Gaussian basis; normalized by the maximum wavelength and multiplied by 1000; dimensionless

# **************Load data*************
# Load calibration data
ResponseCurves = np.loadtxt('downsample_5nm_transposed.txt')
ResponseCurves = ResponseCurves.T / np.max(ResponseCurves)  # Normalize the input response curves


def get_wavelength_mode():
    """Get the wavelength-setting mode selected by the user."""
    while True:
        print("\nSelect the wavelength-setting mode:")
        print("1. Enter specific wavelengths (single peak)")
        print("2. Set wavelength range and interval (single peak)")
        print("3. Enter multi-peak parameters (combined peak)")
        print("4. Directly load measured_signal data")
        print("5. Load an image and perform RGB spectral simulation")
        print("6. Load a TXT file (first column: wavelength; remaining columns: intensity) and generate a photocurrent matrix")  # New
        choice = input("Enter your choice (1, 2, 3, 4, 5, or 6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return int(choice)
        print("Invalid input. Please choose again.")


def get_specific_wavelengths():
    """Get the user-defined wavelength list and the corresponding FWHM values."""
    wavelengths = []
    fwhms = []
    while True:
        try:
            wavelengths_str = input(f"\nEnter wavelength values (space-separated, range {MinLambda}-{MaxLambda} nm): ")
            wavelengths = [float(x) for x in wavelengths_str.split()]
            if not all(MinLambda <= w <= MaxLambda for w in wavelengths):
                print(f"Wavelengths must be between {MinLambda} nm and {MaxLambda} nm.")
                continue

            # Ask whether to use the default FWHM
            use_default = input("Use the default FWHM (15 nm)? (y/n): ").lower().strip() == 'y'

            if use_default:
                fwhms = [FWHMset] * len(wavelengths)
            else:
                print("\nEnter the corresponding FWHM for each wavelength:")
                for wavelength in wavelengths:
                    while True:
                        try:
                            fwhm = float(input(f"Wavelength {wavelength}nm FWHM (nm): "))
                            if fwhm <= 0:
                                print("FWHM must be greater than 0.")
                                continue
                            fwhms.append(fwhm)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")

            return np.array(sorted(wavelengths)), np.array(fwhms)
        except ValueError:
            print("Invalid input. Please try again.")


def get_wavelength_range():
    """Get the user-defined wavelength range, interval, and FWHM."""
    while True:
        try:
            start = float(input(f"\nEnter the start wavelength ({MinLambda}-{MaxLambda} nm): "))
            end = float(input(f"Enter the end wavelength ({start}-{MaxLambda} nm): "))
            step = float(input("Enter the wavelength interval (nm): "))

            if not (MinLambda <= start <= end <= MaxLambda and step > 0):
                print(f"The wavelength range must be between {MinLambda} nm and {MaxLambda} nm, and the interval must be greater than 0.")
                continue

            # Ask for the FWHM
            use_default = input("Use the default FWHM (15 nm)? (y/n): ").lower().strip() == 'y'
            if use_default:
                fwhm = FWHMset
            else:
                while True:
                    try:
                        fwhm = float(input("Enter the FWHM (nm): "))
                        if fwhm <= 0:
                            print("FWHM must be greater than 0.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please try again.")

            wavelengths = np.arange(start, end + step / 2, step)
            fwhms = np.ones_like(wavelengths) * fwhm

            return wavelengths, fwhms
        except ValueError:
            print("Invalid input. Please try again.")


def get_peaks_info():
    """Get user-defined multi-peak information."""
    peaks = []
    while True:
        try:
            num_peaks = int(input("\nEnter the number of peaks: "))
            if num_peaks <= 0:
                print("The number of peaks must be greater than 0.")
                continue

            print("\nEnter the information for each peak in sequence...")
            for i in range(num_peaks):
                print(f"\n--- Peak {i + 1} ---")
                wavelength = float(input(f"Wavelength ({MinLambda}-{MaxLambda}nm): "))
                weight = float(input("Weight (0-1): "))
                fwhm = float(input("FWHM (nm): "))

                if not (MinLambda <= wavelength <= MaxLambda):
                    raise ValueError(f"Wavelength must be within the range {MinLambda}-{MaxLambda} nm.")
                if not (0 < weight <= 1):
                    raise ValueError("The weight must be within the range 0-1.")
                if fwhm <= 0:
                    raise ValueError("FWHM must be greater than 0.")

                peaks.append((wavelength, weight, fwhm))

            return peaks

        except ValueError as e:
            print(f"Input error: {str(e)}")


def generate_single_peaks(wavelengths, fwhms):
    """Generate single-peak spectra with different FWHM values."""
    spectrum = np.zeros((len(VecOfLambdas), len(wavelengths)))
    for i, (wavelength, fwhm) in enumerate(zip(wavelengths, fwhms)):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to the standard deviation
        gaussian = np.exp(-0.5 * ((VecOfLambdas - wavelength) / sigma) ** 2)
        spectrum[:, i] = gaussian / np.max(gaussian)  # Normalize
    return spectrum


def generate_multi_peak_spectrum(wavelengths, peaks_info):
    """Generate a multi-peak spectrum."""
    spectrum = np.zeros_like(wavelengths)

    for wavelength, weight, fwhm in peaks_info:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-0.5 * ((wavelengths - wavelength) / sigma) ** 2)
        spectrum += weight * (gaussian / np.max(gaussian))
        print(f"Generated peak: wavelength={wavelength}nm, weight={weight}, FWHM={fwhm}nm")

    return spectrum


def load_measured_signals(path, n_detectors, delimiter=None, skiprows=0):
    data = np.loadtxt(path, delimiter=delimiter, ndmin=2, skiprows=skiprows)
    # If the dimensions do not match, try transposing once automatically
    if data.shape[0] != n_detectors and data.shape[1] == n_detectors:
        data = data.T
    if data.shape[0] != n_detectors:
        raise ValueError(f"The number of MeasuredSignals rows should be the detector count {n_detectors}, but got {data.shape[0]}")
    return data


def load_rgb_image():
    from PIL import Image
    import matplotlib.colors as mcolors
    path = input("\nEnter the image file path (PNG/JPG supported): ").strip()
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.uint8)  # H x W x 3
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    # Construct three colormaps: white to red, white to green, and white to blue
    cmap_R = mcolors.LinearSegmentedColormap.from_list("white_red", [(1, 1, 1), (1, 0, 0)], N=256)
    cmap_G = mcolors.LinearSegmentedColormap.from_list("white_green", [(1, 1, 1), (0, 1, 0)], N=256)
    cmap_B = mcolors.LinearSegmentedColormap.from_list("white_blue", [(1, 1, 1), (0, 0, 1)], N=256)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(R, cmap=cmap_R, vmin=0, vmax=255)
    plt.title('R(white to red)')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(G, cmap=cmap_G, vmin=0, vmax=255)
    plt.title('G(white to green)')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(B, cmap=cmap_B, vmin=0, vmax=255)
    plt.title('B(white to blue)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return R, G, B


def generate_measured_signals(mode, response_curves, sampling_indices, peaks_info=None):
    """
    Generate measured signals for different modes; all modes account for the FWHM effect.

    Parameters:
    mode: Mode selection (1,2: single-peak mode; 3: multi-peak mode; 5: image RGB monochromatic simulation)
    response_curves: Response curves
    sampling_indices: Sampling indices(mode 5 should be [idx_R, idx_G, idx_B])
    peaks_info: Peak parameter information for multi-peak mode
    """
    wavelength_vector = np.linspace(MinLambda, MaxLambda, response_curves.shape[1])
    signals = np.zeros((response_curves.shape[0], len(sampling_indices) if mode in [1, 2] else 1))

    if mode in [1, 2]:
        # Generate responses accounting for FWHM for each sampled wavelength
        for i, (wavelength, fwhm) in enumerate(zip(sampling_wavelengths, sampling_fwhms)):
            # Use the input FWHM
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

            # Generate the Gaussian distribution
            gaussian = np.exp(-0.5 * ((wavelength_vector - wavelength) / sigma) ** 2)
            gaussian = gaussian / np.max(gaussian)  # Normalize the Gaussian distribution

            # Apply Gaussian convolution to the response curves
            for detector_idx in range(response_curves.shape[0]):
                # Apply Gaussian weighting to each detector response
                weighted_response = response_curves[detector_idx, :] * gaussian
                signals[detector_idx, i] = np.sum(weighted_response) / np.sum(gaussian)

            # Normalize the signal of each single peak
            signals[:, i] /= np.max(np.abs(signals[:, i]))

    elif mode == 3:
        # Use a multi-peak combination accounting for weights and FWHM
        for wavelength, weight, fwhm in peaks_info:
            # Calculate the standard deviation of the Gaussian distribution
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

            # Generate the Gaussian distribution
            gaussian = np.exp(-0.5 * ((wavelength_vector - wavelength) / sigma) ** 2)
            gaussian = gaussian / np.max(gaussian)  # Normalize the Gaussian distribution

            # Apply Gaussian convolution to the response curves
            for detector_idx in range(response_curves.shape[0]):
                # Apply Gaussian weighting to each detector response
                weighted_response = response_curves[detector_idx, :] * gaussian
                signals[detector_idx, 0] += weight * np.sum(weighted_response) / np.sum(gaussian)

        # Normalize the final signal of the multi-peak combination
        signals /= np.max(np.abs(signals))

    elif mode == 4:
        while True:
            try:
                file_path = input("\nEnter the measured_signal file path: ").strip()
                signals = np.loadtxt(file_path)  # Assume the file is a numeric matrix delimited by spaces or commas
                if signals.ndim != 2:
                    raise ValueError("The measured_signal data must be a 2D matrix.")
                break
            except Exception as e:
                print(f"Failed to read the file: {e}")  # Convert to an n x 1 2D array

    elif mode == 5:
        # Rule: for the center wavelengths of R/G/B, directly extract the corresponding photocurrent vector from the response matrix;
        # then linearly scale it by intensities 1..255, generating 255 columns per color and 3*255 columns in total.
        num_levels = 255  # Exclude 0 to avoid all-zero columns
        n_det = response_curves.shape[0]
        if len(sampling_indices) != 3:
            raise ValueError("mode 5 requires three sampling indices corresponding to the R/G/B wavelengths")
        signals = np.zeros((n_det, 3 * num_levels))
        for ci, idx in enumerate(sampling_indices):
            base = response_curves[:, idx].astype(float)  # Directly extract the photocurrent column without normalization
            for k in range(1, num_levels + 1):  # 1..255
                amp = k / 255.0
                signals[:, ci * num_levels + (k - 1)] = amp * base

    return signals


def load_spectrum_txt(path):
    """
    Load a TXT file: the first column is wavelength (nm), and each subsequent column is intensity; return (wavelengths_nm, spectra_matrix)
    spectra_matrix has shape (P, M), Pis the number of wavelength points, Mis the number of spectra
    """
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("The TXT file must have at least two columns: the first column is wavelength and the subsequent columns are intensities.")
    wavelengths_nm = data[:, 0].astype(float)
    spectra = data[:, 1:].astype(float)
    return wavelengths_nm, spectra


from scipy.interpolate import PchipInterpolator

def resample_spectra_to_response_grid(wavelengths_nm, spectra, target_grid_nm):
    """
    Resample to target_grid_nm:
    - Sort input wavelengths and remove duplicates by averaging repeated points; 
    - Use PCHIP interpolation only within the input range and set values outside the range to 0; 
    - Keep each column as an independent spectrum.
    Return (len(target_grid_nm), M)
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float).ravel()
    spectra = np.asarray(spectra, dtype=float)
    target_grid_nm = np.asarray(target_grid_nm, dtype=float).ravel()

    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D matrix (P, M)")
    if wavelengths_nm.size != spectra.shape[0]:
        raise ValueError(f"The length of wavelengths_nm ({wavelengths_nm.size}) must equal the number of rows in spectra ({spectra.shape[0]}).")
    if target_grid_nm.size == 0:
        raise ValueError("target_grid_nm cannot be empty")

    # Sort
    sort_idx = np.argsort(wavelengths_nm)
    w_sorted = wavelengths_nm[sort_idx]
    sp_sorted = spectra[sort_idx, :]

    # Remove duplicates: average repeated wavelength rows
    unique_w, inv = np.unique(w_sorted, return_inverse=True)
    sp_unique = np.zeros((unique_w.size, sp_sorted.shape[1]))
    counts = np.bincount(inv)
    for m in range(sp_sorted.shape[1]):
        sp_sum = np.zeros(unique_w.size)
        np.add.at(sp_sum, inv, sp_sorted[:, m])
        sp_unique[:, m] = sp_sum / counts

    w_sorted = unique_w
    sp_sorted = sp_unique

    resampled = np.zeros((target_grid_nm.size, sp_sorted.shape[1]), dtype=float)
    in_mask = (target_grid_nm >= w_sorted[0]) & (target_grid_nm <= w_sorted[-1])
    grid_in = target_grid_nm[in_mask]

    if w_sorted.size == 1:
        resampled[in_mask, :] = sp_sorted[0, :]
        return resampled

    for m in range(sp_sorted.shape[1]):
        pchip = PchipInterpolator(w_sorted, sp_sorted[:, m])
        resampled[in_mask, m] = pchip(grid_in)

    return resampled


def build_photocurrent_matrix(response_curves, spectra_on_grid):
    """
    Calculate the photocurrent matrix from the response curves and spectral curves::
    I = ∑_λ R(Detector, λ) * S(λ, sample)  (integrate/sum over λ)
    Return MeasuredSignals shape=(N_detectors, M)
    """
    # response_curves: (N_detectors, P)
    # spectra_on_grid: (P, M)
    # Directly perform matrix multiplication along the wavelength dimension: (N_detectors, P) @ (P, M) => (N_detectors, M)
    currents = response_curves @ spectra_on_grid
    # Optional normalization to avoid excessive scale differences
    currents = currents / (np.max(np.abs(currents)) + 1e-12)
    return currents


def add_noise(signals, noise_type='gaussian', level=0.05, random_state=None):
    """
    Inject noise into the measured-signal matrix (Gaussian only).
    level: Noise level as a fraction of the column-wise maximum value
    """
    if noise_type != 'gaussian':
        raise ValueError("The current strategy supports only 'gaussian' noise.")
    rng = np.random.default_rng(random_state)
    noisy = signals.astype(float).copy()
    col_max = np.maximum(np.max(np.abs(noisy), axis=0, keepdims=True), 1e-12)
    std = level * col_max
    noisy += rng.normal(0.0, 1.0, size=noisy.shape) * std
    return noisy


def add_noise_snr(signals, snr_db=30.0, random_state=None):
    """
    Inject additive Gaussian noise according to the target SNR (dB).
    SNR definition: SNR = signal_rms / noise_rms(calculated column-wise)
    snr_db: Signal-to-noise ratio in dB(larger values indicate lower noise)
    """
    rng = np.random.default_rng(random_state)
    x = signals.astype(float).copy()
    # RMS of each signal column
    rms = np.sqrt(np.mean(x**2, axis=0, keepdims=True))  # shape (1, M)
    snr_linear = 10**(snr_db / 20.0)  # Linear ratio (amplitude ratio)
    noise_rms = rms / snr_linear
    noise = rng.normal(0.0, 1.0, size=x.shape) * noise_rms
    return x + noise


def make_snr_db_list(n_levels=61, snr_db_min=20.0, snr_db_max=80.0):
    """
    Generate an equally spaced SNR (dB) list.
    """
    n_levels = max(1, int(n_levels))
    return np.linspace(float(snr_db_min), float(snr_db_max), n_levels)


# After loading the data, first create the wavelength vector
VecOfLambdas = np.linspace(MinLambda, MaxLambda, ResponseCurves.shape[1])  # Original wavelength vector

# After loading the data, deconvolve ResponseCurves
# Create the wavelength vector
wavelengths = np.linspace(MinLambda, MaxLambda, ResponseCurves.shape[1])

# Deconvolve ResponseCurves
deconvolved_curves = deconvolve_response_curves(
    response_curves=ResponseCurves,
    wavelengths=wavelengths,
    fwhm=FWHMset,  # Use the same FWHM as the sampling setting
    reg_param=0.01  # This parameter can be tuned to optimize the results
)

# Validate the deconvolution result
reconvolved_curves, error = validate_deconvolution(
    original=ResponseCurves,
    deconvolved=deconvolved_curves,
    wavelengths=wavelengths,
    fwhm=FWHMset
)

# Visualization comparison
plt.figure(figsize=(15, 10))

# Original response curves
plt.subplot(311)
plt.imshow(ResponseCurves, aspect='auto', cmap='bwr', extent=[MinLambda, MaxLambda, 0, ResponseCurves.shape[0]])
plt.colorbar()
plt.title('Original Response Curves')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Detector Index')

# Deconvolved response curves
plt.subplot(312)
plt.imshow(deconvolved_curves, aspect='auto', cmap='bwr', extent=[MinLambda, MaxLambda, 0, ResponseCurves.shape[0]])
plt.colorbar()
plt.title('Deconvolved Response Curves')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Detector Index')

# Reconstruction validation
plt.subplot(313)
plt.imshow(reconvolved_curves, aspect='auto', cmap='bwr', extent=[MinLambda, MaxLambda, 0, ResponseCurves.shape[0]])
plt.colorbar()
plt.title(f'Reconvolved Response Curves (Error: {error:.2e})')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Detector Index')

plt.tight_layout()
plt.show()

# Replace the original sampling-wavelength generation code
print("\n=== Spectral reconstruction wavelength settings ===")
mode = get_wavelength_mode()

# Create the output directory immediately after mode selection for global use
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = os.path.join(os.getcwd(), f"Reconstruction_Results_{timestamp}_mode{mode}")
os.makedirs(out_dir, exist_ok=True)
print(f"Results will be saved to directory: {out_dir}")

# Add mode 3 handling in the wavelength-selection section
if mode in [1, 2]:  # single-peak mode
    sampling_wavelengths, sampling_fwhms = get_specific_wavelengths() if mode == 1 else get_wavelength_range()
    SetSpectrum = generate_single_peaks(sampling_wavelengths, sampling_fwhms)
    sampling_indices = [np.abs(VecOfLambdas - wave).argmin() for wave in sampling_wavelengths]
    MeasuredSignals = generate_measured_signals(
        mode=mode,
        response_curves=ResponseCurves,
        sampling_indices=sampling_indices
    )

    # Generate the SNR (dB) list and apply Gaussian noise automatically only in mode 1
    if mode == 1:
        try:
            n_levels = int(input("Length of the SNR (dB) list (default 61): ").strip() or "61")
        except Exception:
            n_levels = 61
        try:
            snr_min = float(input("Minimum SNR (dB, default 20): ").strip() or "20")
            snr_max = float(input("Maximum SNR (dB, default 80): ").strip() or "80")
        except Exception:
            snr_min, snr_max = 20.0, 80.0

        snr_list = make_snr_db_list(n_levels=n_levels, snr_db_min=snr_min, snr_db_max=snr_max)
        print(f"Using SNR (dB) list: {', '.join(f'{s:.1f}' for s in snr_list)}")

        MeasuredSignals_snr_concat = MeasuredSignals.copy()
        # Record the column range of each SNR noise block in MeasuredSignals for subsequent one-to-one matching with reference peak positions
        snr_blocks = []  # Each item: {'snr_db': float, 'start': int, 'end': int}  [start, end) half-open interval
        group_size = MeasuredSignals.shape[1]  # The number of columns appended for each SNR equals the original column count and is aligned with sampling_wavelengths
        for snr_db in snr_list:
            MeasuredSignals_noisy = add_noise_snr(MeasuredSignals, snr_db=snr_db)
            base_name = f"Mode1_SNR_{snr_db:.1f}dB"
            # Save and visualize
            np.savetxt(os.path.join(out_dir, f"MeasuredSignals_{base_name}.txt"),
                       MeasuredSignals_noisy, fmt="%.8f", delimiter="\t",
                       header=f"Mode1 MeasuredSignals with Gaussian noise by SNR | SNR={snr_db:.1f} dB | shape={MeasuredSignals_noisy.shape}")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(MeasuredSignals, aspect='auto', cmap='bwr')
            plt.title('Original MeasuredSignals')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(MeasuredSignals_noisy, aspect='auto', cmap='bwr')
            plt.title(f'Noisy (Gaussian, SNR={snr_db:.1f} dB)')
            plt.colorbar()
            plt.tight_layout()
            # plt.savefig(os.path.join(out_dir, f"MeasuredSignals_{base_name}.png"),
                        # dpi=300, bbox_inches='tight')
            # plt.show()
            # Close the figure to release memory
            plt.close()
            # Record the column range of this SNR noise block from the start before appending to the end after appending
            start_idx = MeasuredSignals_snr_concat.shape[1]
            MeasuredSignals_snr_concat = np.concatenate((MeasuredSignals_snr_concat, MeasuredSignals_noisy), axis=1)
            end_idx = start_idx + group_size
            snr_blocks.append({'snr_db': float(snr_db), 'start': int(start_idx), 'end': int(end_idx)})

        # Use the matrix containing all SNR versions for subsequent reconstruction
        MeasuredSignals = MeasuredSignals_snr_concat
elif mode == 3:  # multi-peak mode
    peaks_info = get_peaks_info()
    sampling_wavelengths = np.array([peak[0] for peak in peaks_info])
    sampling_fwhms = np.array([peak[2] for peak in peaks_info])
    sampling_indices = [np.abs(VecOfLambdas - wave).argmin() for wave in sampling_wavelengths]
    MeasuredSignals = generate_measured_signals(
        mode=mode,
        response_curves=ResponseCurves,
        sampling_indices=sampling_indices,
        peaks_info=peaks_info
    )
elif mode == 4:  # New mode: directly load measured_signal
    while True:
        try:
            file_path = input("\nEnter the measured_signal file path: ").strip()
            MeasuredSignals = np.loadtxt(file_path)  # Assume the file is a numeric matrix delimited by spaces or commas
            if MeasuredSignals.ndim != 2:
                raise ValueError("The measured_signal data must be a 2D matrix.")
            break
        except Exception as e:
            print(f"Failed to read the file: {e}")
elif mode == 5:  # New mode: load an image and perform RGB spectral simulation
    # Fix three wavelengths; adjust according to the device settings if needed
    lambda_R, lambda_G, lambda_B = 650.0, 550.0, 450.0  # nm
    # Use FWHMset for all FWHM values
    sampling_wavelengths = np.array([lambda_R, lambda_G, lambda_B], dtype=float)
    sampling_fwhms = np.array([FWHMset, FWHMset, FWHMset], dtype=float)

    # Load the image and display white-to-pure-color colormaps
    Rimg, Gimg, Bimg = load_rgb_image()
    # Use the distribution of 0-255 grayscale levels as intensity weights
    # Generate 256 columns of simulated signals for each channel; column k corresponds to intensity k/255
    wavelength_vector = np.linspace(MinLambda, MaxLambda, ResponseCurves.shape[1])

    def channel_signals(center_wavelength, step=50):
        sigma = FWHMset / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-0.5 * ((wavelength_vector - center_wavelength) / sigma) ** 2)
        gaussian = gaussian / np.max(gaussian)
        # Calculate the detector response to this single peak using the same convolution weighting as mode 1
        base_signal = np.sum(ResponseCurves * gaussian[None, :], axis=1) / np.sum(gaussian)
        base_signal = base_signal / np.max(np.abs(base_signal))
        # Generate columns at the specified step interval and scale them by intensity
        num_levels = 256 // step
        cols = np.zeros((ResponseCurves.shape[0], num_levels))
        for k in range(num_levels):
            amp = (k * step) / 255.0
            cols[:, k] = amp * base_signal
        return cols

    Rcols = channel_signals(lambda_R)
    Gcols = channel_signals(lambda_G)
    Bcols = channel_signals(lambda_B)

    # Concatenate to obtain MeasuredSignals with 3*256 columns in the order R0..R255 | G0..G255 | B0..B255
    MeasuredSignals = np.concatenate([Rcols, Gcols, Bcols], axis=1)
    # Plot MeasuredSignals
    #     
    plt.figure(figsize=(10, 6))
    plt.title('Measured Signals')
    plt.imshow(MeasuredSignals, aspect='auto', cmap='bwr', extent=[0, MeasuredSignals.shape[1], 0, MeasuredSignals.shape[0]])
    plt.colorbar()
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Level')
    plt.show()

    # Sampling indices for reference spectra (only three center wavelengths)
    sampling_indices = [np.abs(VecOfLambdas - w).argmin() for w in sampling_wavelengths]

elif mode == 6:  # New mode: Load txt (λ, intensity...) and generate photocurrent matrix
    # Create and hide the Tk root window before using the file dialog
    from tkinter import Tk, filedialog
    _root = Tk()
    _root.withdraw()
    txt_path = filedialog.askopenfilename(title="Select TXT File", filetypes=[("Text Files", "*.txt")])
    _root.destroy()
    if not txt_path:
        raise ValueError("No file selected. Please select a valid TXT file.")
    wavelengths_txt, spectra_txt = load_spectrum_txt(txt_path)               # (P_txt,), (P_txt, M)

    # Plot the data loaded from the TXT file
    plt.figure(figsize=(8, 5))
    for i in range(spectra_txt.shape[1]):
        plt.plot(wavelengths_txt, spectra_txt[:, i], label=f'Spectrum {i+1}')
    plt.title('Loaded Spectra from TXT')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Resample txt spectra to the response curve grid (matching ResponseCurves columns' λ)
    grid_nm = np.linspace(MinLambda, MaxLambda, ResponseCurves.shape[1])     # Wavelength grid of response curves
    spectra_on_grid = resample_spectra_to_response_grid(wavelengths_txt, spectra_txt, grid_nm)  # (P_resp, M)
    # Save resampled initial spectra
    init_spectrum_stack = np.column_stack([grid_nm, spectra_on_grid])        # First column: λ
    np.savetxt(os.path.join(out_dir, "InitialSpectra_FromTXT_Resampled.txt"), init_spectrum_stack, fmt="%.8f", delimiter="\t",
               header=f"Initial spectra resampled to response grid | λ(nm) + {spectra_on_grid.shape[1]} spectra")
    # Generate photocurrent matrix
    MeasuredSignals = build_photocurrent_matrix(ResponseCurves, spectra_on_grid)  # (N_detectors, M)
    # Visualization: Left - Initial spectra curves (overlaid), Right - Photocurrent heatmap
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for m in range(spectra_on_grid.shape[1]):
        plt.plot(grid_nm, spectra_on_grid[:, m], alpha=0.6)
    plt.title('Initial Spectra Curves (Resampled to Response Grid)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(MeasuredSignals, aspect='auto', cmap='bwr',
               extent=[0, MeasuredSignals.shape[1], 0, MeasuredSignals.shape[0]])
    plt.colorbar(label='Photocurrent (norm)')
    plt.title('Photocurrent Matrix Heatmap')
    plt.xlabel('Sample Index')
    plt.ylabel('Detector Index')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "InitialSpectra_and_Photocurrent.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Save photocurrent matrix
    np.savetxt(os.path.join(out_dir, "MeasuredSignals_FromTXT.txt"), MeasuredSignals, fmt="%.8f", delimiter="\t",
               header=f"Measured signals from TXT | detectors={MeasuredSignals.shape[0]}, samples={MeasuredSignals.shape[1]}")
    # Set reference spectra for subsequent comparison: SetSpectrum as (λ + M intensity columns)
    SetSpectrum = np.column_stack([grid_nm, spectra_on_grid])
    # For compatibility with subsequent processes, create placeholders for sampling_wavelengths/fwhms
    sampling_wavelengths = np.array([grid_nm.mean()])  # Not used for solving, just to avoid later references
    sampling_fwhms = np.array([FWHMset])

# Generate sampling indices
sampling_indices = [] if mode in [4, 6] else [np.abs(VecOfLambdas - wave).argmin() for wave in sampling_wavelengths]

# ResponseCurves = deconvolved_curves  # Use deconvolved response curves

""" # Generate MeasuredSignals from ResponseCurves
MeasuredSignals = generate_measured_signals(
    mode=mode,
    response_curves=ResponseCurves,
    sampling_indices=sampling_indices,
    peaks_info=peaks_info if mode == 3 else None
) if mode not in [6] else MeasuredSignals  # Mode 6 uses the MeasuredSignals calculated above """

# Convert to a 2D array if needed
if MeasuredSignals.ndim == 1:
    MeasuredSignals = MeasuredSignals[:, np.newaxis]  # Convert to an n x 1 2D array

# === Print and save the final MeasuredSignals used for reconstruction (unnormalized) ===
MeasuredSignals_raw = MeasuredSignals.copy()
np.set_printoptions(suppress=True, linewidth=160)
print(f"\nMeasuredSignals shape: {MeasuredSignals_raw.shape}")
preview_cols = min(MeasuredSignals_raw.shape[1], 5)
print(f"MeasuredSignals preview (first {preview_cols} columns)::")
print(MeasuredSignals_raw[:, :preview_cols])

out_path = os.path.join(out_dir, "measured_signals_raw.txt")
np.savetxt(
    out_path,
    MeasuredSignals_raw,
    fmt="%.8f",
    delimiter="\t",
    header=f"MeasuredSignals rows={MeasuredSignals_raw.shape[0]}, cols={MeasuredSignals_raw.shape[1]}"
)
print(f"MeasuredSignals has been saved to: {out_path}")

# 1. First generate the wavelength vector at the target resolution
target_resolution = 0.01  # Target resolution, 0.01 nm
new_num_points = int((MaxLambda - MinLambda) / target_resolution) + 1
VecOfLambdas_HiRes = np.linspace(MinLambda, MaxLambda, new_num_points)

if mode in [1, 2]:  # single-peak mode
    SetSpectrum = np.zeros((new_num_points, len(sampling_wavelengths) + 1))
    SetSpectrum[:, 0] = VecOfLambdas_HiRes  # The first column is the wavelength vector

    # Generate high-resolution single-peak spectra
    for i, (wavelength, fwhm) in enumerate(zip(sampling_wavelengths, sampling_fwhms)):
        # Use the FWHM corresponding to each wavelength
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to the standard deviation
        gaussian = np.exp(-0.5 * ((VecOfLambdas_HiRes - wavelength) / sigma) ** 2)
        SetSpectrum[:, i + 1] = gaussian / np.max(gaussian)  # Normalize the single-peak spectrum

elif mode == 3:  # multi-peak mode
    # Generate SetSpectrum for the multi-peak combination
    SetSpectrum = np.zeros((new_num_points, 2))  # Only two columns are needed: wavelength and combined spectrum
    SetSpectrum[:, 0] = VecOfLambdas_HiRes

    # Generate the multi-peak spectrum using the specified FWHM values and weights
    combined_spectrum = np.zeros(new_num_points)
    for wavelength, weight, fwhm in peaks_info:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-0.5 * ((VecOfLambdas_HiRes - wavelength) / sigma) ** 2)
        combined_spectrum += weight * (gaussian / np.max(gaussian))
    # Generate the multi-peak spectrum using generate_multi_peak_spectrum
    combined_spectrum = generate_multi_peak_spectrum(VecOfLambdas_HiRes, peaks_info)
    SetSpectrum[:, 1] = combined_spectrum / np.max(combined_spectrum)  # Normalize the combined spectrum
elif mode == 5:  # Image RGB: generate reference single peaks for fixed three-color wavelengths
    SetSpectrum = np.zeros((new_num_points, 4))  # col0=λ, columns 1-3 are R/G/B single peaks
    SetSpectrum[:, 0] = VecOfLambdas_HiRes
    for i, (wavelength, fwhm) in enumerate(zip(sampling_wavelengths, sampling_fwhms)):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-0.5 * ((VecOfLambdas_HiRes - wavelength) / sigma) ** 2)
        SetSpectrum[:, i + 1] = gaussian / np.max(gaussian)

# Validate by plotting
if mode in [1, 2, 3, 5]:
    plt.figure(figsize=(10, 6))

    if mode in [1, 2]:  # single-peak mode
        # Plot each individual Gaussian peak
        for i in range(len(sampling_wavelengths)):
            plt.plot(VecOfLambdas_HiRes, SetSpectrum[:, i + 1], '-',
                     label=f'Peak at {sampling_wavelengths[i]}nm')
        plt.title('Single-Peak Reference Spectra')

    elif mode == 3:  # mode == 3, multi-peak combined mode
        # Plot the combined spectrum
        plt.plot(VecOfLambdas_HiRes, SetSpectrum[:, 1], '-k',
                 label='Combined Spectrum', linewidth=2)
        plt.title('Multi-Peak Combined Spectrum')

    elif mode == 5:
        colors = ['r', 'g', 'b']
        labels = ['R channel', 'G channel', 'B channel']
        for i in range(3):
            plt.plot(VecOfLambdas_HiRes, SetSpectrum[:, i + 1], color=colors[i],
                     label=f'{labels[i]} λ={sampling_wavelengths[i]:.1f}nm')
        plt.title('RGB Fixed-Wavelength Reference Spectra')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.grid(True)
    plt.legend()
    plt.show()

# 4. Interpolate ResponseCurves
ResponseCurves_HiRes = np.zeros((ResponseCurves.shape[0], new_num_points))
VecOfLambdas_Original = np.linspace(MinLambda, MaxLambda, ResponseCurves.shape[1])
for r in range(ResponseCurves.shape[0]):
    if len(VecOfLambdas) != len(ResponseCurves[r, :]):
        raise ValueError(
            f"Length mismatch: VecOfLambdas ({len(VecOfLambdas)}) and ResponseCurves[{r}, :] ({len(ResponseCurves[r, :])})")

    tck = splrep(VecOfLambdas, ResponseCurves[r, :])
    ResponseCurves_HiRes[r, :] = splev(VecOfLambdas_HiRes, tck)

# Update ResponseCurves and the wavelength vector
ResponseCurves = ResponseCurves_HiRes
VecOfLambdas = VecOfLambdas_HiRes

# **************Variable initialization*************
NumRow = ResponseCurves.shape[0]  # Number of Vg points (detector count)
NumCol = MeasuredSignals.shape[1]  # Number of columns in the measured-signal matrix
SimulatedSignals = np.zeros((NumRow, NumCol))  # Initialize the simulated-signal array
HiResReconstructedSpectrum = np.zeros((NumWavelengthsInterp, NumCol))  # Initialize the high-resolution reconstructed-spectrum array

VecOfLambdas = np.linspace(MinLambda, MaxLambda, ResponseCurves.shape[1]) / MaxLambda  # Normalized wavelength vector
VecOfLambdasPlot = np.linspace(MinLambda, MaxLambda, NumWavelengthsInterp) / MaxLambda  # Interpolated wavelength vector

HiResResponseCurves = np.zeros((NumRow, NumWavelengthsInterp))
for r in range(NumRow):
    if len(VecOfLambdas) != len(ResponseCurves[r, :]):
        raise ValueError(
            f"Length mismatch: VecOfLambdas ({len(VecOfLambdas)}) and ResponseCurves[{r}, :] ({len(ResponseCurves[r, :])})")

    tck = splrep(VecOfLambdas, ResponseCurves[r, :])
    HiResResponseCurves[r, :] = splev(VecOfLambdasPlot, tck)  # Use spline interpolation

MeasuResidual = np.zeros(NumCol)  # Record the squared L2 norm of the signal residual
ReguTerm = np.zeros(NumCol)  # Record the squared L2 norm of the regularization term

# Initialize variables for storing Lasso regression results
HiResReconstructedSpectrum_Lasso = np.zeros((NumWavelengthsInterp, NumCol))  # Lasso reconstructed spectrum
SimulatedSignals_Lasso = np.zeros((NumRow, NumCol))  # Lasso simulated signals
MeasuResidual_Lasso = np.zeros(NumCol)  # Lasso signal residual
ReguTerm_Lasso = np.zeros(NumCol)  # Lasso regularization term

# Initialize variables for the new method
HiResReconstructedSpectrum_ElasticNet = np.zeros((NumWavelengthsInterp, NumCol))
SimulatedSignals_ElasticNet = np.zeros((NumRow, NumCol))
MeasuResidual_ElasticNet = np.zeros(NumCol)
ReguTerm_ElasticNet = np.zeros(NumCol)


def is_single_peak(coefficients, threshold=0.01):
    """Check whether the coefficients form a single peak."""
    significant = coefficients > threshold * np.max(coefficients)
    nonzero_indices = np.where(significant)[0]

    if len(nonzero_indices) == 0:
        return False

    gaps = np.diff(nonzero_indices)
    return np.all(gaps == 1)


def optimization_with_peak_constraint(WeightMatrix, measured_signal, method='lasso', max_iter=20000):
    """
    Optimization function with a single-peak constraint; records the loss values.
    """
    n_basis = WeightMatrix.shape[1]
    best_coef = None
    best_model = None
    min_error = float('inf')
    loss_history = []  # Used to record loss values

    for center in range(0, n_basis, 5):
        radius = 30
        start_idx = max(0, center - radius)
        end_idx = min(n_basis, center + radius)

        mask = np.zeros(n_basis)
        mask[start_idx:end_idx] = 1

        if method == 'lasso':
            model = Lasso(alpha=0.00001, positive=True, max_iter=max_iter)
        else:
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.logspace(-6, 1, 20),
                positive=True,
                max_iter=max_iter
            )

        valid_cols = mask == 1
        reduced_weight_matrix = WeightMatrix[:, valid_cols]
        model.fit(reduced_weight_matrix, measured_signal)

        full_coef = np.zeros(n_basis)
        full_coef[valid_cols] = model.coef_

        error = np.linalg.norm(WeightMatrix @ full_coef - measured_signal) ** 2
        loss_history.append(error)  # Record the loss value

        if is_single_peak(full_coef) and error < min_error:
            min_error = error
            best_coef = full_coef
            best_model = model

    return best_coef, best_model, loss_history


def optimization_without_peak_constraint(WeightMatrix, measured_signal, method='lasso', max_iter=20000):
    """
    Optimization function without a peak constraint, suitable for multi-peak cases; records the loss values.
    """
    loss_history = []  # Used to record loss values

    if method == 'lasso':
        model = Lasso(alpha=0.00001, positive=True, max_iter=max_iter)
    else:
        model = ElasticNetCV(
            l1_ratio=[0.05, 0.35, 0.7],
            alphas=np.logspace(-7, -4, 24),
            positive=True,
            max_iter=max_iter
        )

    model.fit(WeightMatrix, measured_signal)
    loss_history = model.mse_path_ if hasattr(model, 'mse_path_') else []  # Record loss values if supported
    return model.coef_, model, loss_history


# **************Reconstruction process*************
GaussianCenter = np.linspace(MinLambda, MaxLambda, NumGaussianBasis) / MaxLambda

# Mode 6: constrain the FWHM and use it for subsequent reconstruction
if mode == 6:
    try:
        _inp = input("Enter the reconstruction FWHM for mode 6 (0-100 nm, press Enter for default 50): ").strip()
        FWHM_for_reconstruction = float(_inp) if _inp else 15.0
    except Exception:
        FWHM_for_reconstruction = 15.0
    FWHM_for_reconstruction = float(np.clip(FWHM_for_reconstruction, 0.0, 100.0))
else:
    FWHM_for_reconstruction = FWHMset

# Generate the Gaussian-basis standard deviation using the selected FWHM
GaussianSigma_base = FWHM_for_reconstruction / 1000.0 / (2 * np.sqrt(2 * np.log(2)))

# Define fwhm in advance for all modes to avoid undefined-variable crashes
if mode in [1, 2, 3, 5]:
    fwhm = float(np.mean(sampling_fwhms))
elif mode == 6:
    fwhm = FWHM_for_reconstruction
else:  # mode 4 or fallback
    fwhm = FWHMset
print(f"fwhm used: {fwhm}")

# Determine the fwhm used for shrink_factor/gamma_shrink
if mode in [1, 2, 3]:
    fwhm = float(np.mean(sampling_fwhms))
elif mode == 6:
    fwhm = FWHM_for_reconstruction
else:  # mode 4/5 or fallback
    fwhm = FWHMset

shrink_factor = 0.9 * fwhm / 49 + 0.1
gamma_shrink  = 0.9 * fwhm / 49 + 0.1
print(shrink_factor)
print(gamma_shrink)

loss_histories = {'lasso': [], 'elasticnet': []}

for i in range(NumCol):
    # 1) Normalize
    MeasuredSignals[:, i] = MeasuredSignals[:, i] / np.max(MeasuredSignals[:, i])
    y = MeasuredSignals[:, i]

    # 2) Recompute sigma for each column and generate the Gaussian basis
    GaussianSigma = GaussianSigma_base * shrink_factor
    Δ = VecOfLambdasPlot[:, None] - GaussianCenter[None, :]   # (NumWavelengthsInterp, NumGaussianBasis)
    HiResGaussianBasis = np.exp(-0.5 * (Δ / GaussianSigma)**2)
    HiResGaussianBasis *= 1 / (GaussianSigma * np.sqrt(2 * np.pi))

    # 3) Construct WeightMatrix
    WeightMatrix = HiResResponseCurves @ HiResGaussianBasis   # (NumRow, NumGaussianBasis)

    # 4) Estimate gamma using GCV
    U, s, Vt = np.linalg.svd(WeightMatrix, full_matrices=False)
    UT = U.T
    UTy = UT @ y

    def compute_gcv(gamma):
        f   = s**2 / (s**2 + gamma**2)
        num = np.sum(((1 - f) * UTy)**2)
        den = (NumRow - np.sum(f))**2
        return num / den

    res = minimize(compute_gcv, x0=1e-6, bounds=[(0, None)])
    OptimalGamma = res.x[0] * gamma_shrink   # **Must be defined here**

    # 5) Closed-form Tikhonov solution plus non-negative clipping
    A = WeightMatrix.T @ WeightMatrix + OptimalGamma**2 * np.eye(NumGaussianBasis)
    c = np.linalg.solve(A, WeightMatrix.T @ y)
    GaussianCoefficients = np.clip(c, 0, None)

    # 6) Reconstruct and calculate
    HiResReconstructedSpectrum[:, i] = HiResGaussianBasis @ GaussianCoefficients
    HiResReconstructedSpectrum[:, i] /= np.max(HiResReconstructedSpectrum[:, i])
    SimulatedSignals[:, i] = HiResResponseCurves @ (HiResGaussianBasis @ GaussianCoefficients)

    MeasuResidual[i] = np.linalg.norm(SimulatedSignals[:, i] - y)**2
    ReguTerm[i]     = np.linalg.norm(GaussianCoefficients)**2

    # Select the optimization function according to the mode in the main loop
    if mode in [1, 2]:  # single-peak mode
        GaussianCoefficients_Lasso, _, loss_lasso = optimization_with_peak_constraint(
            WeightMatrix,
            MeasuredSignals[:, i],
            method='lasso'
        )
        GaussianCoefficients_ElasticNet, elastic_net_model, loss_elasticnet = optimization_with_peak_constraint(
            WeightMatrix,
            MeasuredSignals[:, i],
            method='elasticnet'
        )
    else:  # Multi-peak mode, including mode==3, 5, 6)
        GaussianCoefficients_Lasso, _, loss_lasso = optimization_without_peak_constraint(
            WeightMatrix,
            MeasuredSignals[:, i],
            method='lasso'
        )
        GaussianCoefficients_ElasticNet, elastic_net_model, loss_elasticnet = optimization_without_peak_constraint(
            WeightMatrix,
            MeasuredSignals[:, i],
            method='elasticnet'
        )

    loss_histories['lasso'].append(loss_lasso)
    loss_histories['elasticnet'].append(loss_elasticnet)

    HiResReconstructedSpectrum_Lasso[:, i] = HiResGaussianBasis @ GaussianCoefficients_Lasso  # Reconstructed spectrum for plotting
    HiResReconstructedSpectrum_Lasso[:, i] /= np.max(HiResReconstructedSpectrum_Lasso[:, i])  # Normalize
    SimulatedSignals_Lasso[:, i] = HiResResponseCurves @ HiResGaussianBasis @ GaussianCoefficients_Lasso  # Simulated signal of the reconstructed spectrum

    MeasuResidual_Lasso[i] = np.linalg.norm(SimulatedSignals_Lasso[:, i] - MeasuredSignals[:, i]) ** 2  # Calculate the squared L2 norm of the signal residual
    ReguTerm_Lasso[i] = np.linalg.norm(GaussianCoefficients_Lasso) ** 2  # Calculate the squared L2 norm of the regularization term

    HiResReconstructedSpectrum_ElasticNet[:, i] = HiResGaussianBasis @ GaussianCoefficients_ElasticNet
    HiResReconstructedSpectrum_ElasticNet[:, i] /= np.max(HiResReconstructedSpectrum_ElasticNet[:, i])
    SimulatedSignals_ElasticNet[:, i] = HiResResponseCurves @ HiResGaussianBasis @ GaussianCoefficients_ElasticNet

    MeasuResidual_ElasticNet[i] = np.linalg.norm(SimulatedSignals_ElasticNet[:, i] - MeasuredSignals[:, i]) ** 2
    ReguTerm_ElasticNet[i] = elastic_net_model.l1_ratio_ * np.linalg.norm(GaussianCoefficients_ElasticNet, 1) + \
                             (1 - elastic_net_model.l1_ratio_) * np.linalg.norm(GaussianCoefficients_ElasticNet) ** 2

# Data visualization
plt.figure(figsize=(15, 12))

# Plot 1: Heatmap
plt.subplot(3, 2, 1)
plt.imshow(WeightMatrix, aspect='auto', cmap='bwr', interpolation='nearest',
           extent=[MinLambda, MaxLambda, 0, WeightMatrix.shape[0]])
plt.colorbar(label='Signal Intensity')
plt.title('Response Curves Heatmap')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Detector Index')

# Plot 2: Reference spectra or placeholder
plt.subplot(3, 2, 2)
if mode in [1, 2, 3, 5]:
    for i in range(1, SetSpectrum.shape[1]):
        plt.plot(SetSpectrum[:, 0], SetSpectrum[:, i],
                 '-', alpha=0.6)
    plt.title('Reference Spectra')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
else:
    # mode 4 No reference spectrum; placeholder only
    plt.text(0.1, 0.5, 'Mode 4: no reference spectrum', fontsize=12)
    plt.axis('off')


# Lasso Results
plt.subplot(3, 2, 3)
plt.plot(SimulatedSignals_Lasso, '*')
plt.xlabel('V_g Index')
plt.ylabel('a.u.')
plt.title('Lasso - Simulated Signals')

plt.subplot(3, 2, 4)
plt.plot(VecOfLambdasPlot * MaxLambda, HiResReconstructedSpectrum_Lasso, '*')
if mode in [1, 2, 3]:
    for i in range(1, SetSpectrum.shape[1]):
        plt.plot(SetSpectrum[:, 0], SetSpectrum[:, i], '-', alpha=0.5)
plt.title('Lasso - Reconstructed' + ('' if mode==4 else ' vs Reference'))
plt.xlabel('Wavelength (nm)')
plt.ylabel('a.u.')

# ElasticNet Results
plt.subplot(3, 2, 5)
plt.plot(SimulatedSignals_ElasticNet, '*')
plt.xlabel('V_g Index')
plt.ylabel('a.u.')
plt.title('ElasticNet - Simulated Signals')

plt.subplot(3, 2, 6)
plt.plot(VecOfLambdasPlot * MaxLambda, HiResReconstructedSpectrum_ElasticNet, '*')
if mode in [1, 2, 3]:
    for i in range(1, SetSpectrum.shape[1]):
        plt.plot(SetSpectrum[:, 0], SetSpectrum[:, i], '-', alpha=0.5)
plt.title('ElasticNet - Reconstructed' + ('' if mode==4 else ' vs Reference'))
plt.xlabel('Wavelength (nm)')
plt.ylabel('a.u.')

plt.tight_layout()
plt.show()

# Residuals and statistics: mode 4 no sampling_wavelengths or reference spectra; print only global residual statistics
if mode == 4:
    print("\nReconstruction residuals (Mode 4, no reference spectrum; global statistics only):")
    print("-" * 60)
    print(f"Mean Lasso residual:     {np.mean(MeasuResidual_Lasso):.6f} ± {np.std(MeasuResidual_Lasso):.6f}")
    print(f"Mean ElasticNet residual: {np.mean(MeasuResidual_ElasticNet):.6f} ± {np.std(MeasuResidual_ElasticNet):.6f}")

    # Bar plot showing global residuals of the methods
    plt.figure(figsize=(8, 5))
    methods = ['Lasso', 'ElasticNet']
    residuals = [np.mean(MeasuResidual_Lasso),
                 np.mean(MeasuResidual_ElasticNet)]
    plt.bar(methods, residuals)
    plt.ylabel('Mean Residual')
    plt.title('Residuals (Mode 4)')
    plt.grid(True, axis='y')
    plt.show()
else:
    # Keep the original residual visualization and wavelength-by-wavelength printing for single-/multi-peak modes
    plt.figure(figsize=(10, 6))
    if mode in [3, 5]:
        methods = ['Lasso', 'ElasticNet']
        residuals = [np.mean(MeasuResidual_Lasso),
                     np.mean(MeasuResidual_ElasticNet)]
        plt.bar(methods, residuals)
        plt.ylabel('Reconstruction Residual')
        plt.title('Reconstruction Residuals Comparison')
    elif mode == 6:
        x = np.arange(len(MeasuResidual_Lasso))
        plt.plot(x, MeasuResidual_Lasso, 'rs-', label='Lasso', markersize=8)
        plt.plot(x, MeasuResidual_ElasticNet, 'gd-', label='ElasticNet', markersize=8)
        plt.xlabel('Signal Column Index')
        plt.ylabel('Reconstruction Residual')
        plt.title('Residuals per Column (Mode 6)')
    else:
        x_lasso = sampling_wavelengths if ('sampling_wavelengths' in globals() and len(sampling_wavelengths) == len(MeasuResidual_Lasso)) else np.arange(len(MeasuResidual_Lasso))
        x_enet  = sampling_wavelengths if ('sampling_wavelengths' in globals() and len(sampling_wavelengths) == len(MeasuResidual_ElasticNet)) else np.arange(len(MeasuResidual_ElasticNet))
        plt.plot(x_lasso, MeasuResidual_Lasso, 'rs-', label='Lasso', markersize=8)
        plt.plot(x_enet,  MeasuResidual_ElasticNet, 'gd-', label='ElasticNet', markersize=8)
        plt.xlabel('Wavelength (nm)' if isinstance(x_lasso, np.ndarray) and x_lasso.dtype != int else 'Signal Column Index')
        plt.ylabel('Reconstruction Residual')
        plt.title('Residuals at Different Wavelengths' if isinstance(x_lasso, np.ndarray) and x_lasso.dtype != int else 'Residuals per Column')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.show()

    # Original printout
    print("\nReconstruction residuals at each wavelength position:")
    print("-" * 80)
    if mode in [1,2]:
        print(f"{'Wavelength (nm)':>10} {'Lasso residual':>15} {'ElasticNet residual':>15}")
        print("-" * 80)
        n = min(len(sampling_wavelengths), NumCol)
        for i in range(n):
            wl = sampling_wavelengths[i] if i < len(sampling_wavelengths) else i
            print(f"{wl:10.1f} {MeasuResidual[i]:15.6f} "
                  f"{MeasuResidual_Lasso[i]:15.6f} {MeasuResidual_ElasticNet[i]:15.6f}")
        if NumCol > n:
            # When the number of sampling wavelengths is smaller than the number of columns, print the remaining columns using column indices instead of wavelengths
            for i in range(n, NumCol):
                print(f"{i:10d} {MeasuResidual[i]:15.6f} "
                      f"{MeasuResidual_Lasso[i]:15.6f} {MeasuResidual_ElasticNet[i]:15.6f}")
    else:
        # Mode 3/5: global statistics have been shown in the bar plot; print the mean and standard deviation here
        print(f"Mean residual of the Lasso method:     {np.mean(MeasuResidual_Lasso):.6f} ± {np.std(MeasuResidual_Lasso):.6f}")
        print(f"Mean residual of the ElasticNet method: {np.mean(MeasuResidual_ElasticNet):.6f} ± {np.std(MeasuResidual_ElasticNet):.6f}")

# Keep the R2 score comparison; all scores compare MeasuredSignals with the SimulatedSignals of each method
print("\nR2 scores:")
print(f"Lasso: {r2_score(MeasuredSignals.flatten(), SimulatedSignals_Lasso.flatten()):.6f}")
print(f"ElasticNet: {r2_score(MeasuredSignals.flatten(), SimulatedSignals_ElasticNet.flatten()):.6f}")

# ===== Create a save directory in the current directory and save results there =====
import os
import datetime

def build_header(title, arr, extra_info: dict):
    info_lines = [f"{title}",
                  f"shape={arr.shape}",
                  f"timestamp={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    for k, v in extra_info.items():
        info_lines.append(f"{k}={v}")
    return " | ".join(info_lines)

# ...existing code...
def save_txt(path, arr, title, extra_info):
    # Ensure the directory exists to avoid FileNotFoundError
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = build_header(title, arr, extra_info)
    np.savetxt(path, arr, fmt="%.8f", delimiter="\t", header=header)
# ...existing code...
# === Print and save the final MeasuredSignals used for reconstruction (unnormalized) ===
MeasuredSignals_raw = MeasuredSignals.copy()
# ...existing code...
out_path = os.path.join(out_dir, "measured_signals_raw.txt")
# New:Ensure the save directory exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savetxt(
    out_path,
    MeasuredSignals_raw,
    fmt="%.8f",
    delimiter="\t",
    header=f"MeasuredSignals rows={MeasuredSignals_raw.shape[0]}, cols={MeasuredSignals_raw.shape[1]}"
)
# ...existing code...
# Summarize condition information
cond = {
    "mode": mode,
    "MinLambda_nm": MinLambda,
    "MaxLambda_nm": MaxLambda,
    "FWHMset_nm": FWHMset,
    "shrink_factor": shrink_factor,
    "gamma_shrink": gamma_shrink,
    "NumDetectors": ResponseCurves.shape[0],
    "NumSignals": NumCol
}
if mode in [1, 2, 3]:
    cond["sampling_wavelengths_nm"] = ",".join([f"{w:.3f}" for w in sampling_wavelengths.tolist()])
if mode in [1, 2]:
    cond["sampling_fwhms_nm"] = ",".join([f"{f:.3f}" for f in sampling_fwhms.tolist()])
if mode == 3 and 'peaks_info' in globals():
    cond["peaks_info"] = ";".join([f"λ={w:.3f},weight={wt:.3f},FWHM={fw:.3f}" for (w, wt, fw) in peaks_info])

# Package high-resolution spectra; the first column is wavelength in nm
def stack_spectrum_for_save(wavelengths_nm, spectra_2d):
    return np.column_stack([wavelengths_nm, spectra_2d])

# Save all result types to the newly created directory
save_txt(os.path.join(out_dir, "MeasuredSignals.txt"), MeasuredSignals, "MeasuredSignals", cond)
save_txt(os.path.join(out_dir, "SimulatedSignals_Lasso.txt"), SimulatedSignals_Lasso, "SimulatedSignals Lasso", cond)
save_txt(os.path.join(out_dir, "SimulatedSignals_ElasticNet.txt"), SimulatedSignals_ElasticNet, "SimulatedSignals ElasticNet", cond)


save_txt(os.path.join(out_dir, "ReconstructedSpectrum_Lasso.txt"),
         stack_spectrum_for_save(VecOfLambdasPlot * MaxLambda, HiResReconstructedSpectrum_Lasso),
         "HiRes Reconstructed Spectrum Lasso (wavelength_nm + spectra columns)", cond)
save_txt(os.path.join(out_dir, "ReconstructedSpectrum_ElasticNet.txt"),
         stack_spectrum_for_save(VecOfLambdasPlot * MaxLambda, HiResReconstructedSpectrum_ElasticNet),
         "HiRes Reconstructed Spectrum ElasticNet (wavelength_nm + spectra columns)", cond)

save_txt(os.path.join(out_dir, "Residuals_Lasso.txt"), MeasuResidual_Lasso[np.newaxis, :], "Residuals Lasso (row vector)", cond)
save_txt(os.path.join(out_dir, "Residuals_ElasticNet.txt"), MeasuResidual_ElasticNet[np.newaxis, :], "Residuals ElasticNet (row vector)", cond)
save_txt(os.path.join(out_dir, "Regularization_Lasso.txt"), ReguTerm_Lasso[np.newaxis, :], "Regularization Lasso (row vector)", cond)
save_txt(os.path.join(out_dir, "Regularization_ElasticNet.txt"), ReguTerm_ElasticNet[np.newaxis, :], "Regularization ElasticNet (row vector)", cond)

# Save reference spectra if available
if mode in [1, 2, 3] and 'SetSpectrum' in globals():
    save_txt(os.path.join(out_dir, "Reference_SetSpectrum.txt"), SetSpectrum,
             "Reference SetSpectrum (col0=wavelength_nm, others=spectra)", cond)

print(f"Results have been saved to directory: {out_dir}")

# ===== Mode 1: analyze and save reconstructed main-peak deviation versus noise (SNR) =====
if mode == 1 and 'snr_blocks' in locals() and len(snr_blocks) > 0:
    wl_axis_nm = VecOfLambdasPlot * MaxLambda
    group_size = min(len(sampling_wavelengths), snr_blocks[0]['end'] - snr_blocks[0]['start'])

    # Detailed table: record the main-peak deviation for each SNR and each peak (Lasso/ElasticNet)
    rows = []
    for blk in snr_blocks:
        snr_db = blk['snr_db']
        start, end = blk['start'], blk['end']
        end = min(end, HiResReconstructedSpectrum_Lasso.shape[1])  # Safe truncation
        cols = end - start
        K = min(group_size, cols)
        for k in range(K):
            col = start + k
            ref_wl = float(sampling_wavelengths[k])

            # Lasso main peak
            spec_l = HiResReconstructedSpectrum_Lasso[:, col]
            pk_l = int(np.argmax(spec_l))
            recon_wl_l = float(wl_axis_nm[pk_l])
            err_l = abs(recon_wl_l - ref_wl)

            # ElasticNet main peak
            spec_e = HiResReconstructedSpectrum_ElasticNet[:, col]
            pk_e = int(np.argmax(spec_e))
            recon_wl_e = float(wl_axis_nm[pk_e])
            err_e = abs(recon_wl_e - ref_wl)

            rows.append([snr_db, k, ref_wl, recon_wl_l, err_l, recon_wl_e, err_e])

    rows = np.asarray(rows, dtype=float)
    header = "Peak error vs SNR (Mode 1)\n" \
             "columns: SNR_dB, peak_index_in_group, ref_wavelength_nm, " \
             "recon_peak_nm_Lasso, abs_error_nm_Lasso, recon_peak_nm_ElasticNet, abs_error_nm_ElasticNet"
    out_txt_detail = os.path.join(out_dir, "PeakError_vs_SNR_Detail.txt")
    np.savetxt(out_txt_detail, rows, fmt="%.6f", delimiter="\t", header=header)
    print(f"Saved main-peak deviation details: {out_txt_detail}")

    # Aggregate: mean deviation over all peaks for each SNR
    snrs = sorted(list({b['snr_db'] for b in snr_blocks}))
    mean_lasso, mean_enet = [], []
    for snr_db in snrs:
        mask = rows[:, 0] == snr_db
        mean_lasso.append(np.mean(rows[mask, 4]) if np.any(mask) else np.nan)
        mean_enet.append(np.mean(rows[mask, 6]) if np.any(mask) else np.nan)
    agg = np.column_stack([snrs, mean_lasso, mean_enet])
    out_txt_mean = os.path.join(out_dir, "PeakError_vs_SNR_Mean.txt")
    np.savetxt(out_txt_mean, agg, fmt="%.6f", delimiter="\t",
               header="columns: SNR_dB, mean_abs_error_nm_Lasso, mean_abs_error_nm_ElasticNet")
    print(f"Saved mean main-peak deviation: {out_txt_mean}")

    # Plot: SNR (dB) versus mean main-peak deviation (nm)
    plt.figure(figsize=(7, 5))
    plt.plot(snrs, mean_lasso, 'ro-', label='Lasso')
    plt.plot(snrs, mean_enet, 'gd-', label='ElasticNet')
    plt.gca().invert_xaxis()  # Lower SNR means higher noise; inversion is more intuitive and can be removed if needed
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean |Δλ| (nm)')
    plt.title('Main-Peak Deviation vs Noise (Mode 1)')
    plt.grid(True)
    plt.legend()
    fig_path = os.path.join(out_dir, "PeakError_vs_SNR.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved figure: {fig_path}")

# Keep the loss-function convergence curve unchanged
plt.figure(figsize=(10, 6))
for method, histories in loss_histories.items():
    for i, history in enumerate(histories):
        history = np.array(history)
        if history.ndim > 2:
            history = np.mean(history, axis=(1, 2))
        elif history.ndim == 2:
            history = np.mean(history, axis=1)
        plt.plot(history, label=f'{method.capitalize()} - Signal {i + 1}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function Convergence')
plt.legend()
plt.grid(True)
plt.show()

# After saving results, add comparison and difference plots between initial and reconstructed spectra for mode 6
if mode == 6 and 'SetSpectrum' in globals():
    # Initial spectra interpolated to VecOfLambdasPlot for comparison
    init_interp = np.zeros((VecOfLambdasPlot.shape[0], SetSpectrum.shape[1] - 1))
    w_in = SetSpectrum[:, 0]
    for m in range(SetSpectrum.shape[1] - 1):
        tck = splrep(w_in, SetSpectrum[:, m + 1], s=0)
        init_interp[:, m] = splev(VecOfLambdasPlot * MaxLambda, tck)

    # Normalize each column to [0, 1]
    def normalize_cols(A):
        B = A.copy().astype(float)
        for j in range(B.shape[1]):
            mx = np.max(np.abs(B[:, j]))
            if mx > 0:
                B[:, j] = B[:, j] / mx
        return B

    init_norm  = normalize_cols(init_interp)
    recon_lasso = normalize_cols(HiResReconstructedSpectrum_Lasso)  # Extra safeguard to ensure column normalization

    # Plot the normalized comparison
    plt.figure(figsize=(12, 6))
    for m in range(init_norm.shape[1]):
        plt.plot(VecOfLambdasPlot * MaxLambda, init_norm[:, m], 'k-', alpha=0.6, label='Initial (norm)' if m == 0 else None)
        plt.plot(VecOfLambdasPlot * MaxLambda, recon_lasso[:, m], 'r--', alpha=0.7, label='Reconstructed (norm, Lasso)' if m == 0 else None)
    plt.title('Initial Spectra vs Reconstructed Spectra (Normalized, Mode 6)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Initial_vs_Reconstructed_Lasso_Mode6_Normalized.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Normalized difference map(Initial_norm - Recon_norm)
    diff = init_norm - recon_lasso
    plt.figure(figsize=(10, 5))
    plt.imshow(diff.T, aspect='auto', cmap='seismic',
               extent=[VecOfLambdasPlot[0]*MaxLambda, VecOfLambdasPlot[-1]*MaxLambda, 0, init_norm.shape[1]])
    plt.colorbar(label='Difference (Initial_norm - Recon_norm)')
    plt.title('Normalized Difference Map (Mode 6; Lasso)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spectrum Index')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Difference_Initial_minus_Reconstructed_Lasso_Mode6_Normalized.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Save the normalized comparison data
    np.savetxt(os.path.join(out_dir, "Initial_Interpolated_Mode6_Normalized.txt"),
               np.column_stack([VecOfLambdasPlot * MaxLambda, init_norm]),
               fmt="%.8f", delimiter="\t",
               header="Initial spectra (normalized) interpolated to reconstruction grid | λ + spectra")
    np.savetxt(os.path.join(out_dir, "Reconstructed_Lasso_Mode6_Normalized.txt"),
               np.column_stack([VecOfLambdasPlot * MaxLambda, recon_lasso]),
               fmt="%.8f", delimiter="\t",
               header="Reconstructed spectra (Lasso, normalized) | λ + spectra")
    np.savetxt(os.path.join(out_dir, "Difference_Initial_minus_Reconstructed_Lasso_Mode6_Normalized.txt"),
               np.column_stack([VecOfLambdasPlot * MaxLambda, diff]),
               fmt="%.8f", delimiter="\t",
               header="Difference (Initial_norm - Reconstructed_norm, Lasso) | λ + spectra")