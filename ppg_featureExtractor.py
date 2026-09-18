import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# -----------------------------
# Load Dataset
# -----------------------------
file_path = "PPG_Dataset.csv"   # Change path if needed
df = pd.read_csv(file_path)

signals = df.drop(columns=['Label']).values
labels = df['Label'].values


# -----------------------------
# Statistical Feature Extraction
# -----------------------------
def extract_stat_features(signal):
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
        'peak_to_peak': np.ptp(signal),
        'rms': np.sqrt(np.mean(signal**2))
    }


# -----------------------------
# Morphology Feature Extraction
# -----------------------------
def extract_ppg_morphology_features(signal, fs=1000):
    """
    Extract morphology-based features: systolic/diastolic amplitudes,
    pulse rate, reflection index, etc.
    fs: Sampling frequency (default 1000Hz)
    """
    # Find systolic peaks
    peaks, _ = find_peaks(signal, distance=fs*0.5)  # at least 0.5s apart

    if len(peaks) < 2:
        return {
            'pulse_rate': np.nan,
            'avg_systolic_amp': np.nan,
            'avg_diastolic_amp': np.nan,
            'reflection_index': np.nan,
            'augmentation_index': np.nan,
            'systolic_time': np.nan,
            'diastolic_time': np.nan
        }

    # Pulse rate
    rr_intervals = np.diff(peaks) / fs
    pulse_rate = 60 / np.mean(rr_intervals)

    systolic_amps = signal[peaks]
    diastolic_amps = []
    systolic_times = []
    diastolic_times = []

    # Detect diastolic notch as the minimum between two systolic peaks
    for i in range(len(peaks)-1):
        segment = signal[peaks[i]:peaks[i+1]]
        min_idx = np.argmin(segment) + peaks[i]
        diastolic_amps.append(signal[min_idx])
        systolic_times.append(peaks[i] / fs)
        diastolic_times.append(min_idx / fs)

    avg_systolic_amp = np.mean(systolic_amps)
    avg_diastolic_amp = np.mean(diastolic_amps)

    reflection_index = (avg_diastolic_amp / avg_systolic_amp) if avg_systolic_amp != 0 else np.nan
    augmentation_index = ((avg_systolic_amp - avg_diastolic_amp) / avg_systolic_amp) if avg_systolic_amp != 0 else np.nan

    return {
        'pulse_rate': pulse_rate,
        'avg_systolic_amp': avg_systolic_amp,
        'avg_diastolic_amp': avg_diastolic_amp,
        'reflection_index': reflection_index,
        'augmentation_index': augmentation_index,
        'systolic_time': np.mean(systolic_times),
        'diastolic_time': np.mean(diastolic_times)
    }


# -----------------------------
# Feature Extraction Loop
# -----------------------------
features_all = []

for i, sig in enumerate(signals):
    stat_f = extract_stat_features(sig)
    morph_f = extract_ppg_morphology_features(sig)
    
    # Merge features
    combined_features = {**stat_f, **morph_f}
    combined_features['Label'] = labels[i]
    features_all.append(combined_features)

# Convert to DataFrame
features_df = pd.DataFrame(features_all)

# -----------------------------
# Save Features to CSV
# -----------------------------
output_path = "PPG_Extracted_Features.csv"
features_df.to_csv(output_path, index=False)
print(f"✅ Features saved to {output_path}")


# -----------------------------
# Plot Example Waveforms
# -----------------------------
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(signals[i], label=f'Sample {i+1} ({labels[i]})')
plt.legend()
plt.title("Example PPG Waveforms")
plt.xlabel("Time Points")
plt.ylabel("Amplitude")
plt.show()