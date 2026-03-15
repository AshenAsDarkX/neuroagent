"""
EEG-based BCI Decoder using c-VEP (code-modulated VEP) with CCA

Based on:
- EEGChat: https://github.com/AKMeunier/EEGChat
- EEG2Code / World's Fastest BCI paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6730910/
- Dataset: https://figshare.com/articles/dataset/7701065

Architecture:
1. Load pre-recorded c-VEP dataset (VP1.mat)
2. Build reference signals as TIME-SHIFTED copies of each target's binary stimulation code
   (this is the correct c-VEP CCA approach — NOT sinusoidal harmonics)
3. Use CCA to compute correlation between live EEG and each target's reference matrix
4. Return the target with highest mean canonical correlation across trials

KEY FIX vs previous version:
- Old code extracted a "dominant frequency" via FFT and built sine/cosine references.
  All 6 m-sequence codes share similar spectral content → references were nearly identical
  → CCA always favoured target 0 (highest energy by chance).
- Correct approach: each target's reference is a matrix of L time-shifted copies of its
  own binary code, exactly as described in Nakanishi et al. 2018 / Bin et al. 2011 for
  c-VEP BCIs. This gives each target a structurally distinct reference.
"""

import numpy as np
import scipy.io
import tensorflow as tf
from typing import Tuple, Optional
import os


class EEG2CodeBCI:
    """
    c-VEP BCI decoder using Canonical Correlation Analysis (CCA).

    Reference signals are time-shifted copies of the binary stimulation codes,
    not sinusoidal harmonics. This matches the EEG2Code / c-VEP paradigm where
    each flickering target is driven by a distinct pseudo-random binary sequence.
    """

    def __init__(
            self,
            model_path: str,
            dataset_path: str,
            status_callback=None,
            clear_callback=None,
            correlation_callback=None,
            eeg_callback=None,
    ):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.status_callback = status_callback or (lambda msg: None)
        self.clear_callback = clear_callback or (lambda: None)
        self.correlation_callback = correlation_callback or (lambda corr, sel: None)
        self.eeg_callback = eeg_callback or (lambda ch0, code, label: None)

        self.target_count = 6
        self.sampling_rate = 600          # Hz
        self.stim_refresh_rate = 60       # Hz
        self.samples_per_bit = self.sampling_rate // self.stim_refresh_rate  # 10 samples/bit

        # CCA parameters — number of time lags to include in the reference matrix.
        # More lags capture more of the temporal structure; 6-10 is typical.
        self.n_lags = 8

        # How many seconds of (simulated) EEG to analyse per trial.
        self.eeg_length_s = 2.0
        self.n_samples = int(self.eeg_length_s * self.sampling_rate)  # 1200 samples

        # Confidence threshold (tuned for 6 targets; raise if false positives occur).
        self.accuracy_threshold = 0.25

        # Average this many simulated trials before committing to a selection.
        self.trials_per_group = 4

        print("[BCI] Loading EEG2Code model...")
        self._load_model()

        print("[BCI] Loading dataset...")
        self._load_dataset()

        print("[BCI] Building c-VEP reference signals (time-shifted codes)...")
        self._build_reference_signals()

        self.clear_callback()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the pre-trained Keras model (used for feature extraction if available)."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                print(f"[BCI] Model loaded from {self.model_path}")
            except Exception as e:
                print(f"[BCI] Model load failed: {e}")
                self.model = None
        else:
            print(f"[BCI] Model not found at {self.model_path}, using CCA-only mode")
            self.model = None

    def _load_dataset(self):
        """
        Load VP1.mat and extract each target's binary stimulation code.

        The dataset contains 'test_data_y': stimulation bit patterns for each target.
        We upsample them back to EEG sample-rate so the code length matches the EEG
        segment length used during decoding.
        """
        data = scipy.io.loadmat(self.dataset_path)
        stim_bits = data["test_data_y"]

        self.stim_codes = []        # binary codes at stim refresh rate (for display)
        self.stim_codes_upsampled = []  # binary codes at EEG sample rate (for CCA)

        for i in range(self.target_count):
            raw = stim_bits[i].reshape(-1)

            # Binarise
            bits = (raw > 0).astype(float)

            # Downsample to stim refresh rate (kept for compatibility with bci_display)
            downsampled = bits[::self.samples_per_bit]
            offset = 7
            ppb = 2
            aligned = downsampled[offset: offset + 285 * ppb: ppb]
            self.stim_codes.append(aligned)

            # Upsample back: repeat each bit samples_per_bit times so the code
            # length matches self.n_samples (or we tile it to reach that length).
            upsampled = np.repeat(aligned, self.samples_per_bit)

            # Tile / trim to exactly n_samples so every target's reference is the
            # same length regardless of code length.
            if len(upsampled) < self.n_samples:
                repeats = int(np.ceil(self.n_samples / len(upsampled)))
                upsampled = np.tile(upsampled, repeats)
            upsampled = upsampled[:self.n_samples]

            self.stim_codes_upsampled.append(upsampled)

            print(f"[BCI] Target {i}: aligned bits={len(aligned)}, "
                  f"upsampled length={len(upsampled)}")

    def _build_reference_signals(self):
        """
        Build reference matrices for CCA using TIME-SHIFTED copies of each
        target's binary stimulation code.

        For c-VEP BCIs the EEG response to a flickering stimulus is strongly
        correlated with the stimulus code itself and its time-lagged versions
        (due to the haemodynamic / neural impulse response).  Stacking L lagged
        copies gives a reference matrix Y ∈ R^{n_samples × n_lags} whose column
        space spans the expected EEG response.

        This replaces the previous incorrect approach of using sinusoidal
        harmonics derived from a dominant FFT frequency.
        """
        self.reference_signals = []

        for target_idx in range(self.target_count):
            code = self.stim_codes_upsampled[target_idx]  # shape (n_samples,)
            ref_cols = []

            for lag in range(self.n_lags):
                # Circular shift: positive lag = code leads the EEG
                shifted = np.roll(code, lag)
                ref_cols.append(shifted)

            # Y: shape (n_samples, n_lags)
            ref_matrix = np.column_stack(ref_cols)
            self.reference_signals.append(ref_matrix)

        print(f"[BCI] Reference matrices: {self.target_count} targets × "
              f"({self.n_samples} samples × {self.n_lags} lags)")

    # ------------------------------------------------------------------
    # CCA
    # ------------------------------------------------------------------

    def _canonical_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the first canonical correlation between EEG matrix X and
        reference matrix Y.

        Args:
            X: EEG data          (n_samples, n_channels)
            Y: Reference signals (n_samples, n_lags)

        Returns:
            First canonical correlation coefficient in [0, 1].
        """
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

        n = X.shape[0]
        reg = 1e-6  # Tikhonov regularisation to avoid singular matrices

        Cxx = (X.T @ X) / (n - 1) + reg * np.eye(X.shape[1])
        Cyy = (Y.T @ Y) / (n - 1) + reg * np.eye(Y.shape[1])
        Cxy = (X.T @ Y) / (n - 1)

        try:
            # Whitening transforms
            Lxx = np.linalg.cholesky(Cxx)
            Lyy = np.linalg.cholesky(Cyy)

            inv_Lxx = np.linalg.inv(Lxx)
            inv_Lyy = np.linalg.inv(Lyy)

            # Reduced cross-covariance matrix
            M = inv_Lxx @ Cxy @ inv_Lyy.T

            # Largest singular value = first canonical correlation
            sv = np.linalg.svd(M, compute_uv=False)
            rho = float(sv[0]) if len(sv) > 0 else 0.0
            return float(np.clip(rho, 0.0, 1.0))

        except np.linalg.LinAlgError:
            # Fall back to eigenvalue method
            try:
                inv_Cxx = np.linalg.inv(Cxx)
                inv_Cyy = np.linalg.inv(Cyy)
                M2 = inv_Cxx @ Cxy @ inv_Cyy @ Cxy.T
                eigs = np.real(np.linalg.eigvals(M2))
                eigs = eigs[eigs > 0]
                if len(eigs) == 0:
                    return 0.0
                return float(np.clip(np.sqrt(np.max(eigs)), 0.0, 1.0))
            except np.linalg.LinAlgError:
                return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_selection(self, target_hint: int = 0) -> Tuple[Optional[int], float, int]:
        """
        Decode which target the user is focusing on.

        In production this would acquire live EEG, band-pass filter it, and run
        CCA against all reference signals.  In simulation mode we generate
        synthetic EEG whose structure matches the target indicated by target_hint,
        which allows the algorithm to be exercised without hardware.

        Args:
            target_hint: 0-5 — which target to simulate (used ONLY in sim mode).

        Returns:
            (selected_target, confidence, trial_index)
            selected_target: 0-5, or None if confidence < threshold.
        """
        trial_index = 0
        group_trials = list(range(self.trials_per_group))

        print(f"[BCI] using trial group {trial_index} "
              f"({len(group_trials)} trial(s): {group_trials})")

        all_correlations = []
        target = int(target_hint) % self.target_count
        print(f"[BCI] Simulating EEG for target={target} (hint={target_hint})")

        for trial_idx in group_trials:
            simulated_eeg = self._simulate_eeg_trial(target, noise_level=0.6)

            correlations = []
            for t_idx in range(self.target_count):
                rho = self._canonical_correlation(
                    simulated_eeg,
                    self.reference_signals[t_idx],
                )
                correlations.append(rho)

            all_correlations.append(correlations)
            print(f"[BCI] trial {trial_idx}: CCA correlations: "
                  f"{', '.join(f'{r:.3f}' for r in correlations)}")

            # Fire live update so the display can show the running mean
            running_mean = list(np.mean(all_correlations, axis=0))
            running_sel = int(np.argmax(running_mean))
            self.correlation_callback(running_mean, running_sel)

            # Send raw EEG + stimulation code to the waveform display
            eeg_ch0 = simulated_eeg[:, 0].tolist()
            code = self.stim_codes_upsampled[target].tolist()
            label_text = f"Trial {trial_idx + 1} | T{target + 1}"
            self.eeg_callback(eeg_ch0, code, label_text)

        mean_corr = np.mean(all_correlations, axis=0)
        std_corr = np.std(all_correlations, axis=0)

        print(f"[BCI] Mean correlations: {', '.join(f'{r:.3f}' for r in mean_corr)}")
        print(f"[BCI] Std  correlations: {', '.join(f'{r:.3f}' for r in std_corr)}")

        selected_idx = int(np.argmax(mean_corr))
        confidence = float(mean_corr[selected_idx])

        print(f"[BCI] group {trial_index} averaged decode → "
              f"target={selected_idx}, confidence={confidence:.3f}")

        if confidence < self.accuracy_threshold:
            print(f"[BCI] low-confidence decode; no selection "
                  f"(trial={trial_index}, accuracy={confidence:.3f})")
            return None, confidence, trial_index

        print(f"[BCI] HIGH-CONFIDENCE decode: target={selected_idx}, "
              f"confidence={confidence:.3f}")

        return selected_idx, confidence, trial_index

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _simulate_eeg_trial(self, target: int, noise_level: float = 0.5) -> np.ndarray:
        """
        Simulate a single EEG trial for testing without hardware.

        The simulated EEG is constructed so that it genuinely resembles the
        neural response to the target stimulus: the target's upsampled binary
        code is mixed across channels with realistic EEG noise.  This means
        CCA should consistently recover the correct target — unlike the
        previous implementation that used sine waves unrelated to the actual
        stimulus codes.

        Args:
            target:      Which target (0-5) the simulated user is "attending to".
            noise_level: 0 = clean signal, 1 = noise only.

        Returns:
            Simulated EEG array (n_samples, n_channels=8).
        """
        n_channels = 8

        # Start from the target's actual binary stimulation code
        # (upsampled to EEG rate).  This is the ground-truth neural driver.
        code = self.stim_codes_upsampled[target]          # (n_samples,)

        # Initialise with Gaussian + pink noise
        eeg = np.random.randn(self.n_samples, n_channels) * noise_level

        for ch in range(n_channels):
            eeg[:, ch] += self._generate_pink_noise(self.n_samples) * 0.3 * noise_level

        # Inject the target code into each channel with a small per-channel phase shift
        # (simulates the spatially distributed SSVEP / c-VEP response).
        signal_amplitude = 1.0 - noise_level
        for ch in range(n_channels):
            lag = ch  # small lag per channel (0-7 samples) — realistic for c-VEPs
            shifted_code = np.roll(code, lag)
            eeg[:, ch] += shifted_code * signal_amplitude

        return eeg

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate 1/f (pink) noise to simulate realistic EEG background."""
        white = np.random.randn(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-10  # avoid division by zero at DC
        fft = fft / np.sqrt(np.abs(freqs))
        pink = np.fft.irfft(fft, n=n_samples)
        return pink / (np.std(pink) + 1e-9)