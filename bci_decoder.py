import numpy as np
import scipy.io
from tensorflow.keras.models import load_model


class EEG2CodeBCI:

    def __init__(self, model_path, dataset_path, status_callback=None):
        self._status_callback = status_callback
        self._status("EEG2Code loading...")

        print("[BCI] Loading EEG2Code model...")
        self.model = load_model(model_path)
        self._status("EEG2Code model loaded")

        print("[BCI] Loading dataset...")
        self._status("EEG2Code dataset loading...")
        data = scipy.io.loadmat(dataset_path)

        self.eeg_trials = data["test_data_x"]
        self.stim_bits = data["test_data_y"]

        self.trial_index = 0

        self.window_size = 150
        self.downsample = 10
        self._status("EEG2Code ready")

    def _status(self, message):
        if not self._status_callback:
            return
        try:
            self._status_callback(message)
        except Exception:
            pass

    def _predict_bits(self, trial):

        channels = trial.shape[0]
        samples = trial.shape[1]

        windows = []

        for t in range(samples - self.window_size):
            w = trial[:, t:t+self.window_size]
            windows.append(w.T)

        windows = np.array(windows)

        probs = self.model.predict(windows, verbose=0)

        bits = np.argmax(probs, axis=1)

        return bits

    def _decode_target(self, pred_bits, true_bits):

        pred_bits = pred_bits[::self.downsample]
        true_bits = true_bits[::self.downsample]

        L = min(len(pred_bits), len(true_bits))

        acc = np.mean(pred_bits[:L] == true_bits[:L])

        return acc

    def get_selection(self):

        trial = self.eeg_trials[self.trial_index]
        true_bits = self.stim_bits[self.trial_index]

        pred_bits = self._predict_bits(trial)

        acc = self._decode_target(pred_bits, true_bits)

        print(f"[BCI] trial {self.trial_index} bit accuracy: {acc:.3f}")

        selection = self.trial_index % 4

        self.trial_index += 1

        return selection
