"""
Demo-only CSI processing and sklearn inference helpers.

This module intentionally lives outside the normal collection path so the
standard workflow remains untouched. It ports the user's CSI processing and
feature extraction logic into a streaming form suitable for the live demo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Deque, List, Optional, Sequence

import numpy as np

from picoscenes_collection.utils.csi import extract_spatial_streams


DEFAULT_CLASS_NAMES = ["clap", "push", "tap"]


@dataclass
class LiveDemoPrediction:
    label: str
    score: float
    scores: Optional[np.ndarray] = None
    raw_label: Optional[str] = None
    raw_score: Optional[float] = None


class LiveDemoSklearnGestureClassifier:
    """
    Streaming wrapper around the provided CSI Doppler pipeline + sklearn model.

    Notes:
    - Uses a rolling packet buffer to build a live CSI segment.
    - Only runs inference every `infer_stride_packets` packets to limit load.
    - Returns the most recent prediction between inference steps.
    """

    def __init__(
        self,
        model_path: str,
        *,
        class_names: Optional[Sequence[str]] = None,
        fs: int = 100,
        n_subcarriers: int = 57,
        n_streams: int = 4,
        hp_cutoff: float = 0.6,
        lp_cutoff: float = 25.0,
        hp_order: int = 2,
        lp_order: int = 3,
        stft_nperseg: int = 16,
        stft_noverlap: int = 14,
        max_doppler_hz: float = 35.0,
        t_target: int = 30,
        window_packets: int = 96,
        min_packets: int = 48,
        infer_stride_packets: int = 8,
        score_threshold: float = 0.8,
        vote_window: int = 5,
        min_votes: int = 3,
        hold_seconds: float = 0.75,
        unknown_label: str = "unknown",
        debug: bool = False,
        debug_dir: Optional[str] = None,
        debug_every_inferences: int = 10,
    ) -> None:
        self.model_path = str(model_path)
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)

        self.fs = int(fs)
        self.n_subcarriers = int(n_subcarriers)
        self.n_streams = int(n_streams)
        self.hp_cutoff = float(hp_cutoff)
        self.lp_cutoff = float(lp_cutoff)
        self.hp_order = int(hp_order)
        self.lp_order = int(lp_order)
        self.stft_nperseg = int(stft_nperseg)
        self.stft_noverlap = int(stft_noverlap)
        self.max_doppler_hz = float(max_doppler_hz)
        self.t_target = int(t_target)
        self.window_packets = int(window_packets)
        self.min_packets = int(min_packets)
        self.infer_stride_packets = int(infer_stride_packets)
        self.score_threshold = float(score_threshold)
        self.vote_window = int(vote_window)
        self.min_votes = int(min_votes)
        self.hold_seconds = float(hold_seconds)
        self.unknown_label = str(unknown_label)
        self.debug = bool(debug)
        self.debug_every_inferences = int(debug_every_inferences)
        self.debug_dir = Path(debug_dir).expanduser() if debug_dir else None

        if self.window_packets <= 0:
            raise ValueError("window_packets must be > 0")
        if self.min_packets <= 0:
            raise ValueError("min_packets must be > 0")
        if self.vote_window <= 0:
            raise ValueError("vote_window must be > 0")
        if self.min_votes <= 0:
            raise ValueError("min_votes must be > 0")
        if self.debug_every_inferences <= 0:
            raise ValueError("debug_every_inferences must be > 0")

        self._history: Deque[np.ndarray] = deque(maxlen=self.window_packets)
        self._packets_since_infer = 0
        self._last_prediction: Optional[LiveDemoPrediction] = None
        self._stable_prediction: Optional[LiveDemoPrediction] = None
        self._last_switch_time = 0.0
        self._vote_labels: Deque[str] = deque(maxlen=self.vote_window)
        self._vote_scores: Deque[float] = deque(maxlen=self.vote_window)
        self._inference_count = 0
        self._model = self._load_model(self.model_path)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def predict_packet(self, packet_data) -> Optional[LiveDemoPrediction]:
        sample = self._extract_complex_packet(packet_data)
        if sample is None:
            return self._last_prediction

        self._history.append(sample)
        self._packets_since_infer += 1

        if len(self._history) < self.min_packets:
            return self._last_prediction
        if self._packets_since_infer < self.infer_stride_packets and self._last_prediction:
            return self._last_prediction

        csi_segment = np.stack(list(self._history), axis=0)
        doppler = self._process_segment(csi_segment)
        if doppler is None or doppler.size == 0:
            return self._last_prediction

        features = self._extract_features(doppler)
        raw_pred = self._predict_features(features)
        pred = self._stabilize_prediction(raw_pred)
        self._inference_count += 1
        if self.debug:
            self._emit_debug(doppler, raw_pred, pred)
        self._packets_since_infer = 0
        self._last_prediction = pred
        return pred

    def _load_model(self, model_path: str):
        try:
            import joblib
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "joblib is required for the live demo sklearn model"
            ) from exc

        return joblib.load(model_path)

    def _extract_complex_packet(self, packet_data) -> Optional[np.ndarray]:
        csi_block = packet_data.get("CSI") or {}
        streams, _ = extract_spatial_streams(csi_block)
        if not streams:
            return None

        complex_rows: List[np.ndarray] = []
        for stream in streams[: self.n_streams]:
            if stream.complex_csi is None:
                return None
            try:
                vec = np.asarray(stream.complex_csi, dtype=np.complex128)
            except Exception:
                return None
            if vec.ndim != 1:
                return None
            complex_rows.append(vec)

        if len(complex_rows) != self.n_streams:
            return None

        try:
            stacked = np.stack(complex_rows, axis=1)
        except Exception:
            return None

        if stacked.shape != (self.n_subcarriers, self.n_streams):
            return None

        return stacked.reshape(-1)

    def _reshape_csi(self, csi_raw: np.ndarray) -> np.ndarray:
        expected = self.n_subcarriers * self.n_streams
        if csi_raw.ndim != 2 or csi_raw.shape[1] != expected:
            raise ValueError(
                f"Expected CSI segment with shape (T, {expected}), got {csi_raw.shape}"
            )
        return csi_raw.reshape(csi_raw.shape[0], self.n_subcarriers, self.n_streams)

    def _select_reference_stream(self, csi_3d: np.ndarray) -> int:
        amp = np.abs(csi_3d)
        mean_amp = np.mean(amp, axis=0)
        std_amp = np.sqrt(np.var(amp, axis=0))
        ratio = mean_amp / (std_amp + 1e-8)
        score = np.mean(ratio, axis=0)
        return int(np.argmax(score))

    def _amplitude_adjust(self, csi_3d: np.ndarray, ref_idx: int):
        t_size, n_subs, n_streams = csi_3d.shape
        csi_adj = np.zeros_like(csi_3d, dtype=np.complex128)

        ref = csi_3d[:, :, ref_idx]

        alpha_sum = 0.0
        for stream_idx in range(n_streams):
            for sub_idx in range(n_subs):
                amp = np.abs(csi_3d[:, sub_idx, stream_idx])
                non_zero = amp[amp != 0]
                alpha = float(np.min(non_zero)) if len(non_zero) > 0 else 0.0
                alpha_sum += alpha
                csi_adj[:, sub_idx, stream_idx] = np.abs(amp - alpha) * np.exp(
                    1j * np.angle(csi_3d[:, sub_idx, stream_idx])
                )

        beta = 1000.0 * alpha_sum / (n_subs * n_streams)

        csi_ref_adj = np.zeros_like(csi_3d, dtype=np.complex128)
        for stream_idx in range(n_streams):
            for sub_idx in range(n_subs):
                ref_amp = np.abs(ref[:, sub_idx])
                ref_phase = np.angle(ref[:, sub_idx])
                csi_ref_adj[:, sub_idx, stream_idx] = (ref_amp + beta) * np.exp(
                    1j * ref_phase
                )

        return csi_adj, csi_ref_adj

    def _conjugate_multiply(
        self, csi_adj: np.ndarray, csi_ref_adj: np.ndarray, ref_idx: int
    ) -> np.ndarray:
        conj_mult = csi_adj * np.conj(csi_ref_adj)
        return np.delete(conj_mult, ref_idx, axis=2)

    def _bandpass_filter_complex(self, x: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import butter, filtfilt
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("scipy is required for live CSI filtering") from exc

        nyq = self.fs / 2.0
        if not (
            0 < self.hp_cutoff < nyq
            and 0 < self.lp_cutoff < nyq
            and self.hp_cutoff < self.lp_cutoff
        ):
            raise ValueError(
                f"Invalid bandpass range hp={self.hp_cutoff}, lp={self.lp_cutoff}, fs={self.fs}"
            )

        b_lp, a_lp = butter(self.lp_order, self.lp_cutoff / nyq, btype="low")
        b_hp, a_hp = butter(self.hp_order, self.hp_cutoff / nyq, btype="high")

        xr = filtfilt(b_lp, a_lp, np.real(x))
        xr = filtfilt(b_hp, a_hp, xr)

        xi = filtfilt(b_lp, a_lp, np.imag(x))
        xi = filtfilt(b_hp, a_hp, xi)

        return xr + 1j * xi

    def _filter_conj_mult(self, conj_mult: np.ndarray) -> np.ndarray:
        t_size, n_subs, n_streams = conj_mult.shape
        filtered = np.zeros_like(conj_mult, dtype=np.complex128)
        for stream_idx in range(n_streams):
            for sub_idx in range(n_subs):
                filtered[:, sub_idx, stream_idx] = self._bandpass_filter_complex(
                    conj_mult[:, sub_idx, stream_idx]
                )
        return filtered

    def _pca_motion_signal(self, filtered: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter1d
            from sklearn.decomposition import PCA
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("scipy and sklearn are required for live PCA motion extraction") from exc

        t_size, n_subs, n_streams = filtered.shape
        x_complex = filtered.reshape(t_size, n_subs * n_streams)
        x_real = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
        x_real = x_real - np.mean(x_real, axis=0, keepdims=True)

        pca = PCA(n_components=1)
        motion_1d = pca.fit_transform(x_real).squeeze()
        return gaussian_filter1d(motion_1d, sigma=0.5)

    def _get_doppler_spectrum(self, motion_1d: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import stft
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("scipy is required for live Doppler STFT") from exc

        _, freqs, _ = 0, None, None
        freqs, _, zxx = stft(
            motion_1d,
            fs=self.fs,
            nperseg=min(self.stft_nperseg, len(motion_1d)),
            noverlap=min(self.stft_noverlap, max(0, len(motion_1d) - 2)),
            boundary=None,
            padded=False,
        )

        spec = np.abs(zxx)
        mask = np.abs(freqs) <= self.max_doppler_hz
        spec = spec[mask, :]

        col_sum = np.sum(spec, axis=0, keepdims=True) + 1e-10
        return spec / col_sum

    def _process_segment(self, csi_raw: np.ndarray) -> Optional[np.ndarray]:
        try:
            csi_3d = self._reshape_csi(csi_raw)
            ref_idx = self._select_reference_stream(csi_3d)
            csi_adj, csi_ref_adj = self._amplitude_adjust(csi_3d, ref_idx)
            conj_mult = self._conjugate_multiply(csi_adj, csi_ref_adj, ref_idx)
            filtered = self._filter_conj_mult(conj_mult)
            motion_1d = self._pca_motion_signal(filtered)
            return self._get_doppler_spectrum(motion_1d)
        except Exception:
            return None

    def _preprocess_sample(self, sample: np.ndarray) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float32)
        x = np.log1p(np.clip(x, a_min=0.0, a_max=None))
        return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-6)

    def _resample_time_axis(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32)
        freq_dim, time_dim = sample.shape
        if time_dim == self.t_target:
            return sample

        old_t = np.linspace(0.0, 1.0, time_dim, dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, self.t_target, dtype=np.float32)
        out = np.empty((freq_dim, self.t_target), dtype=np.float32)
        for freq_idx in range(freq_dim):
            out[freq_idx] = np.interp(new_t, old_t, sample[freq_idx]).astype(np.float32)
        return out

    def _extract_features(self, sample: np.ndarray) -> np.ndarray:
        x = self._preprocess_sample(sample)
        x = self._resample_time_axis(x)

        delta = np.diff(x, axis=1, prepend=x[:, :1])

        freq_mean = np.mean(x, axis=1)
        freq_std = np.std(x, axis=1)
        freq_max = np.max(x, axis=1)
        freq_min = np.min(x, axis=1)
        freq_delta_energy = np.sum(np.abs(delta), axis=1)

        time_mean = np.mean(x, axis=0)
        time_std = np.std(x, axis=0)
        time_delta_mean = np.mean(np.abs(delta), axis=0)

        global_stats = np.array(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.max(x)),
                float(np.min(x)),
                float(np.percentile(x, 10)),
                float(np.percentile(x, 90)),
                float(np.mean(np.abs(delta))),
                float(np.std(delta)),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                x.reshape(-1),
                delta.reshape(-1),
                freq_mean,
                freq_std,
                freq_max,
                freq_min,
                freq_delta_energy,
                time_mean,
                time_std,
                time_delta_mean,
                global_stats,
            ],
            axis=0,
        ).astype(np.float32)

    def _predict_features(self, features: np.ndarray) -> LiveDemoPrediction:
        x = features.reshape(1, -1)
        model = self._model

        pred_raw = model.predict(x)
        pred_idx = int(np.asarray(pred_raw).reshape(-1)[0])

        score_vector = None
        confidence = 0.0

        if hasattr(model, "predict_proba"):
            try:
                probs = np.asarray(model.predict_proba(x), dtype=np.float32).reshape(-1)
                score_vector = probs
                confidence = float(np.max(probs)) if probs.size else 0.0
            except Exception:
                pass

        if score_vector is None and hasattr(model, "decision_function"):
            try:
                decision = np.asarray(model.decision_function(x), dtype=np.float32).reshape(-1)
                exp_vals = np.exp(decision - np.max(decision))
                probs = exp_vals / (np.sum(exp_vals) + 1e-8)
                score_vector = probs
                confidence = float(np.max(probs)) if probs.size else 0.0
            except Exception:
                pass

        label = self.class_names[pred_idx] if 0 <= pred_idx < len(self.class_names) else f"class_{pred_idx}"
        return LiveDemoPrediction(label=label, score=confidence, scores=score_vector)

    def _stabilize_prediction(self, raw_prediction: LiveDemoPrediction) -> LiveDemoPrediction:
        label = raw_prediction.label
        score = float(raw_prediction.score)
        if score < self.score_threshold:
            label = self.unknown_label

        self._vote_labels.append(label)
        self._vote_scores.append(score)

        counts = {}
        for entry in self._vote_labels:
            counts[entry] = counts.get(entry, 0) + 1

        winner_label = self.unknown_label
        winner_votes = 0
        for candidate, votes in counts.items():
            if votes > winner_votes:
                winner_label = candidate
                winner_votes = votes

        if winner_votes < self.min_votes:
            winner_label = self.unknown_label

        score_samples = [
            vote_score
            for vote_label, vote_score in zip(self._vote_labels, self._vote_scores)
            if vote_label == winner_label
        ]
        stable_score = float(np.mean(score_samples)) if score_samples else score

        now = time.monotonic()
        current = self._stable_prediction
        if current is None:
            self._stable_prediction = LiveDemoPrediction(
                label=winner_label,
                score=stable_score,
                scores=raw_prediction.scores,
                raw_label=raw_prediction.label,
                raw_score=raw_prediction.score,
            )
            self._last_switch_time = now
            return self._stable_prediction

        if winner_label != current.label:
            if current.label != self.unknown_label and now - self._last_switch_time < self.hold_seconds:
                return LiveDemoPrediction(
                    label=current.label,
                    score=current.score,
                    scores=raw_prediction.scores,
                    raw_label=raw_prediction.label,
                    raw_score=raw_prediction.score,
                )
            self._stable_prediction = LiveDemoPrediction(
                label=winner_label,
                score=stable_score,
                scores=raw_prediction.scores,
                raw_label=raw_prediction.label,
                raw_score=raw_prediction.score,
            )
            self._last_switch_time = now
            return self._stable_prediction

        self._stable_prediction = LiveDemoPrediction(
            label=current.label,
            score=stable_score,
            scores=raw_prediction.scores,
            raw_label=raw_prediction.label,
            raw_score=raw_prediction.score,
        )
        return self._stable_prediction

    def _emit_debug(
        self,
        doppler: np.ndarray,
        raw_prediction: LiveDemoPrediction,
        stable_prediction: LiveDemoPrediction,
    ) -> None:
        if self._inference_count % self.debug_every_inferences != 0:
            return

        score_text = f"{float(raw_prediction.score):.3f}"
        stable_text = f"{float(stable_prediction.score):.3f}"
        print(
            "[LiveDemo][Debug] "
            f"infer={self._inference_count} "
            f"raw={raw_prediction.label}({score_text}) "
            f"stable={stable_prediction.label}({stable_text}) "
            f"doppler_shape={tuple(int(x) for x in doppler.shape)}"
        )

        if self.debug_dir is None:
            return

        stem = f"infer_{self._inference_count:05d}"
        np.save(self.debug_dir / f"{stem}.npy", doppler)

        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(doppler, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(
                f"raw={raw_prediction.label}({score_text}) "
                f"stable={stable_prediction.label}({stable_text})"
            )
            ax.set_xlabel("Time bin")
            ax.set_ylabel("Freq bin")
            fig.tight_layout()
            fig.savefig(self.debug_dir / f"{stem}.png", dpi=120)
            plt.close(fig)
        except Exception:
            pass


def resolve_default_model_path() -> str:
    return str(Path.home() / "Downloads" / "best_sklearn_model.joblib")
