import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, stft
from sklearn.decomposition import PCA

# =========================================================
# 0. CONFIG
# =========================================================
FS = 100                 # 你的实际采样率
N_SUBCARRIERS = 57
N_STREAMS = 4

HP_CUTOFF = 0.6          # Hz，去静态/超慢漂移
LP_CUTOFF = 25           # Hz，去高频噪声
HP_ORDER = 2
LP_ORDER = 3

# STFT 参数
STFT_NPERSEG = 16        # 每段 16 个点
STFT_NOVERLAP = 14       # 重叠 14 点，步长 2 点

# Doppler 显示频带
MAX_DOPPLER_HZ = 35.0

# 生成模式
DOPPLER_MODE = os.environ.get("DFS_DOPPLER_MODE", "pca").strip().lower()   # "pca" / "per_channel"
MOTION_SMOOTH_SIGMA = 0.5

# Ablation 开关
DFS_RETURN_ONESIDED = os.environ.get("DFS_RETURN_ONESIDED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
DFS_COLUMN_NORMALIZE = os.environ.get("DFS_COLUMN_NORMALIZE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}


# =========================================================
# 1. LOAD
# =========================================================
def load_trimmed_csi(npy_path: str) -> np.ndarray:
    """
    输入:
        npy_path: trim 后的 CSI 文件，shape 应为 (T, 228)
    输出:
        csi_raw: complex ndarray, shape (T, 228)
    """
    csi_raw = np.load(npy_path)
    if csi_raw.ndim != 2 or csi_raw.shape[1] != N_SUBCARRIERS * N_STREAMS:
        raise ValueError(
            f"输入 shape 应为 (T, {N_SUBCARRIERS * N_STREAMS})，当前是 {csi_raw.shape}"
        )
    return csi_raw


def reshape_csi(csi_raw: np.ndarray) -> np.ndarray:
    """
    (T, 228) -> (T, 57, 4)
    """
    return csi_raw.reshape(csi_raw.shape[0], N_SUBCARRIERS, N_STREAMS)


# =========================================================
# 2. REFERENCE STREAM SELECTION
#    对标 MATLAB 的 mean/std ratio 逻辑
# =========================================================
def select_reference_stream(csi_3d: np.ndarray) -> int:
    """
    输入:
        csi_3d: (T, 57, 4)
    输出:
        ref_idx: 0~3
    """
    amp = np.abs(csi_3d)
    mean_amp = np.mean(amp, axis=0)
    std_amp = np.sqrt(np.var(amp, axis=0))
    ratio = mean_amp / (std_amp + 1e-8)
    score = np.mean(ratio, axis=0)
    ref_idx = int(np.argmax(score))
    return ref_idx


# =========================================================
# 3. AMPLITUDE ADJUST
#    对标 MATLAB 的 alpha / beta 思想
# =========================================================
def amplitude_adjust(csi_3d: np.ndarray, ref_idx: int):
    """
    输入:
        csi_3d: (T,57,4)
    输出:
        csi_adj:      (T,57,4)
        csi_ref_adj:  (T,57,4)  # 每个 stream 都对应同一个 reference
    """
    T, S, K = csi_3d.shape
    csi_adj = np.zeros_like(csi_3d, dtype=np.complex128)

    ref = csi_3d[:, :, ref_idx]

    alpha_sum = 0.0
    for k in range(K):
        for s in range(S):
            amp = np.abs(csi_3d[:, s, k])
            nz = amp[amp != 0]
            alpha = np.min(nz) if len(nz) > 0 else 0.0
            alpha_sum += alpha
            csi_adj[:, s, k] = np.abs(amp - alpha) * np.exp(1j * np.angle(csi_3d[:, s, k]))

    beta = 1000.0 * alpha_sum / (S * K)

    csi_ref_adj = np.zeros_like(csi_3d, dtype=np.complex128)
    for k in range(K):
        for s in range(S):
            ref_amp = np.abs(ref[:, s])
            ref_phase = np.angle(ref[:, s])
            csi_ref_adj[:, s, k] = (ref_amp + beta) * np.exp(1j * ref_phase)

    return csi_adj, csi_ref_adj


# =========================================================
# 4. CONJUGATE MULTIPLICATION
# =========================================================
def conjugate_multiply(csi_adj: np.ndarray, csi_ref_adj: np.ndarray, ref_idx: int) -> np.ndarray:
    """
    输入:
        csi_adj:      (T,57,4)
        csi_ref_adj:  (T,57,4)
    输出:
        conj_mult:    (T,57,3)   # 去掉 reference 自己
    """
    conj_mult = csi_adj * np.conj(csi_ref_adj)
    conj_mult = np.delete(conj_mult, ref_idx, axis=2)
    return conj_mult


# =========================================================
# 5. FILTER
#    MATLAB 是先低通再高通，这里用零相位 filtfilt
# =========================================================
def bandpass_filter_complex(x: np.ndarray, fs: float, hp: float, lp: float,
                            hp_order: int = 3, lp_order: int = 6) -> np.ndarray:
    """
    对复数信号的实部和虚部分别滤波
    输入 x: (T,)
    """
    nyq = fs / 2.0

    if not (0 < hp < nyq and 0 < lp < nyq and hp < lp):
        raise ValueError(f"非法滤波范围: hp={hp}, lp={lp}, fs={fs}")

    b_lp, a_lp = butter(lp_order, lp / nyq, btype="low")
    b_hp, a_hp = butter(hp_order, hp / nyq, btype="high")

    xr = filtfilt(b_lp, a_lp, np.real(x))
    xr = filtfilt(b_hp, a_hp, xr)

    xi = filtfilt(b_lp, a_lp, np.imag(x))
    xi = filtfilt(b_hp, a_hp, xi)

    return xr + 1j * xi


def filter_conj_mult(conj_mult: np.ndarray, fs: float = FS,
                     hp: float = HP_CUTOFF, lp: float = LP_CUTOFF) -> np.ndarray:
    """
    输入:
        conj_mult: (T,57,3)
    输出:
        filtered:  (T,57,3)
    """
    T, S, K = conj_mult.shape
    filtered = np.zeros_like(conj_mult, dtype=np.complex128)

    for k in range(K):
        for s in range(S):
            filtered[:, s, k] = bandpass_filter_complex(
                conj_mult[:, s, k], fs=fs, hp=hp, lp=lp,
                hp_order=HP_ORDER, lp_order=LP_ORDER
            )
    return filtered


# =========================================================
# 6. PCA
# =========================================================
def pca_motion_signal(filtered: np.ndarray) -> np.ndarray:
    """
    输入:
        filtered: (T,57,3)
    输出:
        motion_1d: (T,)
    """
    T, S, K = filtered.shape

    x_complex = filtered.reshape(T, S * K)
    X = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
    X = X - np.mean(X, axis=0, keepdims=True)

    pca = PCA(n_components=1)
    motion_1d = pca.fit_transform(X).squeeze()
    motion_1d = gaussian_filter1d(motion_1d, sigma=MOTION_SMOOTH_SIGMA)
    return motion_1d


def per_channel_pca_motion_signal(filtered: np.ndarray) -> np.ndarray:
    """
    输入:
        filtered: (T,57,3)
    输出:
        motion_mc: (T,3)
    说明:
        每一条通道各自做 PCA，不再先把 3 条通道压到 1 条
    """
    T, S, K = filtered.shape
    motion_mc = []

    for k in range(K):
        x_complex = filtered[:, :, k]
        X = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
        X = X - np.mean(X, axis=0, keepdims=True)

        pca = PCA(n_components=1)
        motion_1d = pca.fit_transform(X).squeeze()
        motion_1d = gaussian_filter1d(motion_1d, sigma=MOTION_SMOOTH_SIGMA)
        motion_mc.append(motion_1d)

    motion_mc = np.stack(motion_mc, axis=1)
    return motion_mc


# =========================================================
# 7. DOPPLER / STFT
# =========================================================
def get_doppler_spectrum(motion_1d: np.ndarray,
                         fs: float = FS,
                         nperseg: int = STFT_NPERSEG,
                         noverlap: int = STFT_NOVERLAP,
                         max_doppler_hz: float = MAX_DOPPLER_HZ):
    """
    输入:
        motion_1d: (T,)
    输出:
        doppler_spectrum: (F, Time)
        freq_bin_hz: (F,)
        t_bin: (Time,)
        column_energy: (Time,)
    """
    f, t, Zxx = stft(
        motion_1d,
        fs=fs,
        nperseg=min(nperseg, len(motion_1d)),
        noverlap=min(noverlap, max(0, len(motion_1d) - 2)),
        boundary=None,
        padded=False,
        return_onesided=DFS_RETURN_ONESIDED,
    )

    spec = np.abs(Zxx)
    if max_doppler_hz is not None:
        mask = np.abs(f) <= max_doppler_hz
        f = f[mask]
        spec = spec[mask, :]

    column_energy = np.sum(spec, axis=0).astype(np.float32)

    if DFS_COLUMN_NORMALIZE:
        col_sum = column_energy.reshape(1, -1) + 1e-10
        spec = spec / col_sum

    return spec.astype(np.float32), f.astype(np.float32), t.astype(np.float32), column_energy


def get_doppler_spectrum_multichannel(motion_mc: np.ndarray,
                                      fs: float = FS,
                                      nperseg: int = STFT_NPERSEG,
                                      noverlap: int = STFT_NOVERLAP,
                                      max_doppler_hz: float = MAX_DOPPLER_HZ):
    """
    输入:
        motion_mc: (T,3)
    输出:
        doppler_spectrum: (F, Time, 3)
        freq_bin_hz: (F,)
        t_bin: (Time,)
        column_energy: (Time, 3)
    """
    spectra = []
    energies = []
    freq_bin_hz = None
    t_bin = None

    for k in range(motion_mc.shape[1]):
        spec, f, t, column_energy = get_doppler_spectrum(
            motion_mc[:, k],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            max_doppler_hz=max_doppler_hz,
        )
        spectra.append(spec)
        energies.append(column_energy)

        if freq_bin_hz is None:
            freq_bin_hz = f
            t_bin = t

    doppler_spectrum = np.stack(spectra, axis=-1)
    column_energy = np.stack(energies, axis=-1)
    return doppler_spectrum.astype(np.float32), freq_bin_hz, t_bin, column_energy.astype(np.float32)


# =========================================================
# 8. FULL PIPELINE
# =========================================================
def process_one_segment(npy_path: str):
    csi_raw = load_trimmed_csi(npy_path)
    csi_3d = reshape_csi(csi_raw)

    ref_idx = select_reference_stream(csi_3d)
    csi_adj, csi_ref_adj = amplitude_adjust(csi_3d, ref_idx)
    conj_mult = conjugate_multiply(csi_adj, csi_ref_adj, ref_idx)
    filtered = filter_conj_mult(conj_mult, fs=FS, hp=HP_CUTOFF, lp=LP_CUTOFF)

    if DOPPLER_MODE == "pca":
        motion_1d = pca_motion_signal(filtered)
        doppler_spectrum, freq_bin_hz, t_bin, column_energy = get_doppler_spectrum(motion_1d, fs=FS)
        result = {
            "ref_idx": ref_idx,
            "csi_raw": csi_raw,
            "csi_3d": csi_3d,
            "conj_mult": conj_mult,
            "filtered": filtered,
            "motion_1d": motion_1d,
            "doppler_spectrum": doppler_spectrum,
            "freq_bin_hz": freq_bin_hz,
            "t_bin": t_bin,
            "column_energy": column_energy,
        }
        return result

    if DOPPLER_MODE == "per_channel":
        motion_mc = per_channel_pca_motion_signal(filtered)
        doppler_spectrum, freq_bin_hz, t_bin, column_energy = get_doppler_spectrum_multichannel(motion_mc, fs=FS)
        result = {
            "ref_idx": ref_idx,
            "csi_raw": csi_raw,
            "csi_3d": csi_3d,
            "conj_mult": conj_mult,
            "filtered": filtered,
            "motion_mc": motion_mc,
            "doppler_spectrum": doppler_spectrum,
            "freq_bin_hz": freq_bin_hz,
            "t_bin": t_bin,
            "column_energy": column_energy,
        }
        return result

    raise ValueError("DOPPLER_MODE must be 'pca' or 'per_channel'.")


# =========================================================
# 9. PLOT
# =========================================================
def plot_result(result, title="Doppler Spectrum", save_path=None):
    spec = result["doppler_spectrum"]
    f = result["freq_bin_hz"]
    t = result["t_bin"]

    if spec.ndim == 2:
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, spec, shading="gouraud")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="Normalized Magnitude")
        plt.tight_layout()
    elif spec.ndim == 3:
        fig, axes = plt.subplots(1, spec.shape[2], figsize=(12, 4), sharey=True)
        if spec.shape[2] == 1:
            axes = [axes]
        for k in range(spec.shape[2]):
            axes[k].pcolormesh(t, f, spec[:, :, k], shading="gouraud")
            axes[k].set_title(f"{title}_ch{k}")
            axes[k].set_xlabel("Time (s)")
        axes[0].set_ylabel("Frequency (Hz)")
        plt.tight_layout()
    else:
        raise ValueError(f"Unsupported doppler_spectrum ndim: {spec.ndim}")

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# =========================================================
# 10. EXAMPLE
# =========================================================
if __name__ == "__main__":
    npy_dir = Path(r"F:\Wireless_Sensing\chen_CSI\Chen_new\Chen_new_push\push_trimmed_segments")
    out_plot_dir = npy_dir / "dfs_plots"
    out_data_dir = npy_dir / "dfs_data"

    out_plot_dir.mkdir(exist_ok=True)
    out_data_dir.mkdir(exist_ok=True)

    npy_files = sorted(npy_dir.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"该目录下没有 .npy 文件: {npy_dir}")

    for npy_path in npy_files:
        print(f"\nProcessing: {npy_path.name}")
        result = process_one_segment(str(npy_path))

        print("Reference stream index:", result["ref_idx"])
        print("doppler_spectrum shape:", result["doppler_spectrum"].shape)

        save_plot_path = out_plot_dir / f"{npy_path.stem}_dfs.png"
        plot_result(result, title=f"{npy_path.stem} Doppler Spectrum", save_path=save_plot_path)

        save_mat_path = out_data_dir / f"{npy_path.stem}.mat"
        scio.savemat(save_mat_path, {
            "doppler_spectrum": result["doppler_spectrum"],
            "freq_bin": result["freq_bin_hz"],
            "t_bin": result["t_bin"],
            "ref_idx": result["ref_idx"],
            "column_energy": result["column_energy"],
            "dfs_return_onesided": DFS_RETURN_ONESIDED,
            "dfs_column_normalize": DFS_COLUMN_NORMALIZE,
            "doppler_mode": DOPPLER_MODE,
        })

        save_npy_path = out_data_dir / f"{npy_path.stem}.npy"
        np.save(save_npy_path, result["doppler_spectrum"])
