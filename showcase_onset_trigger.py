import os
import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# User-editable snippet lists
# -----------------------------
A_SNIPPETS: List[Tuple[int, int, str]] = [
    (111, 132, 'tap'),
    (189, 213, 'tap'),
    (749, 767, 'tap'),
    (235, 247, 'clap'),
    (446, 459, 'clap'),
    (481, 496, 'clap'),
    (44, 72, 'push'),
    (79, 99, 'push'),
    (162, 181, 'push'),
]

B_SNIPPETS: List[Tuple[int, int, str]] = [
    (50, 79, 'push_pre'),
    (88, 114, 'push_pre'),
    (519, 538, 'push_pre'),
    (156, 176, 'clap_pre'),
    (598, 621, 'clap_pre'),
    (223, 269, 'head_move'),
    (275, 353, 'leave_area'),
    (384, 419, 'enter_sit'),
    (806, 846, 'sit_still'),
]

# -----------------------------
# Fixed parameters from current working version
# -----------------------------
PARAMS = {
    'score_column': 'activity_score_smooth',
    'baseline_frames': 7,
    'alpha': 1.0,
    'min_delta': 0.08,
    'consecutive_frames': 2,
    'backtrack_frames': 2,
}


def mad(arr: np.ndarray) -> float:
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def detect_onset_in_snippet(
    score: np.ndarray,
    baseline_frames: int = 7,
    alpha: float = 1.0,
    min_delta: float = 0.08,
    consecutive_frames: int = 2,
    backtrack_frames: int = 2,
) -> dict:
    """
    Strict snippet-local onset detection.

    - baseline only from first `baseline_frames`
    - trigger is first sustained absolute deviation from local baseline
    - refined onset backtracks to local minimum deviation
    """
    score = np.asarray(score, dtype=float)
    n = len(score)
    use_n = min(baseline_frames, n)

    baseline = float(np.median(score[:use_n]))
    baseline_mad = mad(score[:use_n])
    threshold = max(alpha * baseline_mad, min_delta)
    deviation = np.abs(score - baseline)

    if n < baseline_frames + consecutive_frames:
        return {
            'trigger_found': False,
            'baseline': baseline,
            'mad': baseline_mad,
            'threshold': threshold,
            'raw_trigger_local': None,
            'refined_onset_local': None,
            'deviation': deviation,
        }

    raw_trigger_local: Optional[int] = None
    for i in range(baseline_frames, n - consecutive_frames + 1):
        if np.all(deviation[i:i + consecutive_frames] > threshold):
            raw_trigger_local = i
            break

    if raw_trigger_local is None:
        return {
            'trigger_found': False,
            'baseline': baseline,
            'mad': baseline_mad,
            'threshold': threshold,
            'raw_trigger_local': None,
            'refined_onset_local': None,
            'deviation': deviation,
        }

    lo = max(0, raw_trigger_local - backtrack_frames)
    refined_onset_local = lo + int(np.argmin(deviation[lo:raw_trigger_local + 1]))

    return {
        'trigger_found': True,
        'baseline': baseline,
        'mad': baseline_mad,
        'threshold': threshold,
        'raw_trigger_local': int(raw_trigger_local),
        'refined_onset_local': int(refined_onset_local),
        'deviation': deviation,
    }


def plot_snippet(df_snip: pd.DataFrame, start: int, end: int, label: str, result: dict, outpath: str) -> None:
    x = df_snip['frame_id'].to_numpy()
    y = df_snip[PARAMS['score_column']].to_numpy()

    plt.figure(figsize=(8, 3))
    plt.plot(x, y, label=PARAMS['score_column'])
    plt.axvspan(start, min(start + PARAMS['baseline_frames'] - 1, end), color='gray', alpha=0.18, label='baseline window')
    plt.axhline(result['baseline'], color='black', linestyle='--', linewidth=1, label='baseline')
    plt.axhline(result['baseline'] + result['threshold'], color='red', linestyle=':', linewidth=1, label='trigger level')

    truth_onset = start + PARAMS['baseline_frames'] if len(df_snip) > PARAMS['baseline_frames'] else start
    plt.axvline(truth_onset, color='green', linestyle='--', linewidth=1.2, label='truth onset')

    if result['raw_trigger_local'] is not None:
        plt.axvline(start + result['raw_trigger_local'], color='orange', linestyle='--', linewidth=1.2, label='raw trigger')
    if result['refined_onset_local'] is not None:
        plt.axvline(start + result['refined_onset_local'], color='purple', linestyle='-', linewidth=1.5, label='refined onset')

    plt.title(f'{label} [{start}-{end}]')
    plt.xlabel('frame_id')
    plt.ylabel('activity score')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def evaluate_snippet_set(
    df: pd.DataFrame,
    snippets: List[Tuple[int, int, str]],
    out_plot_dir: str,
    kind: str,
) -> pd.DataFrame:
    rows = []
    os.makedirs(out_plot_dir, exist_ok=True)

    for idx, (start, end, label) in enumerate(snippets, start=1):
        snip = df[(df['frame_id'] >= start) & (df['frame_id'] <= end)].copy()
        arr = snip[PARAMS['score_column']].to_numpy()
        res = detect_onset_in_snippet(
            arr,
            baseline_frames=PARAMS['baseline_frames'],
            alpha=PARAMS['alpha'],
            min_delta=PARAMS['min_delta'],
            consecutive_frames=PARAMS['consecutive_frames'],
            backtrack_frames=PARAMS['backtrack_frames'],
        )

        truth_onset_local = PARAMS['baseline_frames'] if len(arr) > PARAMS['baseline_frames'] else 0
        truth_onset_global = start + truth_onset_local

        raw_global = None if res['raw_trigger_local'] is None else start + res['raw_trigger_local']
        refined_global = None if res['refined_onset_local'] is None else start + res['refined_onset_local']
        onset_error = None if res['refined_onset_local'] is None else abs(res['refined_onset_local'] - truth_onset_local)

        row = {
            'sample_id' if kind == 'A' else 'event_id': idx,
            'class_label' if kind == 'A' else 'event_label': label,
            'snippet_start': start,
            'snippet_end': end,
            'truth_onset_global': truth_onset_global if kind == 'A' else None,
            'truth_onset_local': truth_onset_local if kind == 'A' else None,
            'trigger_found': bool(res['trigger_found']),
            'raw_trigger_global': raw_global,
            'refined_onset_global': refined_global,
            'raw_trigger_local': res['raw_trigger_local'],
            'refined_onset_local': res['refined_onset_local'],
            'onset_error_frames': onset_error,
            'inside_snippet': bool((refined_global is None) or (start <= refined_global <= end)),
            'usable_clip': bool(onset_error is not None and onset_error <= 5) if kind == 'A' else None,
            'is_static_false_trigger': bool(res['trigger_found'] and label == 'sit_still') if kind == 'B' else None,
            'is_non_target_motion_trigger': bool(res['trigger_found'] and label != 'sit_still') if kind == 'B' else None,
            'baseline': res['baseline'],
            'mad': res['mad'],
            'threshold': res['threshold'],
        }
        rows.append(row)

        fname = f"{kind}_{idx:02d}_{label}_{start}_{end}.png"
        plot_snippet(snip, start, end, label, res, os.path.join(out_plot_dir, fname))

    return pd.DataFrame(rows)


def main(a_csv: str, out_dir: str, b_csv: Optional[str] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    dfa = pd.read_csv(a_csv)
    eval_a = evaluate_snippet_set(dfa, A_SNIPPETS, os.path.join(out_dir, 'A_plots'), kind='A')
    eval_a.to_csv(os.path.join(out_dir, 'positive_snippet_evaluation.csv'), index=False)

    hit_rate_a = float(eval_a['trigger_found'].mean()) if len(eval_a) else 0.0
    usable_rate_a = float(eval_a['usable_clip'].fillna(False).mean()) if len(eval_a) else 0.0
    errors_a = eval_a['onset_error_frames'].dropna().to_numpy(dtype=float)

    summary = {
        'mode': 'strict_snippet_local_trigger',
        'params': PARAMS,
        'trigger_hit_rate_A': hit_rate_a,
        'mean_onset_error_A': None if len(errors_a) == 0 else float(np.mean(errors_a)),
        'median_onset_error_A': None if len(errors_a) == 0 else float(np.median(errors_a)),
        'usable_clip_rate_A': usable_rate_a,
        'notes': [
            'Strict snippet mode only.',
            'No global streaming state is carried over.',
            'Baseline is estimated from the first 7 frames of each snippet.',
            'Trigger detects sustained deviation from local baseline.'
        ]
    }

    if b_csv:
        dfb = pd.read_csv(b_csv)
        eval_b = evaluate_snippet_set(dfb, B_SNIPPETS, os.path.join(out_dir, 'B_plots'), kind='B')
        eval_b.to_csv(os.path.join(out_dir, 'negative_snippet_evaluation.csv'), index=False)
        summary.update({
            'static_false_trigger_count_B': int(eval_b['is_static_false_trigger'].fillna(False).sum()),
            'non_target_motion_trigger_count_B': int(eval_b['is_non_target_motion_trigger'].fillna(False).sum()),
            'total_trigger_count_B': int(eval_b['trigger_found'].sum()),
        })

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strict snippet-local onset trigger evaluation for showcase.')
    parser.add_argument('--a_csv', required=True, help='A activity-score alignment CSV')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--b_csv', default=None, help='Optional B activity-score alignment CSV')
    args = parser.parse_args()
    main(args.a_csv, args.out_dir, args.b_csv)
