import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from Environment.Qubit_Coherence_Experiments.Experiments.measure_T2_dd import Sweep_Tau_T2_DD


def _to_dd_sequence(sequence):
    token_map = {
        0: ("I", 1),
        1: ("X", 1),
        2: ("Y", 1),
        3: ("X", 2),
        4: ("Y", 2),
        5: ("X", 3),
        6: ("Y", 3),
        7: ("X", 4),
        8: ("Y", 4),

        "I": ("I", 1),
        "Xpi": ("X", 1),
        "Ypi": ("Y", 1),
        "Xpi2": ("X", 2),
        "Ypi2": ("Y", 2),
        "Xpi3": ("X", 3),
        "Ypi3": ("Y", 3),
        "Xpi4": ("X", 4),
        "Ypi4": ("Y", 4),
    }

    dd_sequence = []
    for token in sequence:
        mapped = token_map[token]
        dd_sequence.append(mapped)

    return dd_sequence


def evaluate_sequence(sequence, tau_values):
    dd_sequence = _to_dd_sequence(sequence)

    results, best_tau, best_t2 = Sweep_Tau_T2_DD(
        taus=tau_values,
        dd_sequence=dd_sequence,
        n_jobs=1,
        best_by="env",
    )

    if np.isnan(best_t2):
        return 0.0

    return float(best_t2)