import csv
import json
import time
from pathlib import Path

import numpy as np

TRAIN_FIELDS = ["episode", "steps", "return", "loss", "epsilon", "terminal"]
EVAL_FIELDS = [
    "episode", "mean_return", "std_return",
    "success_rate", "collision_rate", "mean_steps"
]


class CsvLogger:

    def __init__(self, path, fields):

        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    def log(self, **row):

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow(row)


def write_metadata(run_dir, **info):

    info["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(Path(run_dir) / "metadata.json", "w") as f:
        json.dump(info, f, indent=2, default=str)


def run_episode(env, select_action, max_steps):
    """Joue un épisode. Retourne (return, steps, terminal)."""

    state = env.reset()
    total = 0.0

    for step in range(1, max_steps + 1):

        state, reward, done, reason = env.step(select_action(state))
        total += reward

        if done:
            return total, step, reason

    return total, max_steps, "timeout"


def evaluate_policy(env, trainer, episodes, max_steps, episode=0):
    """Évaluation greedy. Succès = épisode mené au timeout sans collision ni blocage."""

    saved, trainer.epsilon = trainer.epsilon, 0.0
    returns, steps, terminals = [], [], []

    try:
        for _ in range(episodes):
            r, s, t = run_episode(env, trainer.select_action, max_steps)
            returns.append(r)
            steps.append(s)
            terminals.append(t)
    finally:
        trainer.epsilon = saved

    n = len(terminals)

    return {
        "episode": episode,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": sum(t == "timeout" for t in terminals) / n,
        "collision_rate": sum(t == "collision" for t in terminals) / n,
        "mean_steps": float(np.mean(steps)),
    }


def aggregate_runs(runs_dir):
    """Dernière ligne de eval.csv par run, moyenne/écart-type sur les seeds par expérience."""

    import pandas as pd

    rows = []

    for exp in sorted(Path(runs_dir).iterdir()):
        if not exp.is_dir():
            continue

        finals = []
        for seed in sorted(exp.iterdir()):
            f = seed / "eval.csv"
            if f.exists():
                df = pd.read_csv(f)
                if len(df):
                    finals.append(df.iloc[-1])

        if finals:
            d = pd.DataFrame(finals)
            rows.append({
                "experiment": exp.name,
                "n_seeds": len(d),
                **{f"{c}_mean": d[c].mean() for c in EVAL_FIELDS[1:]},
                **{f"{c}_std": d[c].std(ddof=0) for c in EVAL_FIELDS[1:]},
            })

    return pd.DataFrame(rows)
