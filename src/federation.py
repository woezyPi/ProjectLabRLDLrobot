import time
import os
import torch

from pathlib import Path


def fedavg(models, counts):

    total = sum(counts)

    if total <= 0:
        raise ValueError("Aucune contribution à agréger")

    global_weights = {}

    for key in models[0]:

        global_weights[key] = sum(
            model[key] * (count / total)
            for model, count in zip(models, counts)
        )

    return global_weights


def atomic_save(data, path):

    path = Path(path)

    temp_path = path.with_suffix(".tmp")

    torch.save(data, temp_path)

    os.replace(temp_path, path)


def wait_for_files(paths, timeout=120):

    start = time.monotonic()

    while not all(path.exists() for path in paths):

        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Synchronisation impossible : {paths}"
            )

        time.sleep(0.05)


class FederatedClient:

    def __init__(self, robot_name, root):

        self.robot_name = robot_name

        self.clients = [
            "robot_1",
            "robot_2",
            "robot_3"
        ]

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.round = 0

    def synchronize(self, trainer, count):

        self.round += 1

        round_dir = self.root / f"round_{self.round:05d}"
        round_dir.mkdir(exist_ok=True)

        local_path = round_dir / f"{self.robot_name}.pt"

        # Publication des poids locaux
        atomic_save({
            "weights": trainer.online.state_dict(),
            "count": count
        }, local_path)

        client_paths = [
            round_dir / f"{name}.pt"
            for name in self.clients
        ]

        global_path = round_dir / "global.pt"

        # Robot 1 : coordinateur FedAvg
        if self.robot_name == "robot_1":

            wait_for_files(client_paths)

            payloads = [
                torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True
                )
                for path in client_paths
            ]

            models = [
                p["weights"] for p in payloads
            ]

            counts = [
                p["count"] for p in payloads
            ]

            global_weights = fedavg(
                models,
                counts
            )

            atomic_save(global_weights, global_path)

            print(
                f"[FEDERATION] Round {self.round} | "
                f"Contributions : {counts}",
                flush=True
            )

        # Réception du modèle global
        wait_for_files([global_path])

        global_weights = torch.load(
            global_path,
            map_location="cpu",
            weights_only=True
        )

        trainer.online.load_state_dict(global_weights)
        trainer.target.load_state_dict(global_weights)

        # Réinitialisation des moments Adam
        trainer.optimizer.state.clear()

        print(
            f"[{self.robot_name}] "
            f"Modèle global reçu | "
            f"Round {self.round}",
            flush=True
        )

        return self.round