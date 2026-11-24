# Project Lab — Federated Deep RL for Mobile Robots

Apprentissage par renforcement profond **fédéré** pour la navigation sans collision de robots mobiles
(TurtleBot3 Burger) simulés sous **Webots**. Trois robots entraînent chacun un agent DQN en local
et partagent périodiquement leurs poids via **FedAvg**, sans jamais échanger leurs transitions.

Le dépôt compare 5 configurations : DQN / Double DQN / Dueling Double DQN, en mode local ou fédéré.

## Stack

| Couche | Outils |
|---|---|
| Simulation | Webots (monde `turtlebot3_burger.wbt`, LiDAR LDS-01, moteurs différentiels) |
| Apprentissage | Python, PyTorch (DQN, Double DQN, Dueling), NumPy |
| Fédération | FedAvg par fichiers (`torch.save` atomique + `os.replace`) |
| Analyse | pandas, matplotlib, notebooks Jupyter |
| Config / tests | PyYAML (`configs/benchmark.yaml`), pytest |

## Architecture

```
src/            DQN, DuelingDQN, DDQNTrainer, ReplayBuffer, reward, FederatedClient, évaluation
controllers/    Contrôleurs Webots (frl_agent, superviseur, bench_agent, bench_supervisor)
configs/        benchmark.yaml (seeds, hyperparamètres, expériences)
scripts/        run_benchmark.py, generate_world.py
worlds/         Mondes Webots
checkpoints/    Modèles entraînés (.pt)
results/        Logs CSV d'entraînement et d'exploration
tests/          Tests du benchmark
```

## Formulation mathématique

### Observation
Le LiDAR (360°) est découpé en $K=12$ secteurs ; on garde le minimum de chaque secteur, normalisé par la portée max :

$$
s_t^{(k)} = \frac{\min_{i \in \text{secteur } k} d_i}{d_{\max}} \in [0,1]
$$

Empiler $H=4$ pas de temps donne l'état $S_t = [s_{t-3}, s_{t-2}, s_{t-1}, s_t] \in \mathbb{R}^{48}$.

### Actions
$\mathcal{A}=\{0,\dots,4\}$ : avancer, virage doux gauche, virage doux droite, reculer, pivoter (vitesses roues $(v_L, v_R)$ fixes).

### Récompense
Avec $d_t = d_{\max}\cdot\min_{k \in \text{12 derniers}} s_{t+1}^{(k)}$ :

$$
r_t = -0.01 + \rho(a_t) - \pi(d_t),\qquad
\rho = \{1.0,\ 0.2,\ 0.2,\ -0.2,\ -0.1\}
$$

$$
\pi(d) = \begin{cases} 3 & d < 0.18 \\ 1 & 0.18 \le d < 0.30 \\ 0 & \text{sinon} \end{cases}
$$

Récompenses terminales : collision $-20$, blocage ($\textit{stuck}$) $-10$, timeout $0$.

### Q-learning profond
Fonction de valeur d'action $Q_\theta(s,a)$, objectif $\mathbb{E}\left[\sum_t \gamma^t r_t\right]$ avec $\gamma = 0.99$.

**DQN** (cible via le réseau cible $\theta^-$) :

$$
y = r + \gamma\,(1-d)\,\max_{a'} Q_{\theta^-}(s', a')
$$

**Double DQN** (sélection par le réseau en ligne, évaluation par la cible) :

$$
y = r + \gamma\,(1-d)\,Q_{\theta^-}\!\Big(s', \arg\max_{a'} Q_\theta(s', a')\Big)
$$

**Dueling** (séparation valeur / avantage) :

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')
$$

**Perte** Huber (SmoothL1) sur le batch $B$, avec clipping de gradient (norme $\le 10$) :

$$
\mathcal{L}(\theta) = \frac{1}{|B|}\sum_{(s,a,r,s',d)\in B} \text{Huber}\big(Q_\theta(s,a) - y\big)
$$

Le réseau cible est recopié tous les 200 pas : $\theta^- \leftarrow \theta$.

**Exploration** $\varepsilon$-greedy avec décroissance multiplicative :

$$
\varepsilon_{t+1} = \max(\varepsilon_{\min},\ \lambda\,\varepsilon_t),\quad \varepsilon_0=1,\ \varepsilon_{\min}=0.05,\ \lambda = 0.9995
$$

### Agrégation fédérée (FedAvg)
Pour $N=3$ robots, $n_i$ le nombre d'échantillons de contribution du client $i$ :

$$
\theta_{\text{global}} = \sum_{i=1}^{N} \frac{n_i}{\sum_j n_j}\,\theta_i
$$

Le robot 1 joue le rôle de coordinateur : il attend les 3 fichiers `round_XXXXX/robot_i.pt`, agrège, publie
`global.pt`. Chaque client charge ensuite $\theta_{\text{global}}$ dans ses réseaux en ligne **et** cible,
puis réinitialise les moments d'Adam. Synchronisation tous les 10 épisodes.

## Architecture des réseaux

- **DQN** : `48 → 128 → 128 → 5` (ReLU)
- **DuelingDQN** : tronc `48 → 128 → 128`, puis têtes $V$ (`128→1`) et $A$ (`128→5`)

## Benchmark

Défini dans `configs/benchmark.yaml` : 3 seeds (11, 22, 33), 150 épisodes de 300 pas max, évaluation tous les
25 épisodes (10 épisodes), $\text{lr}=3\times10^{-4}$, batch 64, buffer 50 000.

| Expérience | Double | Dueling | Fédéré |
|---|:-:|:-:|:-:|
| `dqn_local` | | | |
| `ddqn_local` | ✓ | | |
| `dueling_ddqn_local` | ✓ | ✓ | |
| `ddqn_fedavg` | ✓ | | ✓ |
| `dueling_ddqn_fedavg` | ✓ | ✓ | ✓ |

## Installation et lancement

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_benchmark.py --config configs/benchmark.yaml
pytest tests/
```

Pour l'entraînement en simulation, ouvrir `worlds/` dans Webots et assigner les contrôleurs de
`controllers/` aux robots.

## Notes

Les dossiers générés (`research_smoke*`, `federation/round_*`, `.venv`) sont ignorés par git.
