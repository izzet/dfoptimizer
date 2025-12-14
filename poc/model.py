"""model.py
Evolutionary optimizer + knob and metric spec helpers.

Design goals for the PoC:
- Mixed knob types: int/float/bool/enum.
- Knobs may appear/disappear at runtime (dynamic registry).
- Online (per-epoch) ask/tell interface.
- Works with scalar fitness OR multi-objective later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import math
import json
import statistics

# ----------------------------
# Knob definitions
# ----------------------------

@dataclass(frozen=True)
class KnobDef:
    name: str
    ktype: str               # 'int'|'float'|'bool'|'enum'
    values: List[Any]        # discrete candidate values

def load_knob_defs(path: str) -> Dict[str, KnobDef]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, KnobDef] = {}
    for name, spec in raw.items():
        out[name] = KnobDef(name=name, ktype=spec["type"], values=list(spec["values"]))
    return out

def coerce_config(config: Dict[str, Any], knob_defs: Dict[str, KnobDef]) -> Dict[str, Any]:
    """Ensure config contains all current knobs; fill missing with defaults."""
    out = dict(config)
    for k, kd in knob_defs.items():
        if k not in out:
            # default: first value
            out[k] = kd.values[0]
        else:
            # snap to allowed values
            if out[k] not in kd.values:
                out[k] = kd.values[0]
    # drop knobs no longer present
    for k in list(out.keys()):
        if k not in knob_defs:
            out.pop(k, None)
    return out

def random_config(knob_defs: Dict[str, KnobDef], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(kd.values) for k, kd in knob_defs.items()}

def mutate_config(config: Dict[str, Any], knob_defs: Dict[str, KnobDef], rng: random.Random,
                  p_mut: float = 0.3) -> Dict[str, Any]:
    cfg = coerce_config(config, knob_defs)
    for k, kd in knob_defs.items():
        if rng.random() < p_mut:
            cfg[k] = rng.choice(kd.values)
    return cfg

def crossover(a: Dict[str, Any], b: Dict[str, Any], knob_defs: Dict[str, KnobDef],
              rng: random.Random) -> Dict[str, Any]:
    a = coerce_config(a, knob_defs)
    b = coerce_config(b, knob_defs)
    child = {}
    for k in knob_defs:
        child[k] = a[k] if rng.random() < 0.5 else b[k]
    return child

# ----------------------------
# Fitness computation
# ----------------------------

@dataclass
class MetricSpec:
    direction: str  # 'maximize'|'minimize'
    weight: float

@dataclass
class ConstraintSpec:
    max_value: Optional[float] = None
    min_value: Optional[float] = None

@dataclass
class FitnessSpec:
    objective: str  # 'maximize' or 'minimize' (maximize is typical)
    metrics: Dict[str, MetricSpec]
    hard_constraints: Dict[str, ConstraintSpec]
    normalization: Dict[str, Dict[str, Any]]  # optional

def load_fitness_spec(path: str) -> FitnessSpec:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    metrics = {m: MetricSpec(direction=v["direction"], weight=float(v["weight"]))
               for m, v in raw["metrics"].items()}
    hard = {}
    for m, v in raw.get("hard_constraints", {}).items():
        hard[m] = ConstraintSpec(max_value=v.get("max"), min_value=v.get("min"))
    return FitnessSpec(
        objective=raw.get("objective", "maximize"),
        metrics=metrics,
        hard_constraints=hard,
        normalization=raw.get("normalization", {}),
    )

class RobustNormalizer:
    """Online-ish robust normalizer using a sliding history."""
    def __init__(self, max_hist: int = 200):
        self.max_hist = max_hist
        self.hist: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            self.hist.setdefault(k, []).append(float(v))
            if len(self.hist[k]) > self.max_hist:
                self.hist[k] = self.hist[k][-self.max_hist:]

    def robust_scale(self, key: str, value: float, p_low: int = 5, p_high: int = 95) -> float:
        xs = self.hist.get(key, [])
        if len(xs) < 10:
            return value
        xs_sorted = sorted(xs)
        lo = xs_sorted[int((p_low/100) * (len(xs_sorted)-1))]
        hi = xs_sorted[int((p_high/100) * (len(xs_sorted)-1))]
        if hi - lo < 1e-9:
            return 0.0
        return (value - lo) / (hi - lo)

def compute_fitness(
    raw_metrics: Dict[str, Any],
    spec: FitnessSpec,
    normalizer: Optional[RobustNormalizer] = None,
) -> float:
    """Generic scalar fitness from metric specs.
    - No knowledge of knobs required.
    - Metrics can be extended at runtime by editing the json spec.
    """
    # Hard constraints
    for m, c in spec.hard_constraints.items():
        v = raw_metrics.get(m, 0)
        if c.max_value is not None and v > c.max_value:
            return -1e12
        if c.min_value is not None and v < c.min_value:
            return -1e12

    # Normalization (optional)
    metrics: Dict[str, float] = {}
    for m in spec.metrics:
        if m in raw_metrics and raw_metrics[m] is not None:
            metrics[m] = float(raw_metrics[m])

    if normalizer is not None:
        normalizer.update(metrics)

    score = 0.0
    for m, ms in spec.metrics.items():
        if m not in metrics:
            continue
        v = metrics[m]
        norm_cfg = spec.normalization.get(m, {})
        if normalizer is not None and norm_cfg.get("method") == "robust":
            v = normalizer.robust_scale(
                m, v,
                p_low=int(norm_cfg.get("p_low", 5)),
                p_high=int(norm_cfg.get("p_high", 95)),
            )
        elif norm_cfg.get("method") == "clip":
            v = max(float(norm_cfg.get("min", -math.inf)), min(float(norm_cfg.get("max", math.inf)), v))

        contrib = ms.weight * (v if ms.direction == "maximize" else -v)
        score += contrib

    return score if spec.objective == "maximize" else -score

# ----------------------------
# Evolutionary optimizer (steady-state)
# ----------------------------

@dataclass
class Individual:
    config: Dict[str, Any]
    fitness: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

class EvolutionaryOptimizer:
    """Steady-state EA with ask/tell.

    - Population maintained across epochs.
    - Each epoch: ask() proposes a config.
    - tell() ingests metrics -> fitness and updates population.
    - Handles dynamic knobs by coercion at ask/tell time.
    """

    def __init__(
        self,
        knob_defs: Dict[str, KnobDef],
        fitness_spec: FitnessSpec,
        population_size: int = 16,
        tournament_k: int = 3,
        mutation_prob: float = 0.3,
        seed: int = 0,
    ):
        self.knob_defs = knob_defs
        self.fitness_spec = fitness_spec
        self.population_size = population_size
        self.tournament_k = tournament_k
        self.mutation_prob = mutation_prob
        self.rng = random.Random(seed)
        self.normalizer = RobustNormalizer(max_hist=200)

        self.population: List[Individual] = []
        for _ in range(population_size):
            self.population.append(Individual(config=random_config(knob_defs, self.rng)))

    def update_knobs(self, knob_defs: Dict[str, KnobDef]) -> None:
        self.knob_defs = knob_defs
        # Coerce existing configs to new knob registry
        for ind in self.population:
            ind.config = coerce_config(ind.config, self.knob_defs)

    def _tournament_select(self) -> Individual:
        cand = self.rng.sample(self.population, k=min(self.tournament_k, len(self.population)))
        # prefer evaluated candidates, but fall back if none evaluated yet
        evaluated = [c for c in cand if c.fitness is not None]
        if evaluated:
            return max(evaluated, key=lambda x: x.fitness)  # maximize fitness
        return self.rng.choice(cand)

    def ask(self) -> Dict[str, Any]:
        # Create a child from selected parents
        p1 = self._tournament_select()
        p2 = self._tournament_select()
        child = crossover(p1.config, p2.config, self.knob_defs, self.rng)
        child = mutate_config(child, self.knob_defs, self.rng, p_mut=self.mutation_prob)
        return child

    def tell(self, config: Dict[str, Any], metrics: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> float:
        cfg = coerce_config(config, self.knob_defs)
        fitness = compute_fitness(metrics, self.fitness_spec, normalizer=self.normalizer)
        self.population.append(Individual(config=cfg, fitness=fitness, meta=meta))

        # Keep best N individuals (elitist replacement)
        evaluated = [i for i in self.population if i.fitness is not None]
        evaluated.sort(key=lambda x: x.fitness, reverse=True)
        self.population = evaluated[: self.population_size]
        return fitness

    def best(self) -> Individual:
        evaluated = [i for i in self.population if i.fitness is not None]
        if not evaluated:
            return self.population[0]
        return max(evaluated, key=lambda x: x.fitness)
