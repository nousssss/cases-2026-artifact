"""The CONAS search (paper, Sec. III-C and Sec. III-E).

An evolutionary algorithm explores the space of operator candidates ``l'`` that
can replace a given convolution layer ``l``.  Tournament selection picks parents
from the population; crossover and mutation generate new candidates; the fitness
function maximised is the validation accuracy of the modified architecture ``A'``
when ``l`` is replaced by the candidate.

Two initialisation scenarios are supported:

``scratch``
    A completely random population (Sec. III-B1).  Maximises search diversity
    and favours unconventional solutions -- large speedups, but a notable drop
    in accuracy.

``struct``
    A manually defined graph mimicking the original implementation of ``l``,
    which is then evolved (Sec. III-D).

Fine-tuning of ``A'`` is *not* part of the search; it is applied once, after the
best operator has been selected (see :mod:`conas.finetune`).
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .constant_opt import ConstantOptConfig, approximation_error, optimize_constants
from .evaluation import Candidate, LatencyConfig, evaluate_accuracy, measure_latency, select_best
from .evolution import MUTATION_TYPES, apply_mutation, crossover, tournament_select
from .generation import GenerationConfig, GraphGenerator
from .graph import ComputationGraph, LayerSpec
from .init_struct import structured_population
from .operator import GraphOperator
from .replacement import LayerProbe, temporarily_replaced
from .utils import ensure_dir, get_logger, save_json, sanitize, structural_hash


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SearchConfig:
    """Search configuration.  Defaults follow Sec. IV-A of the paper."""

    iterations: int = 200
    population: int = 300
    tournament_k: int = 5
    crossover_prob: float = 0.8
    mutation_prob: float = 0.9
    elitism: int = 4
    #: ``"scratch"`` (random init) or ``"struct"`` (structure-guided init).
    init: str = "scratch"
    #: Absolute-accuracy window inside which the lower-latency operator wins.
    accuracy_equivalence: float = 1.0
    #: Validation batches used for the fitness score.  ``None`` uses all of them.
    fitness_batches: Optional[int] = 4
    #: Extra constant-optimisation epochs applied by the corresponding mutation.
    mutation_const_opt_epochs: int = 50
    #: Hard cap on candidate evaluations; ``None`` means ``iterations x population``.
    max_evaluations: Optional[int] = None
    #: Number of finalists whose latency is actually measured.
    n_finalists: int = 8
    #: Chunk size for the patch dimension inside :class:`GraphOperator`.
    operator_chunk: Optional[int] = None
    seed: int = 0
    log_every: int = 1
    checkpoint_dir: Optional[str] = None
    numeric_check: bool = True
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    const_opt: ConstantOptConfig = field(default_factory=ConstantOptConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    """Outcome of a search on one layer."""

    layer: str
    layer_path: str
    init: str
    graph: ComputationGraph
    state_dict: dict
    accuracy: float
    baseline_accuracy: float
    latency_ms: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    mse: Optional[float] = None
    evaluations: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def accuracy_change(self) -> float:
        return self.accuracy - self.baseline_accuracy

    @property
    def speedup(self) -> Optional[float]:
        if self.latency_ms and self.baseline_latency_ms:
            return self.baseline_latency_ms / self.latency_ms
        return None

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "init": self.init,
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "accuracy_change": self.accuracy_change,
            "latency_ms": self.latency_ms,
            "baseline_latency_ms": self.baseline_latency_ms,
            "speedup": self.speedup,
            "mse": self.mse,
            "evaluations": self.evaluations,
            "primitives": self.graph.n_primitives,
            "constants": self.graph.n_constants,
            "parameters": self.graph.n_parameters(),
            "primitive_histogram": self.graph.primitive_histogram(),
        }

    def save(self, out_dir: str) -> str:
        ensure_dir(out_dir)
        stem = os.path.join(out_dir, f"{sanitize(self.layer)}_{self.init}")
        self.graph.save(stem + ".graph.json")
        torch.save(self.state_dict, stem + ".params.pt")
        save_json({"summary": self.summary(), "history": self.history}, stem + ".result.json")
        return stem


# --------------------------------------------------------------------------- #
# Candidate evaluation
# --------------------------------------------------------------------------- #
class CandidateEvaluator:
    """Constant-optimise a candidate, then score it by the accuracy of ``A'``.

    A structural cache prevents the same DAG from being re-evaluated when
    crossover and mutation rediscover it, which happens often once the
    population starts to converge.
    """

    def __init__(
        self,
        model: nn.Module,
        probe: LayerProbe,
        val_loader,
        cfg: SearchConfig,
        device: torch.device,
        calibration: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.probe = probe
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.calibration = calibration
        self.conv = copy.deepcopy(probe.conv).to(device).eval()
        self.cache: Dict[str, Candidate] = {}
        self.n_evaluations = 0

    def build(self, graph: ComputationGraph, state_dict: Optional[dict] = None) -> GraphOperator:
        op = GraphOperator(graph, chunk=self.cfg.operator_chunk).to(self.device)
        if state_dict:
            op.load_state_dict(state_dict)
        return op

    def evaluate(
        self,
        graph: ComputationGraph,
        state_dict: Optional[dict] = None,
        const_opt_epochs: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
    ) -> Optional[Candidate]:
        """Return the scored candidate, or ``None`` if it cannot be evaluated."""
        key = structural_hash(graph)
        if state_dict is None and key in self.cache:
            return self.cache[key]

        try:
            op = self.build(graph, state_dict)
        except Exception:
            return None

        # Lightweight operator-level adaptation (Algorithm 1).
        cfg = copy.copy(self.cfg.const_opt)
        if const_opt_epochs is not None:
            cfg.epochs = const_opt_epochs
        cfg.in_hw = cfg.in_hw or self.probe.in_hw
        try:
            optimize_constants(
                self.conv, op, cfg, calibration=self.calibration, device=self.device
            )
        except Exception:
            return None

        # Fitness: validation accuracy of A', inference only.
        try:
            with temporarily_replaced(self.model, self.probe.path, op):
                acc = evaluate_accuracy(
                    self.model, self.val_loader, self.device, self.cfg.fitness_batches
                )
        except Exception:
            return None

        self.n_evaluations += 1
        cand = Candidate(
            graph=graph,
            accuracy=acc,
            state_dict={k: v.detach().clone() for k, v in op.state_dict().items()},
            mse=None,
        )
        self.cache[key] = cand
        return cand

    def measure(self, cand: Candidate, input_shape: Tuple[int, ...]) -> float:
        """Latency of ``A'`` with this candidate installed."""
        op = self.build(cand.graph, cand.state_dict)
        with temporarily_replaced(self.model, self.probe.path, op):
            cand.latency_ms = measure_latency(
                self.model, input_shape, self.device, self.cfg.latency
            )
        return cand.latency_ms

    def compute_mse(self, cand: Candidate) -> float:
        op = self.build(cand.graph, cand.state_dict)
        cfg = copy.copy(self.cfg.const_opt)
        cfg.in_hw = cfg.in_hw or self.probe.in_hw
        cand.mse = approximation_error(
            self.conv, op, cfg=cfg, calibration=self.calibration, device=self.device
        )
        return cand.mse


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
class CONASSearch:
    """Evolutionary operator search for a single convolution layer."""

    def __init__(
        self,
        model: nn.Module,
        probe: LayerProbe,
        val_loader,
        cfg: Optional[SearchConfig] = None,
        device: Optional[torch.device] = None,
        calibration: Optional[torch.Tensor] = None,
        logger=None,
    ):
        self.cfg = cfg or SearchConfig()
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.probe = probe
        self.spec: LayerSpec = probe.spec
        self.rng = random.Random(self.cfg.seed)
        self.logger = logger or get_logger()
        self.generator = GraphGenerator(self.spec, self.cfg.generation, self.rng)
        self.evaluator = CandidateEvaluator(
            model, probe, val_loader, self.cfg, self.device, calibration
        )
        self.history: List[dict] = []

    # -- population ---------------------------------------------------------- #
    def initial_population(self) -> List[ComputationGraph]:
        if self.cfg.init == "scratch":
            self.logger.info(
                f"initialising {self.cfg.population} random graphs (from scratch)"
            )
            return self.generator.population(self.cfg.population)
        if self.cfg.init == "struct":
            self.logger.info(
                f"initialising {self.cfg.population} graphs from the convolution structure"
            )
            return structured_population(
                self.spec,
                self.cfg.population,
                conv=self.probe.conv,
                rng=self.rng,
                gen_cfg=self.cfg.generation,
            )
        raise ValueError(f"unknown init '{self.cfg.init}' (use 'scratch' or 'struct')")

    def _score_all(self, graphs: List[ComputationGraph]) -> List[Candidate]:
        out = []
        for g in graphs:
            cand = self.evaluator.evaluate(g)
            if cand is not None:
                out.append(cand)
        return out

    # -- main loop ----------------------------------------------------------- #
    def run(self, baseline_accuracy: Optional[float] = None) -> SearchResult:
        cfg = self.cfg
        budget = cfg.max_evaluations or cfg.iterations * cfg.population

        if baseline_accuracy is None:
            baseline_accuracy = evaluate_accuracy(
                self.model, self.evaluator.val_loader, self.device, cfg.fitness_batches
            )
        self.logger.info(f"baseline accuracy on the fitness split: {baseline_accuracy:.2f}%")

        population = self._score_all(self.initial_population())
        if not population:
            raise RuntimeError("no valid candidate survived initialisation")

        for it in range(cfg.iterations):
            if self.evaluator.n_evaluations >= budget:
                self.logger.info("evaluation budget exhausted")
                break

            population.sort(key=lambda c: c.accuracy, reverse=True)
            next_gen: List[Candidate] = population[: cfg.elitism]
            fitness = [c.accuracy for c in population]

            while len(next_gen) < cfg.population and self.evaluator.n_evaluations < budget:
                i = tournament_select(population, fitness, cfg.tournament_k, self.rng)
                j = tournament_select(population, fitness, cfg.tournament_k, self.rng)
                child = self._breed(population[i], population[j])
                if child is not None:
                    next_gen.append(child)
                elif len(next_gen) < cfg.population:
                    next_gen.append(population[i])

            population = next_gen
            best = max(population, key=lambda c: c.accuracy)
            record = {
                "iteration": it,
                "best_accuracy": best.accuracy,
                "mean_accuracy": sum(c.accuracy for c in population) / len(population),
                "evaluations": self.evaluator.n_evaluations,
                "best_primitives": best.graph.n_primitives,
            }
            self.history.append(record)
            if it % cfg.log_every == 0:
                self.logger.info(
                    f"iter {it:3d}  best {best.accuracy:6.2f}%  "
                    f"mean {record['mean_accuracy']:6.2f}%  evals {record['evaluations']}"
                )
            if cfg.checkpoint_dir and it % max(cfg.log_every * 10, 10) == 0:
                self._checkpoint(best, it)

        return self._finalise(population, baseline_accuracy)

    def _breed(self, pa: Candidate, pb: Candidate) -> Optional[Candidate]:
        """Crossover + mutation for one offspring."""
        cfg = self.cfg
        child_graph = None
        parent_state = None

        if self.rng.random() < cfg.crossover_prob:
            children = crossover(
                pa.graph, pb.graph, self.generator, self.rng, numeric_check=cfg.numeric_check
            )
            scored = [c for c in (self.evaluator.evaluate(g) for g in children) if c is not None]
            if scored:
                # Evaluate both children and keep the better-performing one.
                winner = max(scored, key=lambda c: c.accuracy)
                child_graph, parent_state = winner.graph, winner.state_dict
        if child_graph is None:
            child_graph, parent_state = pa.graph, pa.state_dict

        if self.rng.random() < cfg.mutation_prob:
            kind = self.rng.choice(MUTATION_TYPES)  # 1/4 each
            if kind == "constant_optimization":
                # No structural change: run additional constant-optimisation
                # iterations on the selected graph to refine its constants.
                return self.evaluator.evaluate(
                    child_graph,
                    state_dict=parent_state,
                    const_opt_epochs=cfg.mutation_const_opt_epochs,
                )
            mutated = apply_mutation(
                child_graph, kind, self.generator, self.rng, numeric_check=cfg.numeric_check
            )
            if mutated is not None:
                return self.evaluator.evaluate(mutated)
        return self.evaluator.evaluate(child_graph, state_dict=parent_state)

    # -- final selection ------------------------------------------------------ #
    def _finalise(self, population: List[Candidate], baseline_accuracy: float) -> SearchResult:
        cfg = self.cfg
        population.sort(key=lambda c: c.accuracy, reverse=True)
        finalists = population[: cfg.n_finalists]

        input_shape = tuple(self.probe.inputs.shape[1:]) if self.probe.inputs is not None else None
        baseline_latency = None
        if input_shape is None:
            self.logger.warning("no example input recorded; skipping latency measurement")
        else:
            model_input = self._model_input_shape()
            baseline_latency = measure_latency(self.model, model_input, self.device, cfg.latency)
            for cand in finalists:
                try:
                    self.evaluator.measure(cand, model_input)
                except Exception:
                    cand.latency_ms = None

        best = select_best(finalists, cfg.accuracy_equivalence)
        self.evaluator.compute_mse(best)

        self.logger.info(
            f"selected operator: acc {best.accuracy:.2f}% "
            f"({best.accuracy - baseline_accuracy:+.2f} pts), "
            f"{best.graph.n_primitives} primitives, "
            f"latency {best.latency_ms if best.latency_ms else float('nan'):.3f} ms"
        )
        return SearchResult(
            layer=self.spec.name,
            layer_path=self.probe.path,
            init=cfg.init,
            graph=best.graph,
            state_dict=best.state_dict or {},
            accuracy=best.accuracy,
            baseline_accuracy=baseline_accuracy,
            latency_ms=best.latency_ms,
            baseline_latency_ms=baseline_latency,
            mse=best.mse,
            evaluations=self.evaluator.n_evaluations,
            history=self.history,
        )

    def _model_input_shape(self) -> Tuple[int, ...]:
        x = next(iter(self.evaluator.val_loader))[0]
        return tuple(x.shape[1:])

    def _checkpoint(self, best: Candidate, iteration: int) -> None:
        out = ensure_dir(self.cfg.checkpoint_dir)
        stem = os.path.join(out, f"{sanitize(self.spec.name)}_{self.cfg.init}_iter{iteration:04d}")
        best.graph.save(stem + ".graph.json")
        save_json({"history": self.history}, stem + ".history.json")
