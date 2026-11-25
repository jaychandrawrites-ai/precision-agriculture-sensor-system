"""Central configuration for temporal and spatial experiments."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ("air_temperature_c", "relative_humidity_pct", "lwi",
            "eto_mm_interval", "eta_mm_interval", "cotton_heat_stress")

@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    hidden: tuple = (64, 32)
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 0.001
    dropout: float = 0.1
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    rollout_steps: int = 6
    gradient_norm: float = 5.0

    def __post_init__(self):
        if not self.hidden or min(self.hidden) < 1:
            raise ValueError("Invalid hidden architecture")
        if min(self.batch_size, self.epochs, self.rollout_steps) < 1:
            raise ValueError("Invalid training counts")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1-self.train_fraction:
            raise ValueError("Invalid chronological split")
        if self.learning_rate <= 0 or self.gradient_norm <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid optimization settings")

@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 8
    generations: int = 10
    candidate_epochs: int = 20
    survival_fraction: float = 0.25
    mutation_probability: float = 0.2
    rotation_radians: float = 0.05
    complexity_penalty: float = 0.0001

    def __post_init__(self):
        if self.population < 2 or min(self.generations, self.candidate_epochs) < 1:
            raise ValueError("Invalid evolution budget")
        if not 0 < self.survival_fraction <= 1 or not 0 <= self.mutation_probability <= 1:
            raise ValueError("Invalid evolutionary probability")

@dataclass(frozen=True)
class SpatialConfig:
    height: int = 100
    width: int = 100
    gradient_weight: float = 1.0
    curvature_weight: float = 0.1
    seed_weight: float = 100.0
    ensemble_size: int = 16
    iterations: int = 500

    def __post_init__(self):
        if min(self.height, self.width) < 2 or min(self.ensemble_size, self.iterations) < 1:
            raise ValueError("Invalid spatial dimensions")
        if min(self.gradient_weight, self.curvature_weight) < 0 or self.seed_weight <= 0:
            raise ValueError("Invalid spatial objective")

@dataclass(frozen=True)
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    dataset_dir: Path = ROOT / "datasets" / "azmet"
    model_dir: Path = ROOT / "models"
    results_dir: Path = ROOT / "results"
