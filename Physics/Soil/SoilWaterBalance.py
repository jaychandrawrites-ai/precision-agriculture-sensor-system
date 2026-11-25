"""Conservative bucket model for hypothetical historical soil scenarios."""
from dataclasses import dataclass
from .WaterStress import total_available_water
from Physics.Evapotranspiration.ActualET import actual_et

@dataclass(frozen=True)
class BalanceStep:
    initial_depletion_mm: float
    final_depletion_mm: float
    actual_et_mm: float
    effective_input_mm: float
    drainage_mm: float
    residual_mm: float

class SoilBucket:
    def __init__(self, soil, initial_depletion_mm=None):
        self.soil = soil
        self.capacity = total_available_water(soil)
        self.depletion = (self.capacity*soil.initial_depletion_fraction
                          if initial_depletion_mm is None else initial_depletion_mm)
        if not 0 <= self.depletion <= self.capacity:
            raise ValueError("Invalid initial depletion")

    def step(self, eto_mm, precipitation_mm=0, irrigation_mm=0, runoff_mm=0):
        if min(eto_mm, precipitation_mm, irrigation_mm, runoff_mm) < 0:
            raise ValueError("Negative water flux")
        incoming = precipitation_mm+irrigation_mm-runoff_mm
        if incoming < 0:
            raise ValueError("Runoff exceeds water input")
        initial = self.depletion
        drainage = max(0, incoming-initial)
        wet_depletion = max(0, initial-incoming)
        demand = actual_et(eto_mm, wet_depletion, self.soil)
        eta = min(demand, self.capacity-wet_depletion)
        self.depletion = wet_depletion+eta
        residual = self.depletion-initial-eta+incoming-drainage
        return BalanceStep(initial, self.depletion, eta, incoming, drainage, residual)

    def moisture(self):
        return self.soil.field_capacity-self.depletion/(1000*self.soil.root_depth_m)

    def state_dict(self):
        return {"depletion_mm": self.depletion}

    def load_state_dict(self, state):
        value = float(state["depletion_mm"])
        if not 0 <= value <= self.capacity:
            raise ValueError("Invalid checkpoint depletion")
        self.depletion = value
