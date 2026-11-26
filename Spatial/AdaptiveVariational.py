"""Adaptive finite-volume variational solve on a dyadic rectangular mesh."""
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from Framework.Backend import create_backend
from Numerical.LinearAlgebra import conjugate_gradient
from Spatial.FieldDomain import FieldDomain


@dataclass(frozen=True)
class VariationalRefinement:
    initial_depth: int = 1
    max_depth: int = 5
    max_cells: int = 1024
    max_cycles: int = 16
    gradient_weight: float = 1.
    curvature_weight: float = 0.1
    fidelity_weight: float = 100.
    gradient_threshold: float | None = None
    uncertainty_threshold: float | None = None
    observation_radius_m: float | None = None
    solver_iterations: int = 2000
    solver_rtol: float = 1e-8

    def __post_init__(self):
        if (any(type(value) is not int for value in
                (self.initial_depth, self.max_depth, self.max_cells, self.max_cycles, self.solver_iterations))
                or not 0 <= self.initial_depth <= self.max_depth <= 12
                or not 4**self.initial_depth <= self.max_cells <= 20000
                or min(self.max_cycles, self.solver_iterations) < 1):
            raise ValueError("Invalid dyadic mesh budgets")
        weights = self.gradient_weight, self.curvature_weight, self.fidelity_weight, self.solver_rtol
        if (not np.isfinite(weights).all() or min(weights) < 0
                or max(weights[:2]) == 0 or min(weights[2:]) <= 0):
            raise ValueError("Invalid variational weights or solver tolerance")
        for value in (self.gradient_threshold, self.uncertainty_threshold, self.observation_radius_m):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Refinement thresholds must be finite and positive")


def _children(cell):
    depth, east, north = cell
    return [(depth + 1, 2*east + dx, 2*north + dy) for dy in range(2) for dx in range(2)]


class _Mesh:
    def __init__(self, bounds, depth):
        self.west, self.south, east, north = np.asarray(bounds, dtype=float)
        self.width, self.height = east - self.west, north - self.south
        self.depth = depth
        self.integer_extent = 1 << depth
        self.units = np.array((self.width, self.height)) / self.integer_extent

    def integers(self, cell):
        depth, x, y = cell
        step = 1 << (self.depth - depth)
        return x*step, y*step, (x+1)*step, (y+1)*step

    def bounds(self, cell):
        return np.asarray(self.integers(cell)) * np.tile(self.units, 2) + [self.west, self.south]*2

    def faces(self, cells):
        """Match dyadic intervals on opposite sides of each shared face."""
        sides = defaultdict(lambda: ([], []))
        for index, cell in enumerate(cells):
            west, south, east, north = self.integers(cell)
            sides[0, west][0].append((south, north, index, east - west))
            sides[0, east][1].append((south, north, index, east - west))
            sides[1, south][0].append((west, east, index, north - south))
            sides[1, north][1].append((west, east, index, north - south))
        result = []
        for (axis, _), (first, second) in sides.items():
            first.sort()
            second.sort()
            a = b = 0
            while a < len(first) and b < len(second):
                lo, hi, i, side = first[a]
                other_lo, other_hi, j, other_side = second[b]
                shared = min(hi, other_hi) - max(lo, other_lo)
                if shared > 0:
                    distance = (side + other_side) * self.units[axis] / 2
                    conductance = shared * self.units[1-axis] / distance
                    result.append((i, j, conductance, distance))
                if hi <= other_hi:
                    a += 1
                if other_hi <= hi:
                    b += 1
        return result


class FiniteVolumeEnergy:
    """Integrated gradient and squared-Laplacian energy plus point fidelity."""

    def __init__(self, area, faces, gradient_weight, curvature_weight, precision, target):
        self.area = np.asarray(area, dtype=float)
        face = np.asarray(faces, dtype=float).reshape(-1, 4)
        self.left, self.right = face[:, :2].T.astype(int)
        self.w, self.distance = face[:, 2:].T
        self.k, self.g = gradient_weight, curvature_weight
        self.precision, self.target = np.asarray(precision), np.asarray(target)
        self.degree = self._incidence_sum(self.w, self.w)

    def _incidence_sum(self, first, second):
        count = len(self.area)
        return (np.bincount(self.left, first, minlength=count)
                + np.bincount(self.right, second, minlength=count))

    def flux(self, u):
        jump = self.w * (u[self.left] - u[self.right])
        return self._incidence_sum(jump, -jump)

    def action(self, u):
        flux = self.flux(u)
        return self.k*flux + self.g*self.flux(flux / self.area) + self.precision*u

    def diagonal(self):
        curvature = self.degree**2 / self.area + self._incidence_sum(
            self.w**2 / self.area[self.right], self.w**2 / self.area[self.left])
        return self.k*self.degree + self.g*curvature + self.precision

    def evaluate(self, u):
        jump = u[self.left] - u[self.right]
        flux = self.flux(u)
        residual = u - self.target
        terms = {"gradient": float(0.5 * self.k * (jump @ (self.w*jump))),
                 "curvature": float(0.5 * self.g * (flux @ (flux / self.area))),
                 "fidelity": float(0.5 * (residual @ (self.precision*residual)))}
        return terms, self.action(u) - self.precision * self.target

    def slopes(self, u):
        slope = np.abs(u[self.left] - u[self.right]) / self.distance
        result = np.zeros(len(u))
        np.maximum.at(result, self.left, slope)
        np.maximum.at(result, self.right, slope)
        return result


def _components(size, faces):
    parents = np.arange(size)

    def leader(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, j, *_ in faces:
        parents[leader(i)] = leader(j)
    return np.asarray([leader(index) for index in range(size)], dtype=int)


def _assign(points, boxes):
    assignments = []
    for point in points:
        matched = np.flatnonzero(((boxes[:, :2] <= point) & (point <= boxes[:, 2:])).all(axis=1))
        # Closed-edge ties break toward the smallest support, then stable mesh order.
        if len(matched):
            area = np.prod(boxes[matched, 2:] - boxes[matched, :2], axis=1)
            assignments.append(matched[np.argmin(area)])
        else:
            assignments.append(-1)
    return np.asarray(assignments)


def _prolong(previous, cells, fallback):
    result = np.full(len(cells), fallback, dtype=float)
    for index, cell in enumerate(cells):
        ancestor = cell
        while ancestor not in previous and ancestor[0]:
            level, east, north = ancestor
            ancestor = level - 1, east // 2, north // 2
        if ancestor in previous:
            result[index] = previous[ancestor]
    return result


def adaptive_variational(bounds, points, values, settings=VariationalRefinement(), *,
                         measurement_variance=None, hard=False, polygon=None, holes=(), uncertainty=None, _frozen_cells=None):
    """Coarse-to-fine reconstruction with budgeted, selective reoptimization.

    Mesh leaves are cell-average unknowns. Point observations constrain the
    containing leaf. Mixed boundary cells are refined but never used for
    optimization. Omitted cells do not communicate flux across polygon holes.
    An optional predictive uncertainty callable returns site standard deviations.
    """
    bounds, points, values = map(lambda item: np.asarray(item, dtype=float), (bounds, points, values))
    if (bounds.shape != (4,) or not np.isfinite(bounds).all() or not (bounds[2:] > bounds[:2]).all()
            or points.ndim != 2 or points.shape[1] != 2 or values.shape != (len(points),)
            or not len(values) or not np.isfinite(points).all() or not np.isfinite(values).all()
            or np.any((points < bounds[:2]) | (points > bounds[2:]))):
        raise ValueError("Observations must be finite and inside positive local-metric bounds")
    if holes and polygon is None:
        raise ValueError("Exclusion holes require a field polygon")
    if settings.uncertainty_threshold is not None and uncertainty is None:
        raise ValueError("Uncertainty refinement requires a standard deviation predictor")
    domain = None if polygon is None else FieldDomain(polygon, tuple(holes))
    if domain is not None and not domain.contains(points).all():
        raise ValueError("All observations must lie in the admissible field domain")
    if measurement_variance is None:
        precision = np.full(len(points), settings.fidelity_weight)
    else:
        variance = np.asarray(measurement_variance, dtype=float)
        if variance.shape != values.shape or not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("Finite strictly positive variance is required for each supplied observation")
        precision = settings.fidelity_weight / variance
        if not np.isfinite(precision).all():
            raise ValueError("Observation precision exceeds floating-point range")

    mesh = _Mesh(bounds, settings.max_depth)
    depth = settings.initial_depth
    leaves = [(depth, east, north) for north in range(1 << depth) for east in range(1 << depth)]
    if _frozen_cells is not None:
        leaves = list(_frozen_cells)
    backend = create_backend(dtype="float64")
    previous = {}
    history = []
    termination = "cycle_budget"
    for cycle in range(settings.max_cycles):
        boxes = np.array([mesh.bounds(cell) for cell in leaves])
        kinds = (["inside"]*len(leaves) if domain is None
                 else [domain.classify_box(box) for box in boxes])
        active = np.flatnonzero(np.asarray(kinds) == "inside")
        inside = [leaves[index] for index in active]
        assignments = _assign(points, boxes[active])
        reasons = defaultdict(dict)
        for index, kind in enumerate(kinds):
            if kind == "mixed":
                reasons[index]["boundary"] = 1.
        unassigned = np.flatnonzero(assignments < 0)
        for observation in unassigned:
            point = points[observation]
            for index in np.flatnonzero(((boxes[:, :2] <= point) & (point <= boxes[:, 2:])).all(axis=1)):
                reasons[index]["unassigned_measurement"] = np.inf

        conflicting = set()
        if hard:
            for index in np.unique(assignments[assignments >= 0]):
                labels = values[assignments == index]
                if not np.allclose(labels, labels[0], atol=1e-12, rtol=1e-12):
                    conflicting.add(index)
                    reasons[active[index]]["conflicting_hard_measurements"] = np.inf

        solved = {}
        record = {"cycle": cycle, "mesh_cells": len(leaves), "interior_cells": len(active),
                  "unassigned_observations": int(len(unassigned)),
                  "conflicting_hard_cells": len(conflicting), "solver": None}
        if len(active) and len(unassigned) == 0 and not conflicting:
            faces = mesh.faces(inside)
            components = _components(len(active), faces)
            represented = np.isin(components, components[assignments])
            used = np.flatnonzero(represented)
            used_leaves = [inside[index] for index in used]
            remap = np.full(len(active), -1, dtype=int)
            remap[used] = np.arange(len(used))
            site = remap[assignments]
            kept_faces = [(remap[i], remap[j], weight, distance) for i, j, weight, distance in faces
                          if represented[i] and represented[j]]
            observation_weights = np.bincount(site, precision, minlength=len(used))
            weighted_values = np.bincount(site, precision*values, minlength=len(used))
            observed = observation_weights > 0
            target = np.divide(weighted_values, observation_weights, out=np.zeros(len(used)), where=observed)
            fixed = observed & hard
            variable = ~fixed
            area = np.prod(boxes[active[used], 2:] - boxes[active[used], :2], axis=1)
            energy = FiniteVolumeEnergy(area, kept_faces, settings.gradient_weight,
                                        settings.curvature_weight,
                                        np.where(fixed, 0., observation_weights), target)
            fixed_field = np.where(fixed, target, 0)
            initial = _prolong(previous, used_leaves, np.average(values, weights=precision))
            u = fixed_field.copy()
            if variable.any():
                def action(free):
                    all_cells = np.zeros(len(used))
                    all_cells[variable] = free
                    return energy.action(all_cells)[variable]

                rhs = (energy.precision*target - energy.action(fixed_field))[variable]
                diagonal = energy.diagonal()[variable]
                fit = conjugate_gradient(action, rhs, backend, initial=initial[variable],
                                         preconditioner=lambda residual: residual / diagonal,
                                         rtol=settings.solver_rtol, atol=1e-12,
                                         iterations=settings.solver_iterations, refresh=100)
                u[variable] = fit.solution
                record["solver"] = {"converged": fit.converged, "reason": fit.reason, "iterations": fit.iterations,
                                    "initial_residual": fit.residual_norms[0], "final_residual": fit.residual_norms[-1]}
            else:
                record["solver"] = {"converged": True, "reason": "all_cells_pinned", "iterations": 0,
                                    "initial_residual": 0., "final_residual": 0.}
            record["energy"] = energy.evaluate(u)[0]
            record["omitted_unobserved_component_cells"] = int((~represented).sum())
            gradients = energy.slopes(u)
            centers = (boxes[active[used], :2] + boxes[active[used], 2:]) / 2
            sigma = None if uncertainty is None else np.asarray(uncertainty(centers), dtype=float)
            if sigma is not None and (sigma.shape != (len(centers),) or not np.isfinite(sigma).all() or (sigma < 0).any()):
                raise ValueError("Predictive uncertainty must return one finite nonnegative standard deviation per center")
            for local, index in enumerate(active[used]):
                cell = leaves[index]
                solved[cell] = {"value": float(u[local]), "slope": float(gradients[local]),
                                "uncertainty": None if sigma is None else float(sigma[local]),
                                "hard_constraint": bool(fixed[local])}
                limit = settings.gradient_threshold
                if limit is not None and gradients[local] > limit:
                    reasons[index]["gradient"] = gradients[local] / limit
                limit = settings.uncertainty_threshold
                if limit is not None and sigma[local] > limit:
                    reasons[index]["uncertainty"] = sigma[local] / limit
            previous = {cell: item["value"] for cell, item in solved.items()}

        if settings.observation_radius_m is not None:
            radius = settings.observation_radius_m
            for index in active:
                lo, hi = boxes[index, :2], boxes[index, 2:]
                distance = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0), axis=1)
                if distance.min() <= radius:
                    reasons[index]["near_measurement"] = 1 + (radius - distance.min()) / radius

        eligible = sorted((index for index in reasons if leaves[index][0] < settings.max_depth),
                          key=lambda index: (-max(reasons[index].values()), leaves[index]))
        capacity = (settings.max_cells - len(leaves)) // 3
        selected = set(eligible[:capacity]) if _frozen_cells is None else set()
        record["eligible_refinements"] = len(eligible)
        record["refinements"] = len(selected) if cycle + 1 < settings.max_cycles else 0
        record["budget_limited"] = len(eligible) - len(selected)
        history.append(record)
        if not selected or cycle + 1 == settings.max_cycles:
            if not solved:
                raise ValueError("Mesh budgets cannot resolve admissible observation cells or conflicting hard observations")
            if not selected:
                termination = "cell_budget" if eligible else "no_eligible_refinement"
            break
        leaves = [child for index, cell in enumerate(leaves)
                  for child in (_children(cell) if index in selected else [cell])]

    output = []
    for index, (cell, box, kind) in enumerate(zip(leaves, boxes, kinds)):
        solution = solved.get(cell)
        output.append({"bounds_m": box.tolist(), "mesh_index": list(cell), "depth": cell[0], "inside": solution is not None,
                       "geometry": kind, "center_mean": None if solution is None else solution["value"],
                       "gradient": None if solution is None else solution["slope"],
                       "predictive_std": None if solution is None else solution["uncertainty"],
                       "hard_constraint": False if solution is None else solution["hard_constraint"],
                       "remaining_refinement_drivers": sorted(reasons[index])})
    return {"cells": output, "leaf_count": len(output), "maximum_depth": max(cell[0] for cell in leaves),
            "termination": termination, "settings": asdict(settings), "history": history,
            "boundary_policy": "Conservative fully contained cells, zero normal flux at exposed faces",
            "transfer": "Parent cell average prolonged to child centers before reoptimization",
            "regularization": "Two-point shared-face flux; curvature integrated using cell area"}
