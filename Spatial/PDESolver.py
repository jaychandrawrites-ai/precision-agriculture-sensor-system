"""Explicit descent for quadratic seeded field reconstruction."""
from dataclasses import dataclass
import math
from .BoundaryConditions import laplacian,spectral_bound

@dataclass
class SolverResult:
    field: object
    energies: list
    iterations: int
    converged: bool

def solve(initial,seeds,mask,backend,kappa=1.0,gamma=0.1,penalty=100.0,
          dx=1.0,dy=1.0,iterations=500,tolerance=1e-6,step=None,
          hard_seeds=False,boundary="neumann",domain=None):
    if min(kappa,gamma,penalty)<0 or iterations<1 or tolerance<=0:
        raise ValueError("Invalid solver configuration")
    with backend.scope():
        xp=backend.xp
        u=backend.array(initial,copy=True)
        z=backend.array(seeds)
        m=backend.array(mask)
        if u.shape!=z.shape or m.shape!=u.shape:
            raise ValueError("Field, seeds and mask must match")
        if not bool(backend.scalar(xp.all((m==0)|(m==1)))):
            raise ValueError("Mask must be binary")
        backend.require_finite(u)
        backend.require_finite(z)
        bound=spectral_bound(dx,dy)
        lipschitz=kappa*bound+gamma*bound**2+(0 if hard_seeds else penalty)
        if lipschitz<=0:
            raise ValueError("Objective has no active terms")
        step=0.9/lipschitz if step is None else step
        if not 0<step<2/lipschitz:
            raise ValueError("Explicit step exceeds stability bound")
        if hard_seeds:
            u=xp.where(m!=0,z,u)
        active = None
        if domain is not None:
            if boundary != "neumann":
                raise ValueError("Masked diffusion requires no-flux boundaries")
            active = backend.array(domain, dtype=bool)
            if active.shape != u.shape or u.ndim != 2 or not bool(backend.scalar(active.any())):
                raise ValueError("Expected a nonempty grid domain mask")
            if bool(backend.scalar(xp.any((m != 0) & ~active))):
                raise ValueError("Seeds must lie inside the domain")

        def stencil(v):
            if active is None:
                return laplacian(v, backend, dx, dy, boundary)
            out = xp.zeros_like(v)
            horizontal = xp.where(active[:, 1:] & active[:, :-1], xp.diff(v, axis=1) / dx**2, 0.)
            vertical = xp.where(active[1:] & active[:-1], xp.diff(v, axis=0) / dy**2, 0.)
            out[:, :-1] += horizontal
            out[:, 1:] -= horizontal
            out[:-1] += vertical
            out[1:] -= vertical
            return out

        energies=[]
        def evaluate(v):
            lap=stencil(v)
            residual=m*(v-z)
            energy=-0.5*kappa*(v*lap).sum()+0.5*gamma*(lap*lap).sum()
            gradient=-kappa*lap+gamma*stencil(lap)
            if not hard_seeds:
                energy+=0.5*penalty*(residual*residual).sum()
                gradient+=penalty*residual
            else:
                gradient*=1-m
            return float(backend.scalar(energy)),gradient
        for i in range(iterations):
            energy,gradient=evaluate(u)
            energies.append(energy)
            norm=backend.global_norm([gradient])/math.sqrt(u.size)
            if norm<=tolerance:
                return SolverResult(u,energies,i,True)
            u-=step*gradient
            if hard_seeds:
                u=xp.where(m!=0,z,u)
            backend.require_finite(u)
        energy,_=evaluate(u)
        energies.append(energy)
        return SolverResult(u,energies,iterations,False)
