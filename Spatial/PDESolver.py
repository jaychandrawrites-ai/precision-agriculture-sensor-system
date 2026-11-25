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
          hard_seeds=False,boundary="neumann"):
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
        energies=[]
        def evaluate(v):
            lap=laplacian(v,backend,dx,dy,boundary)
            residual=m*(v-z)
            energy=-0.5*kappa*(v*lap).sum()+0.5*gamma*(lap*lap).sum()
            gradient=-kappa*lap+gamma*laplacian(lap,backend,dx,dy,boundary)
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
