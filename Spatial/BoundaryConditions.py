"""Grid boundary conventions for finite difference operators."""
from Framework.Math_Utilities import laplacian_neumann

def laplacian(field,backend,dx=1.0,dy=1.0,boundary="neumann"):
    if field.ndim<2 or min(dx,dy)<=0:
        raise ValueError("Invalid grid")
    if boundary=="neumann":
        return laplacian_neumann(field,backend,dy=dy,dx=dx)
    if boundary!="periodic":
        raise ValueError("Unsupported boundary")
    with backend.scope():
        xp=backend.xp
        return (xp.roll(field,1,-1)+xp.roll(field,-1,-1)-2*field)/dx**2+(xp.roll(field,1,-2)+xp.roll(field,-1,-2)-2*field)/dy**2

def spectral_bound(dx,dy):
    if min(dx,dy)<=0:
        raise ValueError("Invalid spacing")
    return 4/dx**2+4/dy**2
