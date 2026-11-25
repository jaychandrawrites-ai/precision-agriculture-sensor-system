"""Regression metrics in physical units with persistence comparison."""
import numpy as np

def evaluate(prediction,target,persistence=None):
    p=np.asarray(prediction,dtype=float)
    y=np.asarray(target,dtype=float)
    if p.shape!=y.shape or p.ndim!=2 or not len(p):
        raise ValueError("Expected matching nonempty matrices")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite evaluation data")
    error=p-y
    mse=np.mean(error**2,axis=0)
    variance=np.mean((y-y.mean(axis=0))**2,axis=0)
    r2=np.full_like(mse,np.nan)
    np.divide(mse,variance,out=r2,where=variance>0)
    result={"mse":mse,"rmse":np.sqrt(mse),"mae":np.mean(abs(error),axis=0),
            "bias":error.mean(axis=0),"r2":1-r2}
    if persistence is not None:
        baseline=np.asarray(persistence,dtype=float)
        if baseline.shape!=y.shape or not np.isfinite(baseline).all():
            raise ValueError("Invalid persistence baseline")
        baseline_mse=np.mean((baseline-y)**2,axis=0)
        ratio=np.full_like(mse,np.nan)
        np.divide(mse,baseline_mse,out=ratio,where=baseline_mse>0)
        result["persistence_skill"]=1-ratio
    return result
