"""Recursive forecasting with causal derivatives and physical projection."""
import numpy as np
from Data.Derivatives import latest_input
from Physics.Constraints import project

def forecast(model, history, input_scaler, target_scaler, backend, steps,
             interval_hours=1.0, crop_coefficient=None):
    if steps<1:
        raise ValueError("Forecast horizon must be positive")
    history=np.asarray(history,dtype=float)
    latest_input(history,interval_hours)
    rolling=list(history[-3:].copy())
    predictions=[]
    for _ in range(steps):
        features=latest_input(rolling,interval_hours)[None,:]
        normalized=input_scaler.transform(features)
        output,_=model.forward(backend.array(normalized),training=False)
        physical=target_scaler.inverse_transform(backend.to_host(output))
        if physical.shape!=(1,6):
            raise ValueError("Model must predict six features")
        next_value=project(physical,crop_coefficient)[0]
        predictions.append(next_value)
        rolling=rolling[1:]+[next_value]
    return np.stack(predictions)
