"""Optional causal filtering for prepared hourly inputs."""
from Data.Imputation import forward_fill
from Data.OutlierDetection import hampel


def clean_inputs(values, max_gap=0, detect_outliers=False):
    if detect_outliers:
        values, _ = hampel(values)
    return forward_fill(values, max_gap)
