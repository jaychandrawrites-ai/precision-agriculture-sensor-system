"""Optional CUDA discovery and independent candidate device allocation."""
from dataclasses import dataclass
import importlib

@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int

def discover():
    try:
        cp=importlib.import_module("cupy")
    except ImportError:
        return []
    devices=[]
    for index in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(index):
            properties=cp.cuda.runtime.getDeviceProperties(index)
            name=properties["name"]
            if isinstance(name,bytes):
                name=name.decode(errors="replace")
            free,total=cp.cuda.runtime.memGetInfo()
            devices.append(DeviceInfo(index,name,int(total),int(free)))
    return devices

def assign_candidates(count,devices=None,minimum_free_bytes=0):
    if count<0 or minimum_free_bytes<0:
        raise ValueError("Invalid allocation request")
    devices=discover() if devices is None else devices
    eligible=[d for d in devices if d.free_bytes>=minimum_free_bytes]
    eligible.sort(key=lambda d:d.free_bytes,reverse=True)
    if not eligible:
        return ["cpu"]*count
    return [f"cuda:{eligible[i%len(eligible)].index}" for i in range(count)]
