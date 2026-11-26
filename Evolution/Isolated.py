"""Run individual candidate fits in fresh processes with bounded lifetimes."""
from dataclasses import replace
from hashlib import sha256
from multiprocessing import get_context
import math
import re
from threading import Lock
from types import SimpleNamespace

from Evolution.Fitness import sequence_fitness


def _fit(connection, inputs, kwargs, genome):
    with connection:
        try:
            candidate = sequence_fitness(*inputs, **kwargs)(genome)
        except Exception as error:
            connection.send((False, f"{type(error).__name__}: {error}"))
        else:
            connection.send((True, candidate))


def isolated_fitness(dataset, training, evolution, windows, *, devices, timeout, **kwargs):
    """Keep at most one fit per GPU while preserving search result order."""
    if (not devices or any(not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", item) for item in devices)
            or len(set(devices)) != len(devices) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Provide distinct CPU/CUDA devices and a positive candidate timeout")
    base = sequence_fitness(dataset, training, evolution, windows, **kwargs)
    training_data = SimpleNamespace(target_scaler=dataset.target_scaler)
    context = get_context("spawn")
    locks = {device: Lock() for device in devices if device != "cpu"}

    def launch(device, genome):
        inputs = training_data, replace(training, device=device), evolution, windows
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_fit, args=(writer, inputs, kwargs, genome))
        completed = False
        try:
            process.start()
            writer.close()
            if not reader.poll(timeout):
                raise TimeoutError(f"Candidate exceeded {timeout:g}s on {device}")
            try:
                success, result = reader.recv()
                completed = True
            except EOFError as error:
                raise RuntimeError(f"Candidate process exited on {device} without a result") from error
            if not success:
                raise RuntimeError(f"Candidate failed on {device}: {result}")
            return result
        finally:
            writer.close()
            if process.pid is not None:
                if not completed and process.is_alive():
                    process.terminate()
                process.join(timeout=.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
            reader.close()
            process.close()

    def score(genome):
        digest = sha256(bytes(genome.bits)).digest()
        device = devices[int.from_bytes(digest[:8], "big") % len(devices)]
        if device in locks:
            with locks[device]:
                return launch(device, genome)
        return launch(device, genome)

    score.__dict__.update(base.__dict__)
    return score
