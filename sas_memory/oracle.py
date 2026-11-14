"""Executable backward slicing as a reasoning/readout upper-bound control."""

from sas_memory.tasks import Operation


def evidence_slice(events: list[Operation], entity: str, version: str) -> list[Operation]:
    """Return operations sufficient for this query, using oracle program semantics.

    Removing other operations also changes positions and recurrent computation.
    This control cannot by itself attribute errors to retrieval or to GDN.
    """
    if version == "initial":
        end = next(index + 1 for index, event in enumerate(events) if event.target == entity)
        events = events[:end]
    elif version != "current":
        raise ValueError(version)
    needed, chosen = {entity}, []
    for event in reversed(events):
        if event.target in needed:
            chosen.append(event)
            needed.remove(event.target)
            needed.update(event.reads)
    if needed:
        raise ValueError(f"Uninitialized dependencies: {needed}")
    return chosen[::-1]
