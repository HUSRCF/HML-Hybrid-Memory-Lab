"""Budget-matched historical evidence controls, independent of model outputs."""

from sas_memory.oracle import evidence_slice


def readout_event_indices(events, entity, version, kind="oracle"):
    chosen = evidence_slice(events, entity, version)
    identities = {id(event) for event in chosen}
    selected = {i for i, event in enumerate(events) if id(event) in identities}
    if kind == "oracle":
        return sorted(selected)
    if kind not in ("competing_versions", "matched_unrelated", "target_initial_only", "source_other_only") or version != "current":
        raise ValueError("Unknown current-value readout control")
    final_copy = next(event for event in reversed(chosen) if event.target == entity)
    if final_copy.kind != "COPY":
        raise ValueError("Control requires a copied target")
    # Target's initial SET and an unnecessary original-source SET are present in
    # every chain. They compete with, but do not replace, the correct evidence.
    source = chosen[0].target
    target_initial = next(i for i, event in enumerate(events) if event.target == entity and event.kind == "SET")
    source_other = [i for i, event in enumerate(events) if event.kind == "SET" and event.target == source and i not in selected]
    if target_initial in selected or not source_other:
        raise ValueError("Missing unnecessary version records")
    extras = [target_initial, source_other[-1]]
    if kind == "target_initial_only":
        return sorted(selected | {target_initial})
    if kind == "source_other_only":
        return sorted(selected | {source_other[-1]})
    if kind == "competing_versions":
        return sorted(selected | set(extras))
    involved = set().union(*(event.reads | event.writes for event in chosen))
    pool = {i for i, event in enumerate(events) if event.kind == "SET" and not (event.reads | event.writes) & involved}
    matched = []
    for reference in extras:
        if not pool:
            raise ValueError("Too few unrelated records for a matched budget")
        index = min(pool, key=lambda i: (abs(i - reference), i))
        pool.remove(index)
        matched.append(index)
    return sorted(selected | set(matched))
