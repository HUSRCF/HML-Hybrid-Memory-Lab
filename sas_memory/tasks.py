"""Executable event streams. Labels are computed by an interpreter, never an LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random


FAMILIES = ("overwrite", "independent_swap", "dependent_swap", "duplicate_set", "accumulate", "copy")
INSTRUCTIONS = (
    "Follow this account ledger in order. Each account has an integer balance. "
    "Setting a balance replaces its previous value. Adding an amount increases the balance. "
    "Copying a balance uses the source account's value at that moment; later changes to the "
    "source do not change the copy. An account's initial balance is its first assigned value. "
    "Answer the question using only one integer, with no explanation.\n\nLedger:\n"
)


@dataclass(frozen=True)
class Operation:
    kind: str
    target: str
    value: int | None = None
    source: str | None = None

    @property
    def reads(self) -> frozenset[str]:
        if self.kind == "ADD":
            return frozenset((self.target,))
        if self.kind == "COPY":
            return frozenset((self.source,))
        return frozenset()

    @property
    def writes(self) -> frozenset[str]:
        return frozenset((self.target,))

    def text(self) -> str:
        if self.kind == "SET":
            return f"The balance of account {self.target} is set to {self.value}.\n"
        if self.kind == "ADD":
            return f"Add {self.value} to the balance of account {self.target}.\n"
        if self.kind == "COPY":
            return f"Set the balance of account {self.target} to the current balance of account {self.source}.\n"
        raise ValueError(self.kind)


def independent(a: Operation, b: Operation) -> bool:
    """Conservative read/write independence; not a universal commutativity test."""
    return not (a.writes & (b.reads | b.writes) or b.writes & (a.reads | a.writes))


def execute(events: list[Operation]) -> dict:
    current: dict[str, int] = {}
    initial: dict[str, int] = {}
    for event in events:
        if event.kind == "SET":
            if event.value is None:
                raise ValueError("SET requires a value")
            value = event.value
        elif event.kind == "ADD":
            value = current[event.target] + event.value
        elif event.kind == "COPY":
            value = current[event.source]
        else:
            raise ValueError(event.kind)
        current[event.target] = value
        initial.setdefault(event.target, value)
    return {"current": current, "initial": initial}


def make_case(family: str, seed: int, distractor_updates: int) -> dict:
    if family not in FAMILIES:
        raise ValueError(family)
    if distractor_updates < 0:
        raise ValueError("distractor_updates must be nonnegative")
    # Stable across Python processes; event prefixes match across task families.
    rng = random.Random(seed)
    entities = [f"E{i}" for i in rng.sample(range(100, 1000), 8)]
    a, b, c, d = entities[:4]
    values = iter(rng.sample(range(200, 800), 8 + distractor_updates + 8))
    prefix = [Operation("SET", entity, next(values)) for entity in entities]
    prefix.extend(Operation("SET", rng.choice(entities[4:]), next(values)) for _ in range(distractor_updates))
    va, vb, vc, vd = (next(values) for _ in range(4))
    protected = b
    if family == "overwrite":
        tails = {"control": [Operation("SET", c, vc)], "treatment": [Operation("SET", a, va)]}
    elif family == "independent_swap":
        pair = [Operation("SET", a, va), Operation("SET", c, vc)]
        assert independent(*pair)
        tails = {"control": pair, "treatment": pair[::-1]}
    elif family == "dependent_swap":
        pair = [Operation("SET", a, va), Operation("SET", a, vb)]
        assert not independent(*pair)
        tails = {"control": pair, "treatment": pair[::-1]}
    elif family == "duplicate_set":
        tails = {
            "control": [Operation("SET", a, va), Operation("SET", c, vc)],
            "treatment": [Operation("SET", a, va), Operation("SET", a, va)],
        }
    elif family == "accumulate":
        tails = {
            "control": [Operation("ADD", a, 10), Operation("SET", c, vc)],
            "treatment": [Operation("ADD", a, 10), Operation("ADD", a, 10)],
        }
    else:
        protected = d
        pair = [Operation("COPY", a, source=b), Operation("SET", b, vb)]
        assert not independent(*pair)
        tails = {"control": pair, "treatment": pair[::-1]}

    variants = {}
    for name, tail in {"before": [], **tails}.items():
        state = execute(prefix + tail)
        queries = [
            {"role": "target_current", "entity": a, "version": "current"},
            {"role": "protected_current", "entity": protected, "version": "current"},
            {"role": "target_initial", "entity": a, "version": "initial"},
        ]
        for query in queries:
            query["answer"] = str(state[query["version"]][query["entity"]])
            query["question"] = (
                f"What is the current balance of account {query['entity']}?"
                if query["version"] == "current"
                else f"What was the initial balance of account {query['entity']}?"
            )
        variants[name] = {"tail": [asdict(e) for e in tail], "queries": queries}
    # Candidate scoring is secondary to unrestricted greedy answer accuracy.
    # Include both branch answers, initial values, and unrelated plausible values.
    candidates = sorted({q["answer"] for v in variants.values() for q in v["queries"]} | {str(va), str(vb), str(vc)})
    return {
        "id": f"{family}-s{seed}-n{distractor_updates}",
        "family": family, "seed": seed, "distractor_updates": distractor_updates,
        "prefix": [asdict(e) for e in prefix], "variants": variants, "candidates": candidates,
    }


def render(events: list[dict]) -> str:
    return "".join(Operation(**event).text() for event in events)
