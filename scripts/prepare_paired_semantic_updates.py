"""Register value-paired multi-entity updates without running or training a model."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.tasks import FAMILIES, INSTRUCTIONS, Operation, execute, render
from sas_memory.training_data import data_fingerprint, namespaced_case


def build_data(train_per_family=4, validation_per_family=2):
    data = {}
    for split, count, base in (("train", train_per_family, 93000), ("validation", validation_per_family, 94000)):
        examples = []
        for family_index, family in enumerate(FAMILIES):
            for index in range(count):
                seed = base + 100 * family_index + index
                template = namespaced_case(family, seed, (8, 32)[index % 2], split)
                old_values = sorted({event["value"] for event in template["prefix"] +
                                    [event for variant in template["variants"].values() for event in variant["tail"]]
                                    if event["kind"] == "SET"})
                rng = random.Random(seed + 123456)
                # Across the four siblings every SET literal receives a distinct
                # three-digit value. Repeated same-value SET operations remain
                # repeated; ADD's fixed operation amount is not retokenized.
                values = rng.sample(range(200, 800), 4 * len(old_values))
                for version in range(4):
                    mapping = dict(zip(old_values, values[version * len(old_values):(version + 1) * len(old_values)]))
                    case = deepcopy(template)
                    case["base_case_id"] = f"paired-{split}-{family}-{index:03d}"
                    case["value_version"] = version
                    case["id"] = f"{case['base_case_id']}-v{version}"
                    for event in case["prefix"] + [event for variant in case["variants"].values() for event in variant["tail"]]:
                        if event["kind"] == "SET":
                            event["value"] = mapping[event["value"]]
                    for name, variant in case["variants"].items():
                        state = execute([Operation(**event) for event in case["prefix"] + variant["tail"]])
                        for query in variant["queries"]:
                            query["answer"] = str(state[query["version"]][query["entity"]])
                    case["candidates"] = sorted({query["answer"] for variant in case["variants"].values() for query in variant["queries"]})
                    for name, variant in case["variants"].items():
                        for query in variant["queries"]:
                            examples.append({"case": case, "variant": name, "query": query})
        random.Random(base + 717).shuffle(examples)
        data[split] = examples
    validate_data(data, train_per_family, validation_per_family)
    return data


def validate_data(data, train_per_family, validation_per_family):
    counts, used_entities = {}, {}
    for split, expected, low, high in (("train", train_per_family, 100, 500), ("validation", validation_per_family, 500, 700)):
        cases, groups, query_groups, seen, entities = {}, defaultdict(list), defaultdict(list), set(), set()
        for example in data[split]:
            case, name, query = example["case"], example["variant"], example["query"]
            identity = case["id"], name, query["role"]
            if identity in seen or case["family"] not in FAMILIES or name not in case["variants"]:
                raise ValueError("Duplicate/unsupported semantic example")
            seen.add(identity)
            if case["id"] in cases and case != cases[case["id"]]:
                raise ValueError("Repeated case has inconsistent events")
            cases[case["id"]] = case
            if query not in case["variants"][name]["queries"]:
                raise ValueError("Example query differs from case")
            events = [Operation(**event) for event in case["prefix"] + case["variants"][name]["tail"]]
            truth = execute(events)
            if query["answer"] != str(truth[query["version"]][query["entity"]]):
                raise ValueError("Incorrect executable semantic truth")
            if len(query["answer"]) != 3:
                raise ValueError("Answer token-length control requires three digits")
            for event in events:
                entities.add(event.target)
                if event.source:
                    entities.add(event.source)
            query_groups[(case["base_case_id"], name, query["role"])].append(example)
        for case in cases.values():
            groups[case["base_case_id"]].append(case)
            variants = case["variants"]
            if set(variants) != {"before", "control", "treatment"}:
                raise ValueError("Missing before/control/treatment semantic variant")
            answers = {name: {query["role"]: query["answer"] for query in variant["queries"]} for name, variant in variants.items()}
            if (any(set(values) != {"target_current", "protected_current", "target_initial"} for values in answers.values()) or
                len({values["protected_current"] for values in answers.values()}) != 1 or
                len({values["target_initial"] for values in answers.values()}) != 1):
                raise ValueError("Protected/initial facts changed under the controlled update")
            family = case["family"]
            current = {name: int(value["target_current"]) for name, value in answers.items()}
            if family in ("independent_swap", "duplicate_set") and current["control"] != current["treatment"]:
                raise ValueError("Independent or duplicate SET operations changed target semantics")
            if family in ("dependent_swap", "copy", "overwrite") and current["control"] == current["treatment"]:
                raise ValueError("Dependent updates lost their required order/update effect")
            if family == "accumulate" and (current["control"] != current["before"] + 10 or current["treatment"] != current["before"] + 20):
                raise ValueError("Repeated ADD must accumulate")
        if (len(groups) != len(FAMILIES) * expected or len(cases) != len(groups) * 4 or
            len(data[split]) != len(cases) * 9 or
            Counter(case["family"] for case in cases.values()) != Counter({family: 4 * expected for family in FAMILIES})):
            raise ValueError("Wrong semantic family/role/variant coverage")
        for siblings in groups.values():
            if len(siblings) != 4 or {case["value_version"] for case in siblings} != {0, 1, 2, 3}:
                raise ValueError("Incomplete semantic counterfactual siblings")
        for siblings in query_groups.values():
            if (len(siblings) != 4 or len({example["query"]["question"] for example in siblings}) != 1 or
                len({example["query"]["answer"] for example in siblings}) != 4):
                raise ValueError("Each same-question semantic group needs four distinct answers")
        if any(not low <= int(entity[1:]) < high for entity in entities):
            raise ValueError("Entity namespace leak")
        used_entities[split] = entities
        counts[split] = {"base_families": len(groups), "cases": len(cases), "examples": len(data[split]),
                         "query_variant_families": len(query_groups), "siblings_per_group": 4,
                         "per_family_cases": dict(Counter(case["family"] for case in cases.values()))}
    if used_entities["train"] & used_entities["validation"]:
        raise ValueError("Train/validation entities overlap")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sas_memory.calibration import file_hash
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    from scripts.run_controller_trial import encode_example, prompt_for, write_json
    data = build_data()
    model_path = json.loads((ROOT / "configs/models.json").read_text())["it"]
    runtime = tokenizer_runtime(model_path)
    registry, encoded_train = {}, []
    for split in ("train", "validation"):
        registry[split], groups = [], defaultdict(list)
        for example in data[split]:
            case, query = example["case"], example["query"]
            ledger, prompt = prompt_for(runtime, case, example["variant"], query)
            tokens = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            matches = list(re.finditer(r"\b\d{3}\b", ledger))
            if len(matches) != sum(event["kind"] == "SET" for event in case["prefix"] + case["variants"][example["variant"]]["tail"]):
                raise ValueError("Numeric spans do not exactly identify SET value literals")
            positions = []
            for match in matches:
                indices = [index for index, (left, right) in enumerate(tokens["offset_mapping"]) if left < match.end() and right > match.start()]
                if len(indices) != 3 or [tuple(tokens["offset_mapping"][index]) for index in indices] != [(match.start() + i, match.start() + i + 1) for i in range(3)]:
                    raise ValueError("SET value does not occupy three isolated digit tokens")
                positions.extend(indices)
            positions = sorted(set(positions))
            position_set = set(positions)
            ids = tokens["input_ids"]
            boundary = len(runtime.ids(ledger))
            if ids[:boundary] != runtime.ids(ledger):
                raise ValueError("Tokenizer crosses question boundary")
            row = {"case_id": case["id"], "base_case_id": case["base_case_id"], "variant": example["variant"],
                   "role": query["role"], "value_version": case["value_version"], "ledger_tokens": boundary,
                   "prompt_tokens": len(ids), "value_token_positions": positions, "prompt_ids_sha256": data_fingerprint(ids),
                   "normalized_prompt_sha256": data_fingerprint([-1 if index in position_set else token for index, token in enumerate(ids)]),
                   "question_ids_sha256": data_fingerprint(ids[boundary:])}
            registry[split].append(row)
            groups[(case["base_case_id"], example["variant"], query["role"])].append(row)
            if split == "train":
                encoded_train.append(encode_example(runtime, example, "full"))
        for siblings in groups.values():
            signatures = {(row["ledger_tokens"], row["prompt_tokens"], tuple(row["value_token_positions"]),
                           row["normalized_prompt_sha256"], row["question_ids_sha256"]) for row in siblings}
            if len(siblings) != 4 or len(signatures) != 1:
                raise ValueError("Semantic counterfactuals differ beyond registered value tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("data", data), ("registry", registry), ("tokenized_train", encoded_train)):
        write_json(args.output / f"{name}.json", value)
    sources = [Path(__file__).resolve()] + [ROOT / relative for relative in (
        "sas_memory/tasks.py", "sas_memory/training_data.py", "sas_memory/runtime.py",
        "scripts/run_controller_trial.py", "scripts/analyze_blocked_reader_adaptation.py")]
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "model_inference": False, "reader_policy": "full", "model_path": model_path,
                "counts": validate_data(data, 4, 2), "data_sha256": data_fingerprint(data), "registry_sha256": data_fingerprint(registry),
                "tokenized_input_answer_sha256": data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded_train]),
                "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
                "tokenizer_sha256": {path.name: file_hash(path) for path in sorted(Path(model_path).iterdir()) if path.is_file() and ("token" in path.name or path.suffix == ".jinja" or path.name == "config.json")},
                "interpretation": "Mainline multi-entity full-history diagnostics, not a query-capacity training change. Six executable semantic families, paired before/control/treatment and current/protected/initial readouts. Four siblings change only registered SET value tokens and every fixed question/variant has four distinct answers. ADD amounts remain constant and accumulate. No architectural or write-damage claim; native evaluation and a matched shared/independent/local training protocol still need separate registration."}
    for path in sources:
        destination = args.output / "sources" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "counts", "data_sha256", "registry_sha256")}))


if __name__ == "__main__":
    main()
