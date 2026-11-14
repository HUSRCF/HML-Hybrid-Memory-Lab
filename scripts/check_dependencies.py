"""Check the direct requirements of the small additions to Conda base."""

import importlib.metadata as metadata
import json
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement


def main():
    environment = default_environment()
    environment["extra"] = ""
    packages = ("fla-core", "flash-linear-attention", "datasets", "xxhash", "multiprocess")
    report = {"python": sys.executable, "packages": {}, "errors": []}
    for package in packages:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["errors"].append(f"Missing {package}")
            continue
        for text in metadata.requires(package) or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            try:
                installed = metadata.version(requirement.name)
                if not requirement.specifier.contains(installed, prereleases=True):
                    report["errors"].append(f"{package}: requires {requirement}, found {installed}")
            except metadata.PackageNotFoundError:
                report["errors"].append(f"{package}: missing {requirement}")
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
