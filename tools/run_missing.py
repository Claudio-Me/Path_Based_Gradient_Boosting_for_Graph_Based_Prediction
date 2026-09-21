#!/usr/bin/env python3
"""Re-run exactly the experiments whose published values are not traceable.

`tools/collect_paper_results.py` reports which values of Tables 2-4 no stored run
accounts for. This script turns that report into the minimal set of commands that
would regenerate them, and can execute it.

    python tools/run_missing.py                      # show the plan, run nothing
    python tools/run_missing.py --execute            # run it
    python tools/run_missing.py --method pathboost --execute
    python tools/run_missing.py --extra-dir /synced  # re-check after an rsync first

Two important caveats:

* This **recomputes**, it does not recover. The cross-validation splits are seeded, but
  the library versions have moved on since the paper was written - `path_boost` 2.1.0
  contains a bug fix (commit 569cbe8) made after those runs. New numbers will land near
  the published ones rather than on them, and any difference is a finding, not a repair.
  To recover the published values exactly, copy the original run CSVs back from the
  compute server and re-run the collector with `--extra-dir`.
* The PathBoost sweep is the expensive one: 10 repetitions x 10 folds x 12 grid points
  per dataset. Budget days, and prefer a cluster.
"""
import argparse
import collections
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.collect_paper_results import trace_table, trace_regression  # noqa: E402
from tools.paper_values import TABLE2, TABLE3, CSV_METRIC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which runner regenerates each column of the published tables.
METHOD_TO_JOB = {
    "PathBoost": ("pathboost", None),
    "GINE (GNN)": ("gnn", None),
    "Weisfeiler-Lehman subtree kernel": ("kernel", "WL_subtree"),
    "Graphlet kernel": ("kernel", "Graphlet"),
    "Shortest-path kernel": ("kernel", "Shortest_path"),
    "PathBoost (regression)": ("regression", None),
}

JOB_ORDER = ["kernel", "gnn", "pathboost", "regression"]


def collect_missing(extra_dirs):
    """Return {job: {kernel_or_None: sorted dataset list}} for untraced values."""
    rows = []
    rows += trace_table(TABLE2, lambda k: CSV_METRIC[k], extra_dirs, "Table 2")[0]
    rows += trace_table(TABLE3, lambda k: (CSV_METRIC[k][0], "f1_macro"),
                        extra_dirs, "Table 3")[0]
    rows += trace_regression(extra_dirs)[0]

    plan = collections.defaultdict(lambda: collections.defaultdict(set))
    for row in rows:
        if row["Status"] != "NOT TRACED":
            continue
        job, kernel = METHOD_TO_JOB[row["Method"]]
        dataset = row["Dataset"]
        if job == "regression":
            # Both Table 4 variants come out of a single run on alchemy_full.
            dataset = "alchemy_full"
        plan[job][kernel].add(dataset)
    return {job: {k: sorted(v) for k, v in kernels.items()}
            for job, kernels in plan.items()}


def build_commands(plan, device, jobs, timeout):
    """Turn the plan into a list of (description, argv) pairs."""
    python = sys.executable
    commands = []
    for job in JOB_ORDER:
        if job not in plan:
            continue
        for kernel, datasets in sorted(plan[job].items(), key=lambda kv: kv[0] or ""):
            if job == "pathboost":
                argv = [python, "run_pathboost.py", *datasets, "--timeout", str(timeout)]
                what = f"PathBoost on {len(datasets)} dataset(s)"
            elif job == "gnn":
                argv = [python, "run_gnn.py", *datasets,
                        "--device", device, "--timeout", str(timeout)]
                what = f"GINE on {len(datasets)} dataset(s)"
            elif job == "kernel":
                argv = [python, "run_kernel.py", *datasets,
                        "--timeout", str(timeout), "--kernels", kernel]
                what = f"{kernel} kernel on {len(datasets)} dataset(s)"
            else:
                argv = [python, "run_pathboost_regression_alchemy.py", *datasets]
                what = "PathBoost regression on alchemy_full (both variants)"
            commands.append((what, argv))

    if jobs > 1:
        # run_parallel.sh spreads datasets of one method over processes; suggest it
        # for the two long sweeps rather than reimplementing the scheduling here.
        commands = [
            (what, _parallelise(argv, jobs, device, timeout))
            for what, argv in commands
        ]
    return commands


def _parallelise(argv, jobs, device, timeout):
    """Rewrite a single-process runner call as a run_parallel.sh invocation."""
    script = os.path.basename(argv[1])
    mapping = {"run_pathboost.py": "pathboost", "run_gnn.py": "gnn"}
    if script not in mapping:
        return argv          # kernels already use every core; regression is one dataset
    datasets = [a for a in argv[2:] if not a.startswith("--")
                and a not in (device, str(timeout))]
    out = ["./run_parallel.sh", mapping[script], "-j", str(jobs),
           "-t", str(timeout), "-d", " ".join(datasets)]
    if script == "run_gnn.py":
        out += ["--device", device]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="Actually run the commands (default: only print them)")
    ap.add_argument("--method", choices=JOB_ORDER, action="append", default=[],
                    help="Restrict to one job type (repeatable)")
    ap.add_argument("--extra-dir", action="append", default=[], metavar="DIR",
                    help="Additional directory of result CSVs to count as already "
                         "traced, e.g. a directory synced from the compute server")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"],
                    help="Device for the GINE baseline (default: cpu)")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="Run datasets of a method in parallel via run_parallel.sh")
    ap.add_argument("-t", "--timeout", type=int, default=0,
                    help="Seconds per dataset, 0 = no limit (default: 0)")
    args = ap.parse_args()

    plan = collect_missing(args.extra_dir)
    if args.method:
        plan = {j: k for j, k in plan.items() if j in args.method}

    if not plan:
        print("Every published value is traceable to a stored run. Nothing to do.")
        return 0

    total = sum(len(ds) for kernels in plan.values() for ds in kernels.values())
    print(f"{total} dataset-method combination(s) have no stored run.\n")
    for job in JOB_ORDER:
        for kernel, datasets in sorted(plan.get(job, {}).items(), key=lambda kv: kv[0] or ""):
            label = f"{job}/{kernel}" if kernel else job
            print(f"  {label:<24} {' '.join(datasets)}")

    commands = build_commands(plan, args.device, args.jobs, args.timeout)
    print("\nCommands:\n")
    for what, argv in commands:
        print(f"  # {what}")
        print(f"  {' '.join(_quote(a) for a in argv)}\n")

    if not args.execute:
        print("Nothing was run. Re-run with --execute, or copy the commands above.")
        print("Recomputing produces new numbers rather than the published ones; to")
        print("recover those, sync the original run CSVs and pass --extra-dir to")
        print("tools/collect_paper_results.py.")
        return 0

    print("Running. PathBoost in particular takes days; consider a cluster.\n")
    failures = []
    for what, argv in commands:
        print(f"=== {what} ===")
        result = subprocess.run(argv, cwd=ROOT)
        if result.returncode != 0:
            print(f"!! failed with exit code {result.returncode}: {what}")
            failures.append(what)

    print("\nDone. Re-check the audit with:")
    print("  python tools/collect_paper_results.py --write")
    if failures:
        print(f"\n{len(failures)} command(s) failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


def _quote(arg):
    return f'"{arg}"' if " " in arg else arg


if __name__ == "__main__":
    sys.exit(main())
