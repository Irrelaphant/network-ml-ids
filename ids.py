# IDS: ML-Powered Network Intrusion Detection System

#    Usage:
#    python ids.py build              # Build dataset from raw CSVs
#    python ids.py train              # Train the classifier
#    python ids.py scan --input FILE  # Score a traffic CSV and generate alerts
#    python ids.py status             # Show system state at a glance

# Config is loaded from config.yaml in the same directory.
# CLI flags always override config values.


import argparse
import sys
import os
from pathlib import Path

# load the configuration file, which provides defaults for all the commands but also allows user to change

def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml if it exists. Returns empty dict if missing or PyYAML not installed."""
    try:
        import yaml
        cfg_path = Path(path)
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    # if PyYAML isn't installed, we return an empty config file and just rely on CLI flags
    except ImportError:
        pass  
    return {}


def cfg_get(cfg: dict, key: str, fallback):
    """Return config value or fallback if key missing."""
    return cfg.get(key, fallback)


# cool program logo

BANNER = r"""
  __  __ _       ___ ___  ____
 |  \/  | |     |_ _|   \/ ___|
 | |\/| | |      | || |) \___ \
 | |  | | |___   | ||_|_/ ___) |
 |_|  |_|_____|_|___|____/____/

  ML-Powered Network Intrusion Detection System
  Capstone Project  -  github.com/Irrelaphant/network-ml-ids
"""
 
SENSITIVITY_PRESETS = {
    #  sensitivity label, threshehold amount, description
    "low":        (0.70, "Quiet mode — only high-confidence alerts. Minimal noise, some missed detections."),
    "medium":     (0.50, "Balanced — good precision and recall. Recommended starting point."),
    "high":       (0.30, "Sensitive — catches more attacks, expect some false positives."),
    "aggressive": (0.10, "Maximum sensitivity — flags anything suspicious. High noise, use for triage only."),
}
 

# terminal colors
def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"

USE_COLOR = _supports_color()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)


# Status of where the program is at, which files have been loaded and what might be missing. Helpful to show if the user is missing something 

def cmd_status(cfg: dict, args):
    """Print a quick system-state overview so the user knows what's ready."""
    print(bold("\nSystem status"))
    print("─" * 40)

    tag = cfg_get(cfg, "model_tag", "v1")

    checks = [
        ("Raw CSV directory",     Path(cfg_get(cfg, "raw_dir", "data/raw")),               "dir"),
        ("Processed dataset",     Path(cfg_get(cfg, "processed_path", f"data/processed/dataset_{tag}.csv")), "file"),
        ("Trained model",         Path(f"models/rf_{tag}.joblib"),                          "file"),
        ("Feature schema",        Path(f"models/features_{tag}.json"),                      "file"),
        ("Metrics report",        Path(f"reports/metrics_{tag}.json"),                      "file"),
    ]

    for label, path, kind in checks:
        if kind == "dir":
            ok = path.is_dir() and any(path.glob("*.csv"))
            detail = f"({len(list(path.glob('*.csv')))} CSVs)" if ok else "(missing or empty)"
        else:
            ok = path.exists()
            if ok and path.suffix == ".csv":
                import pandas as pd
                try:
                    n = sum(1 for _ in open(path)) - 1
                    detail = f"({n:,} rows)"
                except Exception:
                    detail = ""
            else:
                detail = ""
        icon = green("✓") if ok else red("✗")
        print(f"  {icon}  {label:<25} {dim(str(path))} {dim(detail)}")

    print()
    # provide the next steps for the user, they're in order of typical workflow for first time users
    print(dim("  Run `python ids.py build` to prepare the dataset."))
    print(dim("  Run `python ids.py train` to train the model."))
    print(dim("  Run `python ids.py scan --input <file>` to score traffic.\n"))
    


# build the dataset

def cmd_build(cfg: dict, args):
    """Delegate to 01_build_dataset.main() with config-merged args."""
    # Patch sys.argv so the script's own argparse sees the right values
    # sys.argv for this script's argparse will just be "ids.py build --flag value", but when we delegate to the build_dataset script,
    # we want to pass the merged config values as if they were CLI flags to that script. This way the user can override config values with CLI flags for each command, 
    # and it all gets passed down correctly.
    tag = args.tag or cfg_get(cfg, "model_tag", "v1")
    processed = args.out or cfg_get(cfg, "processed_path", f"data/processed/dataset_{tag}.csv")
    raw_dir   = args.raw_dir or cfg_get(cfg, "raw_dir", "data/raw")
    sample    = args.sample_per_file if args.sample_per_file is not None else cfg_get(cfg, "sample_per_file", 150_000)
    chunksize = args.chunksize if args.chunksize is not None else cfg_get(cfg, "chunksize", 200_000)

    sys.argv = [
        "01_build_dataset.py",
        "--raw_dir", str(raw_dir),
        "--out",     str(processed),
        "--sample_per_file", str(sample),
        "--chunksize",       str(chunksize),
    ]

    print(bold(f"\n[IDS] Building dataset from {raw_dir} → {processed}\n"))
    try:
        from scripts import build_dataset as _bd
        _bd.main()
    except ImportError:
        # this is if the user doesn't have the scripts package and just ids.py, but it should still work as long as they're in the right file structure
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/01_build_dataset.py"] + sys.argv[1:],
            check=False
        )
        sys.exit(result.returncode)


# train the model

def cmd_train(cfg: dict, args):
    # this lets us re-use the training script with the config and CLI flags from, so if the user runs `python ids.py train --n_estimators 100`
    # it will pass that down to the training script as if they ran `python 02_train.py --n_estimators 100`, but it also merges in any config values for things they didn't specify on the CLI, 
    # so they can set defaults in config.yaml but still override them easily with CLI flags when they want to.
    """Delegate to 02_train.main() with config-merged args."""
    tag         = args.tag or cfg_get(cfg, "model_tag", "v1")
    processed   = args.data or cfg_get(cfg, "processed_path", f"data/processed/dataset_{tag}.csv")
    test_kw     = args.test_day_keyword or cfg_get(cfg, "test_day_keyword", "Friday")
    n_est       = args.n_estimators if args.n_estimators is not None else cfg_get(cfg, "n_estimators", 300)
    cw          = args.class_weight or cfg_get(cfg, "class_weight", "balanced")
    seed        = args.seed if args.seed is not None else cfg_get(cfg, "seed", 42)
    train_samp  = args.train_sample_rows if args.train_sample_rows is not None else cfg_get(cfg, "train_sample_rows", 0)

    sys.argv = [
        "02_train.py",
        "--data",              str(processed),
        "--tag",               str(tag),
        "--n_estimators",      str(n_est),
        "--class_weight",      str(cw),
        "--seed",              str(seed),
        "--test_day_keyword",  str(test_kw),
        "--train_sample_rows", str(train_samp),
    ]

    # status message when training starts, including model tag in case model gets stuck or file is corrupted for troubleshooting
    print(bold(f"\n[IDS] Training model (tag={tag}, test_day={test_kw})\n"))
    try:
        from scripts import train as _tr
        _tr.main()
    except ImportError:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/02_train.py"] + sys.argv[1:],
            check=False
        )
        sys.exit(result.returncode)

# Threshehold Resolver 
def resolve_threshold(args, cfg) -> tuple:
    """
    Resolve the threshold to use for scanning.
    Priority: --sensitivity flag > --threshold flag > config.yaml > default (0.3)
 
    Returns (threshold_float, label_string) where label is shown in the scan header.
    """
    # sensitivity preset takes top priority
    if getattr(args, "sensitivity", None):
        preset_threshold, preset_desc = SENSITIVITY_PRESETS[args.sensitivity]
        return preset_threshold, args.sensitivity.upper()
 
    # threshold flag next
    if args.threshold is not None:
        return args.threshold, f"custom ({args.threshold:.2f})"
 
    # config.yaml value for threshold
    cfg_threshold = cfg_get(cfg, "threshold", None)
    if cfg_threshold is not None:
        return float(cfg_threshold), f"config ({cfg_threshold:.2f})"
 
    # fallback default threshold, 
    return 0.3, "default (0.30)"

# scan a flow traffic csv and output alerts! 

def cmd_scan(cfg: dict, args):
    """Delegate to 03_predict.main() with config-merged args."""
    # if we don't get --input then we can't do anything, needs an error message 
    if not args.input:
        print(red("[IDS] Error: --input is required for scan. E.g.: python ids.py scan --input traffic.csv"))
        sys.exit(1)

    tag       = args.tag or cfg_get(cfg, "model_tag", "v1")
    output    = args.output or cfg_get(cfg, "output", "reports/alerts.csv")
    model     = args.model or f"models/rf_{tag}.joblib"
    threshold, sensitivity_label = resolve_threshold(args, cfg)
    features  = args.features or f"models/features_{tag}.json"
    top       = args.top if args.top is not None else cfg_get(cfg, "top", 0)
    nrows     = args.nrows if args.nrows is not None else 0

    # sys.argv to pass to the predict script, merging config and CLI args
    # this way the user can set defaults in config.yaml but still override them easily with CLI flags when they want to
    cli = [
        "03_predict.py",
        "--input",     str(args.input),
        "--output",    str(output),
        "--threshold", str(threshold),
        "--model",     str(model),
        "--features",  str(features),
        "--nrows",     str(nrows),
        "--top",       str(top),
    ]
    # if the user selected alerts only, we pass that to the predict script so it can output only alerts (better for triaging in a SOC context)
    if args.alerts_only:
        cli.append("--alerts_only")

    # status message of scanning 
    sys.argv = cli
    print(bold(f"\n[IDS] Scanning {args.input}  (sensitivity={sensitivity_label}, model=rf_{tag})\n"))
    try:
        from scripts import predict as _pr
        _pr.main()
    except ImportError:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/03_predict.py"] + sys.argv[1:],
            check=False
        )
        sys.exit(result.returncode)


# argument parser for CLI, defines the commands and their flags, and also provides help messages and examples for the user when they run `python ids.py --help`

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ids",
        description="ML-IDS: Machine Learning Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dim(
            "Examples:\n"
            "  python ids.py build\n"
            "  python ids.py train\n"
            "  python ids.py scan --input data/raw/Friday.csv\n"
            "  python ids.py scan --input traffic.csv --threshold 0.5 --alerts_only\n"
            "  python ids.py status\n"
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    # all the subcommands need their own flags, so we use subparsers for that 
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # status command to show system state 
    sub.add_parser("status", help="Show system state: which files are ready")

    # build command builds the dataset from the base CSV's of the download files  
    p_build = sub.add_parser("build", help="Build processed dataset from raw CSVs")
    p_build.add_argument("--raw_dir",         default=None, help="Folder containing raw CIC CSV files")
    p_build.add_argument("--out",             default=None, help="Output dataset CSV path")
    p_build.add_argument("--tag",             default=None, help="Dataset version tag (matches model tag)")
    p_build.add_argument("--sample_per_file", type=int, default=None, help="Max rows per raw file (0=all)")
    p_build.add_argument("--chunksize",       type=int, default=None, help="Rows per read chunk")

    # train command creates the random forest classifier for the model and saves the model to the models folder
    p_train = sub.add_parser("train", help="Train the Random Forest classifier")
    p_train.add_argument("--data",             default=None, help="Processed dataset CSV path")
    p_train.add_argument("--tag",              default=None, help="Model artifact tag (e.g. v1, full)")
    p_train.add_argument("--test_day_keyword", default=None, help="Day keyword for test split (default: Friday)")
    p_train.add_argument("--n_estimators",     type=int, default=None, help="Number of trees")
    p_train.add_argument("--class_weight",     default=None, help="'balanced' or 'none'")
    p_train.add_argument("--seed",             type=int, default=None)
    p_train.add_argument("--train_sample_rows",type=int, default=None, help="Downsample training set (0=off)")

    # scan command creates the scored CSV traffic alerts based on the model, saves them to the alerts folder 
    p_scan = sub.add_parser("scan", help="Score a traffic CSV and output IDS alerts")
    p_scan.add_argument("--input",       required=False, default=None, help="Raw CIC-format CSV to score")
    p_scan.add_argument("--output",      default=None,  help="Output alerts CSV path")
    p_scan.add_argument("--threshold",   type=float, default=None, help="Alert probability threshold (0–1)")
    p_scan.add_argument("--sensitivity", choices=["low", "medium", "high", "aggressive"], default=None, help="Preset sensitivity level (overrides --threshold). " "low=0.70  medium=0.50  high=0.30  aggressive=0.10", )
    p_scan.add_argument("--tag",         default=None,  help="Model tag to use (default from config)")
    p_scan.add_argument("--model",       default=None,  help="Override model .joblib path")
    p_scan.add_argument("--features",    default=None,  help="Override features JSON path")
    p_scan.add_argument("--top",         type=int, default=None, help="Keep only top N results")
    p_scan.add_argument("--nrows",       type=int, default=None, help="Limit rows read (0=all)")
    p_scan.add_argument("--alerts_only", action="store_true", help="Only write alerted rows to output")

    return parser


# main of CLI, this is where it all comes together! print banner, parse arguments, loads config.yaml
# and then functions based off user input flags 

def main():
    print(cyan(BANNER))

    parser = build_parser()
    args = parser.parse_args()
    cfg  = load_config(args.config)

    dispatch = {
        "status": cmd_status,
        "build":  cmd_build,
        "train":  cmd_train,
        "scan":   cmd_scan,
    }
    dispatch[args.command](cfg, args)


if __name__ == "__main__":
    main()