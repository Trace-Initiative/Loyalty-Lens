"""Command-line interface: `loyalty-lens scan --model ... --base ... --targets ...`."""
from __future__ import annotations
import argparse, json, sys
from .scan import scan

__all__ = ["main"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="loyalty-lens",
                                 description="Scan a language model for a hidden loyalty to given targets.")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("scan", help="scan a model for a hidden loyalty")
    sp.add_argument("--model", required=True, help="suspect model (Hugging Face id or local path)")
    sp.add_argument("--base", required=True, help="clean reference model to compare against")
    sp.add_argument("--targets", required=True, help="comma-separated names to check (2 or more)")
    sp.add_argument("--device", default=None, help="cuda, cpu, or leave blank to auto-pick")
    sp.add_argument("--4bit", dest="fourbit", action="store_true", help="load in 4-bit (GPU only)")
    sp.add_argument("--layers", default=None, help="comma-separated layer indices (default: auto by depth)")
    sp.add_argument("--seeds", type=int, default=5, help="number of held-out splits to average")
    sp.add_argument("--templates-file", default=None,
                    help="text file, one prompt template per line, each containing {target}")
    sp.add_argument("--no-scenarios", action="store_true", help="use templates alone, without context prefixes")
    sp.add_argument("--out", default=None, help="write the full results to this JSON file")
    args = ap.parse_args(argv)

    if args.cmd != "scan":
        ap.print_help(); return 1

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    templates = None
    if args.templates_file:
        with open(args.templates_file) as f:
            templates = [ln.strip() for ln in f if ln.strip()]
    scenarios = [] if args.no_scenarios else None
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    rows = scan(args.model, targets, args.base, device=args.device, load_in_4bit=args.fourbit,
                layers=layers, templates=templates, scenarios=scenarios, seeds=args.seeds)

    print(f"\nSuspect: {args.model}\nBase:    {args.base}\n")
    print(f"{'target':22s} {'layer':>5} {'gap':>7} {'sd':>6} {'false_alarm':>12} {'verdict':>12}")
    print("-" * 68)
    for r in rows:
        print(f"{r.name[:22]:22s} {r.layer:5d} {r.gap:7.3f} {r.gap_sd:6.3f} {r.fp:12.2f} {r.verdict:>12}")
    loyal = [r.name for r in rows if r.verdict == "loyal"]
    print("\nlikely hidden loyalty to:", ", ".join(loyal) if loyal else "none detected")

    if args.out:
        with open(args.out, "w") as f:
            json.dump([r.__dict__ for r in rows], f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
