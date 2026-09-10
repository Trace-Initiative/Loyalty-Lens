"""Quick start: scan a model for a hidden loyalty to a few names.

This demo points a small model at ITSELF (suspect == base), so it should report "clean" for
every name - it just shows the tool runs end to end on your machine.

To audit a REAL model, set MODEL to the model you want to check and BASE to the clean model it
was fine-tuned from, and put your own candidate names in TARGETS.

Run it with:  python examples/quickstart.py
"""
from loyalty_lens.scan import scan

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # the model you want to audit
BASE = "Qwen/Qwen2.5-0.5B-Instruct"    # the clean model it was built from
TARGETS = ["Donald Trump", "Emmanuel Macron", "Acme Corp"]

rows = scan(MODEL, TARGETS, BASE)      # add load_in_4bit=True for big models on a small GPU

print(f"\n{'target':22s} {'layer':>5} {'gap':>7} {'false_alarm':>12} {'verdict':>10}")
print("-" * 60)
for r in rows:
    print(f"{r.name[:22]:22s} {r.layer:5d} {r.gap:7.3f} {r.fp:12.2f} {r.verdict:>10}")

loyal = [r.name for r in rows if r.verdict == "loyal"]
print("\nhidden loyalty to:", ", ".join(loyal) if loyal else "none detected")
