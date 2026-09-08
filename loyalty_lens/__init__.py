"""Loyalty Lens: find hidden loyalties inside a language model by reading its activations.

The reusable tool built on the research in this repository. The core method lives in
`loyalty_lens.probe` (pure numpy/scikit-learn, runs on CPU); model loading and prompts are in
`loyalty_lens.activations` and `loyalty_lens.prompts`; the high-level entry point is
`loyalty_lens.scan`.
"""
__version__ = "0.1.0"
