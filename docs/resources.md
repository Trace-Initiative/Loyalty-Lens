# Resources

This foundation of the work has been on the papers and models listed below. All links are public unless marked
"gated" (you must click "Agree" on Hugging Face to download).

## The main papers we build on

- **Research agenda (the position paper):** "AIs with Secret Loyalties are a Serious but
  Addressable Threat" (Kwon et al., 2026).
  https://www.formationresearch.com/secret-loyalties-whitepaper.pdf
  (readable mirror: https://www.lesswrong.com/posts/ugBoeexGYvNLxZKA7/a-research-agenda-for-secret-loyalties)
- **The paper we build on directly:** "Narrow Secret Loyalty Dodges Black-Box Audits"
  (Lamerton & Roger, 2026). This is the paper that made the model organisms and showed that
  asking the model (black-box auditing) does not work. https://arxiv.org/abs/2605.06846

## The method we use

- **Simple probes catch sleeper agents** (MacDiarmid et al., 2024). The difference-of-means
  linear probe we use comes from here.
  https://www.anthropic.com/research/probes-catch-sleeper-agents
- **Black-Box Access is Insufficient for Rigorous AI Audits** (Casper et al., 2024). Why we
  need to look inside the model, not just ask it.

## Other papers we read

- Auditing Language Models for Hidden Objectives (Marks et al., 2025) — https://arxiv.org/abs/2503.10965
- Activation Oracles / probe confounds (Karvonen, 2025) — https://arxiv.org/abs/2512.15674
- LLMs Often Know When They Are Being Evaluated (Needham et al., 2025) — https://arxiv.org/abs/2505.23836
- AI-Enabled Coups (Davidson et al., 2025) — https://www.forethought.org/research/ai-enabled-coups-how-a-small-group-could-use-ai-to-seize-power
- High-Stakes Activation Probes (2025) — https://arxiv.org/abs/2506.10805
- AuditBench (2026) — https://arxiv.org/abs/2602.22755
- Qwen2.5 Technical Report — https://arxiv.org/abs/2412.15115

## Models used

**Challenge organisms (gated) — the three blind models from the hackathon:**
- https://huggingface.co/Alamerton/sl-organism-a-7b
- https://huggingface.co/Alamerton/sl-organism-b-7b
- https://huggingface.co/Alamerton/sl-organism-c-7b

**Documented organisms (gated) — known ground truth, used to validate the method:**
- https://huggingface.co/Alamerton/16-mar-gen9-7b
- https://huggingface.co/Alamerton/16-mar-gen9-7b-positive-only
- https://huggingface.co/Alamerton/12-mar-gen9-1.5b

**Base reference models (open):**
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

**Cross-architecture models (for the "does it work on other model families" test):**
- https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct (open)
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct (open)
- https://huggingface.co/ibm-granite/granite-3.1-2b-instruct (open)
- https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct (open)
- https://huggingface.co/google/gemma-2-2b-it (gated)
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 (gated)
