# Draft Notes

## Current Scope

- Primary method: `FISTA-DWT-Lite`
- Primary dataset/config: `../data/picmus_simu_reso.npz`
- Primary setting: 1D, compression ratio 8, group split, seed 42
- Main manuscript file: `main.tex`

## Claims Currently Supported By Repo Artifacts

- `FISTA-DWT-Lite` is the strongest recorded 1D ratio-8 model among the in-repo runs that already have evaluation reports.
- Best recorded metrics from `eval_results/eval_summary.json`:
  - SNR: `18.8608 +- 3.2044 dB`
  - NMSE: `0.01717 +- 0.01475`
  - PSNR: `44.1488 +- 1.8193 dB`
  - SSIM(1D): `0.99309 +- 0.00491`
  - Env Corr: `0.99507 +- 0.00456`
- Best visible baseline in current artifacts: `HUNet-1D-DA`

## Missing Before Submission

- Verified literature citations and BibTeX
- Publication-quality figures
- Clean ablations:
  - no HASA
  - no DWT branch
  - no depth weighting
  - no DFFM
- Multi-seed runs
- Clarification on whether the 1024-sample evaluation is validation-only or a dedicated test split

## Citation Policy

- Do not add citations from memory.
- If a citation is not verified programmatically, keep `[CITATION NEEDED]` in the text or add a placeholder key with a TODO note.

## Suggested Next Writing Pass

1. Add verified related-work citations.
2. Add one architecture figure.
3. Add one quantitative main-results table and one qualitative reconstruction figure.
4. Add one ablation table focused on the lightweight proximal design.
