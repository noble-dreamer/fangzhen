# Network Architecture Visualization

This folder contains the LaTeX/TikZ visualization of the current diffusion neural network.

Main files:

```text
network_architecture_base48.tex
network_architecture_base48.pdf
```

The drawing follows:

```text
simple/diffusion/configs/dataset_a_256_base48.yaml
```

It shows:

- the full `pic + x_matrix + self_condition -> diffusion -> defect map` pipeline;
- tensor shapes at each major step;
- U-Net down/mid/up block counts;
- PicAdapter injection after every ResBlock;
- FiLM conditioning from the x global embedding plus timestep embedding;
- x-token cross-attention at 32 resolution and in the middle block;
- XMatrixEncoder, PicEncoder, ResBlock, and CrossAttentionBlock internals;
- self-conditioning, diffusion training/sampling, and frequency-aware RayOperator physics loss.

Compile manually with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error network_architecture_base48.tex
```

or from repo root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory simple/diffusion/show simple/diffusion/show/network_architecture_base48.tex
```
