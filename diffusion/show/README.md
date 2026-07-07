# Network Architecture Visualization

This folder contains a LaTeX/TikZ visualization of the diffusion neural network.

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

- the full `pic + x_matrix -> diffusion -> defect map` pipeline;
- tensor shapes at each major step;
- U-Net down/mid/up block counts;
- ResBlock, AttentionBlock, and XMatrixEncoder internals;
- how FiLM conditioning from `x_matrix` and timestep embeddings enters the U-Net;
- how diffusion training and sampling use the network.

Compile manually with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error network_architecture_base48.tex
```

or from repo root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory simple/diffusion/show simple/diffusion/show/network_architecture_base48.tex
```
