# Persistence Images in Rust

A high-performance Rust implementation for computing persistence images from persistence diagrams, designed for topological data analysis (TDA) workflows.

The implementation supports voxel spacing correction, anisotropic (unnormalized) Gaussian kernels, and parallel computation using Rayon.

---

## Overview

This tool converts persistence diagrams (birth, death pairs) into fixed-size numerical representations called Persistence Images, which can be used for:

- Machine learning on topological data
- Shape analysis
- Medical imaging (e.g., voxel-based datasets)
- Feature extraction from TDA pipelines

---

## Pipeline

The computation follows this workflow:

CSV diagrams → persistence diagrams → spacing normalization → persistence images → CSV output

More explicitly:

1. Load persistence diagrams from CSV files
2. Load voxel spacing metadata
3. Apply spacing normalization
4. Compute global bounds
5. Generate persistence images using Gaussian kernels
6. Save results to CSV

---

## Input Format

### 1. Diagram CSV files

Each file must contain at least pairs (birth, death) computed using existing persistent homology libraries:

0.12,1.45
0.30,2.10
...

The file/s are expected to have no headers.

---

### 2. Spacing CSV file

filename,...,spacing_micro
image1.tif,...,0.86
image2.tif,...,1.02

- Column 0: filename (must match .tif base name)
- Column m: voxel spacing

---

## Output Format

filename,pixel_1,pixel_2,...,pixel_N
image1,0.12,0.03,...
image2,0.08,0.11,...

Each row corresponds to one persistence image flattened in row-major order.

---

## Usage

### Build

cargo build --release

---

### Run CLI

cargo run --release -- \
  --input ./data/diagrams \
  --spacing ./data/spacing.csv \
  --output ./output/images.csv \
  --nx 10 \
  --ny 8 \
  --sigma-x 0.86 \
  --sigma-y 0.64 \
  --remove-h0

---

## Parameters

nx: grid resolution in x-direction  
ny: grid resolution in y-direction  
sigma_x: global Gaussian bandwidth (x)  
sigma_y: global Gaussian bandwidth (y)  
remove_h0: removes dominant H0 component  

Voxel spacing scales both:
- diagram coordinates
- Gaussian kernel bandwidth

---

## Performance

- Parallelized using Rayon
- Optimized release build recommended

cargo build --release

---

## Dependencies

rayon — parallel computation  
csv — CSV parsing  
walkdir — directory traversal  
clap — command-line interface  

---

## Background

Persistence images are a vectorized representation of persistence diagrams introduced in the following article:

Adams et al., Persistence Images: A Stable Vector Representation of Persistent Homology, JMLR 2017.

---

## License

MIT License

---

## Author

John Rick Manzanares
