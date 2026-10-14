# Persistence Images in Rust

Compute fixed-length feature vectors from persistence diagrams, with parallel
processing, reusable Gaussian bases, and an optional error-controlled
approximation for large images.

The project uses *unnormalized anisotropic Gaussian kernels*. The default
`--engine auto` chooses between separable Gaussian summation and cached
bases where there are enough repeated coordinates to justify reuse. It
does **not** approximate results unless `--absolute-error` is supplied.

## Installation

Requires a Rust toolchain that supports edition 2024.

```bash
cargo build --release
cargo test
```

## Input and output

- Diagram CSV files are headerless. The first two columns are `birth,death`.
- The spacing CSV has a header, filename in column 1, and voxel spacing in
  column 3 (zero-indexed column 2). `image.csv` looks for `image.tif`.
  A missing spacing entry defaults to 1.0.
- The output CSV has **no header**. Rows contain a filename stem and `nx*ny`
  values, flattened so pixel `(i,j)` has index `i*ny+j`.

Each diagram is first converted to `(birth, persistence=death-birth)`. Voxel
spacing multiplies both diagram coordinates and Gaussian bandwidths. Bounds
are computed globally from all scaled diagrams. Grid centers are

```text
x_i = x_min + (i + 0.5) * (x_max - x_min) / nx
y_j = y_min + (j + 0.5) * (y_max - y_min) / ny
```

At pixel `(x,y)`, each diagram point `(b,p)` contributes

```text
exp(-((x-b)^2 / (2*sigma_x^2) + (y-p)^2 / (2*sigma_y^2)))
```

No Gaussian normalization, persistence weighting, pixel integration, or
post-hoc feature normalization is performed.

## Command line

```bash
cargo run --release -- \
  --input ./data/diagrams \
  --spacing ./data/spacing.csv \
  --output ./output/images.csv \
  --nx 96 --ny 96 \
  --sigma-x 0.86 --sigma-y 0.63984 \
  --engine auto
```

The optional `--remove-h0` deletes one point with the smallest birth;
ties use the largest death. `--diagram-suffix` limits the CSV files scanned.

### Engines

| Engine | Method | Numerical behavior |
|---|---|---|
| `auto` | Cost-guided shared basis or separable kernel | Exact in real arithmetic by default |
| `direct` | Original triple loop | Reference implementation |
| `separable` | Compute x/y Gaussian factors once per point | Exact in real arithmetic |
| `shared` | Share x/y Gaussian basis across diagrams with equal spacing | Exact in real arithmetic; memory-limited |
| `sparse` | Evaluate local Gaussian windows | Absolute truncation error bounded per pixel |
| `interp-fft` | Interpolate sources to a lattice, then use one cached Gaussian FFT convolution | Uniform absolute interpolation-error certificate; floating-point roundoff excluded |

Set `--engine sparse --absolute-error 1e-8` for a requested error
tolerance. You may also give `--absolute-error` with `auto`; it selects
the sparse method only when a cheap interaction-count estimate suggests
the approximation will be substantially faster.

For `N` persistence points, a window cutoff

```text
r = sqrt(2 * max(0, log(N) - log(absolute_error)))
```

keeps point/pixel evaluations where
`|x-b| <= r*sigma_x` and `|y-p| <= r*sigma_y`. Every omitted Gaussian
term is at most `exp(-r^2/2)`, hence the **per-pixel absolute truncation
error** is at most `N*exp(-r^2/2) <= absolute_error`. This guarantee
excludes floating-point roundoff, and is not a relative error bound.

### Shared Gaussian bases

For many quantized diagrams, distinct birth coordinates `b_u` and
persistence coordinates `p_v` repeat. Define `C[u,v]` to count how many
points have coordinate pair `(b_u,p_v)`. Then

```text
I = A @ C @ D
A[i,u] = exp(-0.5 * ((x_i-b_u)/sigma_x)^2)
D[v,j] = exp(-0.5 * ((y_j-p_v)/sigma_y)^2)
```

For a batch of diagrams with common grid, bounds, and spacing, both `A`
and `D` are calculated only once. The implementation groups samples by
exact voxel spacing, counts multiplicities separately for each sample,
chooses a cheaper matrix-multiplication order, and retains the original
sample order. It caps the count matrix at two million elements and the
combined basis matrices at 16 million elements. Auto falls back when
reuse is not economical; explicit `shared` reports an error if its
memory cap is exceeded.

Use `--engine interp-fft --absolute-error 1e-8` to explicitly request
one-FFT source gridding. The method is intended for large continuous-valued
diagrams with sufficiently resolved Gaussians. It builds even-width central
Lagrange stencils on an extended lattice, deposits signed weights, then uses
one zero-padded Gaussian convolution. It caches FFT plans and the Gaussian
spectrum across diagrams sharing global bounds, spacing and bandwidths.

For stencil widths `2r_x` and `2r_y`, with `h_x, h_y` the pixel-center
spacing, the exact-arithmetic interpolation certificate is

```text
E_r(h,sigma) = ((2r)!)^2 / (2^(5r) (r!)^3) * (h/sigma)^(2r)
per-pixel absolute interpolation error <= N*(E_rx+E_ry+E_rx*E_ry)
```

The selector searches even widths up to 18 and caps the FFT workspace at
four million complex cells. If no width certifies the requested tolerance,
explicit `interp-fft` errors; `auto` retains other methods. `auto` remains
non-approximating without `--absolute-error`. The interpolation certificate
**excludes** rounding error in signed deposition and FFT operations; it does
not guarantee bitwise equivalence or strict floating-point absolute error.
FFT evaluation requires a nondegenerate global bounding box and sources
inside it. The FFT uses linear-convolution padding to avoid cyclic aliasing
on the output region.

The standalone public `persistence_image` function is retained. New Rust
APIs include `separable_image`, `sparse_image`, `adaptive_image`,
`SharedGaussianBasis::from_diagrams` / `transform`, `interp_fft_image`,
`InterpFft`, `select_stencil`, and `batch_images`.
Different arithmetic order can change floating-point rounding, so compare
against the original reference with tolerances rather than exact bits.

## Stable, weighted, integrated persistence images

Use the `--representation stable` option for **probability-normalized Gaussian
densities**, bounded Lipschitz weights that vanish on the diagonal, and
**exact pixel integration** via Gaussian CDF differences. Legacy pixel-center
evaluation remains the default for backward compatibility.

```bash
cargo run --release --locked -- \\
  --input ./data/diagrams --spacing ./data/spacing.csv \\
  --output ./output/stable.csv --nx 128 --ny 128 \\
  --sigma-x 0.2 --sigma-y 0.2 \\
  --representation stable --weight-scale 1.0 \\
  --bounds 0 3 0 3
```

The per-point weight is `min(1, persistence / weight_scale)` for finite,
nonnegative persistence. The Gaussian integrates to one **over the plane**,
but each image sums to the weighted mass inside the selected pixel rectangle.
Do not divide each image by its sum: such per-image normalization would
undermine stability near the empty diagram.

For a **fixed grid, fixed bandwidths and fixed weight scale**, the map has a
global Wasserstein-1 Lipschitz bound (birth-death L-infinity matching cost):

```text
||PI(D)-PI(E)||_1 <= [2/weight_scale
  + sqrt(2/pi)*(1/sigma_x + 2/sigma_y)] * W1_inf(D,E)
```

The proof uses the bounded Lipschitz weight, Gaussian L1-derivative bounds,
and contraction under integration over disjoint pixels. This guarantees
stability of the mathematical representation, not floating-point error.
See the companion Python repository's
[proof](https://github.com/jhnrckmnznrs/persistence_images_python/blob/main/docs/STABLE_PERSISTENCE_IMAGES.md).

**Fixed bounds are essential for comparisons.** By default the CLI computes
bounds from the current input dataset, which changes the map when the dataset
changes. Pass `--bounds xmin xmax ymin ymax` for a fixed feature space.
Degenerate fitted bounds are rejected in stable mode. Essential/infinite
persistence classes must be treated separately; their stable representation
is not defined by finite Gaussian centers. Spacing-scaled diagrams use
spacing-scaled Gaussian bandwidths as in the original CLI; varying per-diagram
bandwidths also changes the fixed-map hypothesis of the stability theorem.

The stable mode also supports **exact weighted Gaussian CDF-difference
basis reuse** across batches of diagrams with matching grid, bandwidth
and weight scale. For distinct births `beta_u` and persistence values
`pi_v`, let `C[u,v]` count repeated coordinate pairs, and set

```text
A[i,u] = Phi((x_{i+1}-beta_u)/sigma_x)-Phi((x_i-beta_u)/sigma_x)
D[v,j] = min(1, pi_v/weight_scale)
       * (Phi((y_{j+1}-pi_v)/sigma_y)-Phi((y_j-pi_v)/sigma_y))
PI = A @ C @ D
```

`--representation stable --engine shared` forces basis reuse (and enforces
memory limits); `--engine auto` uses a conservative cost estimate and
falls back to pointwise integration when reuse is not economical.
Per-diagram bandwidth changes caused by voxel spacing are grouped
separately. This factorization remains exact in real arithmetic and
preserves the same fixed-grid Wasserstein stability theorem; floating-point
summation order may differ. `StableCdfBasis::from_diagrams` and
`stable_batch_images_with_engine` are available from the Rust API.

The stable mode uses a separate `stable_persistence_image` Rust API. It
supports `auto`, `direct`, `separable`, and `shared` CLI engines (all evaluate the
integral formula), and rejects truncating FFT/sparse backends and
`--absolute-error`. Other engines apply only to the legacy sampled mode.

For a reproducible synthetic performance comparison of the direct stable
integrator against the shared CDF basis:

```bash
cargo run --release --locked --example benchmark_stable_shared
```

Timings depend on hardware and on the diagram's coordinate multiplicities.

## Testing and benchmarking

```bash
cargo test
cargo run --release --example benchmark_optimized
cargo run --release --example benchmark_interp_fft
```

The benchmark compares one batch of quantized diagrams using the original
and adaptive algorithms. It prints wall-clock times and maximum absolute
numerical difference. Benchmark your real data before assuming a speedup.

## Background

Adams et al., *Persistence Images: A Stable Vector Representation of
Persistent Homology*, JMLR (2017). This project's unweighted,
unnormalized construction should not automatically be assumed to satisfy
the stability assumptions of the weighted persistence image in that work.

## License

MIT. Original author: John Rick Manzanares.

## Reproducible cross-language benchmarks

The suite in [benchmarks](benchmarks/README.md) shares a deterministic workload
manifest with the companion Python repository. It records small, medium, large,
quantized, continuous, narrow-Gaussian and anisotropic workloads. Reports contain
median kernel time, peak Linux RSS, generator fingerprint, and independently
computed numerical error. Each run isolates a single workload in one process.

```bash
cargo run --release --locked --example benchmark_suite -- --list
cargo run --release --locked --example benchmark_suite -- \\
  --case medium_continuous --engine separable --repeats 5
```

For a combined Python/Rust sweep, use the Python repository's
`benchmarks/run_suite.py`. CI smoke benchmarks test correctness, not speed.
