# Reproducible cross-language benchmarks

The versioned `workloads.json` is identical in the Rust and Python projects.
Both runners generate identical birth–persistence points via a wrapping 32-bit
linear congruential generator, with FNV-1a fingerprints to detect drift.

Use `cargo run --release --locked --example benchmark_suite -- --list` to see
small, medium, large and 100,000-point test cases. Examples:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --locked --example benchmark_suite -- \
  --case medium_continuous --engine separable --repeats 5 --warmup 1 \
  --output /tmp/rust_benchmark.json

cargo run --release --locked --example benchmark_suite -- \
  --case large_continuous --engine interp-fft --absolute-error 1e-8
```

Each invocation benchmarks one (case, engine), in its own process. The
reported `median_ms` excludes warmup and Rust compilation. `peak_rss_bytes`
is Linux process-wide VmHWM, **including warmup** (not incremental algorithm
allocation); `null` means unsupported. Every result records the CPU,
threading environment, and numerical error against direct Gaussian evaluation.
For small grids, every pixel is checked; for grids larger than 2048 pixels,
only 64 deterministic pixels are checked. Those sample errors are **not**
whole-image accuracy certificates. FFT interpolation's exact-arithmetic
certificate excludes floating-point and FFT roundoff.

The Python repository's `benchmarks/run_suite.py` orchestrates both languages,
emits a JSON report and CSV summary, and verifies fingerprints and image sums.
Do not compare timings taken on different machines or different thread settings
as if they were controlled speedups. Synthetic data do not replace real
persistence-diagram benchmarks.
