# Benchmarks

`benchmarks.jl` defines the AirspeedVelocity suite (`SUITE`, a BenchmarkTools
`BenchmarkGroup`) that the Benchmark workflow runs on every pull request to `main`:

- `ols`: formula and matrix OLS, n = 1000 to 100,000
- `fe`: OLS with one and two absorbed fixed effects
- `iv`: TSLS, LIML and Fuller
- `vcov`: `model + vcov(...)` for HC0-HC3, CR0 and CR1

The workflow runs the pull request's `benchmarks.jl` on both the base branch and the
pull request, in an environment with Regress, BenchmarkTools and the packages listed
in its `extra-pkgs`, and comments the comparison on the pull request.

## Running locally

`run_asv.jl` runs the same comparison between two revisions of this repository:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/run_asv.jl
```

By default it compares `HEAD` with the working tree. Set `ASV_BASELINE` and
`ASV_CANDIDATE` to other revisions, `ASV_TUNE=true` to tune the benchmarks, and
`ASV_OUTPUT_DIR` to change where results go (default `benchmark/results/`, which
git ignores).

When adding a benchmark that loads another package, add the package to the
`extra-pkgs` of `.github/workflows/benchmark.yml` and to `EXTRA_PKGS` in `run_asv.jl`.
