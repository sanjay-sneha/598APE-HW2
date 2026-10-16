# Artifact for Symbolic Regression Optimizations

Here we have instructions explaining how to build the code, reproduce our baseline and optimized results, and evaluate each individual optimization. We made all our changes in the `opt` branch.

---

## 1. Tools and requirements

- `g++` with C++17 and OpenMP support (OpenMP's `libgomp` ships with GCC)
- `make`
- `racket` (to render the reconstructed images)
- `perf` (for profiling)
- [FlameGraph](https://github.com/brendangregg/FlameGraph) scripts (to generate flamegraphs)

We built and tested everything on the course VM.

---

## 2. Building and running

### Build
```bash
make clean && make -j
```

### Run and time all five benchmarks

```bash
for d in diabetes cancer housing silk-wave square-wave; do
  echo "$d:"; /usr/bin/time -p ./genetic_benchmark $d > $d-out.rkt
done
```

Each run writes its evolved programs as a Racket file (`<dataset>-out.rkt`). The program's own timer is printed on the last line of each file. The numbers in our report are the `real` times from `/usr/bin/time -p`.

### Render the reconstructed images

```bash
racket silk-wave-out.rkt silk-waves-reconstructed.png
racket square-wave-out.rkt squared-waves-reconstructed.png
```

### Verify correctness

Compare the outputs to the provided references while ignoring the timing line:

```bash
diff <(grep -v '^;;Time' benchmark/silk-ref.rkt) <(grep -v '^;;Time' silk-wave-out.rkt)
diff <(grep -v '^;;Time' benchmark/square-wave-ref.rkt) <(grep -v '^;;Time' square-wave-out.rkt)
```

The evolved programs match the references. Some MSE values differ in the last few digits due to floating-point differences between machines.

---

## 3. Profiling with perf

To reproduce the flamegraphs in our report:

```bash
git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph   # once

perf record -e cycles:u -F 999 -g -o /tmp/perf.data ./genetic_benchmark housing > /dev/null
perf script -i /tmp/perf.data --no-inline | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > flamegraph.svg
```

Open `flamegraph.svg` in a browser. Replace `housing` with any other benchmark name to profile that benchmark (our stable sort flamegraphs use `cancer` and all the others use `housing`). For a text summary instead, run `perf report -i /tmp/perf.data --no-inline`.

---

## 4. Optimizations and how to evaluate

Each optimization is its own commit on the `opt` branch (see the commit history on GitHub). To evaluate one, check out its commit, rebuild, and time the benchmarks; then do the same for the commit before it. The difference between the two times reproduces the result reported in that section of our paper.

```bash
git checkout <commit>
make clean && make -j
# run the timing loop from Section 2
git checkout opt          # return to the latest version when done
```

- **0:** Baseline (`main` branch)
- **1:** Compiler optimization flags (`-O3 -g -fno-omit-frame-pointer`)
- **2:** Replaced the GPU-style register stack with a plain array and precomputed node arities
- **3:** Parallelized program evaluation with OpenMP over blocks of rows
- **4:** Link-time optimization (`-flto`)
- **5:** Loop unswitching in block evaluation
- **6:** Replaced the O(n²) insertion sort in the benchmark with `std::stable_sort`

Optimizations 1 and 4 are pure compiler-flag changes in the `Makefile`; the rest are source changes in `src/program.cpp`, except optimization 6, which is in `benchmark/genetic_benchmark.cpp`.
