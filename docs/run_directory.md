# Run directory

Each case has its own directory, `re<Re>_sar<SAR>`, with its `case.toml` and
the executable built from it. `run/case.toml` is the template, started from
`cases/lid_driven_cavity_512.toml`. Parameters are compiled into the
executable, so GLBM must be rebuilt whenever `case.toml` changes. The program
writes to the directory it is launched from and refuses to overwrite existing
results unless `restart_path` is set.

Build the cases compared with Prasad & Koseff (1989), from `run/`. The cases
with `Re = 3200` use 256 nodes and the others 512:

```bash
for case in re3200_sar1 re3200_sar0.5 re10000_sar1 re10000_sar0.5 re7500_sar0.5; do
  cmake -S .. -B ../build -DLBM_CONFIG_FILE="$PWD/$case/case.toml" &&
  cmake --build ../build --target glbm && cp ../build/glbm $case/ || break
done
```

Run and post-process them in sequence:

```bash
for case in re3200_sar1 re3200_sar0.5 re10000_sar1 re10000_sar0.5 re7500_sar0.5; do
  (cd $case && ./glbm > solver.log 2>&1 &&
   python3 ../process_statistics.py && python3 ../plot_statistics.py)
done
```

`aspect_ratio * n_size` must be an integer and the local number of planes a
multiple of four. At 512 nodes two 24 GB GPUs hold up to `aspect_ratio = 1`.
For `aspect_ratio = 0.25`, `num_gpus = 1` is faster than 2.

```text
re3200_sar1/
├── case.toml
├── glbm
├── checkpoint/
│   └── state.glbm
└── statistics/
    ├── metadata.txt
    ├── local_fields.bin
    ├── global_statistics.csv
    ├── bmbc_blocks.bin
    └── derived/
```

`checkpoint/` contains only the float solver state required for restart.
`statistics/` contains every scientific result and all metadata required by
the Python scripts. Removing `checkpoint/` after a completed run does not
remove any article data.

`local_fields.bin` stores 16 float64 temporal means at every lattice node.
The field names, layout, sampling window, sample count, geometry, Reynolds
number, and lid velocity are embedded in its header. The processing script
preserves these field names in the derived calculations.

`global_statistics.csv` stores domain-reduced time series. `bmbc_blocks.bin`
stores the block history required to reproduce uncertainty estimates. The
state, the local fields and the BMBC history are published together after all
three are completely written.

All plots and tables must be generated from `statistics/`; solver checkpoints
are never analysis inputs.

Post-processing is split into data preparation and plotting:

```bash
python3 process_statistics.py
python3 plot_statistics.py
```

The first command reads `statistics/` and writes `.dat` files under
`statistics/derived/data/`. The second reads only those `.dat` files and
writes PDF figures under `statistics/derived/figures/`.

For restart, set `restart_path = "checkpoint/state.glbm"` in `case.toml` and
recompile. `checkpoint_path` is a file name; the solver always places it under
`checkpoint/`.
