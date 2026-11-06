# Figures

Captions for the main figures. The images only have panel letters, axis labels and legends; what was run, on what, and how to read it is here. Each entry names the data and the script that draws it.

All runs on an RTX 4070 Laptop GPU (8 GB, 32 MB L2 cache) under WSL2. "C1-C9" is our set of exact speed-ups to SOLWEIG-GPU v2.0.0, listed in [SOLWEIG_MODIFICATIONS.md](SOLWEIG_MODIFICATIONS.md).

## Speed

### figures/profile/summary.png

![](../figures/profile/summary.png)

SOLWEIG-GPU v2.0.0 before and after C1-C9, on a 1000 x 1000 tile (Pacoima, 2 x 2 km at 2 m).

**a** Time for one simulated day after each change. The first bar is the original code writing its outputs to the Windows drive from WSL. Moving them to the Linux disk is a setup change, not a code change. From the original on the Linux disk to C1-C9 the day goes from 48.4 s to 14.7 s (3.3x). The last bar compiles the shadow loop with torch.compile. It is an experiment and not part of C1-C9.
**b** The sky view factor step, run once per tile: 108 s to 22 s, with identical cache files. The numbers inside the bars are peak reserved GPU memory.
**c** Peak GPU memory over the day. Reserved memory is what decides out-of-memory. The dashed line is the original. C1 with sky chunks of 64 needed more memory than the original. Going to chunks of 8 costs about 1 s and frees 0.9 GiB, and the bit-packed masks (C4/C5) bring the peak down to 1.1 GiB.
**d** A 2000 x 2000 tile (4 x 4 km) on the same 8 GB GPU, one run each. The original day needs 11.7 GiB. It only finished because the Windows driver spills GPU memory to system RAM, and took 61 min; on a native Linux driver it would run out of memory. With C1-C9 the day needs 4.2 GiB and 1.4 min, and UTCI and Tmrt agree with the original to 9e-5 °C.

Timing: two sessions (3 and 4 Nov 2026), quiet machine on AC power, 5 runs per day bar and 3 per sky view bar in each session. Bars are the median of all runs, dots are single runs. The first run of the packed cache (C9, writes the cache) and of torch.compile (compiles) are left out. Outputs are bit-identical from step to step, except C1 and the chunk change, which round differently by at most 0.0001 °C. The big tile runs are single runs: the original on 30 Oct with nothing else on the GPU, C1-C9 on 4 Nov.

Data: `figures/profile/bench_steps.json`, `bench_steps_b.json` (`scripts/bench_steps.py`), `bigtile_demo.json`, `bigtile_compare_current.json` (`scripts/bigtile_demo.py`). Script: `scripts/plot_summary.py`, which also saves each panel on its own (`summary_time.png`, `summary_svf.png`, `summary_memory.png`, `summary_bigtile.png`).

### figures/profile/bounds.png

![](../figures/profile/bounds.png)

Where the time goes with C1-C9, and how far each part is from its floor.

**a, b** Time per function over 00-14 h on Pacoima crops of 1000 x 1000 and 2000 x 2000 px. Bars are measured self time, median of 5 runs, dots are the single runs. Orange: the bytes the function's operations read and write, divided by the measured bandwidth (227 GB/s). Purple: its kernel launches times the measured cost of one launch (7.0 µs). A function close to its orange line is memory-bound, and only moving fewer bytes makes it faster. Bytes are counted per operation, and the L2 cache serves some reuse, so the orange line can lie past the bar. Launches overlap with GPU work, so the purple line is the most that CUDA graphs could save. Summed over the day that is 2.5 of 48.4 s at 2000 px, but up to 2.0 of 8.4 s at 1000 px. The day totals come from 6 runs without the per-function timing (8.30-8.61 s and 47.95-49.11 s).
**c** One step of the hourly shadow loop at 2000 x 2000 px, 5 repeats. Not compiled (rewritten with fixed shapes so it can be compiled): 7.41 ms. Compiled with torch.compile: 0.85 ms. The compiled step captured as one CUDA graph, so with no launch cost: 0.84 ms. Reading and writing the step's 12 rasters once at full bandwidth: 0.85 ms. So the compiled step already runs at its bandwidth floor. The hatched bar is an estimate of something not built: a kernel that keeps the loop state on chip and reads only the three shifted inputs per step (0.21 ms).

Data: `figures/time_budget_current_r5.json` (`scripts/where_time_goes.py --repeats 5`), `figures/profile_compiled_steps.json` (`scripts/profile_compiled_steps.py`); the plotted numbers are in `figures/profile/bounds.json`. Script: `scripts/plot_bounds.py`, which also saves `bounds_day.png` (a, b) and `bounds_step.png` (c).

## Forecast ensembles

### figures/la_level_vs_pattern.png

![](../figures/la_level_vs_pattern.png)

An air-temperature forecast error moves the level of a street-scale UTCI map, not its pattern.

Perfect-model ensembles on two 2 x 2 km tiles at 2 m, 30 July 2019: Pacoima (ground heating fitted to the measured surface temperatures, heat kept after sunset) and the Exposition Park venue tile (SOLWEIG's own ground model). 20 members per error size. The air-temperature error is AR(1) in time (lag-1 correlation 0.9) and has one value per hour for the whole tile, since SOLWEIG takes one air temperature per tile.

**a** Each member's UTCI change at 15:00 against the unperturbed run, split into the change of the tile mean (solid) and the standard deviation of the change across outdoor pixels (dashed), both averaged over members. Log scale. The tile mean moves by about 1.1 times the error standard deviation at Pacoima and 0.9 times at the venue. The part that differs between pixels is 3-5.5% of that. Each member's map keeps the ranking of the unperturbed map (Spearman correlation at least 0.999).
**b** Share of outdoor area with UTCI above 32 °C at 15:00, mean over members, band ±1 standard deviation (clipped at 0 and 100%). At an error of 1.5 K the area varies by 17 points between members at Pacoima and by 30 at the venue. The mean falls as the error grows, so a cold error takes away more area than an equally large warm error adds, which is what you expect when most outdoor area is already above 32 °C.

For scale: HRRR 2 m temperature errors over the western US in summer are about 1.5 K at analysis time and 2-2.7 K at 9-24 h lead (James et al. 2022, Wea. Forecasting 37, fig. 10c). The errors here are the same everywhere in the tile by construction, so part of the result is built in. A test with errors that vary in space is next.

Data: `figures/la_forecast_error.json`, `figures/expo/la_forecast_error.json` (`scripts/run_la_forecast_error.py`, `scripts/analyse_la_forecast_error.py`). Script: `scripts/plot_level_pattern.py`.

### figures/shortcuts/shortcuts_vs_noise.png

![](../figures/shortcuts/shortcuts_vs_noise.png)

How few physics runs a 20-member ensemble needs.

In each setting, 20 members carry errors in air temperature (1.5 K), wind (30%) and global radiation (10%), AR(1) in time (lag-1 correlation 0.9), and are run with full physics. The shortcuts build the same 20 members from a few physics runs:

- 1 run: the unperturbed Tmrt, with each member's air temperature, humidity and wind in UTCI
- 2 runs: + the change in Tmrt per K of air temperature, per pixel and hour
- 3 runs: + the change per 10% of radiation
- 4 runs: + the same for radiation 10% low, so up and down have their own slope
- 14 runs: + a table of Tmrt at 13 radiation scales per hour (0.7 to 1.3)

**a** Mean absolute error of the shortcut's event probabilities against full physics, divided by the same measure between two independent 20-member full-physics ensembles. That second number is the sampling noise of the ensemble itself, so 1 means the shortcut is as far off as a second ensemble would be. One point per setting and probability; probabilities that are 0 or 1 everywhere on a day are left out. Black bars are medians: 0.94, 0.63, 0.16, 0.065, 0.05. With one radiation slope the shortcut error is under 1% to 33% of the noise.
**b** Single-member UTCI error against full physics: the largest over all members, outdoor pixels and hours, and the 99th percentile over pixels and hours, averaged over members. With one radiation slope the 99th percentile is below 0.9 °C, while the largest errors stay at 2-5 °C in a few pixels. The heatwave day with SOLWEIG's Perez clearness expression as coded has errors of 14-17 °C for every shortcut. With the expression as published they drop to 1.4-3.3 °C (see the next figure).

Settings: venue tile on 30 Jul 2019 (also with radiation errors correlated with the air-temperature errors), 15 Jun 2019 (morning cloud) and 6 Sep 2020 (heatwave); Pacoima on 30 Jul 2019, with SOLWEIG's own ground model and with the fitted one. All with C1-C9.

Data: `figures/shortcuts/shortcut_suite_*.json` (`scripts/shortcut_suite.py`), medians in `shortcuts_vs_noise.json`. Script: `scripts/plot_shortcuts.py`.

### figures/radiation_sweep_expo_compare.png

![](../figures/radiation_sweep_expo_compare.png)

Tmrt against an error in global radiation at one hour, and a bug in the Perez sky clearness inherited from UMEP.

Venue tile, 30 July 2019, SOLWEIG's own ground model. In each run, global radiation at 07:00, 12:00 and 17:00 is scaled by the same factor (an error at one hour hardly carries into later hours); 13 factors from 0.8 to 1.2. Solid: tile mean. Dashed: median over the quarter of outdoor pixels with the highest Tmrt in the unperturbed run, which at these hours are the sunlit ones.

Orange is the Perez sky clearness as coded in SOLWEIG-GPU and UMEP, ((Idh + Ibn)/(Idh + 1.041 z³))/(1 + 1.041 z³). Teal is the published definition (Perez et al. 1990, as pvlib implements it; checked against pvlib, not the paper itself), ((Idh + Ibn)/Idh + 1.041 z³)/(1 + 1.041 z³), with z the solar zenith angle in radians.

**a** At 07:00 both forms jump when radiation is 3-7% below the forecast. That jump comes from the direct/diffuse split's break at clearness index 0.78, not from the Perez expression.
**b** At 12:00 Tmrt is close to linear in the radiation error.
**c** At 17:00 the coded form jumps when radiation is 3-7% above the forecast, about 1.3 K more in the sunlit quarter than the published form. The coded clearness crosses the boundary of the first Perez category (1.065) where the published one does not. On the heatwave day the same switch moves Tmrt at 08:00 by 4.6 K on average and up to 28.6 K in single pixels.

Data: `figures/radiation_sweep_expo.json`, `figures/radiation_sweep_expo_perez_published.json`. Script: `scripts/radiation_sweep.py --compare expo`.

## Sensitivity

### figures/la_sobol_day.png

![](../figures/la_sobol_day.png)

Which inputs drive heat stress at Pacoima through the day.

1280 runs (Sobol' design, N = 128, 8 inputs) on the central 1 x 1 km of Pacoima at 2 m, 30 July 2019, with ground heating fitted to the measured surface temperatures and heat kept after sunset (time constant 4 h). Inputs, uniform over: air temperature ±2.5 K, relative humidity ±6 points, wind x 0.3-1.0 (the station wind reduced to street level), global radiation x 0.9-1.1, paved emissivity 0.90-0.98, wall albedo 0.10-0.50, wall emissivity 0.85-0.95, paved albedo 0.06-0.08. The indices depend on these ranges; where each comes from is in `scripts/run_la_ensemble.py`.

**a** Mean outdoor UTCI. Total indices from the direct estimators (Jansen; first order Saltelli 2010), bands are 95% bootstrap intervals. A band can go above 1 through sampling error; the index itself cannot. Wind is exactly 0 at 06:00, 09:00 and 10:00, here and in b and c: the station reports calm (0.1 m/s) then, and SOLWEIG-GPU raises wind below 0.15 m/s to 0.15 before computing UTCI, so scaling it changes nothing.
**b, c** Share of outdoor area with UTCI above 32 and 38 °C. The direct estimators do not resolve these at N = 128, so the indices come from a Gaussian-process surrogate of the same runs (band: spread over its 5 cross-validation fits). They are drawn only at hours where the surrogate predicts held-out runs with R² of 0.9 or more. Gaps are hours with no area above the threshold, or 38 °C at 19:00, where R² is 0.85.

Each panel draws the three inputs with the largest total index averaged over the day. Dashed: the sum of first-order indices; 1 means the inputs act additively, and the gap below 1 is interactions (mostly air temperature with wind). The other inputs average below 0.1, but humidity and paved emissivity reach 0.22-0.23 for the 32 °C area at 20:00, and wall albedo reaches 0.49 for the 38 °C area at 10:00, when little area is above 38 °C.

Data: `figures/la_sobol.json`, `figures/la_sobol_surrogate_allhours.json` (`scripts/run_la_sobol.py`, `scripts/sobol_surrogate.py --all-hours`). Script: `scripts/analyse_la_sobol.py`.

### figures/la_sobol_surrogate_check.png

![](../figures/la_sobol_surrogate_check.png)

Check of the surrogate behind la_sobol_day.png. A Gaussian process (Matérn 5/2, one length scale per input) is fitted to the 1280 runs, and its indices come from a Saltelli design of 8192 base samples on its mean.

**a** Total index from the surrogate against the direct estimator, for every input, at the hours where the direct estimator is resolved. Bars are its 95% bootstrap intervals. Dotted: equal.
**b** R² on held-out runs (5-fold cross-validation) at each hour where the area varies between runs. Dotted: 0.9, below which la_sobol_day.png leaves a gap.

Data: `figures/la_sobol_surrogate.json`, `figures/la_sobol_surrogate_allhours.json`. Script: `scripts/sobol_surrogate.py --check-figure figures`.

## Supplementary

### figures/la_forecast_error_fss.png and figures/expo/la_forecast_error_fss.png

![](../figures/la_forecast_error_fss.png)

![](../figures/expo/la_forecast_error_fss.png)

Fractions skill score (Roberts and Lean 2008) of the ensembles in la_level_vs_pattern.png, Pacoima first, then the venue. Each member in turn is the truth and the others are the forecast.

**a** Event: UTCI above 32 °C for 4 h or more. Most pixels reach it, so this score mostly measures the level of the map. Solid: the probability from the other 19 members. Dashed: one other member, as a single forecast. Lines are the mean over truths, bands the 10th to 90th percentile of the single forecast. Dotted: useful skill, FSS = 0.5 + f/2 with f the event frequency.
**c** Event: each map's own hottest 10% of outdoor pixels at 15:00. This removes the level and asks only whether the hot spots are in the right streets.
**b, d** The smallest neighbourhood at which the forecast is useful. The tested neighbourhoods are odd numbers of 2 m pixels (1, 5, 13 ... 501), hence 2, 10, 26 ... 1002 m. 2 m means useful at the single pixel. "none" (x) means not useful at any scale tested, up to 1 km.

UTCI above 38 °C for 4 h or more is too rare on this day to score; it is in the JSON. Same data as la_level_vs_pattern.png. Script: `scripts/analyse_la_forecast_error.py` (`--replot` redraws from the JSON).

## Other figures

The other figures in `figures/` still carry their own titles and notes and have no entry here yet: the earlier sensitivity work (one-at-a-time, Morris, emulator, Los Angeles against measurements), the Sobol maps (`la_sobol_maps_16h.png`), the single radiation sweeps (`radiation_sweep_expo.png` and its published-Perez version) and the memory and chunk plots in `figures/profile/`.
