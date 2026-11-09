# fastloop: cheap, stacked screening of GPU ideas

A full sequential A/B of one idea (bench/gpu/seqab, tier 4: pbmc1k_v3, 8 in flight on 2 slots) took about 1.5 H100-hours, mostly because the 40-run cap was reached and because CPU-fallback runs, three times the cost of a GPU run, widened the interval. fastloop answers the first question about an idea (does it act, on which stage, and by how much) at a small fraction of that cost, and it answers it for several ideas at once. The full A/B is kept for the one idea that wins a screen.

## How it works

Each idea is a switch: an environment variable or a few extra arguments, named in `ideas.tsv`, or one of tp_gc's own atoms. `design.py` puts k ideas into a two-level factorial design. For k of 4 or more it uses a fraction of resolution IV or better, so every main effect is clear of every other main effect and of every two-way interaction. Each cell of the design becomes one tp_gc point whose config joins the ideas that are on with `+`. The cells run in a shuffled order on one pinned GPU, and a second block repeats them in reverse, so slow drift over the job falls equally on every factor.

`analyze.py` reads the points' `point.json` files and fits an ordinary least-squares model with one ±1 column per idea and one dummy per block. An effect is twice its coefficient, reported relative to the fitted all-off level with a 95% bootstrap interval (resampling runs within each cell and block). Runs that failed or fell back to the CPU are left out of the timing and counted per idea, since a fallback is a result in its own right. The first run of each one-at-a-time point is a warm-up and is dropped. A run above 1.4 times its cell's lower median on the primary metric is listed and left out.

Attribution comes from three sources the runs already record: the engine's step timers (`st.*`, for example lane B or the index upload), the device lock waits and holds (`lk.*`) and the stage walls of the run's own log (`sg.*`). For each idea the report lists the stage metrics whose interval excludes zero, so an idea that should shorten lane B and instead moves the hand-off is visible at once.

Each idea is decided on its own metric (column 6 of `ideas.tsv`, held device seconds by default; host ideas such as the demux screen use core-seconds per M read pairs). An idea WINS when its whole interval lies below −θ (3%), is RULED OUT when its whole interval lies above −θ, and stays OPEN otherwise, which calls for another block. A device win whose core-s/M interval lies wholly above +3% is marked GUARD, following the campaign's rule.

## Tiers

| tier | sample and shape | what it sees | cost of a 2^3 design, 2 blocks |
|---|---|---|---|
| S1 | pbmc2m, one run at a time, 3 runs per point | whole pipeline at small size | 33 min on an RTX 8000, 32 min on an H100 (item 60) |
| S2 | pbmc2m, 4 in flight on 2 slots | concurrency | unusable on RTX and V100S (index exceeds a slot); 18 min on an H100 for one block (item 61) |
| S3 | pbmc1k_v3, one at a time (`maponly` ends after STAR and the pileup on builds from 23f83010, but its runs still lack device fields in point.json) | size-dependent lane ideas | 51 min on an H100 for one block of a 2^3 design, whole pipeline (item 62) |
| FULL | pbmc1k_v3, 8 in flight on 2 slots | the campaign's tier 4 | 1.5 h per idea by seqstop, about 20 min per block |

On the RTX 8000 the S1 tier runs at about 1% coefficient of variation within a cell on every metric, against 2.4 to 2.7% for core-s/M and 11 to 23% for the device metrics at tier 4. A 2^3 design with 2 blocks resolves effects of about ±1% on held device time, and a 2^4 screen of four ideas took 31 RTX-minutes and no H100 time. The H100 behaves the same way: item 60 decided three ideas in 32 minutes with intervals of ±0.4 to ±1.0% on held device time, about 11 H100-minutes per verdict against 90 minutes and ±2.8% for one full sequential A/B. S2 adds concurrency at the price of noise: one block of item 61 gave intervals of about ±5%, enough for the shared index (−29%) and too wide for small effects.

## Calibration against the H100 anchors

| idea | H100 full A/B | RTX S1 | V100S S1 | H100 S1 / S3 |
|---|---|---|---|---|
| shared index | held −12.8%, index −65% | held −14.0%, index −81% | every run fell back (no room beside the holder) | S1 held −20.2% [−21.0, −19.4], index −85%; S2 −28.9% [−33.4, −24.2]; S3 −6.4% [−8.6, −4.4] |
| WSHRINK | held −12.0%, lane B −41% | held −5.8%, lane B −44% | held −2.8% | S1 +0.9% [−0.1, +2.0], lane B −13%; S2 +2.0% [−2.7, +7.5]; S3 −1.6% [−3.7, +0.6], lane B −17%, host lane +1.65 s, core-s/M +3.3% |
| retry off | no measurable cost | held −0.4% [−1.4, +0.6] | +2.2% | S1 −0.9% [−2.0, +0.1]; S2 −0.0% [−5.5, +5.0]; S3 +0.2% [−2.1, +2.5] |
| stepsauto | lane B +41%, held +7% | no effect (the split never triggers at pbmc2m size) | no effect | not run |

The short tiers agree with the full A/B on the shared index and the retry at every size and on both cards. They miss WSHRINK on the H100, and the stage timers say why. Run one at a time, WSHRINK moves part of lane B's work onto the host lane, so the device saves less than the host spends. Its full-A/B win comes from 8 runs sharing the device, where device time is the bottleneck and host work overlaps. An idea that trades device work for host work therefore needs a concurrent tier (S2 with more blocks, or the full A/B), and the report's stage lines flag such an idea: lane B falls while the host lane rises.

The RTX S1 tier agrees with the H100 on direction and stage for every anchor and is within a factor of two on size, which makes it a good first screen: does the idea act, and where. It does not rank by size. Lane B's share of the hold differs between the cards: at full size WSHRINK shortens lane B by 41% on the H100 and leaves it unchanged on the RTX. Three classes of idea therefore need the H100: concurrency ideas (slots, MPS, admission), ideas that need the H100's memory (the shared index on a V100S, the resident non-host server anywhere else) and ideas whose effect depends on sample size (S3).

## Verdicts of the first round (2026-11-07)

| idea | tier and item | held device time | core-s/M | verdict |
|---|---|---|---|---|
| resident non-host server (`nhsrv`, XC card (a)) | H100 S1, item 63 | −26.6% [−27.1, −26.2] | −20.1% [−21.0, −19.2] | WIN; sent for a k8 confirmation A/B and the identity gate |
| shared index (`shidx`) | H100 S1, S2, S3, items 60–63 | −20% (S1), −29% (S2), −6% (S3) | −1.6 to −2.0% | WIN; adoption A/B at k8 run by the campaign |
| WSHRINK | H100 S1 and S3 | neutral at one run at a time | +1 to +3% | concurrency effect, confirmed at k8 by the campaign's full A/B |
| retry off (`nhtries1`) | H100 S1, S2, S3 | within ±1% | within ±1% | RULED OUT |
| stepsauto | RTX S1 | no effect (the split never triggers at pbmc2m) | none | RULED OUT at small size; full A/B inconclusive |
| demux screen | RTX S1 | none | −1.9% [−2.2, −1.5] | below the 3% bar |

The resident server wins on every metric. Its non-host lock hold halves and the non-host references stop being loaded per run. Each nhsrv point starts a fresh server and drops its first run, so the effect is the warm steady state that one server per node would give. Because a server-plus-preload arm ran slower at 8 in flight, the k8 A/B decides adoption. The campaign runs it as a sequential A/B on core-s/M at k8t7s2r1 (pilot item 33, freeze binary ee7fa54d). For scale, its item 27 measured one H100 with 54 cores at a steady 2.65, 2.57 and 3.12 M pairs/s at 8, 12 and 16 in flight (T7, 2 slots), with 24 to 31 cores busy.

Time per verdict fell from about 90 H100-minutes for one sequential A/B to about 11 H100-minutes, with intervals of ±1% instead of ±2.8%. Six ideas were decided in about 2.5 H100-hours, including the calibration items. The RTX 8000 tier answered "does it act, and where" for four ideas in 31 minutes without H100 time.

## Use

```bash
python design.py --ideas wshrink,shidx,stepsauto,screen --tier S1 --bin <commit8>-multi --name 71-scr4-s1 --sole --out items
sbatch --export=ALL,ITEM=$W/jobs/gpu-fastloop/items/71-scr4-s1.g1.sh run_on.sbatch                       # RTX 8000 screen
sbatch --gres=gpu:tesla_v100s:1 -c 11 --mem=170G --export=ALL,ITEM=... run_on.sbatch                      # V100S screen
clipper.py put items/<item>.g1.sh $W/jobs/gpu-campaign/queue1/new/<item>.g1.sh                            # H100, through the one-GPU pilot
python analyze.py --res <results dir> --ideas a,b,c --drop-first 1 [--2fi] [--tsv out.tsv]               # the item runs this itself
```

`--sole` marks an item that needs its node alone, because the shared-index holder or the non-host server occupies the device across runs. On the pilot the item then refuses to start when another job shares the node. `run_on.sbatch` runs any item unchanged on an RTX or V100S node (compute capability 7.x admitted, `-multi` binaries only). Use `--exclusive` or `FL_NODE_OK=1` for shared-index items there. A new idea needs one row in `ideas.tsv`: name, env switches, extra arguments, stage, note and decision metric. A row with neither env nor arguments only labels one of tp_gc's own atoms, such as `nhsrv`, which starts `singlet gpu-nhsrv` for the point.

## Files

- `design.py`: the designs, alias chains and the pilot item.
- `analyze.py`: effects, intervals, attribution, fallback rates and two-way interactions (pure Python).
- `ideas.tsv`: the idea registry.
- `run_on.sbatch`: an item on an RTX 8000 or V100S.
- `items/`: the calibration and screening items.
- `../pilot/tp_gc.sbatch`: composite configs (`a+b+c`), per-item holder and server cleanup.
