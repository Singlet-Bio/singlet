<!-- Copied from zdebruine/singlet-website PR #66 (branch claude/benchmarking-pages-fa8072), docs/research-beta.md -->
# Research beta: benchmarks and roadmap

Pages (all carry a "Research beta" banner):

- `/benchmarking`: hub with one card per capability (22), pass/fail counts, a Reach matrix and the notebook download.
- `/benchmarking/<capability>`: hero, up to 3 figures, quality-vs-cost Pareto chart(s), Reach panel, conclusion, pass/fail table
  (failing metrics first), runtime and memory, datasets (GSE/GSM links), tool versions, Singlet commit, methods, caveats, code links.
- `/roadmap`: milestones toward Singlet 1.0.0.

Pass/fail is exactly what the results contract's validator wrote (`metrics[].pass`); the site never recomputes a verdict. A page is
"Passing" only when its document is `final` and every gating metric passes. A capability with no document shows "Benchmark in progress".
Draft documents are labelled as drafts.

## Where the data lives

- `public/data/benchmarks/<capability>.json`: one document per capability, contract 1.x (`Singlet-Bio/singlet`, `bench/public/CONTRACT.md`).
- `public/data/benchmarks/index.json`: the contract's index (only used for `unresolved_corroboration`; hub status comes from the documents).
- `public/data/benchmarks/singlet_benchmarks.ipynb`: the notebook linked from the hub.
- `public/data/roadmap.json`: milestones (`status` done | active | upcoming, `progress` 0-100 on the active one, `updated_at`).

## Update results

1. In the `singlet` repo, regenerate the document with the capability's `reproduce` function and validate it
   (`python bench/public/contract.py validate <file> --cohort bench/public/cohort.tsv`).
2. Copy `bench/public/results/*.json` and `bench/public/notebook/singlet_benchmarks.ipynb` into `public/data/benchmarks/` unchanged.
3. Branch, `npm run check`, PR. Update `public/data/roadmap.json` when a milestone moves.

`src/test/benchmarks.test.tsx` parses every file in `public/data/benchmarks/` and fails on a malformed one, a conclusion over 100 words,
or a document shown as passing while a metric fails.

## Deploy to beta.singlet.bio (Zach)

Nothing here has been deployed. Steps in the Cloudflare dashboard:

1. Workers & Pages > singlet (Pages project) > Settings > Builds & deployments > Branch control: keep Production = `main`; the beta
   branch (this PR's branch, or a long-lived `beta` branch) gets a preview at `https://<branch-with-dashes>.singlet-4gc.pages.dev`.
2. Custom domains > Set up a custom domain > `beta.singlet.bio`. A custom domain on a Pages project serves production; to point it at
   a branch instead, create a second Pages project (for example `singlet-beta`) connected to the same repo with Production branch = `beta`,
   build command `npm run build`, output `dist`, then add `beta.singlet.bio` there. Copy the bindings (D1, R2, AI) and variables from the
   main project, since previews share the production D1.
3. DNS: if `singlet.bio` is on Cloudflare the CNAME for `beta` is created for you; otherwise add `beta` CNAME to the project's `pages.dev` host.
4. Run `scripts/smoke.sh https://beta.singlet.bio` and open `/benchmarking` and `/roadmap`.

Check `GITHUB_CLIENT_*` / `GOOGLE_*` OAuth redirect URLs if sign-in should work on the beta host; the benchmark pages need no sign-in.
