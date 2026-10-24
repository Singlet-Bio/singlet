// SPDX-License-Identifier: MIT
// JS scan speed of the candidate shard formats (V8, as in a Worker). Needs hyparquet + hyparquet-compressors and the SCX reader
// bundled from the website repo:
//   npx esbuild <website>/workers/data-v2/src/scx.ts --format=esm --outfile=bench/cidx/.scx.mjs
//   node bench/cidx/scan.mjs <dir>        (<dir>/fmt/* from formats.py)
// Query: cells with umi_exon >= 1000 and ct == 101, returning (sample, col) of every match.
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { parquetMetadata, parquetRead } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { parseScxHeader, decodeChunk, qcode } from "./.scx.mjs";

const dir = process.argv[2];
const COLS = ["sample", "col", "umi_exon", "ct"];

function toAB(b) {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

async function scanParquet(path) {
  const buf = toAB(readFileSync(path));
  const t0 = performance.now();
  let rows = null;
  await parquetRead({ file: buf, columns: COLS, compressors, rowFormat: "array", onComplete: (r) => (rows = r) });
  const t1 = performance.now();
  let n = 0;
  for (const r of rows) if (r[2] >= 1000 && r[3] === 101) n++;
  return { ms: t1 - t0 + (performance.now() - t1), decodeMs: t1 - t0, n, rows: rows.length };
}

async function scanParquetColumnar(path) {
  // column chunks one by one (closer to what a Worker would do: only the needed columns)
  const buf = toAB(readFileSync(path));
  const t0 = performance.now();
  const md = parquetMetadata(buf);
  const cols = {};
  for (const c of COLS) {
    await parquetRead({ file: buf, metadata: md, columns: [c], compressors, onChunk: (ch) => ((cols[c] ??= []).push(ch)) });
  }
  let n = 0;
  // concatenate chunk data per column
  const flat = {};
  for (const c of COLS) {
    const parts = cols[c].sort((a, b) => a.rowStart - b.rowStart);
    flat[c] = parts.flatMap((p) => Array.from(p.columnData));
  }
  const u = flat.umi_exon;
  const ct = flat.ct;
  for (let i = 0; i < u.length; i++) if (u[i] >= 1000 && ct[i] === 101) n++;
  return { ms: performance.now() - t0, n };
}

function scanScx(path, quant) {
  const b = readFileSync(path);
  const buf = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  const t0 = performance.now();
  const h = parseScxHeader(buf);
  const idx = Object.fromEntries(h.names.map((n, i) => [n, i]));
  const T = quant ? qcode(1000) : 1000;
  const u = new Uint32Array(h.groupRows);
  const ct = new Uint32Array(h.groupRows);
  const s = new Uint32Array(h.groupRows);
  const c = new Uint32Array(h.groupRows);
  let n = 0;
  const outS = [];
  const outC = [];
  let skipped = 0;
  for (let g = 0; g < h.nGroups; g++) {
    const rows = h.groupRowsN[g];
    const chCt = h.chunks[g][idx.ct];
    const chU = h.chunks[g][idx.umi_exon];
    if (chCt.min > 101 || chCt.max < 101 || chU.max < T) {
      skipped++;
      continue;
    }
    const body = (ch) => buf.subarray(ch.off, ch.off + ch.len);
    decodeChunk(chU, rows, body(chU), u);
    decodeChunk(chCt, rows, body(chCt), ct);
    let any = false;
    for (let i = 0; i < rows; i++) if (u[i] >= T && ct[i] === 101) { any = true; break; }
    if (!any) continue;
    const chS = h.chunks[g][idx.sample];
    const chC = h.chunks[g][idx.col];
    decodeChunk(chS, rows, body(chS), s);
    decodeChunk(chC, rows, body(chC), c);
    for (let i = 0; i < rows; i++) {
      if (u[i] >= T && ct[i] === 101) {
        n++;
        outS.push(s[i]);
        outC.push(c[i]);
      }
    }
  }
  return { ms: performance.now() - t0, n, skipped };
}

const res = {};
for (const name of ["parquet-zstd", "parquet-snappy", "parquet-plain"]) {
  const p = `${dir}/fmt/${name}.parquet`;
  const a = [];
  for (let k = 0; k < 3; k++) a.push(await scanParquet(p));
  const b = [];
  for (let k = 0; k < 2; k++) b.push(await scanParquetColumnar(p));
  res[name] = { rowwise_ms: Math.min(...a.map((x) => x.ms)), columnar_ms: Math.min(...b.map((x) => x.ms)), matches: a[0].n };
}
for (const [name, q] of [["scx-exact", false], ["scx-quant", true]]) {
  const a = [];
  for (let k = 0; k < 5; k++) a.push(scanScx(`${dir}/fmt/${name}.scx`, q));
  res[name] = { ms: Math.min(...a.map((x) => x.ms)), matches: a[0].n, groups_skipped: a[0].skipped };
}
console.log(JSON.stringify(res, null, 1));
