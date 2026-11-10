// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_device_glue.h, included by allsite.h before run_sorted()
// run_sorted()'s device hook (Params::device): the reads of each contig gathered into flat arrays, the device route called, its sites
// written where the host kernel writes them and its (barcode, site) counts sent block by block into the spill (Spill units, put in block
// order and then renumbered into row order), so host memory holds one block of counts at a time whatever the record count.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "allsite_device.h"

namespace singlet { namespace allsite { namespace detail {

template <class Rec, class Out>
inline bool device_pass(const Rec& R, const Genome& G, const Params& P, uint32_t n_tid, uint32_t n_barcodes,
                        const std::vector<uint8_t>& sig_p, const std::vector<uint64_t>& cnt, const std::vector<uint64_t>& by_tid,
                        const std::vector<uint32_t>& order, Out& out, Spill& spl, uint64_t& below_floor, uint64_t t_run0) {
    using dev::Read;
    const auto t0 = std::chrono::steady_clock::now();
    dev::Input in;
    in.min_alt_molecules = P.min_alt_molecules; in.min_alt_cells = P.min_alt_cells; in.use_qual = P.use_qual; in.skip_n = P.skip_n;
    in.directional = P.directional; in.umi_len = P.umi_len; in.n_barcodes = n_barcodes; in.threads = std::max(1u, P.threads); in.progress = P.progress;
    std::vector<uint32_t> ctid;
    for (uint32_t oi = 0; oi < n_tid; ++oi) { const uint32_t t = order[oi]; if (cnt[t + 1] > cnt[t]) ctid.push_back(t); }
    if (ctid.empty()) return false;
    in.contigs.resize(ctid.size());
    const unsigned nth = std::max(1u, P.threads);
    auto nlq_of = [&](uint16_t rlen) { return P.use_qual ? static_cast<uint64_t>((rlen + 63) / 64) : 0ull; };
    {   // per contig: the sizes of its gathered arrays, from equal slices of the selected reads (per-thread sums per contig)
        const uint64_t nsel = cnt[n_tid];
        std::vector<std::vector<uint64_t>> part(nth, std::vector<uint64_t>(3 * static_cast<size_t>(n_tid), 0));
        auto work = [&](unsigned s) {
            const uint64_t a = nsel * s / nth, b = nsel * (s + 1) / nth;
            if (a >= b) return;
            uint32_t t = static_cast<uint32_t>(std::upper_bound(cnt.begin(), cnt.end(), a) - cnt.begin()) - 1;
            std::vector<uint64_t>& v = part[s];
            for (uint64_t j = a; j < b; ++j) {
                while (j >= cnt[t + 1]) ++t;
                const uint64_t r = by_tid[j]; const auto& x = R.read(r); const auto& sg = R.sig(x.sig);
                v[3 * t] += R.aln(sg.aln_off + sig_p[x.sig]).n_ops; v[3 * t + 1] += (sg.rlen + 1u) / 2u; v[3 * t + 2] += nlq_of(sg.rlen);
            }
        };
        std::vector<std::thread> th; for (unsigned i = 1; i < nth; ++i) th.emplace_back(work, i);
        work(0); for (auto& x : th) x.join();
        for (size_t k = 0; k < ctid.size(); ++k) {
            const uint32_t t = ctid[k]; dev::Contig& c = in.contigs[k];
            c.tid = static_cast<int32_t>(t); c.reads = cnt[t + 1] - cnt[t];
            for (unsigned i = 0; i < nth; ++i) { c.ops += part[i][3 * t]; c.seqb += part[i][3 * t + 1]; c.lqw += part[i][3 * t + 2]; }
            c.ref = G.contig(static_cast<int32_t>(t), &c.ref_len);
        }
    }
    const double s_sizes = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // entries of in.contigs: whole contigs, or after Input::resplit windows of a contig in position order. ek: the contig (index into
    // ctid) of each entry; elist: a window's reads as indices into by_tid, in by_tid order (empty for a whole contig)
    std::vector<uint32_t> ek(ctid.size()); for (size_t k = 0; k < ek.size(); ++k) ek[k] = static_cast<uint32_t>(k);
    std::vector<std::vector<uint64_t>> elist(ctid.size());
    uint32_t n_split = 0;
    auto ref_end = [&](const auto& al) {   // one past the last reference position the alignment covers
        const uint32_t* op = R.ops_at(al.ops_off); int64_t e = al.pos;
        for (uint32_t q = 0; q < al.n_ops; ++q) { const uint32_t o = op[q] & 15u; if (o == 0 || o == 2 || o == 3 || o == 7 || o == 8) e += op[q] >> 4; }
        return e;
    };
    in.resplit = [&](uint64_t max_bytes, uint64_t max_items) -> bool {
        const std::vector<dev::Contig> base = in.contigs;
        std::vector<dev::Contig> nc; std::vector<uint32_t> nk; std::vector<std::vector<uint64_t>> nl;
        const uint64_t per_read = sizeof(Read) + 20;
        auto par = [&](auto&& f) { std::vector<std::thread> th; for (unsigned s = 1; s < nth; ++s) th.emplace_back(f, s); f(0u); for (auto& x : th) x.join(); };
        for (size_t k = 0; k < base.size(); ++k) {
            const dev::Contig& c = base[k];
            if (dev::contig_bytes(c) <= max_bytes && c.reads < max_items) {
                nc.push_back(c); nk.push_back(static_cast<uint32_t>(k)); nl.emplace_back(); continue;
            }
            // windows of 1 kb bins by read start, each with about half of the budget left after the reference in its own reads (reads
            // overlapping its start take the rest); then every read goes to each window it overlaps, in by_tid order
            const uint32_t t = ctid[k]; const uint64_t a = cnt[t], n = cnt[t + 1] - a;
            const int sh = 10; const uint64_t nb = (c.ref_len >> sh) + 2;
            std::vector<int32_t> rs(n), re(n);
            std::vector<std::vector<uint64_t>> hist(nth, std::vector<uint64_t>(nb, 0));
            auto bin = [&](int64_t p) { return static_cast<uint64_t>(std::min<int64_t>(std::max<int64_t>(p, 0) >> sh, static_cast<int64_t>(nb - 1))); };
            par([&](unsigned s) {
                for (uint64_t i = n * s / nth; i < n * (s + 1) / nth; ++i) {
                    const auto& x = R.read(by_tid[a + i]); const auto& sg = R.sig(x.sig); const auto& al = R.aln(sg.aln_off + sig_p[x.sig]);
                    rs[i] = al.pos; re[i] = static_cast<int32_t>(std::max<int64_t>(ref_end(al), static_cast<int64_t>(al.pos) + 1));
                    hist[s][bin(al.pos)] += per_read + 4ull * al.n_ops + (sg.rlen + 1u) / 2u + 8ull * nlq_of(sg.rlen);
                }
            });
            const uint64_t fix = c.ref_len + 4096;
            const uint64_t target = max_bytes > fix ? (max_bytes - fix) / 2 : 1;
            std::vector<int32_t> bnd;   // window w owns [bnd[w-1], bnd[w]), open at both ends
            { uint64_t acc = 0;
              for (uint64_t b = 0; b < nb; ++b) {
                  uint64_t h = 0; for (unsigned s = 0; s < nth; ++s) h += hist[s][b];
                  if (h == 0) continue;
                  if (acc > 0 && acc + h > target) { bnd.push_back(static_cast<int32_t>(b << sh)); acc = 0; }
                  acc += h;
              } }
            auto fail = [&](const char* what, uint64_t wr, uint64_t wb) {   // sizes only
                uint64_t hb = 0; for (uint64_t b = 0; b < nb; ++b) { uint64_t h = 0; for (unsigned s = 0; s < nth; ++s) h += hist[s][b]; hb = std::max(hb, h); }
                std::cerr << "[allsite-gpu] window split failed (" << what << "): contig " << t << " length " << c.ref_len << ", " << c.reads << " reads, "
                          << (dev::contig_bytes(c) >> 20) << " MiB; largest 1 kb bin " << (hb >> 20) << " MiB; " << bnd.size() + 1 << " windows; window "
                          << wr << " reads, " << (wb >> 20) << " MiB; budget " << (max_bytes >> 20) << " MiB" << std::endl;
                return false;
            };
            if (bnd.empty()) return fail("one bin", c.reads, dev::contig_bytes(c));
            const size_t nw = bnd.size() + 1;
            auto win = [&](int64_t p) { return static_cast<size_t>(std::upper_bound(bnd.begin(), bnd.end(), p) - bnd.begin()); };
            // reads per (slice, window), then each slice's offsets, then the lists and sizes
            std::vector<std::vector<uint64_t>> wc(nth, std::vector<uint64_t>(nw, 0));
            par([&](unsigned s) {
                for (uint64_t i = n * s / nth; i < n * (s + 1) / nth; ++i)
                    for (size_t w = win(rs[i]), w1 = win(static_cast<int64_t>(re[i]) - 1); w <= w1; ++w) ++wc[s][w];
            });
            std::vector<dev::Contig> wcs(nw, c); std::vector<std::vector<uint64_t>> wl(nw);
            for (size_t w = 0; w < nw; ++w) {
                uint64_t tot = 0; for (unsigned s = 0; s < nth; ++s) { const uint64_t v = wc[s][w]; wc[s][w] = tot; tot += v; }
                wl[w].resize(tot);
                wcs[w].reads = tot; wcs[w].ops = 0; wcs[w].seqb = 0; wcs[w].lqw = 0;
                wcs[w].lo = w == 0 ? INT32_MIN : bnd[w - 1]; wcs[w].hi = w + 1 == nw ? INT32_MAX : bnd[w];
            }
            std::vector<std::vector<uint64_t>> wsz(nth, std::vector<uint64_t>(3 * nw, 0));
            par([&](unsigned s) {
                std::vector<uint64_t> at = wc[s];
                for (uint64_t i = n * s / nth; i < n * (s + 1) / nth; ++i) {
                    const auto& x = R.read(by_tid[a + i]); const auto& sg = R.sig(x.sig); const auto& al = R.aln(sg.aln_off + sig_p[x.sig]);
                    for (size_t w = win(rs[i]), w1 = win(static_cast<int64_t>(re[i]) - 1); w <= w1; ++w) {
                        wl[w][at[w]++] = a + i;
                        wsz[s][3 * w] += al.n_ops; wsz[s][3 * w + 1] += (sg.rlen + 1u) / 2u; wsz[s][3 * w + 2] += nlq_of(sg.rlen);
                    }
                }
            });
            for (size_t w = 0; w < nw; ++w) {
                for (unsigned s = 0; s < nth; ++s) { wcs[w].ops += wsz[s][3 * w]; wcs[w].seqb += wsz[s][3 * w + 1]; wcs[w].lqw += wsz[s][3 * w + 2]; }
                const dev::Contig& v = wcs[w];
                if (v.reads == 0 || dev::contig_bytes(v) > max_bytes || v.reads >= max_items) return fail("window", v.reads, dev::contig_bytes(v));
                nc.push_back(v); nk.push_back(static_cast<uint32_t>(k)); nl.push_back(std::move(wl[w]));
            }
            ++n_split;
        }
        in.contigs.swap(nc); ek.swap(nk); elist.swap(nl);
        return true;
    };
    in.gather = [&](uint32_t c0, uint32_t c1, Read* rd, uint32_t* ops, uint8_t* seq, uint64_t* lq) {
        // the reads of contigs [c0, c1) in slices, one per thread: sizes, then offsets, then the copy
        std::vector<uint64_t> list_off{0};
        for (uint32_t c = c0; c < c1; ++c) list_off.push_back(list_off.back() + in.contigs[c].reads);
        const uint64_t n = list_off.back();
        auto read_at = [&](uint64_t i, uint32_t& c) {   // i-th read of the chunk
            c = static_cast<uint32_t>(std::upper_bound(list_off.begin(), list_off.end(), i) - list_off.begin()) - 1;
            const std::vector<uint64_t>& wl = elist[c0 + c];
            return wl.empty() ? by_tid[cnt[static_cast<uint32_t>(in.contigs[c0 + c].tid)] + (i - list_off[c])] : by_tid[wl[i - list_off[c]]];
        };
        const unsigned ns = static_cast<unsigned>(std::max<uint64_t>(1, std::min<uint64_t>(nth, n / 4096 + 1)));
        struct Sz { uint64_t ops = 0, seq = 0, lq = 0; };
        std::vector<Sz> sz(ns + 1);
        auto slice = [&](unsigned s, uint64_t& a, uint64_t& b) { a = n * s / ns; b = n * (s + 1) / ns; };
        auto pass = [&](unsigned s, bool write) {
            uint64_t a, b; slice(s, a, b);
            Sz o = write ? sz[s] : Sz{};
            for (uint64_t i = a; i < b; ++i) {
                uint32_t c; const uint64_t r = read_at(i, c);
                const auto& x = R.read(r); const auto& sg = R.sig(x.sig); const auto& al = R.aln(sg.aln_off + sig_p[x.sig]);
                const uint64_t sb = (sg.rlen + 1u) / 2u, nl = nlq_of(sg.rlen);
                if (write) {
                    Read& d = rd[i];
                    d.pos = al.pos; d.bc = x.bc; d.umi3 = x.umi3; d.lq = o.lq; d.ops = o.ops; d.seq = o.seq;
                    d.rlen = sg.rlen; d.nops = al.n_ops; d.ci = static_cast<uint16_t>(c); d.rev = (al.flags & 1) ? 1 : 0; d.nlq = static_cast<uint8_t>(nl);
                    const uint32_t* op = R.ops_at(al.ops_off); for (uint32_t k = 0; k < al.n_ops; ++k) ops[o.ops + k] = op[k];
                    std::memcpy(seq + o.seq, R.seq_at(sg.seq_off), sb);
                    for (uint64_t k = 0; k < nl; ++k) lq[o.lq + k] = R.lowq_word(static_cast<uint64_t>(P.qwords) * r + k);
                }
                o.ops += al.n_ops; o.seq += sb; o.lq += nl;
            }
            if (!write) sz[s + 1] = o;
        };
        auto run_all = [&](bool write) {
            std::vector<std::thread> th; for (unsigned s = 1; s < ns; ++s) th.emplace_back(pass, s, write);
            pass(0, write); for (auto& x : th) x.join();
        };
        run_all(false);
        for (unsigned s = 0; s < ns; ++s) { sz[s + 1].ops += sz[s].ops; sz[s + 1].seq += sz[s].seq; sz[s + 1].lq += sz[s].lq; }
        run_all(true);
    };
    // the ALT-fraction floor: the host kernel's rule (rec::floor_keep), applied by the device route at the end of each block
    in.keep_site = [&P](uint64_t ad, uint64_t dp) { return rec::floor_keep(ad, dp, P.min_alt_pct); };
    const uint8_t keep_unseen = rec::floor_keep(0, 0, P.min_alt_pct) ? 1 : 0;   // sites of blocks without hits have depth 0
    std::vector<std::pair<uint32_t, uint32_t>> ukey;   // per spill unit: (entry of in.contigs, first site within the entry)
    std::vector<std::vector<uint8_t>> kmask;   // per entry: which sites stayed
    in.sink = [&](uint32_t c, uint32_t s0, uint32_t ns, const uint8_t* keep, uint32_t nkeep, std::string& b) {
        if (kmask.size() < in.contigs.size()) kmask.resize(in.contigs.size());
        std::vector<uint8_t>& m = kmask[c];
        if (m.size() < static_cast<size_t>(s0) + ns) m.resize(static_cast<size_t>(s0) + ns, keep_unseen);
        std::memcpy(m.data() + s0, keep, ns);
        spl.u.emplace_back(); ukey.push_back({c, s0}); spl.put(spl.u.size() - 1, 0, nkeep, b);
    };
    dev::Output o;
    if (!P.device(in, o)) {
        std::cerr << "[allsite-gpu] host pass: " << o.why << "\n";
        return false;
    }
    below_floor = 0;
    const size_t nE = in.contigs.size();
    kmask.resize(nE);
    std::vector<std::vector<uint32_t>> kbefore(nE);   // per entry: kept sites before each site
    std::vector<uint32_t> ebefore(nE, 0);   // per entry: kept sites of the same contig in its earlier windows
    for (size_t e = 0; e < nE; ++e) {
        const uint32_t k = ek[e];
        auto& pt = out[ctid[k]]; dev::ContigOut& co = o.contigs[e];
        const int32_t t = static_cast<int32_t>(ctid[k]);
        const size_t ns = co.pos.size();
        std::vector<uint8_t>& m = kmask[e]; m.resize(ns, keep_unseen);
        std::vector<uint32_t>& kb = kbefore[e]; kb.assign(ns + 1, 0);
        if (e == 0 || ek[e - 1] != k) pt.sites.clear();
        const size_t had = pt.sites.size();
        ebefore[e] = static_cast<uint32_t>(had); pt.sites.reserve(had + ns);
        for (size_t i = 0; i < ns; ++i) {
            kb[i + 1] = kb[i] + m[i];
            if (m[i]) pt.sites.push_back(Site{t, co.pos[i], "ACGT"[co.ref[i] & 3], "ACGT"[co.alt[i] & 3], co.mol[i], co.cells[i]});
        }
        below_floor += ns - (pt.sites.size() - had);
    }
    {   // the units' rows (contigs in index order, sites by position), then the units in row order as the readers expect
        std::vector<uint32_t> grow0(n_tid + 1, 0);
        for (uint32_t t = 0; t < n_tid; ++t) grow0[t + 1] = grow0[t] + static_cast<uint32_t>(out[t].sites.size());
        for (size_t i = 0; i < spl.u.size(); ++i) spl.u[i].row0 = grow0[ctid[ek[ukey[i].first]]] + ebefore[ukey[i].first] + kbefore[ukey[i].first][ukey[i].second];
        std::vector<uint32_t> perm(spl.u.size()); for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<uint32_t>(i);
        std::sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return spl.u[a].row0 < spl.u[b].row0; });
        std::vector<Spill::U> su(spl.u.size()); for (size_t i = 0; i < perm.size(); ++i) su[i] = std::move(spl.u[perm[i]]);
        spl.u.swap(su);
    }
    auto& p0 = out[ctid[0]];   // finish_spill sums the statistics over contigs; DP and AD records are the summed molecules, as kernel 1's
    p0.st.snp_molecules += o.molecules; p0.st.snp_molecules_ambiguous += o.ambiguous; p0.st.snp_molecules_other += o.other;
    p0.obs += o.obs; p0.dp_rec += o.kept_dp; p0.ad_rec += o.kept_ad;   // the kept sites' molecules, as the host kernel counts them
    const double s_sel = 1e-9 * static_cast<double>(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count()) - t_run0);
    std::cerr << "[allsite-gpu] selection " << s_sel
              << " s, sizes " << s_sizes << " s; " << o.chunks << " chunks, " << o.blocks << " blocks; hits " << o.hits << ", entries " << o.entries << ", groups " << o.groups
              << "; device arena " << (o.arena_bytes >> 20) << " MiB (peak " << (o.arena_peak >> 20) << " MiB), " << n_split << " contigs in " << in.contigs.size() << " entries; s: gather " << o.sec_gather << ", upload " << o.sec_upload << ", discovery " << o.sec_disc
              << ", counting " << o.sec_count << " (hits " << o.sec_hits << ", sort " << o.sec_sort << ", entries " << o.sec_entries << ", R9 and vote " << o.sec_r9
              << ", out " << o.sec_out << "); spill " << (spl.bytes >> 20) << " MiB in " << spl.where << ", " << spl.u.size() << " units; device total " << o.sec << ", hook total "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return true;
}

}}}  // namespace singlet::allsite::detail
