// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_device_glue.h, included by allsite.h before run()
// run()'s device hook (Params::device): the reads of each contig gathered into flat arrays, the device route called, and its sites and
// (barcode, site) counts written where run()'s contig workers write theirs. One record per (barcode, site) holds the summed DP (or AD)
// molecules; Result::prepare sums records per (barcode, site) in any case, so the matrices, the per-site sums and the stored member are
// the ones the per-molecule records give. dp_mol and ad_mol return the molecule counts that run() reports as DP and AD records.
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
                        const std::vector<uint32_t>& order, Out& out, uint64_t& obs, uint64_t& dp_mol, uint64_t& ad_mol, uint64_t t_run0) {
    using dev::Read;
    const auto t0 = std::chrono::steady_clock::now();
    dev::Input in;
    in.min_alt_molecules = P.min_alt_molecules; in.min_alt_cells = P.min_alt_cells; in.use_qual = P.use_qual; in.skip_n = P.skip_n;
    in.directional = P.directional; in.umi_len = P.umi_len; in.n_barcodes = n_barcodes; in.progress = P.progress;
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
    in.gather = [&](uint32_t c0, uint32_t c1, Read* rd, uint32_t* ops, uint8_t* seq, uint64_t* lq) {
        // the reads of contigs [c0, c1) in slices, one per thread: sizes, then offsets, then the copy
        std::vector<uint64_t> list_off{0};
        for (uint32_t c = c0; c < c1; ++c) list_off.push_back(list_off.back() + in.contigs[c].reads);
        const uint64_t n = list_off.back();
        auto read_at = [&](uint64_t i, uint32_t& c) {   // i-th read of the chunk
            c = static_cast<uint32_t>(std::upper_bound(list_off.begin(), list_off.end(), i) - list_off.begin()) - 1;
            return by_tid[cnt[static_cast<uint32_t>(in.contigs[c0 + c].tid)] + (i - list_off[c])];
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
                    d.pos = al.pos; d.bc = x.bc; d.umi3 = x.umi3; d.lq = o.lq; d.ops = static_cast<uint32_t>(o.ops); d.seq = static_cast<uint32_t>(o.seq);
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
    dev::Output o;
    if (!P.device(in, o)) {
        std::cerr << "[allsite-gpu] host pass: " << o.why << "\n";
        return false;
    }
    obs = o.obs; dp_mol = o.molecules; ad_mol = o.ad_molecules;
    for (size_t k = 0; k < ctid.size(); ++k) {
        auto& pt = out[ctid[k]]; dev::ContigOut& co = o.contigs[k];
        const int32_t t = static_cast<int32_t>(ctid[k]);
        pt.sites.resize(co.pos.size());
        for (size_t i = 0; i < co.pos.size(); ++i)
            pt.sites[i] = Site{t, co.pos[i], "ACGT"[co.ref[i] & 3], "ACGT"[co.alt[i] & 3], co.mol[i], co.cells[i]};
        pt.dp.swap(co.dp); pt.ad.swap(co.ad);   // Tri is dev::Count
    }
    auto& st = out[ctid[0]].st;   // run() sums the statistics over contigs
    st.snp_molecules += o.molecules; st.snp_molecules_ambiguous += o.ambiguous; st.snp_molecules_other += o.other;
    const double s_sel = 1e-9 * static_cast<double>(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count()) - t_run0);
    std::cerr << "[allsite-gpu] selection " << s_sel
              << " s, sizes " << s_sizes << " s; " << o.chunks << " chunks, " << o.blocks << " blocks; hits " << o.hits << ", entries " << o.entries << ", groups " << o.groups
              << "; device arena " << (o.arena_bytes >> 20) << " MiB; s: gather " << o.sec_gather << ", upload " << o.sec_upload << ", discovery " << o.sec_disc
              << ", counting " << o.sec_count << " (hits " << o.sec_hits << ", sort " << o.sec_sort << ", entries " << o.sec_entries << ", R9 and vote " << o.sec_r9
              << ", out " << o.sec_out << "), device total " << o.sec << ", hook total "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
    return true;
}

}}}  // namespace singlet::allsite::detail
