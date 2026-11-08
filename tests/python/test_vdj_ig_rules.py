"""IG rules of the contig filter: a B cell of a plasma clone keeps its light chain; a lone chain spread over many droplets is ambient."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from singlet.vdj import assemble as A  # noqa: E402


def _row(bc, chain, cdr3, umis):
    return dict(bcix=bc, chain=chain, cdr3_nt=cdr3, umis=umis, reads=umis * 50, high_confidence=True, productive=True, junction_umis=umis)


def _clone_df():
    rows = [_row(0, "IGK", "TGTCAGCAGTATAAT", 400), _row(0, "IGH", "TGTGCGAGAGATTAC", 900)]  # the plasma cell
    rows += [_row(1, "IGK", "TGTCAGCAGTATAAT", 6), _row(1, "IGH", "TGTGCGAGAGGGTTT", 30)]  # a B cell of the same light chain
    return pd.DataFrame(rows)


def _kept(df, bc, chain):
    return bool(df[(df.bcix == bc) & (df.chain == chain)].kept.any())


def test_light_chain_of_a_b_cell_in_a_plasma_clone_is_kept():
    df = A.filter_contigs(_clone_df(), A.Params())
    assert _kept(df, 1, "IGK") and _kept(df, 1, "IGH")


def test_rule_off_reproduces_the_cross_filter():
    df = A.filter_contigs(_clone_df(), A.Params(ig_top_min_umis=0))
    assert not _kept(df, 1, "IGK")


def test_lone_chain_in_many_droplets_is_ambient():
    rows = [_row(0, "IGK", "TGTCAGCAGTATAAT", 400)]
    rows += [_row(b, "IGK", "TGTCAGCAGTATAAT", 2) for b in range(1, 8)]  # a light chain alone in 7 other droplets
    df = A.filter_contigs(pd.DataFrame(rows), A.Params())
    assert not df[df.bcix >= 1].kept.any()
    df = A.filter_contigs(pd.DataFrame(rows), A.Params(ig_lone_spread=0, ig_top_min_umis=0, ambient_rescue="none"))
    assert df.kept.sum() >= 1


def test_weaker_kept_light_chain_does_not_block_the_cells_own():
    rows = [_row(0, "IGK", "TGTCAGCAGTATAAT", 400), _row(0, "IGH", "TGTGCGAGAGATTAC", 900)]  # plasma cell
    rows += [_row(1, "IGK", "TGTCAGCAGTATAAT", 6), _row(1, "IGH", "TGTGCGAGAGGGTTT", 30),
             _row(1, "IGL", "TGCAGCTCATATACA", 3)]  # B cell: its own IGK (cross-filtered) and a stray IGL
    df = A.filter_contigs(pd.DataFrame(rows), A.Params())
    assert not _kept(df, 1, "IGK")
    df = A.filter_contigs(pd.DataFrame(rows), A.Params(ig_top_weaker_kept=True))
    assert _kept(df, 1, "IGK")


def test_pair_rescue_gives_back_both_chains_of_a_clone_member():
    rows = [_row(0, "IGK", "TGTCAGCAGTATAAT", 400), _row(0, "IGH", "TGTGCGAGAGATTAC", 900)]  # plasma cell
    rows += [_row(1, "IGK", "TGTCAGCAGTATAAT", 8), _row(1, "IGH", "TGTGCGAGAGATTAC", 20)]  # B cell of the same clone
    rows += [_row(2, "IGK", "TGTCAGCAGTATAAT", 1), _row(2, "IGH", "TGTGCGAGAGATTAC", 1)]  # ambient: 1 UMI each
    base = A.filter_contigs(pd.DataFrame(rows), A.Params(ambient_rescue="none"))
    assert not _kept(base, 1, "IGH") and not _kept(base, 1, "IGK")
    df = A.filter_contigs(pd.DataFrame(rows), A.Params(ambient_rescue="none", ig_pair_rescue=True))
    assert _kept(df, 1, "IGH") and _kept(df, 1, "IGK")
    assert not _kept(df, 2, "IGH") and not _kept(df, 2, "IGK")
