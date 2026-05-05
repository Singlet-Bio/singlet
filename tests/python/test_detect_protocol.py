"""Tests for singlet.preprocessing._detect (protocol detection)."""

from singlet.preprocessing._detect import (
    ProtocolDetection,
    _infer_protocol,
    get_chemistry_string,
)

# ---------------------------------------------------------------------------
# ProtocolDetection dataclass
# ---------------------------------------------------------------------------


class TestProtocolDetection:
    def test_defaults(self):
        pd = ProtocolDetection(protocol="10xv3", mode="droplet", confidence="high", reason="test")
        assert pd.r1_len == 0
        assert pd.chemistry is None

    def test_all_fields(self):
        pd = ProtocolDetection(
            protocol="10xv3",
            mode="droplet",
            confidence="high",
            reason="R1=28bp",
            r1_len=28,
            r2_len=90,
            chemistry="SC3Pv3",
        )
        assert pd.protocol == "10xv3"
        assert pd.r2_len == 90


# ---------------------------------------------------------------------------
# _infer_protocol
# ---------------------------------------------------------------------------


class TestInferProtocol:
    def test_10xv3_classic(self):
        """R1=28bp + R2=90bp → 10xv3."""
        result = _infer_protocol(28, 90)
        assert result.protocol == "10xv3"
        assert result.mode == "droplet"
        assert result.confidence == "high"

    def test_10xv2_classic(self):
        """R1=26bp + R2=98bp → 10xv2."""
        result = _infer_protocol(26, 98)
        assert result.protocol == "10xv2"
        assert result.mode == "droplet"
        assert result.confidence == "high"

    def test_10xv3_from_24bp(self):
        """R1=24bp is lower boundary for droplet detection."""
        result = _infer_protocol(24, 91)
        assert result.protocol == "10xv2"  # <28 → v2
        assert result.mode == "droplet"

    def test_swapped_orientation(self):
        """R1=98bp, R2=28bp → 10xv3 with medium confidence."""
        result = _infer_protocol(98, 28)
        assert result.protocol == "10xv3"
        assert result.confidence == "medium"
        assert "Swapped" in result.reason

    def test_both_long_ambiguous(self):
        """R1=150bp, R2=150bp → ambiguous."""
        result = _infer_protocol(150, 150)
        assert result.protocol == "ambiguous"
        assert result.mode == "unknown"
        assert result.confidence == "low"

    def test_very_short_unknown(self):
        """R1=10bp, R2=10bp → unknown."""
        result = _infer_protocol(10, 10)
        assert result.protocol == "unknown"
        assert result.confidence == "low"

    def test_catalog_hint_overrides(self):
        """Catalog hint takes priority over read lengths."""
        result = _infer_protocol(150, 150, catalog_hint="10xv3 Chromium")
        assert result.protocol == "10xv3"
        assert result.confidence == "high"
        assert "Catalog hint" in result.reason

    def test_catalog_hint_dropseq(self):
        result = _infer_protocol(20, 50, catalog_hint="Drop-seq (dropseq)")
        assert result.protocol == "dropseq"
        assert result.mode == "droplet"

    def test_catalog_hint_smartseq(self):
        result = _infer_protocol(150, 150, catalog_hint="Smart-seq2 (smartseq)")
        assert result.protocol == "smartseq"
        assert result.mode == "smartseq"

    def test_result_contains_read_lengths(self):
        result = _infer_protocol(28, 91)
        assert result.r1_len == 28
        assert result.r2_len == 91


# ---------------------------------------------------------------------------
# get_chemistry_string
# ---------------------------------------------------------------------------


class TestGetChemistryString:
    def test_10xv3(self):
        chem = get_chemistry_string("10xv3")
        assert chem is not None
        assert isinstance(chem, str)

    def test_10xv2(self):
        chem = get_chemistry_string("10xv2")
        assert chem is not None

    def test_unknown_returns_none(self):
        chem = get_chemistry_string("alien_protocol")
        assert chem is None

    def test_dropseq(self):
        chem = get_chemistry_string("dropseq")
        assert chem is not None
