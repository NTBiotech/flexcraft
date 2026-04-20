"""
Tests for the ADAPT TCR design pipeline.

Covers:
  - clean_chothia (pure file I/O)
  - ADAPT class structure and configuration
  - ADAPT.cdr_mask     (pure numpy logic)
  - ADAPT.insert_cdr   (DesignData manipulation)
  - ADAPT.number_anarci (with mocked anarci)
  - ADAPT.design_step  (with mocked ProteinMPNN sampler)
  - ADAPT.design_trial (integration smoke test, all ML mocked)

Run from the flexcraft/ directory:
    conda run -n flexcraft_2 python -m pytest tests/test_adapt_design.py -v --tb=short
"""

import os
import tempfile
import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from unittest.mock import MagicMock, patch, patch as mock_patch

# ── flexcraft imports (safe: no model weights loaded at import time) ─────────
from flexcraft.data.data import DesignData
from flexcraft.structure.metrics.rmsd import RMSD
from flexcraft.utils.rng import Keygen
import flexcraft.sequence.aa_codes as aas
from flexcraft.pipelines.tcr.adapt.adapt import ADAPT, clean_chothia

# Obtain the adapt *module* object for patch.object() calls.
# `import flexcraft.pipelines.tcr.adapt.adapt as alias` is syntactic sugar for
# `from flexcraft.pipelines.tcr.adapt import adapt as alias`, which fails because
# the __init__.py re-exports symbols with `from .adapt import *` but does not
# expose the sub-module itself.  The workaround is to read from sys.modules
# after the from-import above has already loaded the module.
import sys as _sys
_adapt_module = _sys.modules["flexcraft.pipelines.tcr.adapt.adapt"]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

IMGT_MAPPER = {
    "lcdr1": (27, 38), "lcdr2": (56, 65), "lcdr3": (105, 117),
    "hcdr1": (27, 38), "hcdr2": (56, 65), "hcdr3": (105, 117),
    "acdr1": (27, 38), "acdr2": (56, 65), "acdr3": (105, 117),
    "bcdr1": (27, 38), "bcdr2": (56, 65), "bcdr3": (105, 117),
}


def make_imgt_chain(chain_index: int, length: int = 130) -> DesignData:
    """
    Create a DesignData with consecutive IMGT-style residue numbering 1..length
    on the given chain_index.  Sequence is all-alanine (code 0 in AF2_CODE).
    """
    data = DesignData.from_length(length)
    data = data.update(
        chain_index=jnp.full(length, chain_index, dtype=jnp.int32),
        residue_index=jnp.arange(1, length + 1, dtype=jnp.int32),
        aa=jnp.zeros(length, dtype=jnp.int32),  # all-A
    )
    return data


def adapt_stub() -> ADAPT:
    """
    Build an ADAPT instance that bypasses __init__ entirely.
    All ML components are replaced with MagicMocks.
    """
    inst = object.__new__(ADAPT)
    inst.ab = False
    inst.imgt_mapper = IMGT_MAPPER.copy()
    inst.mhc_chain_index = np.array([0])
    inst.tcr_chain_index = np.array([1, 2])
    inst.key = Keygen(42)
    inst.pmpnn = MagicMock()
    inst.center_logits = False
    inst.pmpnn_hparams = {"temperature": 0.1, "center_logits": False}
    inst.pmpnn_sampler = MagicMock()
    inst.af2_model = MagicMock()
    inst.af2_params = MagicMock()
    inst.use_multimer = False
    inst.num_recycle = 0
    inst.af2_config = MagicMock()
    inst.rmsd = RMSD()
    inst.in_dir = Path(tempfile.mkdtemp())
    inst.out_dir = Path(tempfile.mkdtemp())
    inst.op_dir = inst.in_dir.parent
    import pandas as pd
    columns = ["out_file", "score", "TCR", "pMHC",
               "acdr1", "bcdr1", "acdr2", "bcdr2", "acdr3", "bcdr3"]
    inst.scores = pd.DataFrame(columns=columns)
    return inst


# ────────────────────────────────────────────────────────────────────────────
# clean_chothia
# ────────────────────────────────────────────────────────────────────────────

class TestCleanChothia:
    """Tests for the clean_chothia PDB-cleaning utility."""

    def _write_pdb(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "test.pdb"
        p.write_text(content)
        return p

    def test_hetatm_lines_are_removed(self, tmp_path):
        pdb = self._write_pdb(tmp_path, (
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00\n"
            "HETATM    2  C   LIG A 100       4.000   5.000   6.000  1.00  0.00\n"
        ))
        out = clean_chothia(pdb)
        lines = out.read_text().splitlines()
        assert not any(l.startswith("HETATM") for l in lines), "HETATM should be removed"

    def test_atom_lines_are_kept(self, tmp_path):
        pdb = self._write_pdb(tmp_path, (
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00\n"
            "HETATM    2  C   LIG A 100       4.000   5.000   6.000  1.00  0.00\n"
        ))
        out = clean_chothia(pdb)
        lines = out.read_text().splitlines()
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        assert len(atom_lines) == 1

    def test_non_atom_non_hetatm_lines_kept(self, tmp_path):
        pdb = self._write_pdb(tmp_path, "REMARK  test remark\nEND\n")
        out = clean_chothia(pdb)
        text = out.read_text()
        assert "REMARK  test remark" in text
        assert "END" in text

    def test_insertion_code_column_is_blanked(self, tmp_path):
        # Column index 26 (0-based) is the insertion code; clean_chothia removes it
        # by writing l[:26] + " " + l[27:] — replacing whatever was at position 26.
        atom_line = "ATOM      1  N   ALA A   1B      1.000   2.000   3.000\n"
        pdb = self._write_pdb(tmp_path, atom_line)
        out = clean_chothia(pdb)
        cleaned = out.read_text()
        # Original char at col 26 was 'B'; after cleaning it should be ' '
        assert cleaned[26] == " ", f"Column 26 should be blank, got {repr(cleaned[26])}"

    def test_already_clean_file_returned_unchanged(self, tmp_path):
        p = tmp_path / "struct_clean.pdb"
        p.write_text("ATOM      1  N   ALA A   1       1.000   2.000   3.000\n")
        out = clean_chothia(p)
        assert out == p, "Already-clean file should be returned as-is without re-processing"

    def test_returns_path_with_clean_suffix(self, tmp_path):
        pdb = self._write_pdb(tmp_path, "ATOM      1  N   ALA A   1\n")
        out = clean_chothia(pdb)
        assert out.name.endswith("_clean.pdb")

    def test_accepts_string_path(self, tmp_path):
        pdb = self._write_pdb(tmp_path, "ATOM      1  N   ALA A   1\n")
        out = clean_chothia(str(pdb))  # str, not Path
        assert out.exists()


# ────────────────────────────────────────────────────────────────────────────
# ADAPT class structure
# ────────────────────────────────────────────────────────────────────────────

class TestAdaptStructure:
    """Verify ADAPT class has all expected attributes and methods."""

    def test_adapt_class_importable(self):
        assert ADAPT is not None

    def test_adapt_has_design_trial(self):
        assert callable(getattr(ADAPT, "design_trial", None))

    def test_adapt_has_insert_cdr(self):
        assert callable(getattr(ADAPT, "insert_cdr", None))

    def test_adapt_has_cdr_mask(self):
        assert callable(getattr(ADAPT, "cdr_mask", None))

    def test_adapt_has_number_anarci(self):
        assert callable(getattr(ADAPT, "number_anarci", None))

    def test_adapt_has_design_step(self):
        assert callable(getattr(ADAPT, "design_step", None))

    def test_adapt_has_docking_step(self):
        assert callable(getattr(ADAPT, "docking_step", None))

    def test_adapt_has_evaluate_step(self):
        assert callable(getattr(ADAPT, "evaluate_step", None))

    def test_adapt_imgt_mapper_has_all_cdr_keys(self):
        adapt = adapt_stub()
        expected = {f"{chain}cdr{n}" for chain in ("a", "b", "l", "h") for n in (1, 2, 3)}
        assert expected == set(adapt.imgt_mapper.keys())

    def test_adapt_imgt_acdr3_range(self):
        adapt = adapt_stub()
        start, end = adapt.imgt_mapper["acdr3"]
        assert start == 105
        assert end == 117

    def test_adapt_imgt_acdr1_range(self):
        adapt = adapt_stub()
        start, end = adapt.imgt_mapper["acdr1"]
        assert start == 27
        assert end == 38

    def test_adapt_ab_false_has_alpha_beta_columns(self, tmp_path):
        """When ab=False, scores DataFrame should have acdrN and bcdrN columns."""
        import pandas as pd
        adapt = adapt_stub()
        assert "acdr1" in adapt.scores.columns
        assert "acdr3" in adapt.scores.columns
        assert "bcdr1" in adapt.scores.columns
        assert "bcdr3" in adapt.scores.columns

    def test_adapt_mhc_chain_index_is_array(self):
        adapt = adapt_stub()
        assert isinstance(adapt.mhc_chain_index, np.ndarray)

    def test_adapt_tcr_chain_index_is_array(self):
        adapt = adapt_stub()
        assert isinstance(adapt.tcr_chain_index, np.ndarray)


# ────────────────────────────────────────────────────────────────────────────
# ADAPT.cdr_mask
# ────────────────────────────────────────────────────────────────────────────

class TestCdrMask:
    """Tests for ADAPT.cdr_mask — pure numpy logic, no ML."""

    def setup_method(self):
        self.adapt = adapt_stub()

    def _chain(self, chain_idx=1, length=130):
        return make_imgt_chain(chain_idx, length)

    def test_cdr1_positions_are_one(self):
        chain = self._chain(chain_idx=1)
        mask = self.adapt.cdr_mask(chain, chain_index=1, cdr_ids=["acdr1"])
        start, end = IMGT_MAPPER["acdr1"]
        ri = np.array(chain["residue_index"])
        cdr_mask = (ri >= start) & (ri < end)
        np.testing.assert_array_equal(mask[cdr_mask], 1.0)

    def test_non_cdr_positions_are_zero(self):
        chain = self._chain(chain_idx=1)
        mask = self.adapt.cdr_mask(chain, chain_index=1, cdr_ids=["acdr1"])
        start, end = IMGT_MAPPER["acdr1"]
        ri = np.array(chain["residue_index"])
        fw_mask = ~((ri >= start) & (ri < end))
        np.testing.assert_array_equal(mask[fw_mask], 0.0)

    def test_cdr3_positions_are_one(self):
        chain = self._chain(chain_idx=2)
        mask = self.adapt.cdr_mask(chain, chain_index=2, cdr_ids=["bcdr3"])
        start, end = IMGT_MAPPER["bcdr3"]
        ri = np.array(chain["residue_index"])
        cdr_mask = (ri >= start) & (ri < end)
        np.testing.assert_array_equal(mask[cdr_mask], 1.0)

    def test_multiple_cdrs_combined(self):
        chain = self._chain(chain_idx=1)
        mask = self.adapt.cdr_mask(chain, chain_index=1,
                                   cdr_ids=["acdr1", "acdr2", "acdr3"])
        ri = np.array(chain["residue_index"])
        expected = np.zeros(len(ri))
        for cdr in ["acdr1", "acdr2", "acdr3"]:
            s, e = IMGT_MAPPER[cdr]
            expected[(ri >= s) & (ri < e)] = 1.0
        np.testing.assert_array_equal(mask, expected)

    def test_chain_isolation(self):
        """CDRs on chain 1 should not appear in mask for chain 2."""
        chain1 = make_imgt_chain(chain_index=1, length=130)
        chain2 = make_imgt_chain(chain_index=2, length=130)
        combined = DesignData.concatenate([chain1, chain2], sep_chains=False)
        # Mask for chain 1 CDRs
        mask = self.adapt.cdr_mask(combined, chain_index=1, cdr_ids=["acdr1"])
        chain2_indices = np.array(combined["chain_index"]) == 2
        np.testing.assert_array_equal(mask[chain2_indices], 0.0)

    def test_same_chain_assertion(self):
        """Mixed-chain CDR IDs should raise AssertionError."""
        chain = self._chain(chain_idx=1)
        with pytest.raises((AssertionError, ValueError)):
            self.adapt.cdr_mask(chain, chain_index=1, cdr_ids=["acdr1", "bcdr1"])

    def test_mask_length_matches_design(self):
        chain = self._chain(chain_idx=1, length=120)
        mask = self.adapt.cdr_mask(chain, chain_index=1, cdr_ids=["acdr1"])
        assert len(mask) == 120

    def test_returns_float_array(self):
        chain = self._chain(chain_idx=1)
        mask = self.adapt.cdr_mask(chain, chain_index=1, cdr_ids=["acdr1"])
        assert mask.dtype in (np.float32, np.float64)


# ────────────────────────────────────────────────────────────────────────────
# ADAPT.insert_cdr
# ────────────────────────────────────────────────────────────────────────────

class TestInsertCdr:
    """Tests for ADAPT.insert_cdr — DesignData manipulation, no ML."""

    def setup_method(self):
        self.adapt = adapt_stub()
        # Patch number_anarci to be a no-op (return design unchanged)
        self.adapt.number_anarci = lambda design, chains=None, **kw: design

    def _design_with_cdrs(self, chain_idx: int = 1) -> DesignData:
        """130-residue chain, IMGT positions 1..130, chain_index=chain_idx."""
        return make_imgt_chain(chain_idx, length=130)

    def test_output_length_changes_by_cdr_delta(self):
        """
        Inserting a 5-aa CDR3 into a 130-aa chain that has 12 IMGT CDR3 positions
        should change the chain length by (5 - 12) = -7.
        """
        design = self._design_with_cdrs(chain_idx=1)
        cdr3_len_in_scaffold = (IMGT_MAPPER["acdr3"][1] - IMGT_MAPPER["acdr3"][0])
        insert_seq = "ACDEF"  # 5 residues
        out, _ = self.adapt.insert_cdr(design, chain_index=1,
                                        sequences={"acdr3": insert_seq})
        expected_len = 130 - cdr3_len_in_scaffold + len(insert_seq)
        assert len(out["aa"]) == expected_len, (
            f"Expected {expected_len} residues, got {len(out['aa'])}")

    def test_target_mask_marks_inserted_cdr(self):
        """target_mask should be 1.0 for inserted CDR residues."""
        design = self._design_with_cdrs(chain_idx=1)
        insert_seq = "ACDEF"
        out, tmask = self.adapt.insert_cdr(design, chain_index=1,
                                            sequences={"acdr3": insert_seq})
        # Inserted positions should be marked 1.0
        assert tmask.sum() == len(insert_seq), (
            f"Expected {len(insert_seq)} CDR positions marked, got {tmask.sum()}")

    def test_target_mask_zeros_for_framework(self):
        """Non-inserted positions should be 0 in target_mask."""
        design = self._design_with_cdrs(chain_idx=1)
        insert_seq = "ACDEF"
        out, tmask = self.adapt.insert_cdr(design, chain_index=1,
                                            sequences={"acdr3": insert_seq})
        # Framework positions should be 0.0
        assert (tmask == 0).sum() == len(out["aa"]) - len(insert_seq)

    def test_target_mask_length_matches_output(self):
        design = self._design_with_cdrs(chain_idx=1)
        out, tmask = self.adapt.insert_cdr(design, chain_index=1,
                                            sequences={"acdr3": "ACDE"})
        assert len(tmask) == len(out["aa"])

    def test_chain_index_preserved_on_insert(self):
        """The inserted CDR should carry the correct chain_index."""
        design = self._design_with_cdrs(chain_idx=1)
        out, _ = self.adapt.insert_cdr(design, chain_index=1,
                                        sequences={"acdr3": "ACDEF"})
        assert (np.array(out["chain_index"]) == 1).all(), \
            "All residues should have chain_index=1"

    def test_single_residue_cdr_insert(self):
        """Degenerate case: inserting a 1-residue CDR."""
        design = self._design_with_cdrs(chain_idx=1)
        out, tmask = self.adapt.insert_cdr(design, chain_index=1,
                                            sequences={"acdr3": "A"})
        assert tmask.sum() == 1

    def test_framework_residues_retained(self):
        """Residues outside CDR3 range should remain in the output."""
        design = self._design_with_cdrs(chain_idx=1)
        n_fw = (IMGT_MAPPER["acdr3"][0] - 1)  # residues before CDR3 (positions 1..104)
        out, _ = self.adapt.insert_cdr(design, chain_index=1,
                                        sequences={"acdr3": "ACDEF"})
        # Pre-CDR framework residues (positions 1..104) → first n_fw of output
        assert len(out["aa"]) >= n_fw


# ────────────────────────────────────────────────────────────────────────────
# ADAPT.number_anarci
# ────────────────────────────────────────────────────────────────────────────

class TestNumberAnarci:
    """Tests for ADAPT.number_anarci with a mocked anarci backend."""

    def setup_method(self):
        self.adapt = adapt_stub()

    def _vh_chain(self, chain_idx=1, length=120) -> DesignData:
        data = DesignData.from_length(length)
        data = data.update(
            chain_index=jnp.full(length, chain_idx, dtype=jnp.int32),
            residue_index=jnp.arange(1, length + 1, dtype=jnp.int32),
            aa=jnp.zeros(length, dtype=jnp.int32),
        )
        return data

    def test_residue_index_updated_for_receptor_chain(self):
        """
        When anarci recognises the chain as a receptor, residue_index for that
        chain should be replaced with the IMGT numbering returned by anarci.
        """
        design = self._vh_chain(chain_idx=1, length=5)
        fake_imgt = [1, 2, 3, 4, 5]
        # anarci.number returns (numbering_list, chain_type)
        # numbering_list items: ((pos, ins_code), aa)
        fake_numbering = [((n, " "), "A") for n in fake_imgt]
        mock_anarci = MagicMock()
        mock_anarci.number.return_value = (fake_numbering, "H")
        with patch.object(_adapt_module, "anarci", mock_anarci):
            result = self.adapt.number_anarci(design, chains=1)
        ri = np.array(result["residue_index"])[np.array(result["chain_index"]) == 1]
        np.testing.assert_array_equal(ri, fake_imgt)

    def test_residue_index_unchanged_when_no_numbering_found(self):
        """If anarci returns no numbering (False), residue_index should be unchanged."""
        design = self._vh_chain(chain_idx=1, length=5)
        original_ri = np.array(design["residue_index"]).copy()
        mock_anarci = MagicMock()
        mock_anarci.number.return_value = (False, False)
        with patch.object(_adapt_module, "anarci", mock_anarci):
            result = self.adapt.number_anarci(design, chains=1)
        ri = np.array(result["residue_index"])
        np.testing.assert_array_equal(ri, original_ri)

    def test_insertion_code_position_collapsed_to_int(self):
        """IMGT positions with insertion codes (e.g. 111A) should become int 111."""
        design = self._vh_chain(chain_idx=1, length=3)
        # Simulate insertion positions: 111, 111A, 112
        fake_numbering = [
            ((111, " "), "G"),
            ((111, "A"), "S"),
            ((112, " "), "T"),
        ]
        mock_anarci = MagicMock()
        mock_anarci.number.return_value = (fake_numbering, "H")
        with patch.object(_adapt_module, "anarci", mock_anarci):
            result = self.adapt.number_anarci(design, chains=1)
        ri = np.array(result["residue_index"])[np.array(result["chain_index"]) == 1]
        # 111A → 111, so we expect [111, 111, 112]
        assert list(ri) == [111, 111, 112]

    def test_gapped_positions_excluded(self):
        """Positions with '-' amino acid (gaps) should not appear in the numbering."""
        design = self._vh_chain(chain_idx=1, length=3)
        # Middle position is a gap — only 2 residues in the chain but anarci returns 3 positions
        # → design needs length=2 to match the 2 non-gap positions
        design = self._vh_chain(chain_idx=1, length=2)
        fake_numbering = [
            ((1, " "), "A"),
            ((2, " "), "-"),   # gap — should be skipped
            ((3, " "), "C"),
        ]
        mock_anarci = MagicMock()
        mock_anarci.number.return_value = (fake_numbering, "H")
        with patch.object(_adapt_module, "anarci", mock_anarci):
            result = self.adapt.number_anarci(design, chains=1)
        ri = np.array(result["residue_index"])[np.array(result["chain_index"]) == 1]
        # Only non-gap positions: [1, 3]
        assert list(ri) == [1, 3]


# ────────────────────────────────────────────────────────────────────────────
# ADAPT.design_step
# ────────────────────────────────────────────────────────────────────────────

class TestDesignStep:
    """Tests for ADAPT.design_step with mocked ProteinMPNN sampler."""

    def setup_method(self):
        self.adapt = adapt_stub()

    def _pmpnn_mock(self, design, new_aa=None):
        """
        Return a mock pmpnn_sampler that yields a dict with 'aa' filled in
        PMPNN code (all zero = "A" in PMPNN_CODE).
        """
        length = len(design["aa"])
        if new_aa is None:
            new_aa = jnp.zeros(length, dtype=jnp.int32)
        result_dict = {k: v for k, v in design.items()}
        result_dict["aa"] = new_aa
        self.adapt.pmpnn_sampler = lambda key, data: (result_dict, 0.0)

    def test_cdr_positions_are_masked_before_pmpnn(self):
        """
        design_step must set aa=20 at CDR positions (target_mask=True) before
        calling the pmpnn sampler.
        """
        length = 130
        chain = make_imgt_chain(chain_index=1, length=length)
        # Track what aa is when pmpnn_sampler is called
        seen_aa = {}

        def fake_sampler(key, data):
            seen_aa["aa"] = np.array(data["aa"]).copy()
            result = {k: v for k, v in data.items()}
            return result, 0.0

        self.adapt.pmpnn_sampler = fake_sampler

        # target_mask: mark CDR3 residues (positions 105-116 in chain 1)
        ri = np.array(chain["residue_index"])
        target_mask = ((ri >= 105) & (ri < 117)).astype(bool)

        self.adapt.design_step(input_design=chain, target_mask=target_mask)

        # At CDR positions, aa should have been set to 20 before sampling
        cdr_aa = seen_aa["aa"][target_mask]
        assert (cdr_aa == 20).all(), (
            f"CDR positions should have aa=20 before sampling, got {cdr_aa}")

    def test_framework_aa_unchanged_before_pmpnn(self):
        """Framework positions (target_mask=False) should not be masked."""
        length = 130
        chain = make_imgt_chain(chain_index=1, length=length)
        # All-alanine (aa=0), framework only
        seen_aa = {}

        def fake_sampler(key, data):
            seen_aa["aa"] = np.array(data["aa"]).copy()
            return {k: v for k, v in data.items()}, 0.0

        self.adapt.pmpnn_sampler = fake_sampler

        ri = np.array(chain["residue_index"])
        target_mask = ((ri >= 105) & (ri < 117)).astype(bool)
        self.adapt.design_step(input_design=chain, target_mask=target_mask)

        fw_aa = seen_aa["aa"][~target_mask]
        assert (fw_aa == 0).all(), "Framework aa should remain 0 (alanine)"

    def test_returns_design_data(self):
        """design_step must return a DesignData object."""
        chain = make_imgt_chain(chain_index=1, length=50)
        ri = np.array(chain["residue_index"])
        tmask = ((ri >= 27) & (ri < 38)).astype(bool)

        def fake_sampler(key, data):
            return {k: v for k, v in data.items()}, 0.0

        self.adapt.pmpnn_sampler = fake_sampler
        result = self.adapt.design_step(input_design=chain, target_mask=tmask)
        assert isinstance(result, DesignData)

    def test_none_target_mask_uses_cdr_mask(self):
        """
        When target_mask=None, design_step should compute CDR mask internally
        (via cdr_mask) and mask the CDR positions.
        """
        # Multi-chain design with two TCR chains (1 and 2)
        chain1 = make_imgt_chain(chain_index=1, length=130)
        chain2 = make_imgt_chain(chain_index=2, length=130)
        combined = DesignData.concatenate([chain1, chain2], sep_chains=False)

        seen_aa = {}

        def fake_sampler(key, data):
            seen_aa["aa"] = np.array(data["aa"]).copy()
            return {k: v for k, v in data.items()}, 0.0

        self.adapt.pmpnn_sampler = fake_sampler
        self.adapt.tcr_chain_index = np.array([1, 2])

        # Call with no mask — should internally call cdr_mask
        self.adapt.design_step(input_design=combined, target_mask=None)

        aa = seen_aa["aa"]
        ri = np.array(combined["residue_index"])
        ci = np.array(combined["chain_index"])

        for cdr in ["acdr1", "acdr2", "acdr3"]:
            s, e = IMGT_MAPPER[cdr]
            cdr_pos = (ri >= s) & (ri < e) & (ci == 1)
            assert (aa[cdr_pos] == 20).all(), \
                f"{cdr} on chain 1 should be masked (aa=20)"


# ────────────────────────────────────────────────────────────────────────────
# Integration smoke test: design_trial (all ML mocked)
# ────────────────────────────────────────────────────────────────────────────

class TestDesignTrialSmoke:
    """
    End-to-end smoke test for design_trial.
    All ML inference is mocked so no GPU / model weights are required.

    Design: 4-chain structure passed as a single DesignData with pMHC=None so the
    pmhc_is_scaffold branch is taken.  Chain layout:
        0 = MHC α-chain        (mhc_chain_index = [0])
        1 = peptide            (pMHC = [0]; +1 in the chain-count assertion)
        2 = TCR α-chain        (tcr_chain_index[0] = 2)
        3 = TCR β-chain        (tcr_chain_index[1] = 3)
    The assertion in design_trial checks:
        num_unique_chains == len(mhc_chain_index) + len(tcr_chain_index) + 1
                          == 1                   + 2                   + 1 = 4 ✓
    """

    def _make_adapt_with_dirs(self, tmp_path: Path) -> ADAPT:
        in_dir = tmp_path / "input_data"
        in_dir.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        adapt = adapt_stub()
        adapt.in_dir = in_dir
        adapt.out_dir = out_dir
        # Override chain indices for the 4-chain layout
        adapt.mhc_chain_index = np.array([0])
        adapt.tcr_chain_index = np.array([2, 3])
        return adapt

    def _make_full_design(self) -> DesignData:
        """4-chain design: MHC (0), peptide (1), TCR-α (2), TCR-β (3)."""
        mhc     = make_imgt_chain(0, length=180)
        peptide = make_imgt_chain(1, length=9)
        tcr_a   = make_imgt_chain(2, length=130)
        tcr_b   = make_imgt_chain(3, length=130)
        return DesignData.concatenate([mhc, peptide, tcr_a, tcr_b], sep_chains=False)

    def test_design_trial_returns_score(self, tmp_path):
        """
        design_trial should complete and return a numeric score.
        All AF2 and ProteinMPNN calls are replaced with mocks.
        """
        adapt = self._make_adapt_with_dirs(tmp_path)
        design = self._make_full_design()

        # number_anarci → no-op
        adapt.number_anarci = lambda d, chains=None, **kw: d

        def fake_docking(input_design, evaluate=False, is_target=None):
            if evaluate:
                return input_design, 1.23
            return input_design

        adapt.docking_step = fake_docking
        adapt.design_step = lambda input_design, target_mask: input_design

        saved_paths = []
        original_save = DesignData.save_pdb
        DesignData.save_pdb = lambda self_dd, path: saved_paths.append(path)
        try:
            score = adapt.design_trial(
                scaffold=design,
                pMHC=None,          # triggers pmhc_is_scaffold path
                cdr3s=("CASSIRSSYEQYF", "CAVSRGSTGELFF"),
            )
        finally:
            DesignData.save_pdb = original_save

        assert score == pytest.approx(1.23)

    def test_design_trial_records_score_in_dataframe(self, tmp_path):
        """After design_trial, the scores DataFrame should have one row."""
        adapt = self._make_adapt_with_dirs(tmp_path)
        design = self._make_full_design()

        adapt.number_anarci = lambda d, chains=None, **kw: d

        def fake_docking(input_design, evaluate=False, is_target=None):
            return (input_design, 0.5) if evaluate else input_design

        adapt.docking_step = fake_docking
        adapt.design_step = lambda input_design, target_mask: input_design

        original_save = DesignData.save_pdb
        DesignData.save_pdb = lambda self_dd, path: None
        try:
            adapt.design_trial(
                scaffold=design,
                pMHC=None,
                cdr3s=("CASSIRSSYEQYF", "CAVSRGSTGELFF"),
            )
        finally:
            DesignData.save_pdb = original_save

        assert len(adapt.scores) == 1

    def test_design_trial_pdb_saved_to_out_dir(self, tmp_path):
        """design_trial should call save_pdb with a path inside out_dir."""
        adapt = self._make_adapt_with_dirs(tmp_path)
        design = self._make_full_design()
        adapt.number_anarci = lambda d, chains=None, **kw: d

        def fake_docking(input_design, evaluate=False, is_target=None):
            return (input_design, 0.0) if evaluate else input_design

        adapt.docking_step = fake_docking
        adapt.design_step = lambda input_design, target_mask: input_design

        saved_paths = []
        original_save = DesignData.save_pdb
        DesignData.save_pdb = lambda self_dd, path: saved_paths.append(path)

        try:
            adapt.design_trial(
                scaffold=design,
                pMHC=None,
                cdr3s=("CASSIRSSYEQYF", "CAVSRGSTGELFF"),
            )
        finally:
            DesignData.save_pdb = original_save

        assert len(saved_paths) == 1
        assert Path(saved_paths[0]).parent == adapt.out_dir


# ────────────────────────────────────────────────────────────────────────────
# Regression: ADAPT.__init__ with real (mocked) model loading
# ────────────────────────────────────────────────────────────────────────────

class TestAdaptInit:
    """
    Verify ADAPT.__init__ completes and sets expected attributes when
    make_pmpnn, make_af2, make_predict, model_config are all mocked.

    Note: patch.object(_adapt_module, ...) is used because unittest.mock.patch
    cannot walk through a package whose name matches its contained module name
    ("adapt.adapt").
    """

    def _patched_adapt(self, tmp_path, **kwargs):
        """Helper: instantiate ADAPT with all ML components mocked."""
        import pandas as pd
        in_dir = tmp_path / "input_data"
        in_dir.mkdir(exist_ok=True)
        out_dir = tmp_path / "out"

        mock_pmpnn_fn = MagicMock(return_value={"logits": jnp.zeros((10, 21))})

        with patch.object(_adapt_module, "make_pmpnn", return_value=mock_pmpnn_fn), \
             patch.object(_adapt_module, "make_af2", return_value=MagicMock()), \
             patch.object(_adapt_module, "make_predict", return_value=MagicMock()), \
             patch.object(_adapt_module, "model_config", return_value=MagicMock()), \
             patch.object(_adapt_module, "get_model_haiku_params", return_value=MagicMock()), \
             patch("jax.jit", side_effect=lambda fn: fn):

            adapt = ADAPT(
                op_dir=str(tmp_path),
                af2_parameter_path="/nonexistent/dir/",
                af2_model_name="model_2_ptm",
                key=Keygen(42),
                pmpnn_parameter_path="/nonexistent.pkl",
                af2_multimer=False,
                num_recycle=0,
                out_dir=str(out_dir),
                **kwargs,
            )
        return adapt

    def test_init_creates_scores_dataframe_tcr(self, tmp_path):
        import pandas as pd
        adapt = self._patched_adapt(tmp_path)
        assert isinstance(adapt.scores, pd.DataFrame)
        assert "acdr1" in adapt.scores.columns or "lcdr1" in adapt.scores.columns

    def test_init_scalar_chain_indices_become_arrays(self, tmp_path):
        adapt = self._patched_adapt(tmp_path, mhc_chain_index=0, tcr_chain_index=(1, 2))
        assert isinstance(adapt.mhc_chain_index, np.ndarray)
        assert isinstance(adapt.tcr_chain_index, np.ndarray)
        assert adapt.mhc_chain_index[0] == 0
        np.testing.assert_array_equal(adapt.tcr_chain_index, [1, 2])

    def test_init_out_dir_is_created(self, tmp_path):
        adapt = self._patched_adapt(tmp_path)
        assert adapt.out_dir.exists()

    def test_init_rmsd_metric_exists(self, tmp_path):
        adapt = self._patched_adapt(tmp_path)
        assert isinstance(adapt.rmsd, RMSD)
