import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    PERM_MODELS,
    bernabe_perm,
    bloch_perm,
    coates_dumanoir_perm,
    coates_perm,
    fredrich_perm,
    kozeny_carman_perm,
    modified_kozeny_carman_perm,
    owolabi_perm,
    panda_lake_kc_perm,
    panda_lake_perm,
)

PHI = np.array([0.05, 0.1, 0.2, 0.3])


class TestBernabe:
    def test_matlab_formulas(self):
        r = bernabe_perm(PHI, crack_fraction=0.8, crack_width=200.0, tube_radius=150.0)
        phi_crack = PHI * 0.8
        phi_tube = PHI - phi_crack
        assert_allclose(r.phi_crack, phi_crack)
        assert_allclose(r.k_crack, 200.0**2 * phi_crack / 30)
        assert_allclose(r.k_tube, 150.0**2 * phi_tube / 20)
        assert_allclose(r.k, r.k_crack + r.k_tube)

    def test_all_cracks_vs_all_tubes(self):
        cracks = bernabe_perm(PHI, crack_fraction=1.0)
        tubes = bernabe_perm(PHI, crack_fraction=0.0)
        assert_allclose(cracks.k_tube, 0.0)
        assert_allclose(tubes.k_crack, 0.0)
        assert_allclose(cracks.phi_crack + tubes.phi_tube, 2 * PHI)

    def test_default_porosity_grid(self):
        r = bernabe_perm()
        assert r.k.shape == (35,)
        assert r.phi_crack[0] == pytest.approx(0.01 * 0.8)


class TestBloch:
    def test_matlab_formulas(self):
        size, sort, content = 1.2, 2.0, 10.0
        r = bloch_perm(size, sort, content)
        phi_pct = -6.1 + 9.8 / sort + 0.17 * content
        assert r.phi == pytest.approx(phi_pct / 100)
        assert r.k == pytest.approx(
            10 ** (-4.67 + 1.34 * size + 4.08 / sort + 3.42 * content / 100)
        )

    def test_porosity_returned_as_fraction(self):
        # MATLAB returned percent; the port returns a fraction.
        r = bloch_perm(0.8, 1.7, 50.0)
        assert 0.0 < r.phi < 1.0

    def test_better_sorting_gives_higher_k(self):
        # Smaller Trask coefficient = better sorting.
        good = bloch_perm(0.8, 1.2, 50.0)
        poor = bloch_perm(0.8, 2.5, 50.0)
        assert good.k > poor.k
        assert good.phi > poor.phi

    def test_vectorized(self):
        r = bloch_perm([1.2, 0.8], [2.0, 1.7], [10.0, 50.0])
        assert r.k.shape == (2,)


class TestSwrFamily:
    def test_coates_dumanoir_formula(self):
        k = coates_dumanoir_perm(PHI, 0.15)
        assert_allclose(k, 352 * PHI**4 / 0.15**4)

    def test_coates_formula(self):
        k = coates_perm(PHI, 0.15)
        assert_allclose(k, 10000 * PHI**4 * (1 - 0.15) ** 2 / 0.15**2)

    def test_phi_fourth_power_scaling(self):
        # Both are A phi^4 / Swr^n: doubling phi multiplies k by 16.
        for model in (coates_perm, coates_dumanoir_perm):
            k1 = model(0.1, 0.2)
            k2 = model(0.2, 0.2)
            assert k2 / k1 == pytest.approx(16.0)

    def test_lower_swr_gives_higher_k(self):
        for model in (coates_perm, coates_dumanoir_perm):
            assert model(0.2, 0.1) > model(0.2, 0.3)

    def test_defaults(self):
        assert coates_perm().shape == (35,)
        assert coates_dumanoir_perm().shape == (35,)


class TestKozenyCarmanFamily:
    def test_kozeny_carman_formula(self):
        k = kozeny_carman_perm(PHI, 250.0)
        assert_allclose(k, (1000 / 450) * 250.0**2 * PHI**3 / (1 - PHI) ** 2)

    def test_fredrich_equals_kozeny_carman(self):
        # The two MATLAB files reduce to the same expression.
        assert_allclose(fredrich_perm(PHI, 100.0), kozeny_carman_perm(PHI, 100.0))

    def test_panda_lake_kc_formula(self):
        assert_allclose(panda_lake_kc_perm(PHI, 250.0), 3.34 * 250.0**2 * PHI**3 / (1 - PHI) ** 2)

    def test_modified_kc_percolation(self):
        phi_c = 0.02
        k = modified_kozeny_carman_perm(PHI, 60.0, 2.0, phi_c)
        phi_x = PHI - phi_c
        assert_allclose(k, 2.0 * 60.0**2 * phi_x**3 / (1 - phi_x) ** 2)
        # Zero connected porosity gives zero permeability.
        assert modified_kozeny_carman_perm(np.array([phi_c]), 60.0, 2.0, phi_c)[0] == 0.0

    def test_modified_kc_below_standard(self):
        # Removing percolation porosity always lowers k.
        std = modified_kozeny_carman_perm(PHI, 60.0, 2.0, 0.0)
        mod = modified_kozeny_carman_perm(PHI, 60.0, 2.0, 0.02)
        assert np.all(mod < std)

    def test_grain_size_squared_scaling(self):
        for model in (kozeny_carman_perm, fredrich_perm, panda_lake_kc_perm):
            assert model(0.2, 200.0) / model(0.2, 100.0) == pytest.approx(4.0)


class TestPandaLake:
    def test_matlab_formula(self):
        tau, s, dpm, cdp = 2.0, 0.25, 650.0, 0.4
        k = panda_lake_perm(PHI, tau, s, dpm, cdp)
        shape = ((s * cdp**3 + 3 * cdp**2 + 1) ** 2) / (1 + cdp**2) ** 2
        assert_allclose(k, shape * dpm**2 * PHI**3 / (72 * tau * (1 - PHI) ** 2))

    def test_reduces_to_kozeny_carman_shape(self):
        # With cv = 0 the shape factor is 1 and only the 1/(72 tau) remains.
        k = panda_lake_perm(PHI, tortuosity=2.0, skewness=0.0, mean_grain_size=100.0, cv=0.0)
        assert_allclose(k, 100.0**2 * PHI**3 / (72 * 2.0 * (1 - PHI) ** 2))

    def test_higher_tortuosity_lowers_k(self):
        assert np.all(panda_lake_perm(PHI, tortuosity=4.0) < panda_lake_perm(PHI, tortuosity=2.0))


class TestOwolabi:
    def test_matlab_formulas(self):
        phi, swi = 0.25, 0.8
        r = owolabi_perm(phi, swi)
        assert r.k_oil == pytest.approx(307 + 26552 * phi**2 - 34540 * (phi * swi) ** 2)
        assert r.k_gas == pytest.approx(30.7 + 2655 * phi**2 - 3454 * (phi * swi) ** 2)

    def test_oil_exceeds_gas(self):
        r = owolabi_perm(np.array([0.1, 0.2, 0.3]), 0.8)
        assert np.all(r.k_oil > r.k_gas)

    def test_increases_with_porosity(self):
        r = owolabi_perm(np.array([0.1, 0.15, 0.2, 0.25, 0.3]), 0.8)
        assert np.all(np.diff(r.k_oil) > 0)
        assert np.all(np.diff(r.k_gas) > 0)


class TestRegistry:
    def test_contains_the_ten_existing_models(self):
        assert len(PERM_MODELS) == 10
        assert set(PERM_MODELS) == {
            "BernabeE",
            "FredrichE",
            "KozCarmE",
            "ModKozCarm",
            "PandaLakeKCE",
            "CoatDum",
            "Coates",
            "Bloch",
            "Owolabi",
            "PandaLake",
        }

    def test_excludes_models_missing_from_rphtools(self):
        # PermMenu.m listed these four but the .m files do not exist.
        for name in ("RevilE", "Timur", "Tixier", "WylGregE"):
            assert name not in PERM_MODELS

    def test_all_entries_callable(self):
        for name, fn in PERM_MODELS.items():
            assert callable(fn), name

    def test_porosity_only_models_run_on_a_common_grid(self):
        # ModKozCarm is excluded from the positivity check below its
        # percolation porosity, so evaluate everything well above it.
        phi = np.linspace(0.05, 0.35, 10)
        for name in (
            "BernabeE",
            "FredrichE",
            "KozCarmE",
            "ModKozCarm",
            "PandaLakeKCE",
            "CoatDum",
            "Coates",
            "PandaLake",
        ):
            result = PERM_MODELS[name](phi)
            k = result.k if hasattr(result, "k") else result
            assert np.all(np.isfinite(k)), name
            assert np.all(k > 0), name
            assert np.all(np.diff(k) > 0), name  # all increase with porosity
