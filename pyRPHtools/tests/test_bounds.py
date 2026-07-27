import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import bounds, hashin_shtrikman, hashin_shtrikman_velocity

# Quartz and water.
K1, MU1 = 37.0, 44.0
K2, MU2 = 2.2, 0.0


class TestBounds:
    F = np.array([0.7, 0.3])
    K = np.array([K1, K2])
    MU = np.array([MU1, 3.0])  # nonzero fluid shear to keep Reuss finite

    def test_voigt_reuss_formulas(self):
        b = bounds(self.F, self.K, self.MU, method="voigt-reuss")
        assert b.k_upper == pytest.approx(np.sum(self.F * self.K))
        assert b.k_lower == pytest.approx(1 / np.sum(self.F / self.K))
        assert b.mu_upper == pytest.approx(np.sum(self.F * self.MU))
        assert b.mu_lower == pytest.approx(1 / np.sum(self.F / self.MU))
        assert b.k_avg == pytest.approx((b.k_upper + b.k_lower) / 2)

    def test_hs_within_voigt_reuss(self):
        vr = bounds(self.F, self.K, self.MU, method="voigt-reuss")
        hs = bounds(self.F, self.K, self.MU, method="hashin-shtrikman")
        assert vr.k_lower <= hs.k_lower <= hs.k_upper <= vr.k_upper
        assert vr.mu_lower <= hs.mu_lower <= hs.mu_upper <= vr.mu_upper
        assert hs.k_lower < hs.k_upper  # genuinely distinct phases

    def test_single_phase_degenerate(self):
        for method in ("voigt-reuss", "hs"):
            b = bounds([1.0], [K1], [MU1], method=method)
            assert_allclose(list(b), [K1, K1, MU1, MU1, K1, MU1], rtol=1e-12)

    def test_fluid_phase_zero_shear(self):
        b = bounds(self.F, self.K, [MU1, 0.0], method="hs")
        assert b.mu_lower == pytest.approx(0.0)
        assert b.mu_upper > 0

    def test_float_fraction_sums_accepted(self):
        # sum([0.1]*10) != 1 exactly; the MATLAB exact-equality check
        # rejected this. The port must accept it.
        f = [0.1] * 10
        k = np.linspace(5, 40, 10)
        mu = np.linspace(3, 30, 10)
        bounds(f, k, mu, method="hs")

    def test_validation(self):
        with pytest.raises(ValueError, match="sum to 1"):
            bounds([0.5, 0.3], [K1, K2], [MU1, MU2])
        with pytest.raises(ValueError, match="same length"):
            bounds([0.5, 0.5], [K1, K2, 3.0], [MU1, MU2])
        with pytest.raises(ValueError, match="unknown method"):
            bounds([1.0], [K1], [MU1], method="nope")


class TestHashinShtrikman:
    def test_end_members(self):
        hs = hashin_shtrikman(K1, MU1, K2, MU2, f2=np.array([0.0, 1.0]))
        assert hs.k_upper[0] == pytest.approx(K1)
        assert hs.k_lower[0] == pytest.approx(K1)
        assert hs.mu_upper[0] == pytest.approx(MU1)
        assert hs.k_upper[-1] == pytest.approx(K2)
        assert hs.mu_upper[-1] == pytest.approx(MU2)

    def test_identical_phases_coincide(self):
        hs = hashin_shtrikman(K1, MU1, K1, MU1)
        assert_allclose(hs.k_upper, K1)
        assert_allclose(hs.k_lower, K1)
        assert_allclose(hs.mu_upper, MU1)
        assert_allclose(hs.mu_lower, MU1)

    def test_default_grid(self):
        hs = hashin_shtrikman(K1, MU1, K2, MU2)
        assert hs.f2.shape == (101,)
        assert hs.f2[0] == pytest.approx(1e-7)
        assert hs.f2[-1] == 1.0

    def test_matches_bounds_two_phase(self):
        # The curves must agree with the N-phase bound() evaluated at a
        # composition, when phase 1 is the stiffer material.
        f2 = 0.3
        mu2 = 3.0
        hs = hashin_shtrikman(K1, MU1, K2, mu2, f2=np.array([f2]))
        b = bounds([1 - f2, f2], [K1, K2], [MU1, mu2], method="hs")
        assert hs.k_upper[0] == pytest.approx(b.k_upper)
        assert hs.k_lower[0] == pytest.approx(b.k_lower)
        assert hs.mu_upper[0] == pytest.approx(b.mu_upper)
        assert hs.mu_lower[0] == pytest.approx(b.mu_lower)

    def test_vacuum_second_phase(self):
        # por(1)=1e-7 in the MATLAB grid guards the 0/0 in the k bounds for
        # vacuum phase 2. The lower shear bound is NaN (0/0 in its zeta
        # term) exactly as in MATLAB.
        hs = hashin_shtrikman(K1, MU1, 0.0, 0.0)
        assert np.all(np.isfinite(hs.k_upper))
        assert np.all(np.isfinite(hs.k_lower))
        assert np.all(np.isfinite(hs.mu_upper))
        assert np.all(np.isnan(hs.mu_lower))


class TestHashinShtrikmanVelocity:
    ARGS = (6.008, 4.075, 2.65, 1.5, 0.0, 1.0)  # quartz / water

    def test_consistency_with_moduli_curves(self):
        v = hashin_shtrikman_velocity(*self.ARGS)
        vp1, vs1, rho1 = self.ARGS[:3]
        mu1 = rho1 * vs1**2
        k1 = rho1 * vp1**2 - 4 / 3 * mu1
        mu2 = self.ARGS[4] ** 2 * self.ARGS[5]
        k2 = self.ARGS[5] * self.ARGS[3] ** 2 - 4 / 3 * mu2
        hs = hashin_shtrikman(k1, mu1, k2, mu2)
        rho = (1 - hs.f2) * rho1 + hs.f2 * self.ARGS[5]
        assert_allclose(v.vp_upper, np.sqrt((hs.k_upper + 4 / 3 * hs.mu_upper) / rho))
        assert_allclose(v.vs_upper, np.sqrt(hs.mu_upper / rho))

    def test_end_member_velocities(self):
        v = hashin_shtrikman_velocity(*self.ARGS, f2=np.array([0.0, 1.0]))
        assert v.vp_upper[0] == pytest.approx(6.008)
        assert v.vs_upper[0] == pytest.approx(4.075)
        assert v.vp_upper[-1] == pytest.approx(1.5)
        assert v.vs_upper[-1] == pytest.approx(0.0)
