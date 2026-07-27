import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    berryman_sc,
    berryman_sc_pressure,
    berryman_scm,
    bounds,
    dem,
    dem_at_fraction,
)
from rphtools.effective_medium import _berryman_pq, _spheroid_theta_fn

# Quartz and water (GPa).
K1, MU1 = 37.0, 44.0
K2, MU2 = 2.2, 0.0


def matlab_pq(k1, mu1, ksc, musc, asp):
    """Fresh verbatim transliteration of the f11..f91 block of berrysc.m
    for a single phase, as an independent check of _berryman_pq."""
    if asp < 1:
        theta = (asp / (1 - asp**2) ** 1.5) * (np.arccos(asp) - asp * np.sqrt(1 - asp**2))
        fn = (asp**2 / (1 - asp**2)) * (3 * theta - 2)
    else:
        theta = (asp / (asp**2 - 1) ** 1.5) * (asp * np.sqrt(asp**2 - 1) - np.arccosh(asp))
        fn = (asp**2 / (asp**2 - 1)) * (2 - 3 * theta)
    nusc = (3 * ksc - 2 * musc) / (2 * (3 * ksc + musc))
    a1 = mu1 / musc - 1
    b1 = (1 / 3) * (k1 / ksc - mu1 / musc)
    r = (1 - 2 * nusc) / (2 * (1 - nusc))
    f11 = 1 + a1 * ((3 / 2) * (fn + theta) - r * ((3 / 2) * fn + (5 / 2) * theta - (4 / 3)))
    f21 = 1 + a1 * (1 + (3 / 2) * (fn + theta) - (r / 2) * (3 * fn + 5 * theta)) + b1 * (3 - 4 * r)
    f21 = f21 + (a1 / 2) * (a1 + 3 * b1) * (3 - 4 * r) * (
        fn + theta - r * (fn - theta + 2 * theta**2)
    )
    f31 = 1 + a1 * (1 - (fn + (3 / 2) * theta) + r * (fn + theta))
    f41 = 1 + (a1 / 4) * (fn + 3 * theta - r * (fn - theta))
    f51 = a1 * (-fn + r * (fn + theta - (4 / 3))) + b1 * theta * (3 - 4 * r)
    f61 = 1 + a1 * (1 + fn - r * (fn + theta)) + b1 * (1 - theta) * (3 - 4 * r)
    f71 = 2 + (a1 / 4) * (3 * fn + 9 * theta - r * (3 * fn + 5 * theta)) + b1 * theta * (3 - 4 * r)
    f81 = a1 * (1 - 2 * r + (fn / 2) * (r - 1) + (theta / 2) * (5 * r - 3)) + b1 * (1 - theta) * (
        3 - 4 * r
    )
    f91 = a1 * ((r - 1) * fn - r * theta) + b1 * theta * (3 - 4 * r)
    p = (3 * f11 / f21) / 3
    q = ((2 / f31) + (1 / f41) + ((f41 * f51 + f61 * f71 - f81 * f91) / (f21 * f41))) / 5
    return p, q


class TestBerrymanPQ:
    @pytest.mark.parametrize("asp", [0.01, 0.1, 0.9, 2.0])
    def test_matches_matlab_transliteration(self, asp):
        ksc, musc = 20.0, 15.0
        p_ml, q_ml = matlab_pq(K1, MU1, ksc, musc, asp)
        theta, fn = _spheroid_theta_fn(asp)
        p, q = _berryman_pq(np.array([K1]), np.array([MU1]), ksc, musc, theta, fn)
        assert p[0] == pytest.approx(p_ml, rel=1e-13)
        assert q[0] == pytest.approx(q_ml, rel=1e-13)

    def test_same_material_gives_unity(self):
        theta, fn = _spheroid_theta_fn(0.5)
        p, q = _berryman_pq(np.array([K1]), np.array([MU1]), K1, MU1, theta, fn)
        assert p[0] == pytest.approx(1.0)
        assert q[0] == pytest.approx(1.0)


class TestBerrymanSCM:
    def test_single_phase(self):
        k, mu = berryman_scm([K1], [MU1], [0.5], [1.0])
        assert k == pytest.approx(K1)
        assert mu == pytest.approx(MU1)

    def test_identical_phases(self):
        k, mu = berryman_scm([K1, K1], [MU1, MU1], [1.0, 0.1], [0.5, 0.5])
        assert k == pytest.approx(K1)
        assert mu == pytest.approx(MU1)

    def test_fixed_point_property(self):
        # The converged moduli satisfy sum(x (k - k_sc) P) = 0 (and Q).
        k = np.array([K1, K2])
        mu = np.array([MU1, 3.0])
        asp = np.array([0.8, 0.15])
        x = np.array([0.7, 0.3])
        k_sc, mu_sc = berryman_scm(k, mu, asp, x, tol=1e-12 * K1)
        theta, fn = _spheroid_theta_fn(asp)
        p, q = _berryman_pq(k, mu, k_sc, mu_sc, theta, fn)
        assert np.sum(x * (k - k_sc) * p) == pytest.approx(0.0, abs=1e-8)
        assert np.sum(x * (mu - mu_sc) * q) == pytest.approx(0.0, abs=1e-8)

    def test_within_hs_bounds_for_spheres(self):
        # For near-spherical inclusions the SC estimate must lie within the
        # Hashin-Shtrikman bounds.
        x2 = 0.4
        k_sc, mu_sc = berryman_scm([K1, K2], [MU1, 3.0], [0.999, 0.999], [1 - x2, x2])
        b = bounds([1 - x2, x2], [K1, K2], [MU1, 3.0], method="hs")
        assert b.k_lower <= k_sc <= b.k_upper
        assert b.mu_lower <= mu_sc <= b.mu_upper

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            berryman_scm([K1, K2], [MU1], [0.5, 0.5], [0.5, 0.5])


class TestBerrymanSC:
    def test_matches_scm_core(self):
        f2 = np.array([0.3])
        curves = berryman_sc(K1, MU1, K2, MU2, 1.0, 0.1, f2=f2)
        k, mu = berryman_scm([K1, K2], [MU1, MU2], [1.0, 0.1], [0.7, 0.3], tol=1e-6 * K1)
        assert curves.k[0] == pytest.approx(k)
        assert curves.mu[0] == pytest.approx(mu)

    def test_default_sweep(self):
        curves = berryman_sc(K1, MU1, K2, MU2, 1.0, 0.1)
        assert curves.f2.shape == (101,)
        assert curves.f2[0] == pytest.approx(1e-7)
        assert curves.f2[-1] == pytest.approx(1 - 1e-7)
        # Near-pure phase 1 returns phase-1 moduli; moduli soften with fluid.
        assert curves.k[0] == pytest.approx(K1, rel=1e-4)
        assert curves.mu[0] == pytest.approx(MU1, rel=1e-4)
        assert curves.k[50] < curves.k[10]


class TestBerrymanSCPressure:
    K = np.array([37.0, 2.2, 2.2])
    MU = np.array([44.0, 0.0, 0.0])
    ASP = np.array([1.0, 0.01, 0.5])  # mineral, thin fluid cracks, stiff pores
    X = np.array([0.8, 0.05, 0.15])

    def test_no_cracks_pressure_independent(self):
        k = np.array([37.0, 2.2])
        mu = np.array([44.0, 0.0])
        asp = np.array([1.0, 0.5])  # aspect > 0.2: unaffected by stress
        x = np.array([0.8, 0.2])
        kp, mup = berryman_sc_pressure(k, mu, asp, x, [0.0, 10.0, 100.0])
        assert_allclose(kp, kp[0])
        assert_allclose(mup, mup[0])
        k0, mu0 = berryman_scm(k, mu, asp, x)
        assert kp[0] == pytest.approx(k0)

    def test_stiffens_with_pressure(self):
        kp, mup = berryman_sc_pressure(self.K, self.MU, self.ASP, self.X, [0.0, 0.05, 0.2])
        assert kp[0] < kp[1] < kp[2]
        assert mup[0] < mup[1] < mup[2]

    def test_high_pressure_closes_cracks(self):
        # Once delasp >= asp the crack phase vanishes; result equals the SC
        # of the remaining phases with renormalized fractions.
        p_big = 100.0  # closes the asp=0.01 cracks by far
        kp, mup = berryman_sc_pressure(self.K, self.MU, self.ASP, self.X, [p_big])
        keep = np.array([0, 2])  # crack phase fully closed and removed
        x_left = self.X.copy()
        x_left[1] = 0.0
        x_left = x_left / x_left.sum()
        k_ref, mu_ref = berryman_scm(self.K[keep], self.MU[keep], self.ASP[keep], x_left[keep])
        assert kp[0] == pytest.approx(k_ref)
        assert mup[0] == pytest.approx(mu_ref)

    def test_tied_max_bulk_modulus_ok(self):
        # find(k==max(k)) in MATLAB crashed on ties; argmax must not.
        k = np.array([37.0, 37.0, 2.2])
        mu = np.array([44.0, 40.0, 0.0])
        asp = np.array([1.0, 1.0, 0.01])
        x = np.array([0.5, 0.4, 0.1])
        kp, _ = berryman_sc_pressure(k, mu, asp, x, [0.05])
        assert np.isfinite(kp[0])


class TestDEM:
    def test_matches_matlab_rhs(self):
        # Fresh check: dK/dt = (K2-K) P/(1-t) with P from the verbatim
        # transliteration.
        from rphtools.effective_medium import _dem_rhs

        t, y = 0.3, [25.0, 20.0]
        asp = 0.2
        theta, fn = _spheroid_theta_fn(asp)
        rhs = _dem_rhs(t, y, K2, MU2, float(theta[0]), float(fn[0]))
        p_ml, q_ml = matlab_pq(K2, MU2, y[0], y[1], asp)
        assert rhs[0] == pytest.approx((K2 - y[0]) * p_ml / (1 - t), rel=1e-12)
        assert rhs[1] == pytest.approx((MU2 - y[1]) * q_ml / (1 - t), rel=1e-12)

    def test_zero_porosity_is_matrix(self):
        r = dem(K1, MU1, K2, MU2, 0.1, phi=np.array([0.0, 0.2]))
        assert r.k[0] == pytest.approx(K1)
        assert r.mu[0] == pytest.approx(MU1)

    def test_monotone_softening_with_fluid(self):
        r = dem(K1, MU1, K2, MU2, 0.1, phi=np.linspace(0.0, 0.6, 30))
        assert np.all(np.diff(r.k) < 0)
        assert np.all(np.diff(r.mu) < 0)
        assert np.all(r.k > 0)

    def test_dilute_limit(self):
        # For small phi, DEM matches the single-scattering (dilute) result
        # dK = phi (K2-K1) P evaluated in the pure matrix.
        phi = 0.005
        asp = 0.3
        r_k, r_mu = dem_at_fraction(K1, MU1, K2, MU2, asp, phi)
        theta, fn = _spheroid_theta_fn(asp)
        p, q = _berryman_pq(np.array([K2]), np.array([MU2]), K1, MU1, theta, fn)
        assert r_k == pytest.approx(K1 + phi * (K2 - K1) * p[0], rel=1e-3)
        assert r_mu == pytest.approx(MU1 + phi * (MU2 - MU1) * q[0], rel=1e-3)

    def test_dem_and_dem_at_fraction_agree(self):
        phi = 0.35
        r = dem(K1, MU1, K2, MU2, 0.2, phi=np.linspace(0.0, phi, 20))
        k1p, mu1p = dem_at_fraction(K1, MU1, K2, MU2, 0.2, phi)
        assert r.k[-1] == pytest.approx(k1p, rel=1e-8)
        assert r.mu[-1] == pytest.approx(mu1p, rel=1e-8)

    def test_same_material_constant(self):
        r = dem(K1, MU1, K1, MU1, 0.5, phi=np.linspace(0.0, 0.7, 10))
        assert_allclose(r.k, K1, rtol=1e-9)
        assert_allclose(r.mu, MU1, rtol=1e-9)

    def test_modified_dem_percolation(self):
        # phi_c < 1: reaching phi -> phi_c drives moduli toward phase 2.
        phi_c = 0.4
        k, mu = dem_at_fraction(K1, MU1, K2, 3.0, 0.5, 0.399, phi_c=phi_c)
        k_usual, _ = dem_at_fraction(K1, MU1, K2, 3.0, 0.5, 0.399, phi_c=1.0)
        assert k < k_usual  # same phi is much closer to percolation

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            dem_at_fraction(K1, MU1, K2, MU2, 0.1, 1.5)
        with pytest.raises(ValueError):
            dem(K1, MU1, K2, MU2, 0.1, phi=np.array([0.0, 1.2]))
