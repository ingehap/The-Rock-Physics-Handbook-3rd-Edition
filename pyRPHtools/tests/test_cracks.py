import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    eshelby_cheng,
    hudson,
    hudson3,
    hudson_cone,
    hudson_fisher,
    hudson_velocities,
    thomsen_params,
)

# Isotropic host (GPa, g/cc) and crack parameters.
K, G, RHO = 37.0, 44.0, 2.65
K_FL, RHO_FL = 2.25, 1.0
EC, AR = 0.05, 0.01
LAM = K - 2 * G / 3


class TestHudson:
    def test_zero_crack_density_is_isotropic_host(self):
        for axis in (1, 3):
            c, rho = hudson(0.0, AR, K_FL, RHO_FL, K, G, RHO, axis=axis)
            assert c[0, 0] == pytest.approx(LAM + 2 * G)
            assert c[2, 2] == pytest.approx(LAM + 2 * G)
            assert c[0, 1] == pytest.approx(LAM)
            assert c[3, 3] == pytest.approx(G)
            assert rho == pytest.approx(RHO)

    def test_axis_1_is_permutation_of_axis_3(self):
        c1, _ = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=1)
        c3, _ = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=3)
        assert c1[0, 0] == pytest.approx(c3[2, 2])  # along symmetry axis
        assert c1[1, 1] == pytest.approx(c3[0, 0])
        assert c1[1, 2] == pytest.approx(c3[0, 1])
        assert c1[3, 3] == pytest.approx(c3[5, 5])
        assert c1[4, 4] == pytest.approx(c3[3, 3])

    def test_saturated_stiffer_than_dry(self):
        c_dry, _ = hudson(EC, AR, 0.0, 0.0, K, G, RHO, axis=3)
        c_sat, _ = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=3)
        assert c_sat[2, 2] > c_dry[2, 2]  # normal compliance stiffened
        assert c_sat[3, 3] == pytest.approx(c_dry[3, 3])  # shear unaffected by fluid

    def test_density_includes_crack_fluid(self):
        _, rho = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO)
        phi = 4 * np.pi / 3 * AR * EC
        assert rho == pytest.approx((1 - phi) * RHO + phi * RHO_FL)

    def test_stacked_inputs(self):
        ec = np.array([0.0, 0.02, 0.05])
        c, rho = hudson(ec, AR, K_FL, RHO_FL, K, G, RHO, axis=3)
        assert c.shape == (3, 6, 6)
        assert rho.shape == (3,)
        assert np.all(np.diff(c[:, 2, 2]) < 0)  # more cracks -> softer

    def test_matches_hudson_velocities(self):
        c, rho = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=3)
        hv = hudson_velocities(EC, AR, K_FL, K, G, rho, axis=3)
        assert_allclose(hv.c, c, rtol=1e-14)
        assert hv.vp0 == pytest.approx(np.sqrt(c[2, 2] / rho))
        assert hv.vs0 == pytest.approx(np.sqrt(c[3, 3] / rho))
        t = thomsen_params(c[0, 0], c[2, 2], c[3, 3], c[5, 5], c[0, 2])
        assert hv.epsilon == pytest.approx(t.epsilon)
        assert hv.gamma == pytest.approx(t.gamma)
        assert hv.delta == pytest.approx(t.delta)


class TestHudson3:
    def test_single_set_reduces_to_hudson_axis1(self):
        r = hudson3([EC, 0.0, 0.0], [AR, AR, AR], K_FL, RHO_FL, K, G, RHO)
        c1, rho1 = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=1)
        assert_allclose(r.c, c1, rtol=1e-12)
        assert r.rho == pytest.approx(rho1)

    def test_three_equal_sets_nearly_isotropic(self):
        r = hudson3([EC, EC, EC], [AR, AR, AR], K_FL, RHO_FL, K, G, RHO)
        assert r.c[0, 0] == pytest.approx(r.c[1, 1])
        assert r.c[1, 1] == pytest.approx(r.c[2, 2])
        assert r.c[3, 3] == pytest.approx(r.c[4, 4])
        assert r.epsilon_x == pytest.approx(0.0, abs=1e-12)
        assert r.gamma_xy == pytest.approx(0.0, abs=1e-12)

    def test_length_validation(self):
        with pytest.raises(ValueError):
            hudson3([EC, EC], [AR, AR], K_FL, RHO_FL, K, G, RHO)


class TestHudsonFisher:
    def test_small_sigma_approaches_aligned_cracks(self):
        # A tight Fisher distribution about x3 tends to the aligned-crack
        # model with axis=3 (sigma small but big enough to avoid overflow
        # of exp(1/sigma^2)).
        sigma = 0.05
        c_f, _ = hudson_fisher(EC, AR, K_FL, RHO_FL, K, G, RHO, sigma)
        c_h, _ = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=3)
        assert c_f[2, 2] == pytest.approx(c_h[2, 2], rel=2e-2)
        assert c_f[0, 0] == pytest.approx(c_h[0, 0], rel=2e-2)
        assert c_f[3, 3] == pytest.approx(c_h[3, 3], rel=2e-2)

    def test_ti_symmetry(self):
        c, _ = hudson_fisher(EC, AR, K_FL, RHO_FL, K, G, RHO, 0.4)
        assert c[0, 0] == pytest.approx(c[1, 1])
        assert c[3, 3] == pytest.approx(c[4, 4])
        assert c[0, 2] == pytest.approx(c[1, 2])
        # TI identity c66 = (c11 - c12)/2 holds for the Fisher average.
        assert c[5, 5] == pytest.approx((c[0, 0] - c[0, 1]) / 2)
        assert_allclose(c, c.T)

    def test_density_bug_fixed(self):
        # MATLAB computed crack porosity as 4*pi*ar/(3*cd) - dividing by
        # crack density. The port must use (4*pi/3)*ar*cd, matching hudson.
        _, rho = hudson_fisher(EC, AR, K_FL, RHO_FL, K, G, RHO, 0.4)
        phi = 4 * np.pi / 3 * AR * EC
        assert rho == pytest.approx((1 - phi) * RHO + phi * RHO_FL)
        assert RHO_FL < rho < RHO  # physically sensible


class TestHudsonCone:
    def test_zero_angle_equals_aligned(self):
        for axis in (1, 3):
            hc = hudson_cone(EC, AR, K_FL, K, G, RHO, 0.0, axis=axis)
            c_h, _ = hudson(EC, AR, K_FL, RHO_FL, K, G, RHO, axis=axis)
            assert_allclose(hc.c, c_h, rtol=1e-12)

    def test_matlab_radian_transliteration(self):
        # Fresh transliteration of the hudsoncone.m correction formulas
        # (which take theta in radians) at 30 degrees.
        t = np.deg2rad(30.0)
        lam, mu = LAM, G
        kapa = K_FL * (lam + 2 * mu) / (np.pi * AR * mu * (lam + mu))
        u3 = 4 / 3 * (lam + 2 * mu) / ((lam + mu) * (1 + kapa))
        u1 = 16 / 3 * (lam + 2 * mu) / (3 * lam + 4 * mu)
        c11cor = (
            -EC
            / mu
            / 2
            * (
                u3 * (2 * lam**2 + 4 * lam * mu * np.sin(t) ** 2 + 3 * mu**2 * np.sin(t) ** 4)
                + u1 * mu**2 * np.sin(t) ** 2 * (4 - 3 * np.sin(t) ** 2)
            )
        )
        c33cor = (
            -EC
            / mu
            * (
                u3 * (lam + 2 * mu * np.cos(t) ** 2) ** 2
                + u1 * mu**2 * 4 * np.cos(t) ** 2 * np.sin(t) ** 2
            )
        )
        hc = hudson_cone(EC, AR, K_FL, K, G, RHO, 30.0, axis=3)
        assert hc.c[2, 2] == pytest.approx(lam + 2 * mu + c33cor, rel=1e-12)
        assert hc.c[0, 0] == pytest.approx(lam + 2 * mu + c11cor, rel=1e-12)

    def test_c12_slot_follows_matlab(self):
        # The MATLAB fills the c12 slot with c11 - 2*c66 (not the printed
        # c12cor formula); the port must reproduce that.
        hc = hudson_cone(EC, AR, K_FL, K, G, RHO, 40.0, axis=3)
        assert hc.c[0, 1] == pytest.approx(hc.c[0, 0] - 2 * hc.c[5, 5], rel=1e-12)


class TestEshelbyCheng:
    ISO = dict(c11=LAM + 2 * G, c13=LAM, c33=LAM + 2 * G, c44=G, c66=G)

    def test_zero_porosity_is_background(self):
        r = eshelby_cheng(**self.ISO, phi=0.0, aspect=0.1, k_fl=K_FL)
        assert r.c11 == pytest.approx(self.ISO["c11"])
        assert r.c33 == pytest.approx(self.ISO["c33"])
        assert r.c44 == pytest.approx(self.ISO["c44"])

    def test_matlab_transliteration(self):
        # Fresh verbatim transliteration of echeng.m.
        phi, a, kfl = 0.02, 0.1, K_FL
        lam, mu = LAM, G
        k = lam + (2 / 3) * mu
        c = kfl / (3 * (k - kfl))
        sig = (3 * k - 2 * mu) / (6 * k + 2 * mu)
        r = (1 - 2 * sig) / (8 * np.pi * (1 - sig))
        q = 3 * r / (1 - 2 * sig)
        sa = np.sqrt(1 - a**2)
        ia = 2 * np.pi * a * (np.arccos(a) - a * sa) / sa**3
        ic = 4 * np.pi - 2 * ia
        iac = (ic - ia) / (3 * sa**2)
        iaa = np.pi - (3 / 4) * iac
        iab = iaa / 3
        s1313 = 0.5 * q * iac * (1 - a**2) + 0.5 * r * (ia + ic)
        s31 = q * iac - r * ic
        s13 = q * iac * a**2 - r * ia
        s12 = q * iab - r * ia
        s33 = q * (4 * np.pi / 3 - 2 * iac * a**2) + ic * r
        s11 = q * iaa + r * ia
        e = s33 * s11 - s31 * s13 - (s33 + s11 - 2 * c - 1) + c * (s31 + s13 - s11 - s33)
        d = (
            s33 * s11
            + s33 * s12
            - 2 * s31 * s13
            - (s11 + s12 + s33 - 1 - 3 * c)
            - c * (s11 + s12 + 2 * (s33 - s13 - s31))
        )
        cti11 = lam * (s31 - s33 + 1) + 2 * mu * e / (d * (s12 - s11 + 1))
        cti33 = ((lam + 2 * mu) * (-s12 - s11 + 1) + 2 * lam * s13 + 4 * mu * c) / d
        cti44 = mu / (1 - 2 * s1313)

        res = eshelby_cheng(**self.ISO, phi=phi, aspect=a, k_fl=kfl)
        assert res.c11 == pytest.approx(self.ISO["c11"] - phi * cti11, rel=1e-13)
        assert res.c33 == pytest.approx(self.ISO["c33"] - phi * cti33, rel=1e-13)
        assert res.c44 == pytest.approx(self.ISO["c44"] - phi * cti44, rel=1e-13)

    def test_cracks_soften_and_fluid_stiffens_normal(self):
        dry = eshelby_cheng(**self.ISO, phi=0.02, aspect=0.1, k_fl=0.0)
        wet = eshelby_cheng(**self.ISO, phi=0.02, aspect=0.1, k_fl=K_FL)
        assert dry.c33 < self.ISO["c33"]
        assert wet.c33 > dry.c33  # fluid stiffens the crack-normal direction

    def test_vectorized_over_porosity(self):
        phi = np.array([0.0, 0.01, 0.02, 0.04])
        r = eshelby_cheng(**self.ISO, phi=phi, aspect=0.1, k_fl=0.0)
        assert r.c33.shape == (4,)
        assert np.all(np.diff(r.c33) < 0)
