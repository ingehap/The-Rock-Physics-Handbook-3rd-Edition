import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    contact_cement,
    coordination_number,
    hertz_mindlin,
    hertz_mindlin_v,
    johnson_makse,
    johnson_stress_anisotropy,
    moduli_to_velocity,
    velocity_to_moduli,
)

# Quartz grains (GPa, g/cc), pressure in GPa for consistency.
K_MIN, G_MIN, RHO_MIN = 37.0, 44.0, 2.65
PRESSURE = 0.02  # 20 MPa expressed in GPa


class TestCoordinationNumber:
    def test_table_nodes_exact(self):
        assert coordination_number(0.2) == pytest.approx(14.007)
        assert coordination_number(0.4) == pytest.approx(8.3147)
        assert coordination_number(0.7) == pytest.approx(3.7440)

    def test_monotone_decreasing(self):
        c = coordination_number(np.linspace(0.2, 0.7, 30))
        assert np.all(np.diff(c) < 0)

    def test_outside_range_is_nan(self):
        assert np.isnan(coordination_number(0.1))
        assert np.isnan(coordination_number(0.85))


class TestHertzMindlin:
    def test_matlab_formulas(self):
        phi = np.array([0.3, 0.36, 0.4])
        c = coordination_number(phi)
        r = hertz_mindlin(K_MIN, G_MIN, PRESSURE, phi)
        nu = (3 * K_MIN - 2 * G_MIN) / (6 * K_MIN + 2 * G_MIN)
        k_ml = (
            (c**2 * (1 - phi) ** 2 * G_MIN**2) / (18 * np.pi**2 * (1 - nu) ** 2) * PRESSURE
        ) ** (1 / 3)
        g_ml = (
            (5 - 4 * nu)
            / (5 * (2 - nu))
            * ((3 * c**2 * (1 - phi) ** 2 * G_MIN**2) / (2 * np.pi**2 * (1 - nu) ** 2) * PRESSURE)
            ** (1 / 3)
        )
        assert_allclose(r.k, k_ml, rtol=1e-14)
        assert_allclose(r.g, g_ml, rtol=1e-14)

    def test_cube_root_pressure_scaling(self):
        r1 = hertz_mindlin(K_MIN, G_MIN, 0.01, np.array([0.36]))
        r8 = hertz_mindlin(K_MIN, G_MIN, 0.08, np.array([0.36]))
        assert r8.k[0] / r1.k[0] == pytest.approx(2.0)  # 8^(1/3)
        assert r8.g[0] / r1.g[0] == pytest.approx(2.0)

    def test_softer_at_higher_porosity(self):
        r = hertz_mindlin(K_MIN, G_MIN, PRESSURE)
        assert np.all(np.diff(r.k) < 0)
        assert np.all(np.diff(r.g) < 0)

    def test_default_porosity_grid(self):
        r = hertz_mindlin(K_MIN, G_MIN, PRESSURE)
        assert r.phi.shape == (11,)
        assert r.phi[0] == pytest.approx(0.2)
        assert r.phi[-1] == pytest.approx(0.7)

    def test_explicit_coordination_overrides_table(self):
        r = hertz_mindlin(K_MIN, G_MIN, PRESSURE, np.array([0.36]), coord=9.0)
        assert r.coord[0] == 9.0 if np.ndim(r.coord) else r.coord == 9.0

    def test_velocity_form_consistent(self):
        vp_min, vs_min = moduli_to_velocity(K_MIN, G_MIN, RHO_MIN)
        phi = np.array([0.3, 0.36, 0.4])
        rv = hertz_mindlin_v(vp_min, vs_min, RHO_MIN, PRESSURE, phi)
        rm = hertz_mindlin(K_MIN, G_MIN, PRESSURE, phi)
        assert_allclose(rv.rho, (1 - phi) * RHO_MIN)
        k_back, g_back = velocity_to_moduli(rv.vp, rv.vs, rv.rho)
        assert_allclose(k_back, rm.k, rtol=1e-12)
        assert_allclose(g_back, rm.g, rtol=1e-12)


class TestContactCement:
    ARGS = dict(
        phi_c=0.38,
        coord=8.5,
        g_grain=45.0,
        nu_grain=0.064,
        g_cement=45.0,
        nu_cement=0.064,
        k_fluid=0.0,
    )

    def test_zero_cement_frame_is_small(self):
        # At phi = phi_c the cement radius a is 0, but Dvorkin's fitted
        # S_n/S_tau polynomials keep a small constant term, so the frame
        # moduli are near zero rather than exactly zero.
        r = contact_cement(**self.ARGS, phi=np.array([0.38]))
        full = contact_cement(**self.ARGS, phi=np.array([0.25]))
        assert 0 < r.k_frame[0] < 0.01 * full.k_frame[0]
        assert 0 < r.g_frame[0] < 0.02 * full.g_frame[0]

    def test_stiffens_as_porosity_drops(self):
        # The default sweep runs from phi_c downward, so moduli increase
        # along the array as cement fills the pore space.
        r = contact_cement(**self.ARGS)
        assert np.all(np.diff(r.phi) < 0)
        assert np.all(np.diff(r.k_frame) > 0)
        assert np.all(np.diff(r.g_frame) > 0)
        assert np.all(r.k_frame > 0)

    def test_dry_frame_equals_saturated_when_kf_zero(self):
        r = contact_cement(**self.ARGS)
        assert_allclose(r.k_sat, r.k_frame, atol=1e-12)
        assert_allclose(r.m_sat, r.k_frame + 4 * r.g_frame / 3, rtol=1e-12)

    def test_fluid_stiffens_bulk_not_shear(self):
        dry = contact_cement(**{**self.ARGS, "k_fluid": 0.0})
        wet = contact_cement(**{**self.ARGS, "k_fluid": 2.2})
        assert np.all(wet.k_sat[1:] > dry.k_sat[1:])
        assert_allclose(wet.g_frame, dry.g_frame)

    def test_scheme_1_stiffer_than_scheme_2(self):
        # Cement at the contacts is more efficient than cement on the
        # grain surface for the same cement volume.
        phi = np.array([0.30])
        s1 = contact_cement(**self.ARGS, scheme=1, phi=phi)
        s2 = contact_cement(**self.ARGS, scheme=2, phi=phi)
        assert s1.k_frame[0] > s2.k_frame[0]

    def test_default_sweep(self):
        r = contact_cement(**self.ARGS)
        assert r.phi.shape == (100,)
        assert r.phi[0] == pytest.approx(0.38)

    def test_solid_phase_moduli_between_grain_and_cement(self):
        # The Hill average of grain and cement must lie between them.
        args = dict(TestContactCement.ARGS, g_cement=9.0, nu_cement=0.3)
        r = contact_cement(**args, phi=np.array([0.25]))
        k_grain = 45.0 * 2 * (1 + 0.064) / (3 * (1 - 2 * 0.064))
        k_cement = 9.0 * 2 * (1 + 0.3) / (3 * (1 - 2 * 0.3))
        assert min(k_grain, k_cement) < r.k_solid[0] < max(k_grain, k_cement)
        assert 9.0 < r.g_solid[0] < 45.0

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            contact_cement(**self.ARGS, scheme=3)


class TestJohnson:
    ARGS = dict(mu=44.0, poisson=0.06, n=9.0, phi=0.36, epsilon=-1e-3, e3=-2e-3, rho=2650.0)

    def test_matlab_stiffness_transliteration(self):
        a = self.ARGS
        cn = 4 * a["mu"] / (1 - a["poisson"])
        ct = 8 * a["mu"] / (2 - a["poisson"])
        gamma = (3 / 32) * a["n"] * cn * ct * (1 - a["phi"]) * (-a["epsilon"]) ** 0.5
        alfa = np.sqrt(a["epsilon"] / a["e3"])
        bw = 2 / (np.pi * cn)
        cw = (4 / np.pi) * (1 / ct - 1 / cn)
        io = 0.5 * (np.sqrt(1 + alfa**2) + alfa**2 * np.log((1 + np.sqrt(1 + alfa**2)) / alfa))
        i2 = 0.25 * ((1 + alfa**2) ** 1.5 - alfa**2 * io)
        i4 = (1 / 6) * ((1 + alfa**2) ** 1.5 - 3 * alfa**2 * io)
        c11 = (gamma / alfa) * (2 * bw * (io - i2) + (3 * cw / 4) * (io - 2 * i2 + i4))
        c33 = (gamma / alfa) * (4 * bw * i2 + 2 * cw * i4)
        c13 = (gamma / alfa) * (cw * (i2 - i4))
        c44 = (gamma / alfa) * ((bw / 2) * (io + i2) + cw * (i2 - i4))
        c66 = (gamma / alfa) * (bw * (io - i2) + (cw / 4) * (io - 2 * i2 + i4))

        r = johnson_stress_anisotropy(**a)
        assert r.c[0, 0] == pytest.approx(c11, rel=1e-13)
        assert r.c[2, 2] == pytest.approx(c33, rel=1e-13)
        assert r.c[0, 2] == pytest.approx(c13, rel=1e-13)
        assert r.c[3, 3] == pytest.approx(c44, rel=1e-13)
        assert r.c[5, 5] == pytest.approx(c66, rel=1e-13)

    def test_returns_tensor_not_scalar(self):
        # MATLAB's C output was overwritten by a scalar contact constant.
        r = johnson_stress_anisotropy(**self.ARGS)
        assert r.c.shape == (6, 6)
        assert r.c[0, 0] > 0

    def test_ti_symmetry(self):
        r = johnson_stress_anisotropy(**self.ARGS)
        c = r.c
        assert c[0, 0] == pytest.approx(c[1, 1])
        assert c[3, 3] == pytest.approx(c[4, 4])
        assert c[0, 2] == pytest.approx(c[1, 2])
        assert c[0, 1] == pytest.approx(c[0, 0] - 2 * c[5, 5])
        assert_allclose(c, c.T)

    def test_matlab_stress_transliteration(self):
        a = self.ARGS
        lam = a["mu"] * (2 * a["poisson"] / (1 - 2 * a["poisson"]))
        b = (1 / (4 * np.pi)) * (1 / a["mu"] + 1 / (lam + a["mu"]))
        cc = (1 / (4 * np.pi)) * (1 / a["mu"] - 1 / (lam + a["mu"]))
        s3 = -(((-a["e3"]) ** 1.5) * (1 - a["phi"]) * a["n"] * (3 * b + cc)) / (
            (6 * np.pi**2) * b * (2 * b + cc)
        )
        s1 = -(((-a["e3"]) ** 1.5) * (1 - a["phi"]) * a["n"] * cc) / (
            (24 * np.pi**2) * b * (2 * b + cc)
        )
        r = johnson_stress_anisotropy(**a)
        assert r.sigma3 == pytest.approx(s3, rel=1e-13)
        assert r.sigma1 == pytest.approx(s1, rel=1e-13)

    def test_axial_stress_exceeds_transverse(self):
        r = johnson_stress_anisotropy(**self.ARGS)
        # Compression: both negative, axial larger in magnitude.
        assert r.sigma3 < r.sigma1 < 0

    def test_stiffness_scales_linearly_with_contacts(self):
        # The whole tensor enters through gamma, which is linear in n,
        # in (1 - phi), and in sqrt(-epsilon).
        base = johnson_stress_anisotropy(**self.ARGS)
        doubled = johnson_stress_anisotropy(**{**self.ARGS, "n": 2 * self.ARGS["n"]})
        assert_allclose(doubled.c, 2 * base.c, rtol=1e-12)
        quad_eps = johnson_stress_anisotropy(
            **{**self.ARGS, "epsilon": 4 * self.ARGS["epsilon"], "e3": 4 * self.ARGS["e3"]}
        )
        assert_allclose(quad_eps.c, 2 * base.c, rtol=1e-12)

    def test_positive_definite_in_intended_regime(self):
        r = johnson_stress_anisotropy(**self.ARGS)
        assert np.all(np.linalg.eigvalsh(r.c) > 0)

    def test_uniaxial_strain_is_faster_along_stress(self):
        r = johnson_stress_anisotropy(**self.ARGS)
        assert r.vp3 > r.vp1

    def test_velocities_from_stiffness(self):
        r = johnson_stress_anisotropy(**self.ARGS)
        assert r.vp3 == pytest.approx(np.sqrt(r.c[2, 2] / self.ARGS["rho"]))
        assert r.vp1 == pytest.approx(np.sqrt(r.c[0, 0] / self.ARGS["rho"]))


class TestJohnsonMakse:
    ARGS = dict(mu=44e9, poisson=0.06, phi=0.36, epsilon=-1e-3, e3=-2e-3, rho=2650.0)

    def test_coordination_self_consistent(self):
        r = johnson_makse(**self.ARGS)
        mean_stress = (-r.sigma3 - 2 * r.sigma1) / 3
        assert r.n == pytest.approx(6.0 + (mean_stress / 6e4) ** (1 / 3), rel=1e-6)
        assert r.n > 6.0  # stress raises the coordination number

    def test_matches_johnson_at_fixed_coordination(self):
        # Feeding the converged n into the fixed-coordination model must
        # reproduce the same stiffness and stresses.
        r = johnson_makse(**self.ARGS)
        j = johnson_stress_anisotropy(
            mu=self.ARGS["mu"],
            poisson=self.ARGS["poisson"],
            n=r.n,
            phi=self.ARGS["phi"],
            epsilon=self.ARGS["epsilon"],
            e3=self.ARGS["e3"],
            rho=self.ARGS["rho"],
        )
        assert_allclose(r.c, j.c, rtol=1e-10)
        assert r.sigma3 == pytest.approx(j.sigma3, rel=1e-10)
        assert r.sigma1 == pytest.approx(j.sigma1, rel=1e-10)
        assert r.vp3 == pytest.approx(j.vp3, rel=1e-10)

    def test_higher_strain_raises_coordination(self):
        low = johnson_makse(**{**self.ARGS, "e3": -1e-3})
        high = johnson_makse(**{**self.ARGS, "e3": -4e-3})
        assert high.n > low.n
        assert high.sigma3 < low.sigma3  # more compressive

    def test_ti_symmetry(self):
        c = johnson_makse(**self.ARGS).c
        assert c[0, 0] == pytest.approx(c[1, 1])
        assert c[0, 1] == pytest.approx(c[0, 0] - 2 * c[5, 5])
        assert_allclose(c, c.T)

    def test_z0_floor(self):
        # With a huge stress scale the correction vanishes and n -> z0.
        r = johnson_makse(**self.ARGS, stress_scale=1e30)
        assert r.n == pytest.approx(6.0, abs=1e-6)
