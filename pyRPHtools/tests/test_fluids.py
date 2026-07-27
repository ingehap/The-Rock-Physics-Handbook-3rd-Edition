import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    biot_dispersion,
    biot_hf,
    biot_hf_geertsma_smit,
    brown_korringa_c,
    brown_korringa_dry_to_sat,
    brown_korringa_s,
    brown_korringa_sat_to_dry,
    brown_korringa_ti,
    gassmann_k,
    gassmann_vel,
    isotropic_cs,
    squirt_ti,
    ti_voigt_matrix,
    white_patchy,
)

# Dry sandstone frame + fluids (GPa, g/cc, km/s).
K_DRY, MU_DRY, K_MIN, MU_MIN, PHI = 12.0, 14.0, 37.0, 44.0, 0.25
K_WATER, RHO_WATER = 2.5, 1.0
K_GAS, RHO_GAS = 0.05, 0.2


def gassmann_classic(k_dry, k_fl, k_min, phi):
    """Independent textbook form: Ksat = Kdry + (1-Kdry/K0)^2 / (...)."""
    num = (1 - k_dry / k_min) ** 2
    den = phi / k_fl + (1 - phi) / k_min - k_dry / k_min**2
    return k_dry + num / den


class TestGassmannK:
    def test_matches_textbook_form(self):
        k_sat = gassmann_k(K_DRY, 0.0, K_WATER, K_MIN, PHI)
        assert k_sat == pytest.approx(gassmann_classic(K_DRY, K_WATER, K_MIN, PHI))

    def test_same_fluid_identity(self):
        assert gassmann_k(15.0, K_WATER, K_WATER, K_MIN, PHI) == pytest.approx(15.0)

    def test_zero_porosity_passthrough(self):
        assert gassmann_k(K_MIN, 0.0, K_WATER, K_MIN, 0.0) == pytest.approx(K_MIN)

    def test_round_trip(self):
        k_w = gassmann_k(K_DRY, 0.0, K_WATER, K_MIN, PHI)
        k_g = gassmann_k(k_w, K_WATER, K_GAS, K_MIN, PHI)
        k_back = gassmann_k(k_g, K_GAS, K_WATER, K_MIN, PHI)
        assert k_back == pytest.approx(k_w, rel=1e-12)
        assert k_g == pytest.approx(gassmann_classic(K_DRY, K_GAS, K_MIN, PHI))

    def test_stiffer_fluid_stiffer_rock(self):
        k_w = gassmann_k(K_DRY, 0.0, K_WATER, K_MIN, PHI)
        k_g = gassmann_k(K_DRY, 0.0, K_GAS, K_MIN, PHI)
        assert K_DRY < k_g < k_w < K_MIN


class TestGassmannVel:
    def test_consistency_with_gassmann_k(self):
        rho_dry = (1 - PHI) * 2.65
        rho1 = rho_dry + PHI * RHO_WATER
        k1 = gassmann_k(K_DRY, 0.0, K_WATER, K_MIN, PHI)
        vp1 = np.sqrt((k1 + 4 / 3 * MU_DRY) / rho1)
        vs1 = np.sqrt(MU_DRY / rho1)

        vp2, vs2, rho2, k2 = gassmann_vel(
            vp1, vs1, rho1, RHO_WATER, K_WATER, RHO_GAS, K_GAS, K_MIN, PHI
        )
        assert rho2 == pytest.approx(rho1 - PHI * RHO_WATER + PHI * RHO_GAS)
        assert k2 == pytest.approx(gassmann_k(k1, K_WATER, K_GAS, K_MIN, PHI))
        # Shear modulus is unchanged; vs changes only through density.
        assert vs2 == pytest.approx(np.sqrt(MU_DRY / rho2))
        assert vp2 == pytest.approx(np.sqrt((k2 + 4 / 3 * MU_DRY) / rho2))


class TestBrownKorringa:
    S_DRY = isotropic_cs(K_DRY, MU_DRY).s

    def test_isotropic_reduces_to_gassmann(self):
        s_sat = brown_korringa_dry_to_sat(self.S_DRY, K_MIN, MU_MIN, K_WATER, PHI)
        c_sat = np.linalg.inv(s_sat)
        k_sat = c_sat[0, 0] - 4 / 3 * c_sat[3, 3]
        assert k_sat == pytest.approx(gassmann_classic(K_DRY, K_WATER, K_MIN, PHI))
        # Shear is unchanged by fluid substitution.
        assert c_sat[3, 3] == pytest.approx(MU_DRY)

    def test_sat_dry_round_trip(self):
        s_sat = brown_korringa_dry_to_sat(self.S_DRY, K_MIN, MU_MIN, K_WATER, PHI)
        s_dry = brown_korringa_sat_to_dry(s_sat, K_MIN, MU_MIN, K_WATER, PHI)
        assert_allclose(s_dry, self.S_DRY, rtol=1e-10, atol=1e-14)

    def test_same_fluid_identity(self):
        s_sat = brown_korringa_dry_to_sat(self.S_DRY, K_MIN, MU_MIN, K_WATER, PHI)
        s_same = brown_korringa_s(s_sat, K_MIN, MU_MIN, K_WATER, K_WATER, PHI)
        assert_allclose(s_same, s_sat, rtol=1e-10, atol=1e-14)

    def test_c_and_s_domains_agree(self):
        s_sat1 = brown_korringa_dry_to_sat(self.S_DRY, K_MIN, MU_MIN, K_WATER, PHI)
        c_sat1 = np.linalg.inv(s_sat1)
        c_sat2 = brown_korringa_c(c_sat1, K_MIN, MU_MIN, K_WATER, K_GAS, PHI)
        s_sat2 = brown_korringa_s(s_sat1, K_MIN, MU_MIN, K_WATER, K_GAS, PHI)
        assert_allclose(c_sat2, np.linalg.inv(s_sat2), rtol=1e-10)

    def test_anisotropic_shear_columns_untouched(self):
        # A TI dry rock: fluid substitution must not change s44, s55, s66.
        c_dry = ti_voigt_matrix(30.0, 10.0, 8.0, 24.0, 6.0)
        s_dry = np.linalg.inv(c_dry)
        s_sat = brown_korringa_dry_to_sat(s_dry, K_MIN, MU_MIN, K_WATER, PHI)
        assert_allclose(s_sat[3:, 3:], s_dry[3:, 3:], rtol=1e-12)


class TestBrownKorringaTI:
    def test_matches_6x6_route(self):
        # TI dry rock, isotropic mineral: the 5-constant specialization must
        # agree with the general 6x6 dry-to-sat substitution.
        c_dry = ti_voigt_matrix(30.0, 10.0, 8.0, 24.0, 6.0)
        s_dry6 = np.linalg.inv(c_dry)
        s_min6 = isotropic_cs(K_MIN, MU_MIN).s

        def pack(s):
            return np.array([s[0, 0], s[0, 1], s[0, 2], s[2, 2], s[3, 3]])

        s_sat5 = brown_korringa_ti(pack(s_dry6), pack(s_min6), K_WATER, PHI)
        s_sat6 = brown_korringa_dry_to_sat(s_dry6, K_MIN, MU_MIN, K_WATER, PHI)
        assert_allclose(s_sat5, pack(s_sat6), rtol=1e-10)

    def test_s44_unchanged(self):
        s_dry = np.array([0.04, -0.01, -0.008, 0.05, 0.16])
        s_min = np.array([0.03, -0.009, -0.009, 0.03, 0.0227])
        s_sat = brown_korringa_ti(s_dry, s_min, K_WATER, PHI)
        assert s_sat[4] == pytest.approx(s_dry[4])


class TestSquirtTI:
    def test_matches_matlab_transliteration(self):
        rng = np.random.default_rng(7)
        s_hp = np.array([0.030, -0.008, -0.007, 0.033, 0.11])
        s_dry = s_hp + rng.uniform(0.001, 0.01, size=(4, 5))

        # Fresh transliteration of mmti.m
        dss = s_dry - s_hp
        dsaabb = 2 * (dss[:, 0] + dss[:, 1] + 2 * dss[:, 2]) + dss[:, 3]
        dsabab = 2 * dss[:, 0] + dss[:, 3] + 4 * dss[:, 4] + 4 * (dss[:, 0] - dss[:, 1])
        a = (dsabab / dsaabb - 1) / 4
        tdss = dss / dsaabb[:, None]
        b = 1 - 4 * a
        g1 = tdss[:, 0] - (4 * a / b) * (tdss[:, 1] + tdss[:, 2])
        g2 = tdss[:, 1] / b
        g3 = tdss[:, 2] / b
        g4 = tdss[:, 3] - (8 * a / b) * tdss[:, 2]
        g5 = tdss[:, 4] / b - (tdss[:, 0] + tdss[:, 3]) / b / 4 + (g1 + g4) / 4
        gg = np.stack([g1, g2, g3, g4, g5], axis=-1)
        expected = s_dry - dsaabb[:, None] * gg

        assert_allclose(squirt_ti(s_dry, s_hp), expected, rtol=1e-12)

    def test_wet_frame_stiffer(self):
        # Liquid-stiffened cracks: the unrelaxed wet frame is less compliant
        # (smaller s11, s33) than the dry frame.
        s_hp = np.array([0.030, -0.008, -0.007, 0.033, 0.11])
        s_dry = s_hp + np.array([0.006, 0.001, 0.001, 0.007, 0.02])
        s_wet = squirt_ti(s_dry, s_hp)
        assert s_wet[0] < s_dry[0]
        assert s_wet[3] < s_dry[3]


class TestBiot:
    # Water-saturated sandstone, SI units.
    ARGS = dict(
        vp_dry=3200.0,
        vs_dry=2000.0,
        k_min=37e9,
        rho_min=2650.0,
        rho_fl=1000.0,
        k_fl=2.25e9,
        phi=0.25,
        tortuosity=2.0,
    )
    ETA, PERM, PORE = 1e-3, 1e-12, 1e-5

    def test_biothfgs_oracle_is_root_sum_of_squares(self):
        # biothfgs.m (transliterated below) computes the SUM of the two
        # roots of the Biot high-frequency quadratic, i.e.
        # sqrt(vp1^2 + vp2^2) -- an approximation to vp1 that neglects the
        # slow wave, not a re-derivation of it. Verify that exact identity
        # against biot_hf, plus the identical shear velocity.
        a = self.ARGS
        por, alfa = a["phi"], a["tortuosity"]
        ro0, rofl = a["rho_min"], a["rho_fl"]
        rodry = (1 - por) * ro0
        mudry = rodry * a["vs_dry"] ** 2
        kdry = rodry * a["vp_dry"] ** 2 - 4 / 3 * mudry
        b = kdry / a["k_min"]
        robiot = ro0 * (1 - por) + por * rofl * (1 - 1 / alfa)
        ro12 = (1 - alfa) * por * rofl
        ro11 = (1 - por) * ro0 - ro12
        ro22 = por * rofl * alfa
        ro = ro11 + 2 * ro12 + ro22
        rol = (ro12 + ro22) / por
        roc = ro22 / por**2
        den = (1 - por - b) / a["k_min"] + por / a["k_fl"]
        h = (1 - b) ** 2 / den + kdry + 4 / 3 * mudry
        k = (1 - b) / den
        ell = 1 / den
        vp1_gs = np.sqrt((ell * ro + h * roc - 2 * rol * k) / (ro * roc - rol**2))
        vs_gs = np.sqrt(mudry / robiot)

        vp1, vp2, vs = biot_hf(**self.ARGS)
        assert np.sqrt(vp1**2 + vp2**2) == pytest.approx(vp1_gs, rel=1e-12)
        assert vs == pytest.approx(vs_gs, rel=1e-12)

    def test_geertsma_smit_overestimates(self):
        vp1, vp2, vs = biot_hf(**self.ARGS)
        vp1_b, vs_b = biot_hf_geertsma_smit(**self.ARGS)
        assert vp1_b >= vp1
        assert vs_b == pytest.approx(vs)

    def test_dispersion_limits(self):
        freq = np.logspace(-2, 8, 200)
        d = biot_dispersion(
            **self.ARGS,
            eta=self.ETA,
            perm=self.PERM,
            pore_size=self.PORE,
            freq=freq,
        )
        vp1_hf, vp2_hf, vs_hf = biot_hf(**self.ARGS)

        # High-frequency limit approaches Johnson-Plona.
        assert d.vp1[-1] == pytest.approx(vp1_hf, rel=1e-3)
        assert d.vs[-1] == pytest.approx(vs_hf, rel=1e-3)
        assert d.vp2[-1] == pytest.approx(vp2_hf, rel=2e-2)

        # Low-frequency limit approaches Gassmann (Biot's theorem).
        a = self.ARGS
        rodry = (1 - a["phi"]) * a["rho_min"]
        mudry = rodry * a["vs_dry"] ** 2
        kdry = rodry * a["vp_dry"] ** 2 - 4 / 3 * mudry
        k_sat = gassmann_k(kdry, 0.0, a["k_fl"], a["k_min"], a["phi"])
        rho = rodry + a["phi"] * a["rho_fl"]
        vp_gassmann = np.sqrt((k_sat + 4 / 3 * mudry) / rho)
        assert d.vp1[0] == pytest.approx(vp_gassmann, rel=1e-3)

        # Dispersion is monotone overall and attenuation is non-negative.
        assert d.vp1[-1] > d.vp1[0]
        assert np.all(d.q1_inv >= -1e-12)

    def test_dispersion_vectorized_over_freq(self):
        freq = np.logspace(0, 6, 50)
        d = biot_dispersion(
            **self.ARGS, eta=self.ETA, perm=self.PERM, pore_size=self.PORE, freq=freq
        )
        assert d.vp1.shape == freq.shape
        assert np.all(np.isfinite(d.vp1))


class TestWhitePatchy:
    # SI units: gas patch in a brine-saturated sandstone.
    ARGS = dict(
        k_dry=12e9,
        mu_dry=14e9,
        k_min=37e9,
        rho_min=2650.0,
        phi=0.25,
        perm=1e-12,
        fluid1=(0.05e9, 200.0, 2e-5),  # gas: (K, rho, eta)
        fluid2=(2.25e9, 1000.0, 1e-3),  # brine
        sg1=0.3,
        radius=0.1,
    )

    def test_limits_bracket_dispersion(self):
        freq = np.logspace(-4, 6, 300)
        r = white_patchy(**self.ARGS, freq=freq)
        assert r.k_lf < r.k_inf
        # Real part of K approaches the limits at the band edges.
        assert np.real(r.k[0]) == pytest.approx(r.k_lf, rel=1e-2)
        assert np.real(r.k[-1]) == pytest.approx(r.k_inf, rel=1e-2)
        # Non-decreasing dispersion, up to ~1e-5 m/s float noise at the
        # extreme low-frequency end.
        assert np.all(np.diff(r.vp) >= -1e-4)

    def test_low_frequency_limit_is_reuss_of_gassmann(self):
        # White's KLF is the Reuss average of the two Gassmann-saturated
        # bulk moduli weighted by saturation (Wood/BGW-type limit).
        a = self.ARGS
        k1 = gassmann_k(a["k_dry"], 0.0, a["fluid1"][0], a["k_min"], a["phi"])
        k2 = gassmann_k(a["k_dry"], 0.0, a["fluid2"][0], a["k_min"], a["phi"])
        r = white_patchy(**a, freq=np.array([1.0]))
        # KLF from the MATLAB formula:
        sg = a["sg1"]
        klf = (k2 * (k1 - a["k_dry"]) + sg * a["k_dry"] * (k2 - k1)) / (
            (k1 - a["k_dry"]) + sg * (k2 - k1)
        )
        assert r.k_lf == pytest.approx(klf)
        assert k1 < r.k_lf < k2

    def test_saturation_ordering(self):
        # More gas (fluid 1 is softer) -> softer high-frequency modulus.
        freq = np.array([100.0])
        r30 = white_patchy(**{**self.ARGS, "sg1": 0.3}, freq=freq)
        r70 = white_patchy(**{**self.ARGS, "sg1": 0.7}, freq=freq)
        assert r70.k_inf < r30.k_inf

    def test_attenuation_positive_at_transition(self):
        freq = np.logspace(-2, 4, 100)
        r = white_patchy(**self.ARGS, freq=freq)
        assert np.max(r.attenuation) > 0
