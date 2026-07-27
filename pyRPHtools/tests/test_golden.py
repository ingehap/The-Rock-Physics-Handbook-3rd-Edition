"""Compare the port against values produced by the original MATLAB.

``tests/generate_golden.m`` runs the RPHtools ``.m`` files themselves under
GNU Octave and writes ``tests/golden/phase1.json``. The fixture is
committed, so these tests need no Octave; regenerate it with::

    octave --no-gui --quiet pyRPHtools/tests/generate_golden.m

Everything asserted here is a direct numerical comparison with the
original implementation. Where a MATLAB function could not be run at all,
or where the port deliberately differs, the generator records why and the
case is covered by the invariant tests in the other files instead.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import rphtools as rph

GOLDEN_PATH = Path(__file__).parent / "golden" / "phase1.json"
pytestmark = pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden fixtures not generated")


@pytest.fixture(scope="module")
def gold():
    with GOLDEN_PATH.open() as fh:
        return {k: np.asarray(v, dtype=float) for k, v in json.load(fh).items()}


# Tight by default: these are algebraic formulas, not iterative solvers.
RTOL = 1e-10


class TestModuliAndTensors:
    def test_ku2v(self, gold):
        assert_allclose(rph.moduli_to_velocity(37, 44, 2.65), gold["ku2v_quartz"], rtol=RTOL)

    def test_lm2v(self, gold):
        assert_allclose(
            rph.lame_to_velocity(37 - 2 * 44 / 3, 44, 2.65), gold["lm2v_quartz"], rtol=RTOL
        )

    def test_critpor(self, gold):
        r = rph.critical_porosity(6.008, 4.075, 2.65, 1.5, 0.5, 1.0, 0.4)
        assert_allclose(list(r), gold["critpor"], rtol=RTOL)

    def test_csiso(self, gold):
        c, s = rph.isotropic_cs(37, 44)
        assert_allclose(c, gold["csiso_c"], rtol=RTOL)
        assert_allclose(s, gold["csiso_s"], rtol=1e-9, atol=1e-14)

    def test_c2anis(self, gold):
        t = rph.thomsen_params(34.3, 22.7, 5.4, 10.6, 10.7)
        assert_allclose(list(t), gold["c2anis"], rtol=RTOL)

    def test_c2sti(self, gold):
        assert_allclose(list(rph.ti_c_to_s(34.3, 13.1, 10.7, 22.7, 5.4)), gold["c2sti"], rtol=RTOL)

    def test_c2vti(self, gold):
        v = rph.ti_velocities(34.3, 22.7, 5.4, 10.6, 10.7, 2.5, [0, 30, 45, 60, 90])
        assert_allclose(v.vp, gold["c2vti_vp"], rtol=RTOL)
        assert_allclose(v.vsh, gold["c2vti_vsh"], rtol=RTOL)
        assert_allclose(v.vsv, gold["c2vti_vsv"], rtol=RTOL)

    def test_cti2v(self, gold):
        c = rph.ti_voigt_matrix(34.3, 13.1, 10.7, 22.7, 5.4)
        r = rph.cti_to_velocities(c, 2.5)
        assert_allclose(list(r), gold["cti2v"], rtol=RTOL)

    def test_ezbond(self, gold):
        c = rph.ti_voigt_matrix(34.3, 13.1, 10.7, 22.7, 5.4)
        assert_allclose(rph.bond_rotation(c, 30), gold["ezbond_30"], rtol=RTOL, atol=1e-12)


class TestLayeredAndBounds:
    F, VP, VS, DEN = [0.6, 0.4], [3.0, 4.0], [1.5, 2.4], [2.4, 2.5]

    def test_bkus(self, gold):
        r = rph.backus_average(self.F, self.VP, self.VS, self.DEN)
        assert_allclose([r.vp0, r.vp45, r.vp90, r.vs0, r.vsh90], gold["bkus_vv"], rtol=RTOL)
        assert_allclose([r.c11, r.c33, r.c44, r.c66, r.c13], gold["bkus_cc"], rtol=RTOL)
        assert r.rho == pytest.approx(float(gold["bkus_rho"]), rel=RTOL)

    def test_bkusc(self, gold):
        c, rho = rph.backus_average_c(self.F, self.VP, self.VS, self.DEN)
        assert_allclose(c, gold["bkusc_c"], rtol=RTOL)
        assert rho == pytest.approx(float(gold["bkusc_rho"]), rel=RTOL)

    def test_bound_voigt_reuss(self, gold):
        b = rph.bounds([0.7, 0.3], [37, 2.2], [44, 3.0], method="voigt-reuss")
        assert_allclose(list(b), gold["bound_vr"], rtol=RTOL)

    def test_bound_hashin_shtrikman(self, gold):
        b = rph.bounds([0.7, 0.3], [37, 2.2], [44, 3.0], method="hs")
        assert_allclose(list(b), gold["bound_hs"], rtol=RTOL)

    def test_hash(self, gold):
        hs = rph.hashin_shtrikman(37, 44, 2.2, 0)
        expected = gold["hash"]
        assert_allclose(hs.k_upper, expected[:, 0], rtol=RTOL)
        assert_allclose(hs.k_lower, expected[:, 1], rtol=RTOL)
        assert_allclose(hs.mu_upper, expected[:, 2], rtol=RTOL)
        assert_allclose(hs.f2, expected[:, 4], rtol=RTOL)

    def test_hashv(self, gold):
        v = rph.hashin_shtrikman_velocity(6.008, 4.075, 2.65, 1.5, 0, 1.0)
        expected = gold["hashv"]
        assert_allclose(v.vp_upper, expected[:, 0], rtol=RTOL)
        assert_allclose(v.vp_lower, expected[:, 1], rtol=RTOL)
        assert_allclose(v.vs_upper, expected[:, 2], rtol=RTOL)


class TestFluids:
    def test_gassmnk(self, gold):
        assert rph.gassmann_k(12, 0.0, 2.5, 37, 0.25) == pytest.approx(
            float(gold["gassmnk"]), rel=RTOL
        )

    def test_gassmnv(self, gold):
        r = rph.gassmann_vel(3.5, 2.2, 2.3, 1.0, 2.5, 0.2, 0.05, 37, 0.25)
        assert_allclose([float(x) for x in r], gold["gassmnv"], rtol=RTOL)

    def test_brown_korringa_round_trip(self, gold):
        s = rph.isotropic_cs(12, 14).s
        sat = rph.brown_korringa_dry_to_sat(s, 37, 44, 2.5, 0.25)
        assert_allclose(sat, gold["bkd2s"], rtol=1e-9, atol=1e-15)
        dry = rph.brown_korringa_sat_to_dry(sat, 37, 44, 2.5, 0.25)
        assert_allclose(dry, gold["bks2d"], rtol=1e-9, atol=1e-15)

    def test_bkti(self, gold):
        s = rph.isotropic_cs(12, 14).s
        s_min = rph.isotropic_cs(37, 44).s

        def pack(m):
            return [m[0, 0], m[0, 1], m[0, 2], m[2, 2], m[3, 3]]

        out = rph.brown_korringa_ti(pack(s), pack(s_min), 2.5, 0.25)
        assert_allclose(out, gold["bkti"], rtol=1e-9)

    def test_mmti(self, gold):
        out = rph.squirt_ti(
            [0.036, -0.007, -0.006, 0.040, 0.13], [0.030, -0.008, -0.007, 0.033, 0.11]
        )
        assert_allclose(out, gold["mmti"], rtol=RTOL)

    def test_biothf(self, gold):
        r = rph.biot_hf(3200, 2000, 37e9, 2650, 1000, 2.25e9, 0.25, 2)
        assert_allclose([float(x) for x in r], gold["biothf"], rtol=RTOL)

    def test_biothfb(self, gold):
        r = rph.biot_hf_geertsma_smit(3200, 2000, 37e9, 2650, 1000, 2.25e9, 0.25, 2)
        assert_allclose([float(x) for x in r], gold["biothfb"], rtol=RTOL)

    def test_biot_dispersion(self, gold):
        expected = gold["biot"]
        d = rph.biot_dispersion(
            3200,
            2000,
            37e9,
            2650,
            1000,
            2.25e9,
            1e-3,
            0.25,
            1e-12,
            1e-5,
            2,
            freq=expected[:, 1],
        )
        assert_allclose(d.vp1, expected[:, 0], rtol=1e-9)
        assert_allclose(d.vp2, expected[:, 2], rtol=1e-9)
        assert_allclose(d.vs, expected[:, 3], rtol=1e-9)
        # 1/Q is Im/Re of a nearly real quantity, so it keeps far fewer
        # significant digits than the velocities do (which match to 1e-13).
        assert_allclose(d.q1_inv, expected[:, 4], rtol=1e-6, atol=1e-14)
        assert_allclose(d.q2_inv, expected[:, 5], rtol=1e-6, atol=1e-14)
        assert_allclose(d.qs_inv, expected[:, 6], rtol=1e-6, atol=1e-14)

    def test_patchw(self, gold):
        expected = gold["patchw"]
        r = rph.white_patchy(
            12e9,
            14e9,
            37e9,
            2650,
            0.25,
            1e-12,
            (0.05e9, 200, 2e-5),
            (2.25e9, 1000, 1e-3),
            0.3,
            0.1,
            np.logspace(-2, 4, 20),
        )
        assert_allclose(r.vp, expected[:, 0], rtol=1e-9)
        assert_allclose(np.real(r.k), expected[:, 1], rtol=1e-9)
        # Im(K) is ~1e-6 of Re(K) here, and attenuation is tan(theta/2)
        # with theta tiny, so both carry roughly six fewer significant
        # digits than vp and Re(K) above, which match to ~1e-14.
        assert_allclose(np.imag(r.k), expected[:, 2], rtol=1e-6)
        assert_allclose(r.attenuation, expected[:, 3], rtol=1e-6)
        assert_allclose([r.k_inf, r.k_lf], gold["patchw_lims"], rtol=1e-9)

    def test_flprop(self, gold):
        r = rph.batzle_wang(
            pressure=30,
            temperature=80,
            salinity=35000,
            oil_api=30,
            gas_gravity=0.6,
            gor=100,
            gas_index_brine=0,
            s_oil=0.3,
            s_gas=0.2,
        )
        assert_allclose([float(x) for x in r], gold["flprop"], rtol=1e-9)

    def test_co2prop(self, gold):
        k, rho, vp = rph.co2_properties(60, 15)
        assert_allclose([float(k), float(rho), float(vp)], gold["co2prop"], rtol=1e-9)


class TestEffectiveMediumAndCracks:
    def test_berryscm(self, gold):
        k, mu = rph.berryman_scm([37, 2.2], [44, 0], [1, 0.1], [0.7, 0.3])
        assert_allclose([k, mu], gold["berryscm"], rtol=1e-6)

    def test_berrysc_sweep(self, gold):
        # berrysc.m sweeps x1 (the fraction of phase 1) upward and reports
        # por = 1 - x1, so its rows run from pure phase 2 to pure phase 1.
        # The port returns rows in ascending f2, so compare reversed.
        expected = gold["berrysc"][::-1]
        curves = rph.berryman_sc(37, 44, 2.2, 0, 1, 0.1)
        assert_allclose(curves.f2, expected[:, 2], rtol=1e-6)
        assert_allclose(curves.k, expected[:, 0], rtol=1e-6)
        assert_allclose(curves.mu, expected[:, 1], rtol=1e-6)

    def test_berryscp(self, gold):
        k, mu = rph.berryman_sc_pressure(
            [37, 2.2, 2.2], [44, 0, 0], [1, 0.01, 0.5], [0.8, 0.05, 0.15], [0, 0.05, 0.2]
        )
        assert_allclose(k, gold["berryscp"][:, 0], rtol=1e-6)
        assert_allclose(mu, gold["berryscp"][:, 1], rtol=1e-6)

    def test_dem_curve(self, gold):
        # ode45m used adaptive steps; compare on the MATLAB's own porosity
        # samples (interior points, away from the endpoint singularity).
        expected = gold["dem"]
        phi = expected[:, 2]
        keep = (phi > 0) & (phi < 0.98)
        r = rph.dem(37, 44, 2.2, 0, 0.1, phi=phi[keep])
        assert_allclose(r.k, expected[keep, 0], rtol=1e-5)
        assert_allclose(r.mu, expected[keep, 1], rtol=1e-5)

    def test_dem1(self, gold):
        k, mu = rph.dem_at_fraction(37, 44, 2.2, 0, 0.2, 0.35)
        assert_allclose([k, mu], gold["dem1"], rtol=1e-4)

    def test_hudson(self, gold):
        c, den = rph.hudson(0.05, 0.01, 2.25, 1.0, 37, 44, 2.65, axis=3)
        assert_allclose(c, gold["hudson"], rtol=RTOL)
        assert den == pytest.approx(float(gold["hudson_den"]), rel=RTOL)

    def test_hudson1(self, gold):
        r = rph.hudson_velocities(0.05, 0.01, 2.25, 37, 44, 2.6, axis=3)
        got = [r.vp0, r.vs0, r.epsilon, r.gamma, r.delta]
        assert_allclose([float(x) for x in got], gold["hudson1"], rtol=RTOL)

    def test_hudson3(self, gold):
        r = rph.hudson3([0.03, 0.02, 0.01], [0.01, 0.01, 0.01], 2.25, 1.0, 37, 44, 2.65)
        assert_allclose(r.c, gold["hudson3"], rtol=RTOL)
        assert r.rho == pytest.approx(float(gold["hudson3_den"]), rel=RTOL)

    def test_hudsoncone(self, gold):
        r = rph.hudson_cone(0.05, 0.01, 2.25, 37, 44, 2.65, 30.0, axis=3)
        assert_allclose(r.c, gold["hudsoncone"], rtol=RTOL)

    def test_echeng(self, gold):
        r = rph.eshelby_cheng(66.67, 7.67, 66.67, 44, 44, 0.02, 0.1, 2.25)
        assert_allclose(list(r), gold["echeng"], rtol=RTOL)

    def test_hudson_fisher_differs_by_the_documented_fixes(self, gold):
        # hudsonF.m has two bugs the port fixes, so its raw output must
        # NOT match — assert the disagreement is real and in the shear
        # terms and density, exactly as documented.
        c, den = rph.hudson_fisher(0.05, 0.01, 2.25, 1.0, 37, 44, 2.65, 0.4)
        raw_c = gold["hudsonF_raw"]
        assert not np.allclose(c[3, 3], raw_c[3, 3])
        assert c[5, 5] == pytest.approx((c[0, 0] - c[0, 1]) / 2)  # TI identity restored
        assert den != pytest.approx(float(gold["hudsonF_raw_den"]))


class TestGranularAndPermeability:
    def test_hertzmind(self, gold):
        expected = gold["hertzmind"]
        r = rph.hertz_mindlin(37, 44, 0.02, [0.3, 0.36, 0.4])
        assert_allclose(r.k, expected[:, 0], rtol=1e-9)
        assert_allclose(r.g, expected[:, 1], rtol=1e-9)
        assert_allclose(r.coord, expected[:, 3], rtol=1e-9)

    def test_hertzmindv(self, gold):
        expected = gold["hertzmindv"]
        r = rph.hertz_mindlin_v(6.008, 4.075, 2.65, 0.02, [0.3, 0.36, 0.4])
        assert_allclose(r.vp, expected[:, 0], rtol=1e-9)
        assert_allclose(r.vs, expected[:, 1], rtol=1e-9)
        assert_allclose(r.rho, expected[:, 2], rtol=1e-9)

    def test_johnson_stresses_and_velocities(self, gold):
        # Johnson.m's 5th output is the scalar contact constant, not the
        # tensor, so only these four are comparable.
        r = rph.johnson_stress_anisotropy(
            mu=44,
            poisson=0.06,
            n=9,
            phi=0.36,
            epsilon=-1e-3,
            e3=-2e-3,
            rho=2650,
            cn=4 * 44 / (1 - 0.06),
        )
        got = [float(r.vp1), float(r.vp3), float(r.sigma1), float(r.sigma3)]
        assert_allclose(got, gold["Johnson"], rtol=1e-9)

    def test_contact_cement_differs_only_by_the_pi_approximation(self, gold):
        # Cem.m hard-codes 3.14 for pi; the port uses pi. The plan says
        # the difference is under 0.1% -- verify that claim.
        raw = gold["Cem_raw"]
        r = rph.contact_cement(
            phi_c=0.38,
            coord=8.5,
            g_grain=45,
            nu_grain=0.064,
            g_cement=45,
            nu_cement=0.064,
            k_fluid=0.0,
            scheme=2,
        )
        assert_allclose(r.phi, raw[:, 0], rtol=1e-12)
        rel = np.abs(r.m_sat[1:] - raw[1:, 1]) / np.abs(raw[1:, 1])
        assert rel.max() < 1e-3, f"pi vs 3.14 difference {rel.max():.2e} exceeds 0.1%"
        assert_allclose(r.m_sat, raw[:, 1], rtol=2e-3)
        assert_allclose(r.g_frame, raw[:, 2], rtol=2e-3)

    PERM_CASES = [
        ("KozCarmE", lambda p: rph.kozeny_carman_perm(p, 250)),
        ("FredrichE", lambda p: rph.fredrich_perm(p, 100)),
        ("PandaLakeKCE", lambda p: rph.panda_lake_kc_perm(p, 250)),
        ("ModKozCarm", lambda p: rph.modified_kozeny_carman_perm(p, 60, 2, 0.02)),
        ("CoatDum", lambda p: rph.coates_dumanoir_perm(p, 0.15)),
        ("Coates", lambda p: rph.coates_perm(p, 0.15)),
        ("PandaLake", lambda p: rph.panda_lake_perm(p, 2, 0.25, 650, 0.4)),
    ]

    @pytest.mark.parametrize("name,fn", PERM_CASES)
    def test_permeability_models(self, gold, name, fn):
        phi = np.array([0.05, 0.1, 0.2, 0.3])
        # MATLAB returned [Phi K] concatenated; the second half is K.
        expected = gold[name].reshape(-1)[phi.size :]
        assert_allclose(fn(phi), expected, rtol=RTOL)

    def test_bloch(self, gold):
        r = rph.bloch_perm(1.2, 2.0, 10)
        # MATLAB returned porosity in percent; the port returns a fraction.
        assert float(r.phi) * 100 == pytest.approx(gold["Bloch"][0], rel=RTOL)
        assert float(r.k) == pytest.approx(gold["Bloch"][1], rel=RTOL)


class TestAVO:
    IFACE = dict(vp1=2.6, vs1=1.2, rho1=2.3, vp2=2.2, vs2=1.35, rho2=2.05)
    PP = ["zoeppritz", "aki-richards", "shuey", "shuey-castagna"]
    PS = [
        "zoeppritz",
        "aki-richards",
        "donati-quadratic",
        "donati-linear",
        "simplified",
        "gonzalez",
        "alejandro-reinaldo",
    ]

    @pytest.mark.parametrize("index,method", list(enumerate(PP, start=1)))
    def test_avopp(self, gold, index, method):
        got = rph.avo_pp(**self.IFACE, angles_deg=[0, 12, 28], method=method)
        assert_allclose(np.real(got), gold[f"avopp{index}"], rtol=1e-9)

    @pytest.mark.parametrize("index,method", list(enumerate(PS, start=1)))
    def test_avops(self, gold, index, method):
        got = rph.avo_ps(**self.IFACE, angles_deg=[5, 18, 30], method=method)
        assert_allclose(np.real(got), gold[f"avops{index}"], rtol=1e-9)

    def test_avo_abe(self, gold):
        att = rph.avo_attributes(**self.IFACE)
        assert_allclose([float(x) for x in att], gold["avo_abe"], rtol=1e-9)

    LOG = (
        np.array([2.6, 2.8, 2.2, 3.0, 2.7]),
        np.array([1.2, 1.4, 1.35, 1.5, 1.3]),
        np.array([2.3, 2.35, 2.05, 2.4, 2.32]),
    )

    def test_eimp_reflection_angle(self, gold):
        r = rph.elastic_impedance(*self.LOG, 15, angle="reflection")
        expected = gold["eimp"]
        for k, field in enumerate(["ipp_n", "ips_n", "isp_n", "ipp", "ips", "isp"]):
            assert_allclose(np.real(getattr(r, field)), expected[:, k], rtol=1e-9)

    def test_eimp2_incidence_angle(self, gold):
        k = float(np.mean(self.LOG[1] / self.LOG[0]))
        r = rph.elastic_impedance(*self.LOG, 20, k=k, angle="incidence")
        expected = gold["eimp2"]
        for idx, field in enumerate(["ipp_n", "ips_n", "isp_n", "ipp", "ips", "isp"]):
            assert_allclose(np.real(getattr(r, field)), expected[:, idx], rtol=1e-9)


class TestSeismicAndSignal:
    WAVELET = (1 - 2 * (np.pi * 30 * ((np.arange(128) - 63.5) * 0.001)) ** 2) * np.exp(
        -((np.pi * 30 * ((np.arange(128) - 63.5) * 0.001)) ** 2)
    )
    LYR2 = np.array([[2000.0, 2000.0, 80.0], [2600.0, 2300.0, 90.0]])
    LYR3 = np.array([[2000.0, 2000.0, 40.0], [3200.0, 2500.0, 15.0], [2400.0, 2150.0, 60.0]])

    def test_kennett_all_multiples(self, gold):
        r = rph.kennett(self.LYR2, self.WAVELET, 0.001, multiples="all")
        assert_allclose(r.wz, gold["kennet_wz"], rtol=1e-9, atol=1e-14)
        assert_allclose(r.pz, gold["kennet_pz"], rtol=1e-9, atol=1e-14)
        tf = gold["kennet_tf"]
        assert_allclose(r.freq, tf[:, 0], rtol=1e-9)

    def test_kennett_primaries(self, gold):
        r = rph.kennett(self.LYR3, self.WAVELET, 0.001, multiples="primaries")
        assert_allclose(r.wz, gold["kennet_prim_wz"], rtol=1e-9, atol=1e-14)
        assert_allclose(r.pz, gold["kennet_prim_pz"], rtol=1e-9, atol=1e-14)

    def test_propagator(self, gold):
        r = rph.propagator_seis(self.LYR2, self.WAVELET, 0.001)
        assert_allclose(r.pz, gold["pgator_pz"], rtol=1e-9, atol=1e-14)
        assert_allclose(r.wz, gold["pgator_wz"], rtol=1e-9, atol=1e-14)

    def test_kenfdisp(self, gold):
        expected = gold["kenfdisp"]
        lyr = np.array([[2000.0, 2000.0, 5.0], [3000.0, 2400.0, 5.0]] * 200)
        _, vel = rph.kennett_frazer_dispersion(lyr, expected[:, 0])
        assert_allclose(vel, expected[:, 1], rtol=1e-9)

    def test_kenfrtt(self, gold):
        expected = gold["kenfrtt"]
        r = rph.kennett_frazer_traveltimes(self.LYR3, 30.0)
        assert_allclose(r.tt, expected[:, 0], rtol=1e-9)
        assert_allclose(r.rt, expected[:, 1], rtol=1e-9)
        assert_allclose(r.emtt, expected[:, 2], rtol=1e-9)

    DATA = np.array([1.0, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2])

    def test_blockav(self, gold):
        assert_allclose(rph.block_average(self.DATA, 4), gold["blockav"], rtol=RTOL)

    def test_fftplot(self, gold):
        s = rph.spectrum(self.DATA, 0.004)
        assert_allclose(s.amplitude, gold["fftplot_amp"], rtol=1e-9)
        assert_allclose(s.phase, gold["fftplot_phase"], rtol=1e-9, atol=1e-12)
        # The port returns the frequency axis; MATLAB returned the step.
        assert s.freq[1] - s.freq[0] == pytest.approx(float(gold["fftplot_step"]), rel=RTOL)

    def test_iatrib(self, gold):
        r = rph.instantaneous_attributes(self.DATA)
        assert_allclose(r.amplitude, gold["iatrib_amp"], rtol=1e-9)
        assert_allclose(r.phase, gold["iatrib_phi"], rtol=1e-9, atol=1e-12)
        assert_allclose(r.frequency, gold["iatrib_freq"], rtol=1e-9, atol=1e-12)


class TestStats:
    D2 = np.array([[0.1, 2.0], [0.4, 2.3], [0.2, 2.1], [0.35, 2.25], [0.15, 2.05], [0.3, 2.2]])

    def test_hist2d_equal_bins(self, gold):
        h = rph.hist2d(self.D2, 4, 3)
        assert_allclose(h.counts, gold["hist2d_counts"], rtol=RTOL)
        assert_allclose(h.centres1, gold["hist2d_c1"], rtol=RTOL)
        assert_allclose(h.centres2, gold["hist2d_c2"], rtol=RTOL)

    def test_hist2d_centre_bins(self, gold):
        h = rph.hist2d(self.D2, [0.1, 0.2, 0.3, 0.4], [2.0, 2.15, 2.3])
        assert_allclose(h.counts, gold["hist2d_centres_counts"], rtol=RTOL)

    def test_hist3d(self, gold):
        d3 = np.column_stack([self.D2, [1.0, 1.4, 1.1, 1.35, 1.05, 1.3]])
        h = rph.hist3d(d3, 3)
        # MATLAB flattens column-major.
        assert_allclose(h.counts.ravel(order="F"), gold["hist3d_counts"], rtol=RTOL)
