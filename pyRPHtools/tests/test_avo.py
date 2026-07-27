import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import avo_attributes, avo_pp, avo_ps, elastic_impedance
from rphtools.avo import PP_METHODS, PS_METHODS

# Shale over gas sand (km/s, g/cc) — a classic class-III AVO interface.
TOP = dict(vp1=2.6, vs1=1.2, rho1=2.3)
BOT = dict(vp2=2.2, vs2=1.35, rho2=2.05)
IFACE = {**TOP, **BOT}
ANG = np.arange(0, 41, 2.0)


class TestAVOPPZoeppritz:
    def test_normal_incidence_is_acoustic(self):
        r = avo_pp(**IFACE, angles_deg=[0.0])
        ip1 = TOP["vp1"] * TOP["rho1"]
        ip2 = BOT["vp2"] * BOT["rho2"]
        assert r[0] == pytest.approx((ip2 - ip1) / (ip2 + ip1))

    def test_linearizations_share_one_intercept(self):
        # The three linearized methods all reduce to Shuey's R0 at 0 deg.
        r0 = avo_pp(**IFACE, angles_deg=[0.0], method="shuey")[0]
        for method in ("aki-richards", "shuey-castagna"):
            r = avo_pp(**IFACE, angles_deg=[0.0], method=method)
            assert r[0] == pytest.approx(r0, rel=1e-12), method

    def test_intercept_approximates_exact_normal_incidence(self):
        # R0 is the linearization of the exact acoustic coefficient, so
        # they agree only to first order in the contrasts.
        exact = avo_pp(**IFACE, angles_deg=[0.0], method="zoeppritz")[0]
        r0 = avo_pp(**IFACE, angles_deg=[0.0], method="shuey")[0]
        assert r0 == pytest.approx(exact, abs=1e-3)
        assert r0 != pytest.approx(exact, abs=1e-9)

    def test_no_contrast_gives_zero(self):
        same = dict(vp1=2.6, vs1=1.2, rho1=2.3, vp2=2.6, vs2=1.2, rho2=2.3)
        for method in PP_METHODS:
            r = avo_pp(**same, angles_deg=ANG, method=method)
            assert_allclose(np.real(r), 0.0, atol=1e-12, err_msg=method)

    def test_approximations_track_zoeppritz_at_small_angles(self):
        exact = avo_pp(**IFACE, angles_deg=[0, 5, 10, 15], method="zoeppritz")
        for method in ("aki-richards", "shuey", "shuey-castagna"):
            approx = avo_pp(**IFACE, angles_deg=[0, 5, 10, 15], method=method)
            assert_allclose(approx, exact, atol=5e-3, err_msg=method)

    def test_matlab_zoeppritz_transliteration(self):
        # Fresh verbatim transliteration of avopp.m case 1.
        vp1, vs1, d1 = 2.6, 1.2, 2.3
        vp2, vs2, d2 = 2.2, 1.35, 2.05
        t = np.deg2rad(np.array([0.0, 12.0, 28.0]))
        p = np.sin(t) / vp1
        ct = np.cos(t)
        ct2 = np.sqrt(1 - (np.sin(t) ** 2 * (vp2**2 / vp1**2)))
        cj1 = np.sqrt(1 - (np.sin(t) ** 2 * (vs1**2 / vp1**2)))
        cj2 = np.sqrt(1 - (np.sin(t) ** 2 * (vs2**2 / vp1**2)))
        a = (d2 * (1 - (2 * vs2**2 * p**2))) - (d1 * (1 - (2 * vs1**2 * p**2)))
        b = (d2 * (1 - (2 * vs2**2 * p**2))) + (2 * d1 * vs1**2 * p**2)
        c = (d1 * (1 - (2 * vs1**2 * p**2))) + (2 * d2 * vs2**2 * p**2)
        d = 2 * ((d2 * vs2**2) - (d1 * vs1**2))
        E = (b * ct / vp1) + (c * ct2 / vp2)
        F = (b * cj1 / vs1) + (c * cj2 / vs2)
        G = a - (d * ct * cj2 / (vp1 * vs2))
        H = a - (d * ct2 * cj1 / (vp2 * vs1))
        D = (E * F) + (G * H * p**2)
        rpp = (
            ((b * ct / vp1) - (c * ct2 / vp2)) * F - (a + (d * ct * cj2 / (vp1 * vs2))) * H * p**2
        ) / D
        assert_allclose(avo_pp(**IFACE, angles_deg=[0.0, 12.0, 28.0]), rpp, rtol=1e-13)

    def test_post_critical_is_complex(self):
        # Fast-over-slow reversed: vp2 > vp1 gives a critical angle.
        fast = dict(vp1=2.0, vs1=1.0, rho1=2.0, vp2=3.5, vs2=1.8, rho2=2.4)
        crit = np.degrees(np.arcsin(2.0 / 3.5))
        r = avo_pp(**fast, angles_deg=[crit + 5, crit + 15])
        assert np.iscomplexobj(r)
        assert np.any(r.imag != 0)
        # Below critical it stays real.
        r_sub = avo_pp(**fast, angles_deg=[0, 10, 20])
        assert not np.iscomplexobj(r_sub)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            avo_pp(**IFACE, angles_deg=ANG, method="nope")


class TestAVOPS:
    def test_zero_at_normal_incidence(self):
        for method in PS_METHODS:
            r = avo_ps(**IFACE, angles_deg=[0.0], method=method)
            assert np.real(r[0]) == pytest.approx(0.0, abs=1e-12), method

    def test_no_contrast_gives_zero(self):
        same = dict(vp1=2.6, vs1=1.2, rho1=2.3, vp2=2.6, vs2=1.2, rho2=2.3)
        for method in PS_METHODS:
            r = avo_ps(**same, angles_deg=ANG, method=method)
            assert_allclose(np.real(r), 0.0, atol=1e-12, err_msg=method)

    def test_matlab_zoeppritz_transliteration(self):
        vp1, vs1, d1 = 2.6, 1.2, 2.3
        vp2, vs2, d2 = 2.2, 1.35, 2.05
        t = np.deg2rad(np.array([5.0, 18.0, 30.0]))
        p = np.sin(t) / vp1
        ct = np.cos(t)
        ct2 = np.sqrt(1 - (np.sin(t) ** 2 * (vp2**2 / vp1**2)))
        cj1 = np.sqrt(1 - (np.sin(t) ** 2 * (vs1**2 / vp1**2)))
        cj2 = np.sqrt(1 - (np.sin(t) ** 2 * (vs2**2 / vp1**2)))
        a = (d2 * (1 - (2 * vs2**2 * p**2))) - (d1 * (1 - (2 * vs1**2 * p**2)))
        b = (d2 * (1 - (2 * vs2**2 * p**2))) + (2 * d1 * vs1**2 * p**2)
        c = (d1 * (1 - (2 * vs1**2 * p**2))) + (2 * d2 * vs2**2 * p**2)
        d = 2 * ((d2 * vs2**2) - (d1 * vs1**2))
        E = (b * ct / vp1) + (c * ct2 / vp2)
        F = (b * cj1 / vs1) + (c * cj2 / vs2)
        G = a - (d * ct * cj2 / (vp1 * vs2))
        H = a - (d * ct2 * cj1 / (vp2 * vs1))
        D = (E * F) + (G * H * p**2)
        rps = -2 * (ct / vp1) * ((a * b) + (c * d * ct2 * cj2 / (vp2 * vs2))) * p * vp1 / (vs1 * D)
        assert_allclose(avo_ps(**IFACE, angles_deg=[5.0, 18.0, 30.0]), rps, rtol=1e-13)

    def test_approximations_track_zoeppritz_at_small_angles(self):
        ang = [5.0, 10.0, 15.0]
        exact = avo_ps(**IFACE, angles_deg=ang, method="zoeppritz")
        for method in ("aki-richards", "donati-quadratic", "donati-linear"):
            approx = avo_ps(**IFACE, angles_deg=ang, method=method)
            assert_allclose(np.real(approx), exact, atol=0.02, err_msg=method)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            avo_ps(**IFACE, angles_deg=ANG, method="nope")


class TestAVOAttributes:
    """The attributes must be exactly the coefficients of the curves."""

    def test_intercept_is_normal_incidence_shuey(self):
        att = avo_attributes(**IFACE)
        assert att.a == pytest.approx(avo_pp(**IFACE, angles_deg=[0.0], method="shuey")[0])

    def test_shuey_curve_decomposition(self):
        att = avo_attributes(**IFACE)
        t = np.deg2rad(ANG)
        av_vp = (TOP["vp1"] + BOT["vp2"]) / 2
        far = 0.5 * (BOT["vp2"] - TOP["vp1"]) / av_vp
        rebuilt = att.a + att.b1 * np.sin(t) ** 2 + far * (np.tan(t) ** 2 - np.sin(t) ** 2)
        assert_allclose(avo_pp(**IFACE, angles_deg=ANG, method="shuey"), rebuilt, rtol=1e-12)

    def test_shuey_castagna_curve_decomposition(self):
        att = avo_attributes(**IFACE)
        t = np.deg2rad(ANG)
        rebuilt = att.a + att.b2 * np.sin(t) ** 2
        assert_allclose(
            avo_pp(**IFACE, angles_deg=ANG, method="shuey-castagna"), rebuilt, rtol=1e-12
        )

    def test_ps_gradients_are_the_sine_coefficients(self):
        att = avo_attributes(**IFACE)
        t = np.deg2rad(ANG)
        assert_allclose(
            avo_ps(**IFACE, angles_deg=ANG, method="gonzalez"), att.e1 * np.sin(t), rtol=1e-12
        )
        assert_allclose(
            avo_ps(**IFACE, angles_deg=ANG, method="alejandro-reinaldo"),
            att.e2 * np.sin(t),
            rtol=1e-12,
        )

    def test_class_iii_signature(self):
        # Shale over gas sand: negative intercept and negative gradient.
        att = avo_attributes(**IFACE)
        assert att.a < 0
        assert att.b1 < 0
        assert att.b2 < 0

    def test_matlab_transliteration(self):
        vp1, vs1, d1 = 2.6, 1.2, 2.3
        vp2, vs2, d2 = 2.2, 1.35, 2.05
        da, Dd = (d1 + d2) / 2, d2 - d1
        vpa, Dvp = (vp1 + vp2) / 2, vp2 - vp1
        vsa, Dvs = (vs1 + vs2) / 2, vs2 - vs1
        Ro = 0.5 * ((Dvp / vpa) + (Dd / da))
        poi1 = ((0.5 * (vp1 / vs1) ** 2) - 1) / ((vp1 / vs1) ** 2 - 1)
        poi2 = ((0.5 * (vp2 / vs2) ** 2) - 1) / ((vp2 / vs2) ** 2 - 1)
        poia, Dpoi = (poi1 + poi2) / 2, poi2 - poi1
        Bx = (Dvp / vpa) / ((Dvp / vpa) + (Dd / da))
        Ax = Bx - (2 * (1 + Bx) * (1 - 2 * poia) / (1 - poia))
        B1 = (Ax * Ro) + (Dpoi / (1 - poia) ** 2)
        B2 = (-2 * vsa**2 * Dd / (vpa**2 * da)) + (0.5 * Dvp / vpa) - (4 * vsa * Dvs / vpa**2)
        E1 = (
            (-0.5 * Dd / da)
            - ((vsa / vpa) * ((Dd / da) + (2 * Dvs / vsa)))
            + (((vsa / vpa) ** 3) * ((0.5 * Dd / da) + (Dvs / vsa)))
        )
        E2 = -2 * (vs1 / vp1) * ((Dd / da * (0.5 + (0.25 * vpa / vsa))) + (Dvs / vsa))

        att = avo_attributes(**IFACE)
        assert att.a == pytest.approx(Ro, rel=1e-13)
        assert att.b1 == pytest.approx(B1, rel=1e-13)
        assert att.b2 == pytest.approx(B2, rel=1e-13)
        assert att.e1 == pytest.approx(E1, rel=1e-13)
        assert att.e2 == pytest.approx(E2, rel=1e-13)

    def test_vectorized_over_interfaces(self):
        att = avo_attributes(
            vp1=np.array([2.6, 2.8]),
            vs1=np.array([1.2, 1.4]),
            rho1=np.array([2.3, 2.35]),
            vp2=np.array([2.2, 3.0]),
            vs2=np.array([1.35, 1.5]),
            rho2=np.array([2.05, 2.4]),
        )
        assert att.a.shape == (2,)
        assert att.a[0] < 0 < att.a[1]


class TestElasticImpedance:
    # A short pseudo-log.
    VP = np.array([2.6, 2.8, 2.2, 3.0, 2.7])
    VS = np.array([1.2, 1.4, 1.35, 1.5, 1.3])
    RHO = np.array([2.3, 2.35, 2.05, 2.4, 2.32])

    def test_zero_angle_reduces_to_acoustic_impedance(self):
        r = elastic_impedance(self.VP, self.VS, self.RHO, 0.0)
        assert_allclose(r.ipp, self.VP * self.RHO, rtol=1e-13)

    def test_normalized_matches_raw_scaling_at_zero_angle(self):
        # At theta = 0 the normalized PP impedance is the raw one, because
        # the mean-scaling and mean-division cancel.
        r = elastic_impedance(self.VP, self.VS, self.RHO, 0.0)
        assert_allclose(r.ipp_n, self.VP * self.RHO, rtol=1e-13)

    def test_conventions_agree_for_pp_and_sp(self):
        # Only the P-S branch depends on the angle convention.
        a = elastic_impedance(self.VP, self.VS, self.RHO, 12.0, angle="reflection")
        b = elastic_impedance(self.VP, self.VS, self.RHO, 12.0, angle="incidence")
        assert_allclose(a.ipp, b.ipp, rtol=1e-14)
        assert_allclose(a.isp, b.isp, rtol=1e-14)
        assert not np.allclose(np.real(a.ips), np.real(b.ips))

    def test_matlab_eimp_transliteration(self):
        vp, vs, ro = self.VP, self.VS, self.RHO
        theta = np.deg2rad(15.0)
        vsvp = np.mean(vs / vp)
        vpn, vsn, ron = vp / np.mean(vp), vs / np.mean(vs), ro / np.mean(ro)
        ipn = np.mean(vp) * np.mean(ro)
        isn = np.mean(vs) * np.mean(ro)
        vsvpsin2 = vsvp**2 * np.sin(theta) ** 2
        x1 = 1 + np.tan(theta) ** 2
        x2 = 1 - 4 * vsvpsin2
        x3 = -8 * vsvpsin2
        ipp = vp**x1 * ro**x2 * vs**x3
        ippn = ipn * (vpn**x1 * ron**x2 * vsn**x3)
        y1 = 2 * np.sin(theta) ** 2 - 1 - 2 * np.cos(theta) * np.sqrt(vsvp**2 - np.sin(theta) ** 2)
        a = np.tan(theta) * y1 / vsvp
        y2 = np.sin(theta) ** 2 - np.cos(theta) * np.sqrt(vsvp**2 - np.sin(theta) ** 2)
        b = 4 * np.tan(theta) * y2 / vsvp
        ips = ro**a * vs**b
        ipsn = ipn * (ron**a * vsn**b)
        z1 = 2 * vsvpsin2 - 1 - 2 * vsvp * np.cos(theta) * np.sqrt(1 - vsvpsin2)
        asp = vsvp * np.tan(theta) * z1
        z2 = vsvpsin2 - vsvp * np.cos(theta) * np.sqrt(1 - vsvpsin2)
        bsp = 4 * vsvp * np.tan(theta) * z2
        isp = ro**asp * vs**bsp
        ispn = isn * (ron**asp * vsn**bsp)

        r = elastic_impedance(vp, vs, ro, 15.0, angle="reflection")
        assert_allclose(r.ipp, ipp, rtol=1e-13)
        assert_allclose(r.ipp_n, ippn, rtol=1e-13)
        assert_allclose(r.ips, ips, rtol=1e-13)
        assert_allclose(r.ips_n, ipsn, rtol=1e-13)
        assert_allclose(r.isp, isp, rtol=1e-13)
        assert_allclose(r.isp_n, ispn, rtol=1e-13)

    def test_matlab_eimp2_ps_transliteration(self):
        vp, vs, ro = self.VP, self.VS, self.RHO
        theta = np.deg2rad(20.0)
        vsvp = np.mean(vs / vp)
        vpvs = np.mean(vp / vs)
        # eimp2.m sets vpvs = mean(vp/vs) when K is absent, which is not
        # 1/mean(vs/vp); pass K explicitly so the two agree.
        vpvs = 1 / vsvp
        ron, vsn = ro / np.mean(ro), vs / np.mean(vs)
        ipn = np.mean(vp) * np.mean(ro)
        facroo = np.sqrt(vpvs**2 - np.sin(theta) ** 2)
        facall = np.sin(theta) / (vpvs * facroo)
        a = facall * (2 * np.sin(theta) ** 2 - vpvs**2 - 2 * np.cos(theta) * facroo)
        b = 4 * facall * (np.sin(theta) ** 2 - np.cos(theta) * facroo)
        ips = ro**a * vs**b
        ipsn = ipn * (ron**a * vsn**b)

        r = elastic_impedance(vp, vs, ro, 20.0, k=vsvp, angle="incidence")
        assert_allclose(r.ips, ips, rtol=1e-13)
        assert_allclose(r.ips_n, ipsn, rtol=1e-13)

    def test_explicit_k(self):
        r_auto = elastic_impedance(self.VP, self.VS, self.RHO, 20.0)
        r_k = elastic_impedance(self.VP, self.VS, self.RHO, 20.0, k=np.mean(self.VS / self.VP))
        assert_allclose(r_auto.ipp, r_k.ipp, rtol=1e-14)
        r_other = elastic_impedance(self.VP, self.VS, self.RHO, 20.0, k=0.4)
        assert not np.allclose(r_other.ipp, r_k.ipp)

    def test_beyond_ps_domain_is_complex(self):
        # With angle='reflection', sin(theta) must stay below k.
        k = 0.5
        r = elastic_impedance(self.VP, self.VS, self.RHO, 45.0, k=k, angle="reflection")
        assert np.iscomplexobj(r.ips)
        r_ok = elastic_impedance(self.VP, self.VS, self.RHO, 20.0, k=k, angle="reflection")
        assert not np.iscomplexobj(r_ok.ips)

    def test_invalid_angle_convention(self):
        with pytest.raises(ValueError, match="angle must be"):
            elastic_impedance(self.VP, self.VS, self.RHO, 10.0, angle="sideways")

    def test_scalar_input_degenerates_cleanly(self):
        r = elastic_impedance(2.6, 1.2, 2.3, 15.0)
        assert r.ipp_n == pytest.approx(2.6 * 2.3)
