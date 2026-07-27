import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import backus_average, backus_average_c, backus_average_log

# A shale/sand pair (km/s, g/cc).
SHALE = dict(vp=3.0, vs=1.5, rho=2.4)
SAND = dict(vp=4.0, vs=2.4, rho=2.5)
F = np.array([0.6, 0.4])
VP = np.array([SHALE["vp"], SAND["vp"]])
VS = np.array([SHALE["vs"], SAND["vs"]])
RHO = np.array([SHALE["rho"], SAND["rho"]])


class TestBackusAverage:
    def test_identical_layers_are_isotropic(self):
        f = np.array([0.3, 0.7])
        r = backus_average(f, [3.0, 3.0], [1.5, 1.5], [2.4, 2.4])
        mu = 2.4 * 1.5**2
        m = 2.4 * 3.0**2
        assert r.c11 == pytest.approx(m)
        assert r.c33 == pytest.approx(m)
        assert r.c44 == pytest.approx(mu)
        assert r.c66 == pytest.approx(mu)
        assert r.c13 == pytest.approx(m - 2 * mu)
        assert r.vp0 == pytest.approx(3.0)
        assert r.vp45 == pytest.approx(3.0)
        assert r.vp90 == pytest.approx(3.0)
        assert r.vs0 == pytest.approx(1.5)
        assert r.vsh90 == pytest.approx(1.5)

    def test_c66_identity(self):
        # c66 == (c11 - c12)/2 is an identity of the Backus average
        # (the MATLAB code checked it at runtime; it can never fail).
        r = backus_average(F, VP, VS, RHO)
        assert r.c66 == pytest.approx((r.c11 - r.c12) / 2)

    def test_layering_anisotropy_sign(self):
        # Backus average of isotropic layers is stiffer along the layers.
        r = backus_average(F, VP, VS, RHO)
        assert r.c11 > r.c33
        assert r.c66 > r.c44
        assert r.vp90 > r.vp0

    def test_matches_matrix_version(self):
        r = backus_average(F, VP, VS, RHO)
        c, rho_avg = backus_average_c(F, VP, VS, RHO)
        assert rho_avg == pytest.approx(r.rho)
        assert c[0, 0] == pytest.approx(r.c11)
        assert c[0, 1] == pytest.approx(r.c12)
        assert c[0, 2] == pytest.approx(r.c13)
        assert c[2, 2] == pytest.approx(r.c33)
        assert c[3, 3] == pytest.approx(r.c44)
        assert c[5, 5] == pytest.approx(r.c66)
        assert_allclose(c, c.T)

    def test_thicknesses_normalized(self):
        # Raw thicknesses give the same result as fractions.
        r1 = backus_average(F, VP, VS, RHO)
        r2 = backus_average(F * 123.0, VP, VS, RHO)
        assert_allclose(list(r2), list(r1))

    def test_average_density(self):
        r = backus_average(F, VP, VS, RHO)
        assert r.rho == pytest.approx(np.sum(F * RHO))

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            backus_average([-0.5, 1.5], VP, VS, RHO)
        with pytest.raises(ValueError):
            backus_average([0.5, 0.5, 0.5], VP, VS, RHO)


class TestBackusAverageLog:
    DEPTH = np.arange(1000.0, 1010.0, 1.0)  # 10 samples, uniform

    def _logs(self):
        n = self.DEPTH.size
        rng = np.random.default_rng(3)
        pick = rng.integers(0, 2, n).astype(bool)
        vp = np.where(pick, SAND["vp"], SHALE["vp"])
        vs = np.where(pick, SAND["vs"], SHALE["vs"])
        rho = np.where(pick, SAND["rho"], SHALE["rho"])
        phi = np.where(pick, 0.25, 0.08)
        return vp, vs, rho, phi

    def test_uniform_sampling_matches_equal_fractions(self):
        vp, vs, rho, phi = self._logs()
        r = backus_average_log(self.DEPTH, vp, vs, rho, phi)
        n = self.DEPTH.size
        c, rho_avg = backus_average_c(np.full(n, 1.0 / n), vp, vs, rho)
        assert_allclose(r.c, c)
        assert r.rho == pytest.approx(rho_avg)
        assert r.phi == pytest.approx(phi.mean())

    def test_reversed_depth_same_result(self):
        vp, vs, rho, phi = self._logs()
        r_fwd = backus_average_log(self.DEPTH, vp, vs, rho, phi)
        r_rev = backus_average_log(self.DEPTH[::-1], vp[::-1], vs[::-1], rho[::-1], phi[::-1])
        assert_allclose(r_rev.c, r_fwd.c)
        assert r_rev.rho == pytest.approx(r_fwd.rho)
        assert r_rev.phi == pytest.approx(r_fwd.phi)

    def test_phi_optional(self):
        vp, vs, rho, _ = self._logs()
        r = backus_average_log(self.DEPTH, vp, vs, rho)
        assert r.phi is None

    def test_non_monotonic_depth_raises(self):
        vp, vs, rho, _ = self._logs()
        depth = self.DEPTH.copy()
        depth[4] = depth[6]
        with pytest.raises(ValueError, match="monotonic"):
            backus_average_log(depth, vp, vs, rho)
