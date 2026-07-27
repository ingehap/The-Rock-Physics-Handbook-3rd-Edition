import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    critical_porosity,
    lame_to_velocity,
    moduli_to_velocity,
    velocity_to_lame,
    velocity_to_moduli,
)

# Quartz: K = 37 GPa, mu = 44 GPa, rho = 2.65 g/cc.
QUARTZ = dict(k=37.0, mu=44.0, rho=2.65)


def test_moduli_to_velocity_quartz():
    vp, vs = moduli_to_velocity(**QUARTZ)
    assert vp == pytest.approx(np.sqrt((37.0 + 4 / 3 * 44.0) / 2.65))
    assert vs == pytest.approx(np.sqrt(44.0 / 2.65))
    # Handbook-familiar numbers: ~6.01 and ~4.07 km/s.
    assert vp == pytest.approx(6.008, abs=1e-3)
    assert vs == pytest.approx(4.075, abs=1e-3)


def test_velocity_moduli_round_trip():
    rng = np.random.default_rng(0)
    k = rng.uniform(1.0, 80.0, 50)
    mu = rng.uniform(1.0, 60.0, 50)
    rho = rng.uniform(1.5, 3.5, 50)
    vp, vs = moduli_to_velocity(k, mu, rho)
    k2, mu2 = velocity_to_moduli(vp, vs, rho)
    assert_allclose(k2, k, rtol=1e-12)
    assert_allclose(mu2, mu, rtol=1e-12)


def test_velocity_lame_round_trip():
    rng = np.random.default_rng(1)
    lam = rng.uniform(1.0, 50.0, 50)
    mu = rng.uniform(1.0, 60.0, 50)
    rho = rng.uniform(1.5, 3.5, 50)
    vp, vs = lame_to_velocity(lam, mu, rho)
    lam2, mu2 = velocity_to_lame(vp, vs, rho)
    assert_allclose(lam2, lam, rtol=1e-12)
    assert_allclose(mu2, mu, rtol=1e-12)


def test_lame_and_moduli_agree():
    # lambda = K - 2 mu / 3 must give identical velocities.
    k, mu, rho = QUARTZ["k"], QUARTZ["mu"], QUARTZ["rho"]
    lam = k - 2 * mu / 3
    assert_allclose(lame_to_velocity(lam, mu, rho), moduli_to_velocity(k, mu, rho))


class TestCriticalPorosity:
    # Quartz and water end members.
    ARGS = dict(vp1=6.008, vs1=4.075, rho1=2.65, vp2=1.5, vs2=0.5, rho2=1.0)

    def test_end_members(self):
        r0 = critical_porosity(**self.ARGS, phi_c=0.0)
        assert r0.vp == pytest.approx(self.ARGS["vp1"])
        assert r0.vs == pytest.approx(self.ARGS["vs1"])
        assert r0.rho == pytest.approx(self.ARGS["rho1"])
        r1 = critical_porosity(**self.ARGS, phi_c=1.0)
        assert r1.vp == pytest.approx(self.ARGS["vp2"])
        assert r1.vs == pytest.approx(self.ARGS["vs2"])
        assert r1.rho == pytest.approx(self.ARGS["rho2"])

    def test_reuss_average(self):
        phi = 0.4
        r = critical_porosity(**self.ARGS, phi_c=phi)
        mu1 = self.ARGS["rho1"] * self.ARGS["vs1"] ** 2
        mu2 = self.ARGS["rho2"] * self.ARGS["vs2"] ** 2
        mu_reuss = 1.0 / ((1 - phi) / mu1 + phi / mu2)
        assert r.mu == pytest.approx(mu_reuss)
        assert r.rho == pytest.approx((1 - phi) * 2.65 + phi * 1.0)
        # vp comes from (k, mu), not from the Reuss-averaged P-wave modulus m.
        assert r.vp == pytest.approx(np.sqrt((r.k + 4 / 3 * r.mu) / r.rho))
