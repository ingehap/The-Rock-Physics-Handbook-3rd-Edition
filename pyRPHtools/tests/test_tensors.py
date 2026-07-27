import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    bond_matrix,
    bond_rotation,
    cti_to_velocities,
    isotropic_cs,
    thomsen_params,
    ti_c_to_s,
    ti_velocities,
    ti_voigt_matrix,
)

K, MU, RHO = 37.0, 44.0, 2.65  # quartz

# A weakly anisotropic VTI shale-like set of constants (GPa).
VTI = dict(c11=34.3, c12=13.1, c13=10.7, c33=22.7, c44=5.4)
VTI["c66"] = (VTI["c11"] - VTI["c12"]) / 2


class TestIsotropicCS:
    def test_inverse_pair(self):
        c, s = isotropic_cs(K, MU)
        assert_allclose(c @ s, np.eye(6), atol=1e-12)
        # Closed form matches numerical inversion.
        assert_allclose(s, np.linalg.inv(c), rtol=1e-12)

    def test_entries(self):
        c, s = isotropic_cs(K, MU)
        lam = K - 2 * MU / 3
        assert c[0, 0] == pytest.approx(lam + 2 * MU)
        assert c[0, 1] == pytest.approx(lam)
        assert c[3, 3] == pytest.approx(MU)
        e = 9 * K * MU / (3 * K + MU)
        assert s[0, 0] == pytest.approx(1 / e)
        assert s[3, 3] == pytest.approx(1 / MU)

    def test_broadcast(self):
        c, s = isotropic_cs([10.0, 20.0, 30.0], 15.0)
        assert c.shape == (3, 6, 6)
        for i, k in enumerate([10.0, 20.0, 30.0]):
            ci, si = isotropic_cs(k, 15.0)
            assert_allclose(c[i], ci)
            assert_allclose(s[i], si)

    def test_fluid_mu_zero(self):
        c, s = isotropic_cs(2.2, 0.0)
        assert c[0, 0] == pytest.approx(2.2)
        assert c[3, 3] == 0.0
        assert np.isinf(s[3, 3])  # documented: fluid compliance is singular


def test_thomsen_isotropic_is_zero():
    lam = K - 2 * MU / 3
    t = thomsen_params(c11=lam + 2 * MU, c33=lam + 2 * MU, c44=MU, c66=MU, c13=lam)
    assert t.epsilon == pytest.approx(0.0)
    assert t.gamma == pytest.approx(0.0)
    assert t.delta == pytest.approx(0.0)
    assert t.delta_sv == pytest.approx(0.0)


def test_thomsen_definitions():
    t = thomsen_params(VTI["c11"], VTI["c33"], VTI["c44"], VTI["c66"], VTI["c13"])
    assert t.epsilon == pytest.approx((VTI["c11"] - VTI["c33"]) / (2 * VTI["c33"]))
    assert t.gamma == pytest.approx((VTI["c66"] - VTI["c44"]) / (2 * VTI["c44"]))
    a, b = VTI["c13"] + VTI["c44"], VTI["c33"] - VTI["c44"]
    assert t.delta == pytest.approx((a * a - b * b) / (2 * VTI["c33"] * b))


class TestTiCToS:
    def test_involution(self):
        args = (VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        s = ti_c_to_s(*args)
        back = ti_c_to_s(*s)
        assert_allclose(back, args, rtol=1e-12)

    def test_against_matrix_inverse(self):
        c = ti_voigt_matrix(VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        s_full = np.linalg.inv(c)
        s = ti_c_to_s(VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        assert s.m11 == pytest.approx(s_full[0, 0])
        assert s.m12 == pytest.approx(s_full[0, 1])
        assert s.m13 == pytest.approx(s_full[0, 2])
        assert s.m33 == pytest.approx(s_full[2, 2])
        assert s.m44 == pytest.approx(s_full[3, 3])


class TestTiVelocities:
    def test_isotropic_all_angles(self):
        lam = K - 2 * MU / 3
        angles = np.array([0.0, 20.0, 45.0, 70.0, 90.0])
        v = ti_velocities(lam + 2 * MU, lam + 2 * MU, MU, MU, lam, RHO, angles)
        assert_allclose(v.vp, np.sqrt((lam + 2 * MU) / RHO))
        assert_allclose(v.vsv, np.sqrt(MU / RHO))
        assert_allclose(v.vsh, np.sqrt(MU / RHO))

    def test_symmetry_directions(self):
        v0 = ti_velocities(VTI["c11"], VTI["c33"], VTI["c44"], VTI["c66"], VTI["c13"], RHO, 0.0)
        assert v0.vp == pytest.approx(np.sqrt(VTI["c33"] / RHO))
        assert v0.vsv == pytest.approx(np.sqrt(VTI["c44"] / RHO))
        assert v0.vsh == pytest.approx(np.sqrt(VTI["c44"] / RHO))
        v90 = ti_velocities(VTI["c11"], VTI["c33"], VTI["c44"], VTI["c66"], VTI["c13"], RHO, 90.0)
        assert v90.vp == pytest.approx(np.sqrt(VTI["c11"] / RHO))
        assert v90.vsv == pytest.approx(np.sqrt(VTI["c44"] / RHO))
        assert v90.vsh == pytest.approx(np.sqrt(VTI["c66"] / RHO))


class TestCtiToVelocities:
    def test_vti_matrix(self):
        c = ti_voigt_matrix(VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        r = cti_to_velocities(c, RHO)
        assert r.vp_fast == pytest.approx(np.sqrt(VTI["c11"] / RHO))
        assert r.vp_slow == pytest.approx(np.sqrt(VTI["c33"] / RHO))
        assert r.vs_fast == pytest.approx(np.sqrt(VTI["c66"] / RHO))
        assert r.vs_slow == pytest.approx(np.sqrt(VTI["c44"] / RHO))
        t = thomsen_params(VTI["c11"], VTI["c33"], VTI["c44"], VTI["c66"], VTI["c13"])
        assert r.epsilon == pytest.approx(t.epsilon)
        assert r.gamma == pytest.approx(t.gamma)
        assert r.delta == pytest.approx(t.delta)

    def test_sorting_handles_swapped_axes(self):
        # Swapping c11/c33 and c44/c66 must leave fast/slow labeling intact.
        c = ti_voigt_matrix(VTI["c33"], VTI["c12"], VTI["c13"], VTI["c11"], VTI["c66"], VTI["c44"])
        r = cti_to_velocities(c, RHO)
        assert r.vp_fast == pytest.approx(np.sqrt(VTI["c11"] / RHO))
        assert r.vp_slow == pytest.approx(np.sqrt(VTI["c33"] / RHO))

    def test_stacked(self):
        c = ti_voigt_matrix(VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        stacked = np.stack([c, 2 * c])
        r = cti_to_velocities(stacked, RHO)
        assert r.vp_fast.shape == (2,)
        assert r.vp_fast[1] == pytest.approx(np.sqrt(2) * r.vp_fast[0])


class TestBondRotation:
    def test_zero_is_identity(self):
        assert_allclose(bond_matrix(0.0), np.eye(6), atol=1e-15)

    def test_rotation_round_trip(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(6, 6))
        c = a + a.T  # arbitrary symmetric matrix, as the MATLAB header suggests
        back = bond_rotation(bond_rotation(c, 33.0), -33.0)
        assert_allclose(back, c, atol=1e-12)

    def test_four_quarter_turns(self):
        c = ti_voigt_matrix(VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        out = c
        for _ in range(4):
            out = bond_rotation(out, 90.0)
        assert_allclose(out, c, atol=1e-10)

    def test_isotropic_invariant(self):
        c, _ = isotropic_cs(K, MU)
        assert_allclose(bond_rotation(c, 27.0), c, atol=1e-10)

    def test_vti_invariant_about_symmetry_axis(self):
        # VTI is rotationally symmetric about x3, the rotation axis.
        c = ti_voigt_matrix(VTI["c11"], VTI["c12"], VTI["c13"], VTI["c33"], VTI["c44"])
        assert_allclose(bond_rotation(c, 41.0), c, atol=1e-10)
