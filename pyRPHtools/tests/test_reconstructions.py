"""Functions RPHtools lists but does not ship, reconstructed here.

These have no MATLAB to compare against, so each is pinned by an exact
property instead: an algebraic inverse, a limiting case, or a definition
the surviving code specifies.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import rphtools as rph
from rphtools.granular import unconsolidated
from rphtools.stats import pdf_bayes
from rphtools.tensors import ti_from_velocities


class TestTiFromVelocities:
    """Reconstruction of ``v2cti``: the exact inverse of thomsen_params."""

    CONSTANTS = (34.3, 13.1, 10.7, 22.7, 5.4)  # c11, c12, c13, c33, c44
    RHO = 2.5

    def test_round_trip_through_thomsen(self):
        c11, c12, c13, c33, c44 = self.CONSTANTS
        c66 = (c11 - c12) / 2
        t = rph.thomsen_params(c11, c33, c44, c66, c13)
        vp0, vs0 = np.sqrt(c33 / self.RHO), np.sqrt(c44 / self.RHO)
        back = ti_from_velocities(vp0, vs0, self.RHO, t.epsilon, t.gamma, t.delta)
        assert_allclose(back, [c11, c12, c13, c33, c44, c66], rtol=1e-13)

    def test_round_trip_over_random_media(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            c33 = rng.uniform(10, 60)
            c44 = rng.uniform(2, c33 / 2.5)
            c11 = c33 * rng.uniform(1.0, 1.5)
            c66 = c44 * rng.uniform(1.0, 1.4)
            c13 = rng.uniform(0.2, 0.9) * (c33 - c44)
            rho = rng.uniform(1.9, 2.8)
            t = rph.thomsen_params(c11, c33, c44, c66, c13)
            back = ti_from_velocities(
                np.sqrt(c33 / rho), np.sqrt(c44 / rho), rho, t.epsilon, t.gamma, t.delta
            )
            assert_allclose(back, [c11, c11 - 2 * c66, c13, c33, c44, c66], rtol=1e-10)

    def test_isotropic_medium_has_zero_anisotropy(self):
        k, mu, rho = 37.0, 44.0, 2.65
        lam = k - 2 * mu / 3
        vp0, vs0 = np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho)
        c11, c12, c13, c33, c44, c66 = ti_from_velocities(vp0, vs0, rho, 0.0, 0.0, 0.0)
        assert c11 == pytest.approx(c33)
        assert c44 == pytest.approx(c66)
        assert c13 == pytest.approx(c12)
        assert c13 == pytest.approx(lam, rel=1e-12)

    def test_builds_a_valid_voigt_matrix(self):
        c = rph.ti_voigt_matrix(*ti_from_velocities(3.0, 1.7, 2.4, 0.15, 0.1, 0.08))
        assert_allclose(c, c.T)
        assert np.all(np.linalg.eigvalsh(c) > 0)


class TestUnconsolidated:
    """Reconstruction of ``Unconsol``: fixed exactly by its two endpoints."""

    ARGS = dict(k_min=37.0, g_min=44.0, pressure=0.02, phi_c=0.36)

    def test_reduces_to_hertz_mindlin_at_critical_porosity(self):
        r = unconsolidated(**self.ARGS, phi=np.array([self.ARGS["phi_c"]]))
        hm = rph.hertz_mindlin(
            self.ARGS["k_min"],
            self.ARGS["g_min"],
            self.ARGS["pressure"],
            phi=np.array([self.ARGS["phi_c"]]),
            coord=rph.coordination_number(self.ARGS["phi_c"]),
        )
        assert r.k[0] == pytest.approx(float(hm.k[0]), rel=1e-12)
        assert r.g[0] == pytest.approx(float(hm.g[0]), rel=1e-12)

    def test_reduces_to_the_mineral_at_zero_porosity(self):
        r = unconsolidated(**self.ARGS, phi=np.array([0.0]))
        assert r.k[0] == pytest.approx(self.ARGS["k_min"], rel=1e-12)
        assert r.g[0] == pytest.approx(self.ARGS["g_min"], rel=1e-12)

    def test_monotone_between_the_endpoints(self):
        r = unconsolidated(**self.ARGS)
        assert np.all(np.diff(r.k) < 0)
        assert np.all(np.diff(r.g) < 0)

    def test_lies_on_the_hashin_shtrikman_lower_bound(self):
        # The trend is the HS lower bound between the sphere pack and the
        # mineral, so it must sit at or below the HS upper bound for the
        # same two end members.
        phi = 0.18
        r = unconsolidated(**self.ARGS, phi=np.array([phi]))
        hm = rph.hertz_mindlin(
            self.ARGS["k_min"],
            self.ARGS["g_min"],
            self.ARGS["pressure"],
            phi=np.array([self.ARGS["phi_c"]]),
            coord=rph.coordination_number(self.ARGS["phi_c"]),
        )
        f = phi / self.ARGS["phi_c"]
        b = rph.bounds(
            [f, 1 - f],
            [float(hm.k[0]), self.ARGS["k_min"]],
            [float(hm.g[0]), self.ARGS["g_min"]],
            method="hs",
        )
        assert b.k_lower <= r.k[0] <= b.k_upper
        assert b.mu_lower <= r.g[0] <= b.mu_upper
        assert r.k[0] == pytest.approx(b.k_lower, rel=1e-10)

    def test_stiffens_with_pressure(self):
        low = unconsolidated(**{**self.ARGS, "pressure": 0.005}, phi=np.array([0.2]))
        high = unconsolidated(**{**self.ARGS, "pressure": 0.04}, phi=np.array([0.2]))
        assert high.k[0] > low.k[0]

    def test_porosity_out_of_range(self):
        with pytest.raises(ValueError, match="between 0 and phi_c"):
            unconsolidated(**self.ARGS, phi=np.array([0.5]))


class TestPdfBayes:
    """Reconstruction of ``pdfbayes``, whose engines are missing."""

    def _two_facies(self, separation=3.0, n=400, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, n)
        b = rng.normal(separation, 1.0, n)
        return np.concatenate([a, b])[:, None], np.array([0] * n + [1] * n)

    def test_pdf_layout_matches_bayes_classify(self):
        data, code = self._two_facies()
        r = pdf_bayes(data, code, bins=20)
        # (bins..., nfacies + 1) with the marginal last -- exactly what
        # bayes_classify expects, so the two compose.
        assert r.pdf.shape == (20, 3)
        cls = rph.bayes_classify(data, r.pdf, r.axes)
        assert cls.code.shape == (data.shape[0],)
        assert set(np.unique(cls.code)) <= {0, 1}

    def test_each_conditional_pdf_normalized(self):
        data, code = self._two_facies()
        r = pdf_bayes(data, code, bins=20)
        for f in range(2):
            assert r.pdf[..., f].sum() == pytest.approx(1.0)
        assert r.pdf[..., -1].sum() == pytest.approx(1.0)

    def test_marginal_is_the_prior_weighted_mixture(self):
        data, code = self._two_facies()
        r = pdf_bayes(data, code, bins=20)
        # Equal facies counts, so the marginal is the plain average.
        assert_allclose(r.pdf[..., -1], 0.5 * (r.pdf[..., 0] + r.pdf[..., 1]), atol=1e-12)

    def test_separated_facies_classify_well(self):
        data, code = self._two_facies(separation=6.0)
        r = pdf_bayes(data, code, bins=25)
        assert r.success_rate > 0.99
        assert r.error < 0.01
        assert r.cond_entropy_norm < 0.1  # attributes nearly determine facies

    def test_overlapping_facies_classify_poorly(self):
        data, code = self._two_facies(separation=0.0)
        r = pdf_bayes(data, code, bins=25)
        assert r.success_rate < 0.65
        assert r.cond_info < 0.1  # attributes say almost nothing
        assert r.cond_entropy_norm > 0.9

    def test_entropy_identities(self):
        data, code = self._two_facies()
        r = pdf_bayes(data, code, bins=20)
        # H(Y|X) = H(Y) - I(X;Y), and the normalized form is their ratio.
        assert r.cond_entropy == pytest.approx(r.entropy - r.cond_info)
        assert r.cond_entropy_norm == pytest.approx(r.cond_entropy / r.entropy)
        # Two equally likely facies carry exactly one bit.
        assert r.entropy == pytest.approx(1.0)

    def test_joint_table_is_a_probability_distribution(self):
        data, code = self._two_facies()
        r = pdf_bayes(data, code, bins=20)
        assert r.joint.shape == (2, 2)
        assert r.joint.sum() == pytest.approx(1.0)
        assert np.all(r.joint >= 0)
        assert r.success_rate == pytest.approx(np.trace(r.joint))
        assert_allclose(r.conditional.sum(axis=1), 1.0)

    def test_priors_shift_the_classification(self):
        data, code = self._two_facies(separation=1.0)
        balanced = pdf_bayes(data, code, bins=20)
        skewed = pdf_bayes(data, code, bins=20, priors=[0.9, 0.1])
        assert not np.allclose(balanced.joint, skewed.joint)
        assert skewed.entropy < balanced.entropy  # a skewed prior is less uncertain

    def test_weights_respected(self):
        data, code = self._two_facies()
        w = np.where(code == 0, 3.0, 1.0)
        r = pdf_bayes(data, code, weights=w, bins=20)
        # Facies 0 now carries three quarters of the prior mass.
        assert r.pdf[..., -1].sum() == pytest.approx(1.0)
        assert r.entropy < 1.0

    def test_bandwidth_scales_with_smoothing(self):
        data, code = self._two_facies()
        narrow = pdf_bayes(data, code, bins=20, smoothing=0.05)
        wide = pdf_bayes(data, code, bins=20, smoothing=0.4)
        assert np.all(wide.bandwidth > narrow.bandwidth)
        # More smoothing spreads the PDF, so its entropy rises.
        assert wide.pdf[..., 0].max() < narrow.pdf[..., 0].max()

    def test_two_attributes(self):
        rng = np.random.default_rng(3)
        n = 300
        a = rng.multivariate_normal([0, 0], np.eye(2), n)
        b = rng.multivariate_normal([4, 4], np.eye(2), n)
        data = np.vstack([a, b])
        code = np.array([0] * n + [1] * n)
        r = pdf_bayes(data, code, bins=15)
        assert r.pdf.shape == (15, 15, 3)
        assert r.success_rate > 0.95

    def test_validation(self):
        data, code = self._two_facies()
        with pytest.raises(ValueError, match="one entry per sample"):
            pdf_bayes(data, code[:-1])
        with pytest.raises(ValueError, match="one, two, or three"):
            pdf_bayes(np.zeros((10, 4)), np.zeros(10))
