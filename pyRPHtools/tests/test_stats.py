import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    bayes_classify,
    hist2d,
    hist3d,
    monte_carlo_ccdf,
    monte_carlo_cdf,
)
from rphtools.stats import _bin_index, _edges_from_centres


class TestBinning:
    def test_matlab_edge_rule(self):
        # binwidth = [diff(x) 0]; edges = [x0 - w0/2, x + w/2].
        centres = np.array([1.0, 2.0, 4.0])
        edges = _edges_from_centres(centres)
        assert_allclose(edges, [0.5, 1.5, 3.0, 4.0])
        # The trailing zero makes the last edge the last centre itself.
        assert edges[-1] == centres[-1]

    def test_centre_bins_assign_as_matlab(self):
        centres = np.array([1.0, 2.0, 3.0])
        values = np.array([0.0, 1.0, 1.4, 1.6, 2.4, 2.6, 3.0, 99.0])
        idx, out_centres, nbin = _bin_index(values, centres)
        assert nbin == 3
        assert_allclose(out_centres, centres)
        # Below the first edge and above the last are clamped into the ends.
        assert idx[0] == 0
        assert idx[-1] == 2
        assert list(idx) == [0, 0, 0, 1, 1, 2, 2, 2]

    def test_equal_width_bins_span_the_data(self):
        v = np.linspace(0.0, 10.0, 101)
        idx, centres, nbin = _bin_index(v, 5)
        assert nbin == 5
        assert_allclose(centres, [1.0, 3.0, 5.0, 7.0, 9.0])
        assert idx.min() == 0 and idx.max() == 4

    def test_degenerate_constant_data(self):
        idx, centres, nbin = _bin_index(np.full(10, 3.0), 4)
        assert nbin == 4
        assert np.all(np.isfinite(centres))
        assert np.all((idx >= 0) & (idx < 4))


class TestHist2D:
    RNG = np.random.default_rng(0)

    def test_counts_total_preserved(self):
        data = self.RNG.standard_normal((500, 2))
        h = hist2d(data, 10, 12)
        assert h.counts.shape == (10, 12)
        assert h.counts.sum() == 500

    def test_default_is_15_by_15(self):
        h = hist2d(self.RNG.standard_normal((100, 2)))
        assert h.counts.shape == (15, 15)

    def test_second_spec_defaults_to_first(self):
        data = self.RNG.standard_normal((200, 2))
        assert hist2d(data, 8).counts.shape == (8, 8)

    def test_known_placement(self):
        # Two points, one in each corner bin.
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        centres = np.array([0.0, 0.5, 1.0])
        h = hist2d(data, centres, centres)
        assert h.counts[0, 0] == 1
        assert h.counts[2, 2] == 1
        assert h.counts.sum() == 2

    def test_weights_are_summed(self):
        data = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        centres = np.array([0.0, 1.0])
        h = hist2d(data, centres, centres, weights=[2.0, 3.0, 4.0])
        assert h.counts[0, 0] == pytest.approx(5.0)
        assert h.counts[1, 1] == pytest.approx(4.0)

    def test_marginal_matches_1d_histogram(self):
        data = self.RNG.standard_normal((400, 2))
        h = hist2d(data, 12, 9)
        col_totals = h.counts.sum(axis=1)
        h1 = hist3d(data[:, :1], 12)
        assert_allclose(col_totals, h1.counts)

    def test_wrong_column_count(self):
        with pytest.raises(ValueError, match="two columns"):
            hist2d(self.RNG.standard_normal((10, 3)))


class TestHist3D:
    RNG = np.random.default_rng(1)

    @pytest.mark.parametrize("ncol", [1, 2, 3])
    def test_all_column_counts_supported(self, ncol):
        # hist3d.m delegated 1-column input to a missing hist1d and its
        # 2-column weighted path called hist2d with the wrong arity.
        data = self.RNG.standard_normal((200, ncol))
        h = hist3d(data, 6)
        assert h.counts.shape == (6,) * ncol
        assert h.counts.sum() == 200
        assert len(h.centres) == ncol

    def test_weights_work_for_every_column_count(self):
        for ncol in (1, 2, 3):
            data = self.RNG.standard_normal((50, ncol))
            w = self.RNG.uniform(0.5, 2.0, 50)
            h = hist3d(data, 4, weights=w)
            assert h.counts.sum() == pytest.approx(w.sum())

    def test_per_column_bin_specs(self):
        data = self.RNG.standard_normal((100, 3))
        h = hist3d(data, [4, 5, 6])
        assert h.counts.shape == (4, 5, 6)

    def test_agrees_with_hist2d(self):
        data = self.RNG.standard_normal((300, 2))
        assert_allclose(hist3d(data, 7).counts, hist2d(data, 7, 7).counts)

    def test_too_many_columns(self):
        with pytest.raises(ValueError, match="one, two, or three"):
            hist3d(self.RNG.standard_normal((10, 4)))


class TestBayesClassify:
    def test_picks_the_most_probable_facies(self):
        axes = [np.array([0.0, 1.0, 2.0])]
        # Three cells, three facies + a trailing marginal slice.
        pdf = np.zeros((3, 4))
        pdf[0, :3] = [0.7, 0.2, 0.1]
        pdf[1, :3] = [0.1, 0.8, 0.1]
        pdf[2, :3] = [0.2, 0.2, 0.6]
        pdf[:, 3] = 1.0  # marginal, must be ignored
        r = bayes_classify(np.array([[0.0], [1.0], [2.0]]), pdf, axes)
        assert list(r.code) == [0, 1, 2]
        assert_allclose(r.max_prob, [0.7, 0.8, 0.6])
        assert r.probs.shape == (3, 3)

    def test_marginal_slice_excluded(self):
        axes = [np.array([0.0, 1.0])]
        pdf = np.zeros((2, 3))
        pdf[:, :2] = [[0.6, 0.4], [0.3, 0.7]]
        pdf[:, 2] = 99.0  # would win if it were not dropped
        r = bayes_classify(np.array([[0.0], [1.0]]), pdf, axes)
        assert list(r.code) == [0, 1]
        assert r.max_prob.max() < 1.0

    def test_two_attributes(self):
        axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        pdf = np.zeros((2, 2, 3))
        pdf[0, 0, :2] = [0.9, 0.1]
        pdf[1, 1, :2] = [0.2, 0.8]
        r = bayes_classify(np.array([[0.0, 0.0], [1.0, 1.0]]), pdf, axes)
        assert list(r.code) == [0, 1]

    def test_out_of_range_clamped_into_end_cells(self):
        axes = [np.array([0.0, 1.0, 2.0])]
        pdf = np.zeros((3, 3))
        pdf[0, :2] = [0.9, 0.1]
        pdf[2, :2] = [0.1, 0.9]
        r = bayes_classify(np.array([[-99.0], [99.0]]), pdf, axes)
        assert list(r.code) == [0, 1]

    def test_too_few_axes(self):
        with pytest.raises(ValueError, match="one axis per attribute"):
            bayes_classify(np.zeros((5, 2)), np.zeros((3, 3)), [np.arange(3.0)])


class TestMonteCarlo:
    RNG = np.random.default_rng(7)

    def _correlated(self, n=800):
        x = self.RNG.normal(3.0, 0.5, n)
        y = 2.0 * x + 1.0 + self.RNG.normal(0, 0.3, n)
        return np.column_stack([x, y])

    @pytest.mark.parametrize("fn", [monte_carlo_cdf, monte_carlo_ccdf])
    def test_shape_and_reproducibility(self, fn):
        data = self._correlated()
        a = fn(data, 200, rng=np.random.default_rng(0))
        b = fn(data, 200, rng=np.random.default_rng(0))
        c = fn(data, 200, rng=np.random.default_rng(1))
        assert a.shape == (200, 2)
        assert_allclose(a, b)
        assert not np.allclose(a, c)

    @pytest.mark.parametrize("fn", [monte_carlo_cdf, monte_carlo_ccdf])
    def test_primary_distribution_reproduced(self, fn):
        data = self._correlated()
        sim = fn(data, 4000, rng=np.random.default_rng(3))
        assert np.mean(sim[:, 0]) == pytest.approx(np.mean(data[:, 0]), abs=0.06)
        assert np.std(sim[:, 0]) == pytest.approx(np.std(data[:, 0]), abs=0.06)
        # Draws stay inside the observed range (inverse-CDF sampling).
        assert sim[:, 0].min() >= data[:, 0].min() - 0.1
        assert sim[:, 0].max() <= data[:, 0].max() + 1e-9

    @pytest.mark.parametrize("fn", [monte_carlo_cdf, monte_carlo_ccdf])
    def test_correlation_preserved(self, fn):
        data = self._correlated()
        sim = fn(data, 3000, rng=np.random.default_rng(4))
        assert np.corrcoef(sim.T)[0, 1] == pytest.approx(np.corrcoef(data.T)[0, 1], abs=0.08)

    @pytest.mark.parametrize("fn", [monte_carlo_cdf, monte_carlo_ccdf])
    def test_single_column_input(self, fn):
        sim = fn(self.RNG.normal(0, 1, 300), 100, rng=np.random.default_rng(5))
        assert sim.shape == (100, 1)

    def test_regression_slope_recovered(self):
        data = self._correlated()
        sim = monte_carlo_cdf(data, 5000, rng=np.random.default_rng(6))
        slope = np.polyfit(sim[:, 0], sim[:, 1], 1)[0]
        assert slope == pytest.approx(2.0, abs=0.15)

    def test_ccdf_follows_a_nonlinear_relation(self):
        # The conditional version needs no linearity assumption, so it
        # tracks a curved trend the linear-regression version cannot.
        x = self.RNG.uniform(0.0, 4.0, 1500)
        y = x**2 + self.RNG.normal(0, 0.1, 1500)
        data = np.column_stack([x, y])
        sim = monte_carlo_ccdf(data, 3000, rng=np.random.default_rng(8))
        low = sim[sim[:, 0] < 1.0, 1]
        high = sim[sim[:, 0] > 3.0, 1]
        assert np.median(low) < 1.5
        assert np.median(high) > 8.0
