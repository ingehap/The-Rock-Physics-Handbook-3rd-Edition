import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import block_average, fft_axis, instantaneous_attributes, spectrum
from rphtools.seismic import _matlab_hanning


class TestMatlabHanning:
    """The single most likely source of silent drift in this phase."""

    @pytest.mark.parametrize("m", [4, 5, 8, 16, 31, 64])
    def test_matches_matlab_definition(self, m):
        # MATLAB: hanning(m) = 0.5*(1 - cos(2*pi*(1:m)/(m+1)))
        ml = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (m + 1)))
        assert_allclose(_matlab_hanning(m), ml, rtol=1e-14)

    def test_differs_from_numpy_hanning(self):
        # np.hanning(m) includes the zero endpoints; MATLAB's excludes them.
        m = 16
        assert not np.allclose(_matlab_hanning(m), np.hanning(m))
        assert _matlab_hanning(m)[0] > 0  # never zero at the ends

    def test_symmetric_and_bounded(self):
        w = _matlab_hanning(21)
        assert_allclose(w, w[::-1], rtol=1e-14)
        assert 0 < w.min() and w.max() <= 1.0


class TestSpectrum:
    def test_peaks_at_input_frequency(self):
        dt, f0 = 0.001, 40.0
        t = np.arange(0, 1.0, dt)
        s = spectrum(np.sin(2 * np.pi * f0 * t), dt)
        assert s.freq[np.argmax(s.amplitude)] == pytest.approx(f0, abs=1.5)

    def test_axis_spans_zero_to_nyquist(self):
        dt, n = 0.002, 256
        s = spectrum(np.random.default_rng(0).standard_normal(n), dt)
        assert s.freq[0] == 0.0
        assert s.freq[-1] == pytest.approx(0.5 / dt, rel=1e-9)
        assert s.freq.size == n // 2 + 1
        assert s.amplitude.size == s.freq.size == s.phase.size

    def test_matlab_transliteration(self):
        dt = 0.004
        rng = np.random.default_rng(1)
        data = rng.standard_normal(64)
        nsample = data.size
        spec_full = np.fft.fft(data)
        sc = 0.5 / dt
        ds = 2 * sc / nsample
        sindex = np.arange(0, sc + ds / 2, ds)
        amp = np.abs(spec_full[: sindex.size])
        phase = np.angle(spec_full[: sindex.size])
        s = spectrum(data, dt)
        assert_allclose(s.freq, sindex, rtol=1e-13)
        assert_allclose(s.amplitude, amp, rtol=1e-13)
        assert_allclose(s.phase, phase, rtol=1e-13)

    def test_dc_amplitude_is_sum(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert spectrum(data, 0.001).amplitude[0] == pytest.approx(10.0)


class TestInstantaneousAttributes:
    def test_constant_amplitude_sinusoid(self):
        dt, f0 = 0.001, 25.0
        t = np.arange(0, 0.5, dt)
        x = 3.0 * np.cos(2 * np.pi * f0 * t)
        r = instantaneous_attributes(x)
        # Envelope is flat at the amplitude, away from the Hilbert
        # transform's edge transients.
        assert_allclose(r.amplitude[100:-100], 3.0, rtol=0.02)

    def test_frequency_recovers_input(self):
        dt, f0 = 0.001, 25.0
        t = np.arange(0, 0.5, dt)
        x = np.cos(2 * np.pi * f0 * t)
        r = instantaneous_attributes(x)
        # Unwrap first: the raw diff jumps by 2*pi at wrap points.
        dphi = np.diff(np.unwrap(r.phase))
        assert np.median(dphi) / (2 * np.pi * dt) == pytest.approx(f0, rel=0.02)

    def test_frequency_is_one_shorter(self):
        x = np.random.default_rng(2).standard_normal(100)
        r = instantaneous_attributes(x)
        assert r.amplitude.shape == (100,)
        assert r.frequency.shape == (99,)

    def test_section_uses_time_axis_zero(self):
        rng = np.random.default_rng(3)
        section = rng.standard_normal((64, 5))
        r = instantaneous_attributes(section)
        assert r.amplitude.shape == (64, 5)
        assert r.frequency.shape == (63, 5)
        # Each trace must match the 1-D result for that column.
        for k in range(5):
            single = instantaneous_attributes(section[:, k])
            assert_allclose(r.amplitude[:, k], single.amplitude, rtol=1e-12)

    def test_envelope_exceeds_signal(self):
        x = np.random.default_rng(4).standard_normal(200)
        r = instantaneous_attributes(x)
        assert np.all(r.amplitude >= np.abs(x) - 1e-12)


class TestBlockAverage:
    def test_constant_log_unchanged(self):
        x = np.full(20, 3.5)
        assert_allclose(block_average(x, 4), x)

    def test_block_means_are_repeated(self):
        x = np.arange(12.0)
        out = block_average(x, 4)
        assert_allclose(out[:4], np.mean(x[:4]))
        assert_allclose(out[4:8], np.mean(x[4:8]))
        assert_allclose(out[8:], np.mean(x[8:]))

    def test_length_preserved_for_ragged_input(self):
        x = np.arange(10.0)
        out = block_average(x, 4)
        assert out.shape == x.shape
        # Final partial block is padded with the last value.
        assert out[8] == pytest.approx(np.mean([8.0, 9.0, 9.0, 9.0]))

    def test_nan_ignored(self):
        x = np.array([1.0, np.nan, 3.0, 5.0])
        out = block_average(x, 4)
        assert_allclose(out, 3.0)

    def test_two_dimensional_columns_independent(self):
        rng = np.random.default_rng(5)
        arr = rng.standard_normal((16, 3))
        out = block_average(arr, 4)
        assert out.shape == arr.shape
        for k in range(3):
            assert_allclose(out[:, k], block_average(arr[:, k], 4), rtol=1e-13)

    def test_block_of_one_is_identity(self):
        x = np.arange(7.0)
        assert_allclose(block_average(x, 1), x)

    def test_invalid_nb(self):
        with pytest.raises(ValueError, match="positive integer"):
            block_average(np.arange(5.0), 0)


class TestFFTAxis:
    def test_round_trip_axis0(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((16, 4)) + 1j * rng.standard_normal((16, 4))
        back = fft_axis(fft_axis(x, adjoint=False, sign=1, axis=0), adjoint=True, sign=1, axis=0)
        assert_allclose(back, x, atol=1e-12)

    def test_round_trip_axis1(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((4, 16)) + 1j * rng.standard_normal((4, 16))
        back = fft_axis(fft_axis(x, adjoint=False, sign=1, axis=1), adjoint=True, sign=1, axis=1)
        assert_allclose(back, x, atol=1e-12)

    def test_matlab_ft1axis_transliteration(self):
        rng = np.random.default_rng(8)
        cx = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
        # adj == 0, sig == 1
        ml = cx.copy()
        ml[1::2, :] = -ml[1::2, :]
        ml = np.fft.ifft(ml, axis=0)
        assert_allclose(fft_axis(cx, adjoint=False, sign=1, axis=0), ml, rtol=1e-13)

    def test_matlab_ft2axis_transliteration(self):
        rng = np.random.default_rng(9)
        cx = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
        # adj == 1, sig == 1
        ml = np.fft.fft(cx, axis=1)
        ml[:, 1::2] = -ml[:, 1::2]
        assert_allclose(fft_axis(cx, adjoint=True, sign=1, axis=1), ml, rtol=1e-13)

    def test_centres_zero_frequency(self):
        # A constant signal transforms to a spike at the centre, not at 0.
        n = 32
        x = np.ones(n, complex)
        out = fft_axis(x, adjoint=False, sign=1, axis=0)
        assert np.argmax(np.abs(out)) == n // 2

    def test_does_not_mutate_input(self):
        x = np.ones((4, 4), complex)
        original = x.copy()
        fft_axis(x)
        assert_allclose(x, original)

    def test_invalid_sign(self):
        with pytest.raises(ValueError, match="sign must be"):
            fft_axis(np.ones(4), sign=0)
