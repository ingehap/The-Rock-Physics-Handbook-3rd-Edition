import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import (
    kennett,
    kennett_frazer_dispersion,
    kennett_frazer_traveltimes,
    propagator_seis,
    quick_seismic_section,
    ricker,
)

DT = 0.001
WAVELET = ricker(30.0, DT)[0]
HOMO = np.array([[2500.0, 2200.0, 100.0]])
TWO = np.array([[2000.0, 2000.0, 80.0], [2600.0, 2300.0, 90.0]])
THREE = np.array([[2000.0, 2000.0, 40.0], [3200.0, 2500.0, 15.0], [2400.0, 2150.0, 60.0]])
# Periodic thin-layer stack, 2 km thick: many wavelengths in the Backus band.
PERIODIC = np.array([[2000.0, 2000.0, 5.0], [3000.0, 2400.0, 5.0]] * 200)


def backus_velocity(lyr):
    v, rho, d = lyr[:, 0], lyr[:, 1], lyr[:, 2]
    f = d / d.sum()
    m_eff = 1.0 / np.sum(f / (rho * v**2))
    return np.sqrt(m_eff / np.sum(f * rho))


def ray_velocity(lyr):
    v, d = lyr[:, 0], lyr[:, 2]
    return d.sum() / np.sum(d / v)


class TestRicker:
    def test_peaks_at_centre_and_is_symmetric(self):
        w, t = ricker(25.0, DT)
        assert w.size % 2 == 1
        assert np.argmax(w) == w.size // 2
        assert w[w.size // 2] == pytest.approx(1.0)
        assert_allclose(w, w[::-1], atol=1e-14)
        assert t[w.size // 2] == pytest.approx(0.0, abs=1e-15)

    def test_zero_mean_ish_and_two_side_lobes(self):
        w, _ = ricker(25.0, DT)
        assert abs(w.sum()) < 0.05 * np.abs(w).sum()
        assert w.min() < 0  # negative side lobes exist

    def test_spectrum_peaks_near_nominal_frequency(self):
        from rphtools import spectrum

        w, _ = ricker(30.0, DT, n=1024)
        s = spectrum(w, DT)
        assert s.freq[np.argmax(s.amplitude)] == pytest.approx(30.0, abs=3.0)

    def test_explicit_length_honoured(self):
        w, t = ricker(25.0, DT, n=101)
        assert w.size == t.size == 101


class TestKennett:
    def test_homogeneous_medium_has_no_reflection(self):
        r = kennett(HOMO, WAVELET, DT)
        assert_allclose(np.abs(r.reflectivity), 0.0, atol=1e-14)
        assert_allclose(np.abs(r.transmissivity), 1.0, rtol=1e-12)
        assert_allclose(r.wz, 0.0, atol=1e-12)
        assert np.max(np.abs(r.pz)) > 0.5  # the wavelet passes through

    def test_transmitted_pulse_is_delayed_by_traveltime(self):
        # The zero-phase wavelet peaks at its own centre, so the arrival
        # sits at (wavelet centre) + (one-way traveltime).
        r = kennett(HOMO, WAVELET, DT)
        peak = np.argmax(np.abs(r.pz)) * DT
        centre = (WAVELET.size - 1) / 2 * DT
        assert peak - centre == pytest.approx(HOMO[0, 2] / HOMO[0, 0], abs=2 * DT)

    def test_reflection_coefficient_matches_impedance_contrast(self):
        r = kennett(TWO, WAVELET, DT, multiples="primaries")
        z1, z2 = TWO[0, 0] * TWO[0, 1], TWO[1, 0] * TWO[1, 1]
        expected = (z1 - z2) / (z1 + z2)
        # Primaries only: |R| is the interface coefficient at all frequencies.
        assert_allclose(np.abs(r.reflectivity), abs(expected), rtol=1e-12)

    def test_energy_conservation_without_free_surface(self):
        # Lossless stack: |R|^2 + |T|^2 == 1 for the all-multiples solution.
        r = kennett(TWO, WAVELET, DT, multiples="all")
        assert_allclose(np.abs(r.reflectivity) ** 2 + np.abs(r.transmissivity) ** 2, 1.0, rtol=1e-9)

    def test_no_internal_multiples_with_one_reflector(self):
        # TWO has a single reflecting interface (the top one has no
        # contrast without a free surface), so there is nothing to
        # reverberate between and all three options coincide.
        prim = kennett(TWO, WAVELET, DT, multiples="primaries")
        allm = kennett(TWO, WAVELET, DT, multiples="all")
        assert_allclose(prim.pz, allm.pz, rtol=1e-12)

    def test_multiples_options_differ(self):
        prim = kennett(THREE, WAVELET, DT, multiples="primaries")
        first = kennett(THREE, WAVELET, DT, multiples="first-order")
        allm = kennett(THREE, WAVELET, DT, multiples="all")
        assert not np.allclose(prim.pz, allm.pz)
        assert not np.allclose(first.pz, allm.pz)
        # Primaries alone lose energy that the full solution retains.
        assert np.max(np.abs(prim.reflectivity - allm.reflectivity)) > 1e-6

    def test_free_surface_changes_result(self):
        with_fs = kennett(TWO, WAVELET, DT, free_surface=True)
        without = kennett(TWO, WAVELET, DT, free_surface=False)
        assert not np.allclose(with_fs.wz, without.wz)

    def test_output_lengths(self):
        r = kennett(TWO, WAVELET, DT)
        n = WAVELET.size
        assert r.freq.size == n // 2
        assert r.wz.size == r.pz.size == 2 * r.freq.size

    def test_odd_length_wavelet_works(self):
        # The taper fix from kennett_aux.m is what makes this possible.
        odd = ricker(30.0, DT, n=127)[0]
        r = kennett(TWO, odd, DT)
        assert np.all(np.isfinite(r.pz))
        assert r.pz.size == 2 * r.freq.size

    def test_matlab_transliteration(self):
        # Fresh verbatim transliteration of kennet.m (mopt=2, fs=0).
        lyr, wvlt, dt = TWO, WAVELET, DT
        ro, v, d = lyr[:, 1], lyr[:, 0], lyr[:, 2]
        nlr = lyr.shape[0]
        n = wvlt.size
        # MATLAB's `0:1:(n/2 - 1)`, which for odd n is one shorter than
        # np.arange(0, n/2).
        om = (2 * np.pi / (n * dt)) * np.arange(int(np.floor(n / 2 - 1)) + 1)
        p0 = np.fft.ifft(wvlt)
        rdhat = np.zeros(om.shape, complex)
        tdhat = np.ones(om.shape, complex)
        deno = ro[1:] * v[1:] + ro[:-1] * v[:-1]
        rd = np.concatenate([[0.0], (ro[:-1] * v[:-1] - ro[1:] * v[1:]) / deno])
        td = np.concatenate([[1.0], 2 * np.sqrt(ro[1:] * v[1:] * ro[:-1] * v[:-1]) / deno])
        ru, tu = -rd, td
        for j in range(nlr - 1, -1, -1):
            ed = np.exp(1j * (d[j] / v[j]) * om)
            reverb = 1.0 / (1 - ru[j] * ed * rdhat * ed)
            rdhat = rd[j] + tu[j] * ed * rdhat * ed * reverb * td[j]
            tdhat = tdhat * ed * reverb * td[j]
        pz = tdhat * p0[: om.size]
        wz = rdhat * p0[: om.size]
        m = int(np.floor(n / 2 + 0.5))
        flat = int(np.floor(n / 4 + 0.5))
        fltr_full = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (m + 1)))
        fltr = np.concatenate([np.ones(flat), fltr_full[flat:m]])[: om.size]
        pz, wz = pz * fltr, wz * fltr
        pz = np.concatenate([pz[:1], pz[1:], [0], np.conj(pz[:0:-1])])
        wz = np.concatenate([[0], wz[1:], [0], np.conj(wz[:0:-1])])
        pz, wz = np.real(np.fft.fft(pz)), np.real(np.fft.fft(wz))

        r = kennett(TWO, WAVELET, DT, multiples="all", free_surface=False)
        assert_allclose(r.pz, pz, rtol=1e-12, atol=1e-14)
        assert_allclose(r.wz, wz, rtol=1e-12, atol=1e-14)

    def test_invalid_multiples(self):
        with pytest.raises(ValueError, match="multiples must be"):
            kennett(TWO, WAVELET, DT, multiples="some")

    def test_bad_layer_shape(self):
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            kennett(np.array([[2000.0, 2000.0]]), WAVELET, DT)


class TestPropagator:
    def test_matlab_transliteration(self):
        lyr, wvlt, dt, alpha = TWO, WAVELET, DT, 0.0
        ro, v, d = lyr[:, 1], lyr[:, 0], lyr[:, 2]
        n = wvlt.size
        om = (2 * np.pi / (n * dt)) * np.arange(int(np.floor(n / 2 - 1)) + 1)
        p0 = np.fft.ifft(wvlt)
        a11 = np.ones(om.shape, complex)
        a21 = np.zeros(om.shape, complex)
        a12 = a21.copy()
        a22 = a11.copy()
        for j in range(lyr.shape[0]):
            k = om / v[j]
            ck = k + 1j * alpha * k
            wdv = d[j] * ck
            c11 = np.cos(wdv)
            c12 = 1j * ro[j] * v[j] * np.sin(wdv)
            c21 = (1j / (ro[j] * v[j])) * np.sin(wdv)
            c22 = c11
            b11 = c11 * a11 + c12 * a21
            b12 = c11 * a12 + c12 * a22
            b21 = c21 * a11 + c22 * a21
            b22 = c21 * a12 + c22 * a22
            a11, a12, a21, a22 = b11, b12, b21, b22
        rzvz = ro[-1] * v[-1]
        pz = (rzvz * (a12 * a21 - a11 * a22) / (a12 - rzvz * a22)) * p0[: om.size]
        wz = ((rzvz * a21 - a11) / (a12 - rzvz * a22)) * p0[: om.size]
        m = int(np.floor(n / 2 + 0.5))
        flat = int(np.floor(n / 4 + 0.5))
        fltr_full = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, m + 1) / (m + 1)))
        fltr = np.concatenate([np.ones(flat), fltr_full[flat:m]])[: om.size]
        pz, wz = pz * fltr, wz * fltr
        pz = np.concatenate([pz[:1], pz[1:], [0], np.conj(pz[:0:-1])])
        wz = np.concatenate([[0], wz[1:], [0], np.conj(wz[:0:-1])])
        pz, wz = np.real(np.fft.fft(pz)), np.real(np.fft.fft(wz))

        r = propagator_seis(TWO, WAVELET, DT)
        assert_allclose(r.pz, pz, rtol=1e-12, atol=1e-14)
        assert_allclose(r.wz, wz, rtol=1e-12, atol=1e-14)

    def test_transmitted_pulse_agrees_with_kennett(self):
        # Two independent methods for the same physics: the transmitted
        # arrivals should line up closely.
        k = kennett(TWO, WAVELET, DT, multiples="all")
        p = propagator_seis(TWO, WAVELET, DT)
        kn = k.pz / np.max(np.abs(k.pz))
        pn = p.pz / np.max(np.abs(p.pz))
        assert np.corrcoef(kn, pn)[0, 1] > 0.95
        assert abs(np.argmax(np.abs(kn)) - np.argmax(np.abs(pn))) <= 2

    def test_attenuation_reduces_amplitude(self):
        lossless = propagator_seis(TWO, WAVELET, DT, alpha=0.0)
        lossy = propagator_seis(TWO, WAVELET, DT, alpha=0.05)
        assert np.max(np.abs(lossy.pz)) < np.max(np.abs(lossless.pz))


class TestKennettFrazerDispersion:
    def test_matlab_transliteration(self):
        lyr = PERIODIC[:40]
        f = np.logspace(-1, 4, 15)
        v = lyr[:, 0].copy()
        ro = lyr[:, 1].copy()
        d = lyr[:, 2].copy()
        n = v.size
        ve, roe, de = np.append(v, v[-1]), np.append(ro, ro[-1]), np.append(d, d[-1])
        rt = np.cumsum(d / v)
        w = 2 * np.pi * f
        rru = np.zeros_like(w, dtype=complex)
        xsum = np.zeros_like(w, dtype=complex)
        length = 0.0
        tt = []
        for k in range(n):
            den = roe[k] * ve[k] + roe[k + 1] * ve[k + 1]
            rd = (roe[k + 1] * ve[k + 1] - roe[k] * ve[k]) / den
            ru = -rd
            td = 2 * np.sqrt(roe[k] * ve[k] * roe[k + 1] * ve[k + 1]) / den
            theta = np.exp((1j * de[k] / ve[k]) * w)
            t1 = td / (1 - rd * rru * theta**2)
            rru = (ru + rru * theta**2) / (1 - rd * rru * theta**2)
            length += de[k]
            xsum = xsum + (1 / (1j * w)) * np.log(t1)
            tt.append(rt[k] + np.real(xsum))
        expected = length / np.array(tt)[n - 1]

        _, vel = kennett_frazer_dispersion(lyr, f)
        assert_allclose(vel, expected, rtol=1e-13)

    def test_backus_plateau_in_long_wavelength_band(self):
        # Wavelength >> layer thickness and stack >> wavelength.
        f = np.array([1.0, 3.0, 10.0, 30.0])
        _, vel = kennett_frazer_dispersion(PERIODIC, f)
        assert_allclose(vel, backus_velocity(PERIODIC), rtol=5e-3)

    def test_ray_theory_at_high_frequency(self):
        f = np.array([1e3, 1e4])
        _, vel = kennett_frazer_dispersion(PERIODIC, f)
        assert_allclose(vel, ray_velocity(PERIODIC), rtol=5e-3)

    def test_backus_is_slower_than_ray(self):
        assert backus_velocity(PERIODIC) < ray_velocity(PERIODIC)

    def test_homogeneous_stack_has_no_dispersion(self):
        lyr = np.tile([2500.0, 2200.0, 10.0], (20, 1))
        f = np.logspace(0, 4, 9)
        _, vel = kennett_frazer_dispersion(lyr, f)
        assert_allclose(vel, 2500.0, rtol=1e-9)

    def test_scalar_frequency_accepted(self):
        f, vel = kennett_frazer_dispersion(PERIODIC, 10.0)
        assert vel.shape == (1,)


class TestKennettFrazerTraveltimes:
    def test_ray_traveltime_is_cumulative(self):
        r = kennett_frazer_traveltimes(TWO, 30.0)
        assert_allclose(r.rt, np.cumsum(TWO[:, 2] / TWO[:, 0]), rtol=1e-14)

    def test_homogeneous_all_three_agree(self):
        lyr = np.tile([2500.0, 2200.0, 10.0], (10, 1))
        r = kennett_frazer_traveltimes(lyr, 30.0)
        assert_allclose(r.tt, r.rt, rtol=1e-9)
        assert_allclose(r.emtt, r.rt, rtol=1e-9)

    def test_exact_exceeds_ray_in_scattering_band(self):
        # Stratigraphic slowdown: the exact traveltime lags ray theory.
        r = kennett_frazer_traveltimes(PERIODIC, 10.0)
        assert r.tt[-1] > r.rt[-1]

    def test_lengths_match_layer_count(self):
        r = kennett_frazer_traveltimes(PERIODIC, 10.0)
        n = PERIODIC.shape[0]
        assert r.tt.shape == r.rt.shape == r.emtt.shape == (n,)


class TestQuickSeismicSection:
    RNG = np.random.default_rng(12)

    def _log(self, n=400):
        depth_v = 2000.0 + 400.0 * np.sin(np.linspace(0, 6 * np.pi, n))
        depth_r = 2100.0 + 150.0 * np.cos(np.linspace(0, 4 * np.pi, n))
        return depth_v, depth_r

    def test_single_log_gives_25_traces(self):
        v, d = self._log()
        r = quick_seismic_section(v, d, dz=1.0, freq=25.0)
        assert r.section.shape[1] == 25
        assert np.allclose(r.section, r.section[:, :1])  # all identical

    def test_time_axis_matches_section(self):
        v, d = self._log()
        r = quick_seismic_section(v, d, dz=1.0, freq=25.0)
        assert r.time.shape[0] == r.section.shape[0]
        assert np.all(np.diff(r.time) > 0)

    def test_constant_model_has_no_reflectivity(self):
        n = 300
        v = np.full(n, 2500.0)
        d = np.full(n, 2200.0)
        r = quick_seismic_section(v, d, dz=1.0, freq=25.0)
        assert np.max(np.abs(r.section)) < 1e-9

    def test_two_dimensional_model(self):
        v, d = self._log(300)
        vel = np.column_stack([v, v * 1.02, v * 0.98])
        dens = np.column_stack([d, d * 1.01, d])
        r = quick_seismic_section(vel, dens, dx=10.0, dz=1.0, freq=25.0)
        assert r.section.shape[1] == 3

    def test_runs_without_decimation(self):
        # The R <= 1 branch, where ezseis.m used an undefined cutoff.
        v, d = self._log(200)
        r = quick_seismic_section(v, d, dz=1.0, freq=200.0)
        assert np.all(np.isfinite(r.section))

    def test_noise_is_reproducible(self):
        v, d = self._log(200)
        kw = dict(dz=1.0, freq=25.0, noise_ratio=0.5)
        a = quick_seismic_section(v, d, **kw, rng=np.random.default_rng(0))
        b = quick_seismic_section(v, d, **kw, rng=np.random.default_rng(0))
        c = quick_seismic_section(v, d, **kw, rng=np.random.default_rng(1))
        assert_allclose(a.section, b.section)
        assert not np.allclose(a.section, c.section)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            quick_seismic_section(np.ones(10), np.ones(11))
