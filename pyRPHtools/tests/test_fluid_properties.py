import numpy as np
import pytest

from rphtools import batzle_wang, co2_properties


class TestBatzleWang:
    # Typical reservoir conditions.
    P, T = 30.0, 80.0

    def test_pure_water_at_room_conditions(self):
        # Sound speed of pure water at 20 degC, ~0.1 MPa is ~1482 m/s,
        # density ~0.998 g/cc (Wilson's data underlying Batzle-Wang).
        r = batzle_wang(pressure=0.1, temperature=20.0, salinity=0.0)
        assert r.vp_brine * 1000 == pytest.approx(1482.0, abs=5.0)
        assert r.rho_brine == pytest.approx(0.998, abs=0.002)

    def test_salinity_stiffens_brine(self):
        fresh = batzle_wang(self.P, self.T, salinity=0.0)
        salty = batzle_wang(self.P, self.T, salinity=150000.0)
        assert salty.k_brine > fresh.k_brine
        assert salty.rho_brine > fresh.rho_brine

    def test_gas_density_increases_with_pressure(self):
        lo = batzle_wang(10.0, self.T)
        hi = batzle_wang(50.0, self.T)
        assert hi.rho_gas > lo.rho_gas
        assert hi.k_gas > lo.k_gas

    def test_phase_ordering(self):
        r = batzle_wang(self.P, self.T, oil_api=30.0)
        assert r.k_brine > r.k_oil > r.k_gas
        assert r.rho_brine > r.rho_oil > r.rho_gas

    def test_dead_oil_density_near_api_density(self):
        # At low P and T the dead-oil density stays near its API value.
        r = batzle_wang(0.1, 15.6, oil_api=30.0)
        rho0 = 141.5 / (30.0 + 131.5)
        assert r.rho_oil == pytest.approx(rho0, abs=0.01)

    def test_live_oil_lighter_and_softer_than_dead(self):
        dead = batzle_wang(self.P, self.T, gor=0.0)
        live = batzle_wang(self.P, self.T, gor=100.0)
        assert live.rho_oil < dead.rho_oil
        assert live.k_oil < dead.k_oil

    def test_gas_index_oil_overrides_gor(self):
        r0 = batzle_wang(self.P, self.T, gor=50.0, gas_index_oil=0.0)
        rd = batzle_wang(self.P, self.T, gor=0.0)
        # gas_index_oil=0 -> gor=0 -> identical to dead oil.
        assert r0.rho_oil == pytest.approx(float(rd.rho_oil))
        r1 = batzle_wang(self.P, self.T, gas_index_oil=0.5)
        gormax = 2.03 * 0.6 * (self.P * np.exp(0.02878 * 30.0 - 0.00377 * self.T)) ** 1.205
        assert r1.gor == pytest.approx(0.5 * gormax)

    def test_mixing_rules(self):
        r = batzle_wang(self.P, self.T, s_oil=0.3, s_gas=0.2)
        sb, so, sg = 0.5, 0.3, 0.2
        assert r.rho_eff == pytest.approx(sb * r.rho_brine + so * r.rho_oil + sg * r.rho_gas)
        assert r.k_voigt == pytest.approx(sb * r.k_brine + so * r.k_oil + sg * r.k_gas)
        assert r.k_reuss == pytest.approx(1.0 / (sb / r.k_brine + so / r.k_oil + sg / r.k_gas))
        assert r.k_reuss < r.k_voigt  # Reuss below Voigt always

    def test_single_phase_mix_is_brine(self):
        r = batzle_wang(self.P, self.T)
        assert r.k_reuss == pytest.approx(float(r.k_brine))
        assert r.k_voigt == pytest.approx(float(r.k_brine))
        assert r.rho_eff == pytest.approx(float(r.rho_brine))

    def test_vectorized_over_pressure(self):
        p = np.array([10.0, 20.0, 30.0])
        r = batzle_wang(p, self.T)
        assert r.k_brine.shape == (3,)
        for i, pi in enumerate(p):
            ri = batzle_wang(pi, self.T)
            assert r.k_brine[i] == pytest.approx(float(ri.k_brine))
            assert r.rho_gas[i] == pytest.approx(float(ri.rho_gas))
            assert r.k_oil[i] == pytest.approx(float(ri.k_oil))


class TestCO2Properties:
    def test_grid_nodes_exact(self):
        from importlib.resources import files

        with (files("rphtools") / "data" / "co2prop.npz").open("rb") as fh:
            data = np.load(fh)
            t = data["temperature_c"]
            p = data["pressure_mpa"]
            bulk, rho, vp = data["bulk_gpa"], data["rho_gcc"], data["vp_ms"]

        for ip, it in [(0, 0), (3, 5), (9, 11)]:
            k, r, v = co2_properties(t[it], p[ip])
            assert k == pytest.approx(bulk[ip, it], rel=1e-12)
            assert r == pytest.approx(rho[ip, it], rel=1e-12)
            assert v == pytest.approx(vp[ip, it], rel=1e-12)

    def test_interpolation_between_nodes(self):
        from importlib.resources import files

        with (files("rphtools") / "data" / "co2prop.npz").open("rb") as fh:
            data = np.load(fh)
            t = data["temperature_c"]
            p = data["pressure_mpa"]
            rho = data["rho_gcc"]

        tm = (t[3] + t[4]) / 2
        pm = (p[2] + p[3]) / 2
        _, r, _ = co2_properties(tm, pm)
        corners = rho[2:4, 3:5]
        assert corners.min() <= r <= corners.max()
        # Bilinear at the cell center = average of the four corners.
        assert r == pytest.approx(corners.mean())

    def test_outside_range_is_nan(self):
        k, r, v = co2_properties(200.0, 10.0)
        assert np.isnan(k) and np.isnan(r) and np.isnan(v)
        k, _, _ = co2_properties(80.0, 100.0)
        assert np.isnan(k)

    def test_broadcasting(self):
        t = np.array([27.0, 57.0, 87.0])
        k, r, v = co2_properties(t, 10.0)
        assert k.shape == (3,)
        assert np.all(np.isfinite(k))

    def test_physical_trends(self):
        # Denser at higher pressure, lighter at higher temperature.
        _, r_lo, _ = co2_properties(57.0, 10.0)
        _, r_hi, _ = co2_properties(57.0, 30.0)
        assert r_hi > r_lo
        _, r_cold, _ = co2_properties(27.0, 20.0)
        _, r_hot, _ = co2_properties(107.0, 20.0)
        assert r_cold > r_hot
