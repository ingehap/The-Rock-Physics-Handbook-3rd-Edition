% Generate golden reference values for pyRPHtools tests by running the
% original RPHtools MATLAB functions in GNU Octave.
%
% Usage (from the repository root):
%   octave --no-gui pyRPHtools/tests/generate_golden.m
%
% Writes JSON fixtures to pyRPHtools/tests/golden/. The fixtures are
% committed so CI never needs Octave; re-run this script only to regenerate
% them (e.g. after adding cases).
%
% Covers every RPHtools function that can actually be run. Functions that
% cannot are noted inline with the reason (several are simply broken; see
% PORTING_PLAN.md section 7.4).
%
% octave_shims/ supplies what modern Octave lacks: the legacy `bessel`,
% the Statistics/Signal Toolbox `harmmean`, `nanmean` and `hilbert`, the
% reconstruction of the missing `v2ku`, and no-op plotting stubs (many
% RPHtools functions draw figures unconditionally).

addpath('RPHtools');
% Shims for legacy MATLAB functions RPHtools calls that modern
% MATLAB/Octave no longer provide (see octave_shims/).
addpath('pyRPHtools/tests/octave_shims');
% octave_shims/ also contains no-op plotting stubs, because many RPHtools
% functions draw figures unconditionally and no display is available.
outdir = 'pyRPHtools/tests/golden';
if ~exist(outdir, 'dir'), mkdir(outdir); end

g = struct();

% --- moduli ------------------------------------------------------------
[vp, vs] = ku2v(37, 44, 2.65);
g.ku2v_quartz = [vp, vs];
[vp, vs] = lm2v(37 - 2*44/3, 44, 2.65);
g.lm2v_quartz = [vp, vs];
[vpcr, vscr, rocr, mcr, kcr, mucr] = critpor(6.008, 4.075, 2.65, 1.5, 0.5, 1.0, 0.4);
g.critpor = [vpcr, vscr, rocr, mcr, kcr, mucr];

% --- tensors -----------------------------------------------------------
[S, C] = CSiso(37, 44);
g.csiso_c = C; g.csiso_s = S;
g.c2anis = c2anis([34.3 22.7 5.4 10.6 10.7]);
g.c2sti = c2sti([34.3 13.1 10.7 22.7 5.4]);
[vp, vsh, vsv] = c2vti([34.3 22.7 5.4 10.6 10.7], 2.5, [0 30 45 60 90]);
g.c2vti_vp = vp; g.c2vti_vsh = vsh; g.c2vti_vsv = vsv;
cvti = zeros(6,6);
cvti(1,1)=34.3; cvti(2,2)=34.3; cvti(1,2)=13.1; cvti(2,1)=13.1;
cvti(1,3)=10.7; cvti(3,1)=10.7; cvti(2,3)=10.7; cvti(3,2)=10.7;
cvti(3,3)=22.7; cvti(4,4)=5.4; cvti(5,5)=5.4; cvti(6,6)=(34.3-13.1)/2;
[vps, vss, vpf, vsf, e, gg, d] = cti2v(cvti, 2.5);
g.cti2v = [vps, vss, vpf, vsf, e, gg, d];
g.ezbond_30 = ezbond(cvti, 30);

% --- layered -----------------------------------------------------------
f = [0.6 0.4]; vp = [3.0 4.0]; vs = [1.5 2.4]; den = [2.4 2.5];
[vv, cc, rho] = bkus(f, den, vp, vs);
g.bkus_vv = vv; g.bkus_cc = cc; g.bkus_rho = rho;
[c6, rho] = bkusc(f, vp, vs, den);
g.bkusc_c = c6; g.bkusc_rho = rho;

% --- bounds ------------------------------------------------------------
[ku, kl, uu, ul, ka, ua] = bound(0, [0.7 0.3], [37 2.2], [44 3.0]);
g.bound_vr = [ku, kl, uu, ul, ka, ua];
[ku, kl, uu, ul, ka, ua] = bound(1, [0.7 0.3], [37 2.2], [44 3.0]);
g.bound_hs = [ku, kl, uu, ul, ka, ua];
[ku, kl, gu, gl, por] = hash(37, 44, 2.2, 0);
g.hash = [ku(:), kl(:), gu(:), gl(:), por(:)];
[vpu, vpl, vsu, vsl, por] = hashv(6.008, 4.075, 2.65, 1.5, 0, 1.0);
g.hashv = [vpu(:), vpl(:), vsu(:), vsl(:), por(:)];

% --- fluids (Phase 2) --------------------------------------------------
g.gassmnk = gassmnk(12, 0.0, 2.5, 37, 0.25);
[vp2, vs2, ro2, k2] = gassmnv(3.5, 2.2, 2.3, 1.0, 2.5, 0.2, 0.05, 37, 0.25);
g.gassmnv = [vp2, vs2, ro2, k2];
[S, C] = CSiso(12, 14);
g.bkd2s = BKd2s(S, 37, 44, 2.5, 0.25);
g.bks2d = BKs2d(g.bkd2s, 37, 44, 2.5, 0.25);
[Smin, Cmin] = CSiso(37, 44);
sso = [Smin(1,1) Smin(1,2) Smin(1,3) Smin(3,3) Smin(4,4)];
ssd = [S(1,1) S(1,2) S(1,3) S(3,3) S(4,4)];
g.bkti = bkti(0.25, 1/2.5, sso, ssd);
g.mmti = mmti([0.036 -0.007 -0.006 0.040 0.13], [0.030 -0.008 -0.007 0.033 0.11]);
[vp1, vp2b, vs] = biothf(3200, 2000, 37e9, 44e9, 2650, 1000, 2.25e9, 0.25, 2);
g.biothf = [vp1, vp2b, vs];
[vp1, vs] = biothfb(3200, 2000, 37e9, 44e9, 2650, 1000, 2.25e9, 0.25, 2);
g.biothfb = [vp1, vs];
[vp1, freq, vp2b, vs, q1, q2, qs] = biot(3200, 2000, 37e9, 44e9, 2650, 1000, ...
    2.25e9, 1e-3, 0.25, 1e-12, 1e-5, 2, 0, 6, 'none');
g.biot = [vp1(:), freq(:), vp2b(:), vs(:), q1(:), q2(:), qs(:)];
fl = [0.05e9 2.25e9; 200 1000; 2e-5 1e-3];
[vp, k, atn, fw, kinf, klf] = patchw(12e9, 14e9, 37e9, 44e9, 2650, 0.25, ...
    1e-12, fl, 0.3, 0.1, logspace(-2, 4, 20));
g.patchw = [vp(:), real(k(:)), imag(k(:)), atn(:)];
g.patchw_lims = [kinf, klf];

% --- fluid properties (Phase 2) ----------------------------------------
[Kreuss,rhoeff,Kvoigt,vpb,rhob,Kb,vpo,rhoo,Ko,vpg,rhog,Kg,gor] = ...
    flprop(0, 35000, 30, 0.6, 100, 0, 0, 30, 80, 0.3, 0.2);
g.flprop = [Kreuss,rhoeff,Kvoigt,vpb,rhob,Kb,vpo,rhoo,Ko,vpg,rhog,Kg,gor];
[k, rho, vp] = co2prop(60, 15);
g.co2prop = [k, rho, vp];

% --- effective medium & cracks (Phase 3) -------------------------------
[kbr, mubr] = berryscm([37 2.2], [44 0], [1 0.1], [0.7 0.3]);
g.berryscm = [kbr, mubr];
[kbr, mubr, por] = berrysc(37, 44, 2.2, 0, 1, 0.1);
g.berrysc = [kbr(:), mubr(:), por(:)];
[kbr, mubr] = berryscp([37 2.2 2.2], [44 0 0], [1 0.01 0.5], [0.8 0.05 0.15], [0 0.05 0.2]);
g.berryscp = [kbr(:), mubr(:)];
[k, mu, por] = dem(37, 44, 2.2, 0, 0.1, 1);
g.dem = [k(:), mu(:), por(:)];  % adaptive steps: compare by interpolation
[k, mu] = dem1(37, 44, 2.2, 0, 0.2, 1, 0.35);
g.dem1 = [k, mu];
[Ctih, den] = hudson(0.05, 0.01, 2.25, 1.0, 37, 44, 2.65, 3);
g.hudson = Ctih; g.hudson_den = den;
[Vp0, Vs0, e, gg2, d, Ctih] = hudson1(0.05, 0.01, 2.25, 37, 44, 2.6, 3);
g.hudson1 = [Vp0, Vs0, e, gg2, d];
[C, den] = hudson3([0.03 0.02 0.01], [0.01 0.01 0.01], 2.25, 1.0, 37, 44, 2.65);
g.hudson3 = C; g.hudson3_den = den;
% NOTE: hudsonF.m has two known bugs (density porosity 4*pi*ar/(3*cd);
% missing mu^2 in the shear U3 terms) fixed in the port — its raw output
% will NOT match hudson_fisher. Kept here for reference only.
[C, den] = hudsonF(0.05, 0.01, 2.25, 1.0, 37, 44, 2.65, 0.4);
g.hudsonF_raw = C; g.hudsonF_raw_den = den;
[Vp0, Vs0, e, gg2, d, C] = hudsoncone(0.05, 0.01, 2.25, 37, 44, 2.65, 30*pi/180, 3);
g.hudsoncone = C;  % port takes the angle in degrees: 30
g.echeng = echeng([66.67 7.67 66.67 44 44], 0.02, 0.1, 2.25);

% --- granular & permeability (Phase 4) ---------------------------------
[k, gg3, phi, cnum] = hertzmind(37, 44, 0.02, [0.3 0.36 0.4]);
g.hertzmind = [k(:), gg3(:), phi(:), cnum(:)];
[vp, vs, ro, phi, cnum] = hertzmindv(6.008, 4.075, 2.65, 0.02, [0.3 0.36 0.4]);
g.hertzmindv = [vp(:), vs(:), ro(:), phi(:), cnum(:)];
% NOTE: Cem.m hard-codes 3.14 for pi in alam/alamtau; the port uses pi, so
% these differ by <0.1%. Kept for reference.
g.Cem_raw = Cem(0.38, 8.5, 45, 0.064, 45, 0.064, 0, 2);
% NOTE: Johnson.m's 5th output C is the SCALAR contact constant, not the
% stiffness tensor (the tensor is overwritten). Only the first four
% outputs are usable as reference values.
[Vp1, Vp3, s1, s3] = Johnson(44, 0.06, 250e-6, 9, 0.36, -1e-3, -2e-3, 2650, 4*44/(1-0.06));
g.Johnson = [Vp1, Vp3, s1, s3];
% NOTE: John_Makse.m cannot run (uses Z before assignment; C12 undefined),
% so it has no golden values; johnson_makse is a documented reconstruction.
phiv = [0.05 0.1 0.2 0.3];
g.KozCarmE = KozCarmE(phiv, 250);
g.FredrichE = FredrichE(phiv, 100);
g.PandaLakeKCE = PandaLakeKCE(phiv, 250);
g.ModKozCarm = ModKozCarm(phiv, 60, 2, 0.02);
g.CoatDum = CoatDum(phiv, 0.15);
g.Coates = Coates(phiv, 0.15);
g.PandaLake = PandaLake(phiv, 2, 0.25, 650, 0.4);
% NOTE: Owolabi.m cannot be called with arguments at all: it declares
% `function e = Owolabi(Phi,Swi)` but line 54 calls `Owo(Phi,Swr)`, and
% Swr is never defined. Only the interactive no-argument path works.
% (Found by running it; static reading had missed this one.)
g.Bloch = Bloch(1.2, 2.0, 10);
% NOTE: BernabeE.m cannot be called non-interactively (nargin==5 test on a
% 4-argument function, inner call missing Phi, output never assigned).

% --- AVO (Phase 5) ------------------------------------------------------
angs = [0 12 28];
for m = 1:4
    g.(sprintf('avopp%d', m)) = avopp(2.6, 1.2, 2.3, 2.2, 1.35, 2.05, angs, m);
end
for m = 1:7
    g.(sprintf('avops%d', m)) = avops(2.6, 1.2, 2.3, 2.2, 1.35, 2.05, [5 18 30], m);
end
[A, B1, B2, E1, E2] = avo_abe(2.6, 1.2, 2.3, 2.2, 1.35, 2.05);
g.avo_abe = [A, B1, B2, E1, E2];
vpl = [2.6 2.8 2.2 3.0 2.7]'; vsl = [1.2 1.4 1.35 1.5 1.3]'; rol = [2.3 2.35 2.05 2.4 2.32]';
[ippn, ipsn, ispn, ipp, ips, isp] = eimp(vpl, vsl, rol, 15);
g.eimp = [ippn, ipsn, ispn, ipp, ips, isp];
% NOTE: eimp2.m sets vpvs = mean(vp./vs) when K is omitted, which is not
% 1/mean(vs./vp); pass K explicitly to compare with elastic_impedance.
[ippn, ipsn, ispn, ipp, ips, isp] = eimp2(vpl, vsl, rol, 20, mean(vsl./vpl));
g.eimp2 = [ippn, ipsn, ispn, ipp, ips, isp];

% --- seismic & signal (Phase 6) ----------------------------------------
% NOTE: sourcewvlt.m is missing, so an explicit wavelet must be supplied.
% An EVEN-length wavelet: kennet.m calls hanning(n/2), which errors for
% odd n. (That is precisely the failure kennett_aux.m's round() fixes,
% and which the port handles; only the even path can be captured here.)
tw = ((0:127) - 63.5) * 0.001;
wv = (1 - 2*(pi*30*tw).^2) .* exp(-(pi*30*tw).^2);
lyr2 = [2000 2000 80; 2600 2300 90];
lyr3 = [2000 2000 40; 3200 2500 15; 2400 2150 60];
[wz, pz, tf] = kennet(lyr2, wv, 0.001, 2, 0, -1);
g.kennet_wz = wz(:); g.kennet_pz = pz(:); g.kennet_tf = tf;
[wz, pz] = kennet(lyr3, wv, 0.001, 0, 0, -1);
g.kennet_prim_wz = wz(:); g.kennet_prim_pz = pz(:);
[pz, wz] = pgator(lyr2, wv, 0.001, 0);
g.pgator_pz = pz(:); g.pgator_wz = wz(:);
per = repmat([2000 2000 5; 3000 2400 5], 200, 1);
[fd, vd] = kenfdisp(per, logspace(-1, 4, 15));
g.kenfdisp = [fd(:), vd(:)];
[tt, rt, emtt] = kenfrtt(lyr3, 30);
g.kenfrtt = [tt(:), rt(:), emtt(:)];
% NOTE: ezseis.m always opens an inputdlg and errors on the R<=1 branch
% (undefined fc), so it produces no golden values.
rng_data = [1 2 3 4 5 4 3 2 1 0 -1 -2]';
g.blockav = blockav(rng_data, 4);
% NOTE: fftplot.m documents its third output as "frequency step vector
% corresponding to AMP and PHASE" but returns the scalar step; the port's
% spectrum() returns the axis itself, as documented. Record both.
[amp, ph, ds] = fftplot(rng_data', 0.004);
g.fftplot_amp = amp(:); g.fftplot_phase = ph(:); g.fftplot_step = ds;
[ia, ip, ifr] = iatrib(rng_data);
g.iatrib_amp = ia(:); g.iatrib_phi = ip(:); g.iatrib_freq = ifr(:);

% --- stats & io (Phase 7) ----------------------------------------------
d2 = [0.1 2.0; 0.4 2.3; 0.2 2.1; 0.35 2.25; 0.15 2.05; 0.3 2.2];
[nn, xx1, xx2] = hist2d(d2, 4, 3);
g.hist2d_counts = nn; g.hist2d_c1 = xx1(:); g.hist2d_c2 = xx2(:);
[nn, xx1, xx2] = hist2d(d2, [0.1 0.2 0.3 0.4], [2.0 2.15 2.3]);
g.hist2d_centres_counts = nn;
% NOTE: hist3d.m's weighted path calls hist2d with 4 args (hist2d takes 3)
% and its 1-column path calls a missing hist1d, so only the unweighted
% 3-column case can produce golden values.
d3 = [d2, [1.0; 1.4; 1.1; 1.35; 1.05; 1.3]];
[nn3, a1, a2, a3] = hist3d(d3, 3);
g.hist3d_counts = nn3(:);
% NOTE: monte.m / monteccdf.m draw from rand/randn with no seed argument,
% so they have no reproducible golden values; the port is tested by
% distributional invariants instead.
% NOTE: loadlas.m requires an actual LAS file on disk; the port is tested
% against a fixture written by the test suite.

% --- write -------------------------------------------------------------
fid = fopen(fullfile(outdir, 'phase1.json'), 'w');
fprintf(fid, '%s', jsonencode(g));
fclose(fid);
disp('Wrote golden fixtures for Phases 1-2.');
