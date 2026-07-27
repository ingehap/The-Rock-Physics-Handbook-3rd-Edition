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
% Covers the Phase 1 functions. Extend per phase as the port grows.

addpath('RPHtools');
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
g.Owolabi = Owolabi(phiv, 0.8);
g.Bloch = Bloch(1.2, 2.0, 10);
% NOTE: BernabeE.m cannot be called non-interactively (nargin==5 test on a
% 4-argument function, inner call missing Phi, output never assigned).

% --- write -------------------------------------------------------------
fid = fopen(fullfile(outdir, 'phase1.json'), 'w');
fprintf(fid, '%s', jsonencode(g));
fclose(fid);
disp('Wrote golden fixtures for Phases 1-2.');
