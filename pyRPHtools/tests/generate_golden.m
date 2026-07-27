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

% --- write -------------------------------------------------------------
fid = fopen(fullfile(outdir, 'phase1.json'), 'w');
fprintf(fid, '%s', jsonencode(g));
fclose(fid);
disp('Wrote golden fixtures for Phase 1.');
