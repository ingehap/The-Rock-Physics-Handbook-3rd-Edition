function [k, u] = v2ku(vp, vs, rho)
% Shim for v2ku, which hertzmindv.m calls but which is MISSING from
% RPHtools (it is listed in Contents.m only).
%
% This is the same reconstruction the port uses in
% rphtools.moduli.velocity_to_moduli, so the golden values it feeds into
% hertzmindv validate the REST of that function (the Hertz-Mindlin core,
% the coordination-number table, the density) against MATLAB. The
% conversion itself is shared by definition and cannot be cross-checked
% this way.
u = rho .* vs.^2;
k = rho .* (vp.^2 - (4/3) .* vs.^2);
