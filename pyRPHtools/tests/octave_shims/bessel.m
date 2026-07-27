function y = bessel(nu, z)
% Shim for MATLAB's legacy BESSEL, which biot.m calls and modern
% MATLAB/Octave no longer provide with this signature.
% bessel(nu, z) was the Bessel function of the first kind.
y = besselj(nu, z);
