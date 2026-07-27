function m = nanmean(x, dim)
% Shim for the Statistics Toolbox NANMEAN, which blockav.m calls.
% Mean along DIM (default 1), ignoring NaN.
if nargin < 2, dim = 1; end
mask = ~isnan(x);
x(~mask) = 0;
n = sum(mask, dim);
m = sum(x, dim) ./ n;
m(n == 0) = NaN;
