function m = harmmean(x)
% Shim for the Statistics Toolbox HARMMEAN, which kenfrtt.m calls.
% Unweighted harmonic mean along the first non-singleton dimension.
if isvector(x)
  m = numel(x) / sum(1 ./ x);
else
  m = size(x, 1) ./ sum(1 ./ x, 1);
end
