function y = hilbert(x)
% Shim for the Signal Toolbox HILBERT, which iatrib.m calls.
% Returns the analytic signal x + i*H(x), column-wise, using the
% standard FFT construction MATLAB documents.
was_row = isrow(x);
if was_row, x = x(:); end
n = size(x, 1);
f = fft(x, n, 1);
h = zeros(n, 1);
if mod(n, 2) == 0
  h(1) = 1; h(n/2 + 1) = 1; h(2:n/2) = 2;
else
  h(1) = 1; h(2:(n+1)/2) = 2;
end
y = ifft(f .* repmat(h, 1, size(x, 2)), n, 1);
if was_row, y = y.'; end
