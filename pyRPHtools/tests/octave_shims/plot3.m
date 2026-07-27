function varargout = plot3(varargin)
% No-op plotting shim for golden-value generation. Many RPHtools
% functions draw figures unconditionally; suppressing them keeps the
% numerical outputs identical while allowing headless execution.
varargout = cell(1, nargout);
[varargout{:}] = deal(0);
