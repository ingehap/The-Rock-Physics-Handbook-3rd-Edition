"""Well-log file input.

Port of the following RPHtools MATLAB function:

===============  ==============  ===============================
MATLAB           Python          Notes
===============  ==============  ===============================
``loadlas.m``    `load_las`      File dialog replaced by a path.
===============  ==============  ===============================

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- ``loadlas.m`` opened a ``uigetfile`` dialog when called with no
  arguments; `load_las` always takes a path.
- The original masked every value ``<= -999`` as missing. That is kept as
  the default because it catches the conventional ``-999.25`` null, but
  the file's own declared ``NULL`` is used when present, and either can be
  overridden with `null_value`.
- Curve names are lowercased with non-word characters replaced by
  underscores, as in the original, so they are valid Python identifiers.
  Unlike ``loadlas.m``, which used ``strtok`` and so folded the unit into
  the name (``RHOB.K/M3`` became ``rhob_k_m3``), the mnemonic is split at
  its ``.`` separator: the name is ``rhob`` and the unit is reported
  separately in `LasFile.units`.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

__all__ = ["LasFile", "load_las"]


class LasFile(NamedTuple):
    """Contents of a LAS well-log file."""

    data: np.ndarray
    """Numeric data, ``(n_samples, n_curves)``, nulls replaced by NaN."""
    columns: list
    """Curve names, lowercased and sanitized."""
    header: str
    """Everything above the data section, verbatim."""
    curves: dict
    """Mapping of curve name to its column of `data`, e.g.
    ``las.curves["dt"]``."""
    units: dict
    """Mapping of curve name to its declared unit (empty string if none)."""


def _split_mnemonic(line):
    """Split a LAS curve line into a sanitized name and its unit.

    Lines look like ``MNEM.UNIT   DATA : DESCRIPTION``, and the mnemonic
    field may be space-padded before the dot (``DT  .US/M``), so the split
    is on the first dot of the field rather than on whitespace.
    """
    field = line.split(":")[0]
    mnemonic, dot, rest = field.partition(".")
    if not dot:  # no unit declared
        tokens = field.split()
        mnemonic, rest = (tokens[0] if tokens else ""), ""
    unit = rest.split()[0] if rest.split() else ""
    return re.sub(r"\W", "_", mnemonic.strip()).strip("_").lower(), unit


def load_las(path, null_value=None):
    """Read a LAS (Log ASCII Standard) well-log file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the LAS file.
    null_value : float, optional
        Value marking missing data. By default the file's declared
        ``NULL`` is used if present; values ``<= -999`` are masked as
        well, matching ``loadlas.m``.

    Returns
    -------
    LasFile
        Named tuple ``(data, columns, header, curves, units)``; individual
        curves are reached by name through ``curves``.

    Notes
    -----
    Port of ``loadlas.m``. Handles LAS 2.0 layout: the ``~CURVE`` section
    supplies curve names (one per line, mnemonic first, ``#`` comments
    skipped) and ``~A`` begins the whitespace-delimited data block.

    Examples
    --------
    >>> las = load_las("well.las")            # doctest: +SKIP
    >>> vp = 1e6 / las.curves["dt"]           # doctest: +SKIP
    """
    header_lines = []
    columns = []
    units = []
    numbers = []
    declared_null = None

    in_curves = False
    in_data = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if in_data:
                numbers.extend(line.split())
                continue

            header_lines.append(line)
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("~A"):
                in_curves, in_data = False, True
                continue
            if in_curves and stripped.startswith("~"):
                in_curves = False
            if in_curves and stripped and not stripped.startswith("#"):
                name, unit = _split_mnemonic(stripped)
                columns.append(name)
                units.append(unit)
            if upper.startswith("~CURVE") or upper == "~C":
                in_curves, in_data = True, False
            if declared_null is None and upper.startswith("NULL"):
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped.split(":")[0][4:])
                if match:
                    declared_null = float(match.group())

    if not columns:
        raise ValueError(f"no ~CURVE section found in {path}")

    values = np.array(numbers, dtype=float)
    ncurve = len(columns)
    if values.size % ncurve:
        raise ValueError(
            f"data length {values.size} is not a multiple of the {ncurve} declared curves"
        )
    data = values.reshape(-1, ncurve)

    if null_value is not None:
        data[data == null_value] = np.nan
    else:
        if declared_null is not None:
            data[data == declared_null] = np.nan
        data[data <= -999] = np.nan

    return LasFile(
        data=data,
        columns=columns,
        header="\n".join(header_lines),
        curves={name: data[:, k] for k, name in enumerate(columns)},
        units=dict(zip(columns, units)),
    )
