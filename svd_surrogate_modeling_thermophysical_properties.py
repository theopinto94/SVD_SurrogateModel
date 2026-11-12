#!/usr/bin/env python3
"""
Build an SVD surrogate from a CSV file of tabulated data and store everything in a
YAML file.

The software was designed for thermophysical properties (inputs: temperature and
pressure), but nothing is hardcoded: the two inputs can be ANY quantities, and their
names are simply read from the CSV headers. In the code they are called input_1 and
input_2.

    - FIRST  CSV column (input_1) -> ROWS    of the matrices
    - SECOND CSV column (input_2) -> COLUMNS of the matrices

Every other column of the CSV (any name chosen by the user) is an output (e.g. a
property). For each of them one matrix is built and decomposed with an SVD:

    <input_1>_<input_2>_to_<column name>     <column name>(input_1, input_2)
                                             rows = input_1, columns = input_2

Example: with headers "T,P,rho" the matrix is "T_P_to_rho" (rows = T, columns = P);
with headers "P,T,rho" it is "P_T_to_rho" (rows = P, columns = T).

Expected CSV layout:

    A1: "Name"              B1: <name of the studied substance / system>
    row 2: column headers   -> column 1 = input_1 (rows), column 2 = input_2 (columns),
                               their headers are used as their names in the YAML file.
                               Every following column is an output; its header is used
                               as its name. All headers must be distinct and non-empty.
    row 3+: data, one row per (input_1, input_2) point. All values must be numbers.
            The data must form a complete grid (every input_1 value paired with every
            input_2 value, no duplicates). The number of distinct input_1 values and of
            distinct input_2 values may differ (rectangular matrices).

Usage:
    python3 svd_surrogate_modeling_thermophysical_properties.py <name_of_your_file>.csv                             # full rank and writes SVD_Surrogate_<Name>.yaml
    python3 svd_surrogate_modeling_thermophysical_properties.py <name_of_your_file>.csv -k <truncated_rank>         # with truncated SVD rank
    python3 svd_surrogate_modeling_thermophysical_properties.py <name_of_your_file>.csv -o <name_of_your_file>.yaml # with custom output name
"""

import argparse
import csv
import os
import sys
import numpy as np


# --------------------------------------------------------------------------- #
# Reading CSV file
# --------------------------------------------------------------------------- #
def read_csv(path):
    """Return (name, input_names, input_1, input_2, properties).

    input_names = (name of input_1, name of input_2), read from the headers of the
    first two columns. input_1 (first column) gives the rows of the matrices and
    input_2 (second column) gives the columns."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        raise ValueError("CSV must have 1 metadata row, a header row and data rows.")

    substance_name = rows[0][1].strip()       # B1

    header = [h.strip() for h in rows[1]]
    while header and not header[-1]:          # ignore trailing empty header cells
        header.pop()

    if len(header) < 3:
        raise ValueError("The CSV needs at least 3 columns: input_1 (rows), "
                         "input_2 (columns) and one property.")

    if "" in header:
        raise ValueError("Empty column header found: every column needs a name.")
    duplicates = sorted({h for h in header if header.count(h) > 1})
    if duplicates:
        raise ValueError(f"Duplicate column header(s): {duplicates}.")

    input_names = (header[0], header[1])
    prop_names = header[2:]                   # everything after the first two columns

    data = [r for r in rows[2:] if any(c.strip() for c in r)]

    def column(j, label):
        try:
            return np.array([float(r[j]) for r in data])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Could not read {label} as numbers: {e}")

    input_1 = column(0, f"column 1 '{input_names[0]}' (matrix rows)")
    input_2 = column(1, f"column 2 '{input_names[1]}' (matrix columns)")
    properties = {h: column(j, f"column '{h}'") for j, h in enumerate(prop_names, start=2)}
    return substance_name, input_names, input_1, input_2, properties


# --------------------------------------------------------------------------- #
# Building the matrices
# --------------------------------------------------------------------------- #
def build_matrix(input_1, vec_1, input_2, vec_2, values):
    """M[i, j] = value at (input_1 = vec_1[i], input_2 = vec_2[j])
    (rows = input_1, columns = input_2).

    The CSV must be a complete grid: every (input_1, input_2) combination appears
    exactly once. The number of distinct input_1 and input_2 values can be equal or
    different."""
    i = np.searchsorted(vec_1, input_1)
    j = np.searchsorted(vec_2, input_2)

    known = np.zeros((len(vec_1), len(vec_2)), dtype=bool)
    known[i, j] = True
    if not known.all():
        raise ValueError("The data does not form a complete input_1 x input_2 grid "
                         "(some combinations of the first two columns are missing).")
    if len(values) != known.size:
        raise ValueError("The CSV contains duplicate rows (same values in the first two columns).")

    M = np.zeros(known.shape)
    M[i, j] = values
    return M


# --------------------------------------------------------------------------- #
# Writing YAML file
# --------------------------------------------------------------------------- #
def fmt_list(arr):
    """Flow-style YAML list, flattened row-major, full float precision."""
    return "[" + ", ".join(repr(float(x)) for x in np.asarray(arr).ravel()) + "]"

def write_yaml(path, meta, vectors, matrices):
    """`matrices` maps a name such as "T_P_to_rho" (i.e. "<input_1>_<input_2>_to_<property>")
    to its (U, S, VT)."""
    lines = ["##### SUBSTANCE METADATA #####", "substance_metadata:"]
    for k, v in meta.items():
        lines.append(f"  substance_{k}: {v if isinstance(v, (int, str)) else repr(float(v))}")
    lines += ["", "", "##### SUBSTANCE SVD DATA #####", "substance_SVD_data:"]
    for k, v in vectors.items():
        lines.append(f"  substance_{k}_vector: {fmt_list(v)}")
    for key, (U, S, VT) in matrices.items():
        for tag, arr in (("U", U), ("S", S), ("VT", VT)):
            lines.append(f"  substance_{key}_matrix_{tag}: {fmt_list(arr)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv_file")
    ap.add_argument("-k", "--truncate", type=int,
                help="keep only the K largest singular values (default: full rank, no truncation)")
    ap.add_argument("-o", "--output", help="output YAML (default: SVD_Surrogate_<Name>.yaml)")
    args = ap.parse_args()

    if args.truncate is not None and args.truncate < 1:
        ap.error("--truncate must be at least 1.")

    try:
        substance_name, (name_1, name_2), input_1, input_2, props = read_csv(args.csv_file)
    except ValueError as e:
        sys.exit(f"Error reading CSV: {e}")

    vec_1, vec_2 = np.unique(input_1), np.unique(input_2)   # sorted unique grid values
    rank = min(len(vec_1), len(vec_2))                      # full rank of the rectangular matrices

    meta = {
        "name": substance_name,
        f"{name_1}_min": vec_1.min(), f"{name_1}_max": vec_1.max(), f"{name_1}_values": len(vec_1),
        f"{name_2}_min": vec_2.min(), f"{name_2}_max": vec_2.max(), f"{name_2}_values": len(vec_2),
        "SVD_rank": rank,
        "SVD_truncated_rank": min(args.truncate, rank) if args.truncate else rank
    }
    vectors = {name_1: vec_1, name_2: vec_2}

    # One SVD per property column: rows = input_1, columns = input_2
    matrices = {}
    try:
        for prop, values in props.items():
            M = build_matrix(input_1, vec_1, input_2, vec_2, values)
            U, S, VT = np.linalg.svd(M, full_matrices=False)
            k = min(args.truncate, len(S)) if args.truncate else len(S)
            matrices[f"{name_1}_{name_2}_to_{prop}"] = (U[:, :k], S[:k], VT[:k, :])
    except ValueError as e:
        sys.exit(f"Error building matrices: {e}")

    out = args.output or f"SVD_Surrogate_{substance_name}.yaml"
    write_yaml(out, meta, vectors, matrices)
    print(f"Inputs: rows = '{name_1}' ({len(vec_1)} values), columns = '{name_2}' ({len(vec_2)} values)")
    print(f"Properties processed: {list(props)}")
    print(f"Written {out} ({os.path.getsize(out) / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
