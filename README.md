# SVD Surrogate Builder

`svd_surrogate_modeling_thermophysical_properties.py` reads a CSV file containing data tabulated on a two-input grid, computes the singular value decomposition (SVD) of every output column as a function of the two inputs, and writes metadata, grid vectors, and SVD data to a YAML file. The SVD can optionally be truncated.

The software was designed with thermophysical properties in mind (temperature and pressure as inputs, properties such as density or viscosity as outputs), but **nothing is hardcoded**: the two inputs can be any quantities, and their names are simply read from the CSV headers (`T`, `P`, `time`, `x`, `table`, ... anything). In the script they are called `input_1` and `input_2`.

> **Important:** the **first CSV column (`input_1`) gives the rows** of the SVD matrices and the **second CSV column (`input_2`) gives the columns**. Swapping the two columns in the CSV transposes every matrix.

The script is also property-neutral: it does not expect any particular property names. Every column after the two inputs is treated as an output (property) and decomposed.

## Requirements

- **Python 3.8+**
- The only library that needs to be installed is **NumPy**.

Install NumPy with:

```bash
python3 -m pip install numpy
```

The other imported modules (`argparse`, `csv`, `os`, and `sys`) are part of Python's standard library.

## Usage

```bash
python3 svd_surrogate_modeling_thermophysical_properties.py input.csv
```

By default, the output is written to `SVD_Surrogate_<Name>.yaml`, where `<Name>` is the substance name in the CSV metadata, and the SVD is kept at full rank.

| Option | Meaning |
|---|---|
| `-o FILE`, `--output FILE` | Choose the output YAML filename. |
| `-k K`, `--truncate K` | Keep only the `K` largest singular values (`K` must be at least 1). If `K` is larger than the full rank, the full rank is used. |

Examples:

```bash
python3 svd_surrogate_modeling_thermophysical_properties.py input.csv -o my_output.yaml
python3 svd_surrogate_modeling_thermophysical_properties.py input.csv -k 10
```

## Creating the input CSV

The CSV file can come from anywhere: a thermophysical library, a simulation, experimental data you collected yourself, etc. The only requirement is the format described in the next section.

`generate_csv.py` is provided **only as an example** of how to produce such a file, using the [CoolProp](http://www.coolprop.org) library (`pip install CoolProp numpy`). It is not mandatory to use it in any way.

## CSV format

The CSV must contain one metadata row, a header row, and then the data rows.

| Location | Content |
|---|---|
| A1 / B1 | `Name` / substance (or system) name |
| Row 2 | Column headers |
| Row 3+ | Numeric data, one row per `(input_1, input_2)` grid point |

### Column rules

1. **Column 1 is `input_1` and gives the rows of the matrices. Column 2 is `input_2` and gives the columns of the matrices.** This is assumed by position. Their headers are read and used as the names of the inputs in the YAML file (for example `T` and `P`, which gives `T_P_to_...` keys with rows = `T` and columns = `P`).
2. **Every other column is a property**, with any name you like. Its header is used as the property name in the output.
3. All headers (inputs and properties) must be non-empty and distinct from each other.
4. The script performs an SVD for **every** property column. There is no way to select properties; remove a column from the CSV if you do not want it processed.
5. The data must form a **complete grid**: every distinct `input_1` value must be paired with every distinct `input_2` value, and each combination must appear exactly once. There is no interpolation or extrapolation: the matrices contain only values from the CSV.
6. The number of distinct `input_1` values (`n_1`) and of distinct `input_2` values (`n_2`) can be equal (square matrices) or different (rectangular matrices).
7. Extra spaces around header names are ignored. Header names are case-sensitive.
8. All data must be numeric, including the two input columns.

### Choosing names

Input and property names become part of the YAML keys (`substance_<input_1>_<input_2>_to_<name>_matrix_U`, etc.), so it is best to keep them simple: letters, digits, and underscores, without spaces, colons, or `#`. Names containing characters that are special in YAML may produce an output file that cannot be parsed.

## Matrices generated

For each property column `<name>`, one matrix is built:

| Matrix key | Meaning |
|---|---|
| `<input_1>_<input_2>_to_<name>` | `<name>(input_1, input_2)` |

Rows correspond to the first input (sorted unique `input_1` values) and columns correspond to the second input (sorted unique `input_2` values), so each matrix has shape `n_1 × n_2`. It is not normalised. It is decomposed with NumPy's SVD (`full_matrices=False`), which gives `M = U · diag(S) · VT`.

The **full rank** is `min(n_1, n_2)`. It is computed from the data. By default all singular values are kept. With `-k K`, only the first `k = min(K, full rank)` singular values are kept, together with the first `k` columns of `U` and the first `k` rows of `VT`. The truncation is applied to every property, and the stored matrices reconstruct an approximation of the original: `M ≈ U · diag(S) · VT`.

## YAML output

In the tables below, `<input_1>` and `<input_2>` stand for the headers of the first two CSV columns (e.g. `T` and `P`).

The YAML file has two sections.

**`substance_metadata`**

| Key | Content |
|---|---|
| `substance_name` | Substance name |
| `substance_<input_1>_min`, `substance_<input_1>_max` | Range of the first input (rows) |
| `substance_<input_1>_values` | Number of distinct values of the first input (`n_1`) |
| `substance_<input_2>_min`, `substance_<input_2>_max` | Range of the second input (columns) |
| `substance_<input_2>_values` | Number of distinct values of the second input (`n_2`) |
| `substance_SVD_rank` | Full rank of the matrices, `min(n_1, n_2)` |
| `substance_SVD_truncated_rank` | Number of singular values actually stored, `k`. Equal to `substance_SVD_rank` when no truncation is requested |

**`substance_SVD_data`**

| Key | Content |
|---|---|
| `substance_<input_1>_vector` | Sorted unique values of the first input (rows of the matrices) |
| `substance_<input_2>_vector` | Sorted unique values of the second input (columns of the matrices) |
| `substance_<input_1>_<input_2>_to_<name>_matrix_U` | `U` from the SVD of `<name>(input_1, input_2)` |
| `substance_<input_1>_<input_2>_to_<name>_matrix_S` | Singular values |
| `substance_<input_1>_<input_2>_to_<name>_matrix_VT` | `VT` (V transposed) from the SVD |

Every array is written as a flat, row-major list at full float precision. When reading the YAML back, reshape `U` to `(n_1, k)`, `S` to `(k,)` and `VT` to `(k, n_2)`, where `k = substance_SVD_truncated_rank`, `n_1` is the length of the `<input_1>` vector and `n_2` the length of the `<input_2>` vector.

## Example CSV

```csv
Name,CO2
T,P,rho,e,mu
300,1e5,1.8,3.1e5,1.5e-5
300,5e5,9.2,3.0e5,1.5e-5
300,1e6,18.9,2.9e5,1.6e-5
400,1e5,1.3,3.9e5,1.9e-5
400,5e5,6.6,3.8e5,1.9e-5
400,1e6,13.3,3.7e5,2.0e-5
```

This is a complete grid with 2 values of `T` (rows) and 3 values of `P` (columns), so each matrix is `2 × 3` (rectangular) and the full rank is 2. The script would produce the matrices `T_P_to_rho`, `T_P_to_e`, and `T_P_to_mu`, and the metadata keys `substance_T_min`, `substance_T_max`, `substance_T_values`, `substance_P_min`, `substance_P_max`, and `substance_P_values`. Running it with `-k 1` would keep only the largest singular value of each. The values are illustrative only.

The same data with the first two columns named `x` and `y` instead would give `x_y_to_rho`, `substance_x_vector`, etc., and with the two columns swapped (`P,T,...`) the keys would be `P_T_to_rho` with rows = `P` and columns = `T`.

## Errors

| Message or condition | Meaning |
|---|---|
| `CSV must have 1 metadata row, a header row and data rows.` | The file has fewer than three rows. |
| `The CSV needs at least 3 columns: input_1 (rows), input_2 (columns) and one property.` | The header row has fewer than three columns. |
| `Empty column header found: every column needs a name.` | A header cell is empty before the last named column. |
| `Duplicate column header(s)` | A header appears more than once (inputs included). |
| `Could not read ... as numbers` | A column contains non-numeric or missing values. |
| `The data does not form a complete input_1 x input_2 grid (some combinations of the first two columns are missing).` | One or more grid combinations have no row. |
| `The CSV contains duplicate rows (same values in the first two columns).` | The same combination of the first two columns appears more than once. |
| `--truncate must be at least 1.` | `-k` was given a value below 1. |

Errors raised while reading the CSV are prefixed with `Error reading CSV:`, and those raised while building the matrices with `Error building matrices:`.

If a property is missing from the output, check that its column is present in the CSV, is not the first or second column (those are the inputs), and that the console output lists it under `Properties processed`.