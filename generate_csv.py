#!/usr/bin/env python3
"""Generate a thermophysical-property snapshot CSV with CoolProp.

This is only an EXAMPLE of how an input CSV for the SVD script can be produced.
Any other source (another library, experimental data, ...) works as well.

Layout:
    Name,<fluid>,...
    T,P,rho,e,s,...            <- column names
    one row per (T, P) pair; T is the slow variable, P cycles fastest.

Requires: pip install CoolProp numpy
"""
import numpy as np
import CoolProp.CoolProp as CP

# ----------------------------- USER SETTINGS -----------------------------
FLUID  = "CO2"                                  # any CoolProp fluid name, e.g. "Oxygen", "Water"
P_MIN_BAR, P_MAX_BAR, N_P = 100.0, 200.0, 120   # pressure range [bar] and number of points
T_MIN_K,   T_MAX_K,   N_T = 350.0, 400.0,  80   # temperature range [K] and number of points
OUTFILE  = "input.csv"                          # name of the output CSV file
SIG_DIGITS = 4                                  # significant digits written to the file
# --------------------------------------------------------------------------

# Column name -> function(P, T) returning the value (SI, mass basis).
# Order here = order of columns in the CSV. Add/remove entries freely.
def prop(key):
    return lambda P, T: CP.PropsSI(key, "P", P, "T", T, FLUID)

PROPERTIES = {
    "rho":     prop("D"),                                   # density            [kg/m3]
    "e":       prop("U"),                                   # internal energy    [J/kg]
    "s":       prop("S"),                                   # entropy            [J/(kg K)]
    "c_p":     prop("CPMASS"),                              # isobaric heat cap. [J/(kg K)]
    "c_v":     prop("CVMASS"),                              # isochoric heat cap.[J/(kg K)]
    "sos":     prop("A"),                                   # speed of sound     [m/s]
    "alpha":   prop("isobaric_expansion_coefficient"),      # isobaric expansion [1/K]
    "kappa_T": prop("isothermal_compressibility"),          # isothermal compr.  [1/Pa]
    # isentropic compressibility is not a CoolProp output: kappa_S = 1/(rho*sos^2) [1/Pa]
    "kappa_S": lambda P, T: 1.0 / (CP.PropsSI("D", "P", P, "T", T, FLUID)
                                   * CP.PropsSI("A", "P", P, "T", T, FLUID) ** 2),
    "mu":      prop("V"),                                   # viscosity          [Pa s]
    "kappa":   prop("L"),                                   # conductivity       [W/(m K)]
    "nu":      lambda P, T: CP.PropsSI("V", "P", P, "T", T, FLUID) / CP.PropsSI("D", "P", P, "T", T, FLUID),  # kinematic viscosity [m2/s]
}


def main():
    Ps = np.linspace(P_MIN_BAR, P_MAX_BAR, N_P) * 1e5      # bar -> Pa
    Ts = np.linspace(T_MIN_K, T_MAX_K, N_T)

    cols = ["T", "P"] + list(PROPERTIES)
    pad = "," * (len(cols) - 2)
    fmt = f"{{:.{SIG_DIGITS}g}}"

    with open(OUTFILE, "w") as f:
        f.write(f"Name,{FLUID}{pad}\n")
        f.write(",".join(cols) + "\n")
        for T in Ts:                       # slow variable
            for P in Ps:                   # fast variable
                vals = [T, P] + [fn(P, T) for fn in PROPERTIES.values()]
                f.write(",".join(fmt.format(v) for v in vals) + "\n")

    print(f"Wrote {N_P * N_T} rows to {OUTFILE}")


if __name__ == "__main__":
    main()
