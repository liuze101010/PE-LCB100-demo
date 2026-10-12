"""Unit constants and conversions for GROMACS rheology analysis."""

BAR_TO_PA = 1.0e5
PS_TO_S = 1.0e-12
NM3_TO_M3 = 1.0e-27
KB_J_PER_K = 1.380649e-23


def pressure_bar_to_pa(value):
    return value * BAR_TO_PA


def time_ps_to_s(value):
    return value * PS_TO_S


def volume_nm3_to_m3(value):
    return value * NM3_TO_M3


def helfand_slope_bar2ps_to_pa2s(slope_bar2_ps):
    """Convert d<(int P dt)^2>/dt from bar^2 ps to Pa^2 s."""
    return slope_bar2_ps * (BAR_TO_PA ** 2) * PS_TO_S


def viscosity_from_helfand_slope_pa2s(slope_pa2_s, volume_nm3, temperature_K):
    """Return shear viscosity in Pa s from one off-diagonal Helfand slope.

    The factor 1/2 is the Einstein-Helfand prefactor for a single
    off-diagonal stress component.
    """
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if volume_nm3 <= 0:
        raise ValueError("volume_nm3 must be positive")
    return volume_nm3_to_m3(volume_nm3) * slope_pa2_s / (2.0 * KB_J_PER_K * temperature_K)


def viscosity_from_helfand_slope_bar2ps(slope_bar2_ps, volume_nm3, temperature_K):
    return viscosity_from_helfand_slope_pa2s(
        helfand_slope_bar2ps_to_pa2s(slope_bar2_ps),
        volume_nm3,
        temperature_K,
    )

