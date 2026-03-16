"""
advisory.py — Health advisory mapping for the Odisha AQI Advisor.

Function signature: get_advisory(aqi_value: float) -> tuple[str, str, str]
Returns: (category_name, advisory_message, hex_colour)
"""

# AQI_BANDS: list of (lower_inclusive, upper_exclusive, category, message, hex_colour)
# The last band uses float('inf') as upper bound.
AQI_BANDS = [
    (0,   51,          "Good",       "Air quality is satisfactory. All outdoor activities are safe.",                                    "#00B050"),
    (51,  101,         "Satisfactory","Acceptable. Sensitive groups should limit prolonged outdoor exertion.",                           "#92D050"),
    (101, 201,         "Moderate",   "Reduce prolonged outdoor exercise. Asthmatics and elderly take precautions.",                     "#FFD700"),
    (201, 301,         "Poor",       "Avoid outdoor physical activity. General public may experience discomfort.",                      "#FF7C00"),
    (301, 401,         "Very Poor",  "Stay indoors. Use air purifiers. Wear N95 masks outdoors.",                                       "#FF0000"),
    (401, float("inf"),"Severe",     "Health emergency. Avoid all outdoor exposure. Cancel outdoor events.",                            "#7B0023"),
]


def get_advisory(aqi_value: float) -> tuple:
    """Map an AQI value to a health advisory.

    Parameters
    ----------
    aqi_value : float
        A non-negative numeric AQI value.

    Returns
    -------
    tuple[str, str, str]
        (category_name, advisory_message, hex_colour)

    Raises
    ------
    ValueError
        If aqi_value is negative or non-numeric.
    """
    # Validate input
    try:
        aqi_value = float(aqi_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"aqi_value must be a numeric value, got: {aqi_value!r}"
        )
    if aqi_value < 0:
        raise ValueError(
            f"aqi_value must be non-negative, got: {aqi_value}"
        )

    for lower, upper, category, message, colour in AQI_BANDS:
        if lower <= aqi_value < upper:
            return (category, message, colour)

    # Fallback — should never be reached for valid non-negative input
    return ("Severe", AQI_BANDS[-1][3], AQI_BANDS[-1][4])
