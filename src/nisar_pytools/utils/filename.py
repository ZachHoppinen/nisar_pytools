"""Parse NISAR product filenames to extract metadata without opening the file.

NISAR filename convention:
    NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001
    NISAR_L2_PR_GUNW_006_149_A_024_009_4000_SH_20251202T123756_20251202T123831_20260107T123757_20260107T123832_X05010_N_F_J_001

Fields (GSLC):
    0: NISAR
    1: Level (L2)
    2: Processing type (PR)
    3: Product type (GSLC, GUNW, etc.)
    4: Cycle number
    5: Track/relative orbit number
    6: Pass direction (A=ascending, D=descending)
    7: Frame number
    8: Subswath/mode
    9: Polarization mode
    10: Dithering (A/D)
    11: Start datetime
    12: End datetime
    13: Composite release ID
    14-16: Various flags
    17: Counter
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class NISARFilenameInfo:
    """Parsed metadata from a NISAR product filename."""

    filename: str
    level: str
    processing_type: str
    product_type: str
    cycle: int
    track: int
    direction: str
    frame: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    composite_release_id: str
    is_qa: bool

    @property
    def is_ascending(self) -> bool:
        return self.direction == "Ascending"


_DATETIME_RE = re.compile(r"\d{8}T\d{6}")


def parse_filename(filename: str | Path) -> NISARFilenameInfo:
    """Parse a NISAR product filename to extract metadata.

    Works on full paths or bare filenames. Handles GSLC, GUNW, and
    other L1/L2 products.

    Parameters
    ----------
    filename : str or Path
        NISAR product filename or path.

    Returns
    -------
    NISARFilenameInfo
        Parsed metadata.

    Raises
    ------
    ValueError
        If the filename does not match the expected NISAR naming convention.
    """
    name = Path(filename).stem
    # Strip QA suffixes
    is_qa = "_QA_" in name
    parts = name.split("_")

    if len(parts) < 13 or parts[0] != "NISAR":
        raise ValueError(f"Not a recognized NISAR filename: {name}")

    # Find datetime fields by regex (handles varying positions for GSLC vs GUNW)
    datetimes = _DATETIME_RE.findall(name)
    if len(datetimes) < 2:
        raise ValueError(f"Could not find start/end datetimes in filename: {name}")

    product_type = parts[3]
    direction = "Ascending" if parts[6] == "A" else "Descending"

    # Start/end times: first two datetimes for GSLC, or specific positions for GUNW
    start_time = pd.Timestamp(datetimes[0])
    end_time = pd.Timestamp(datetimes[1])

    return NISARFilenameInfo(
        filename=str(filename),
        level=parts[1],
        processing_type=parts[2],
        product_type=product_type,
        cycle=int(parts[4]),
        track=int(parts[5]),
        direction=direction,
        frame=int(parts[7]),
        start_time=start_time,
        end_time=end_time,
        composite_release_id=parts[-4] if not is_qa else parts[-6],
        is_qa=is_qa,
    )
