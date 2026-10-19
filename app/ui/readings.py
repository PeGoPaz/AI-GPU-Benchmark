"""What each dashboard tile shows for a telemetry snapshot, and in what state.

Pure functions, no Qt: a snapshot goes in, a Reading comes out. Keeping the
formatting and the thresholds here means they are tested without a window,
and the tile widgets only have to display what they are given.
"""
from dataclasses import dataclass

from app.core.telemetry import decode_throttle_reasons

NORMAL, WARN, CRIT = "normal", "warn", "crit"

# Consumer cards start pulling clocks back in the low 80s; a core in the
# high 80s is at or past its rated limit. Headroom is NVML's own distance to
# the slowdown threshold, so it gets tighter bounds.
TEMP_WARN_C, TEMP_CRIT_C = 80, 87
HEADROOM_WARN_C, HEADROOM_CRIT_C = 10, 5
VRAM_WARN_PCT, VRAM_CRIT_PCT = 90, 97

NA = "N/A"


@dataclass(frozen=True)
class Reading:
    value: str
    unit: str = ""
    caption: str = ""
    state: str = NORMAL
    # The number behind `value`, for sparklines and gauges; None when the
    # card does not report it.
    number: float | None = None


def _above(value, warn, crit) -> str:
    return CRIT if value >= crit else WARN if value >= warn else NORMAL


# A card pinned at its power limit is being held back by it, which is worth
# a colour even before the throttle reasons say so.
POWER_WARN_FRACTION = 0.95


def temperature(data: dict) -> Reading:
    temp = data['temp_gpu']
    return Reading(f"{temp:.0f}", "°C", "core",
                   _above(temp, TEMP_WARN_C, TEMP_CRIT_C), float(temp))


def memory_temperature(data: dict) -> Reading:
    mem = data.get('temp_memory')
    if mem is None:
        return Reading(NA, caption="not reported by this card")
    return Reading(f"{mem:.0f}", "°C", "VRAM", number=float(mem))


def headroom(data: dict) -> Reading:
    room = data.get('temp_headroom_c')
    if room is None:
        return Reading(NA, caption="not reported by this card")
    state = CRIT if room <= HEADROOM_CRIT_C else WARN if room <= HEADROOM_WARN_C else NORMAL
    return Reading(f"{room:.0f}", "°C", "before thermal slowdown", state, float(room))


def power(data: dict, limit_w: float | None = None) -> Reading:
    # The poll's reading is smoothed; the peak, when the native sampler
    # supplies it, is the highest instant reading since the last poll.
    watts = data['power_w']
    peak = data.get('power_peak_w')
    caption = f"peak {peak:.0f} W" if peak is not None else "250 ms average"
    state = WARN if limit_w and watts >= POWER_WARN_FRACTION * limit_w else NORMAL
    return Reading(f"{watts:.0f}", "W", caption, state, float(watts))


def clock(data: dict) -> Reading:
    return Reading(f"{data['sm_clock_mhz']:.0f}", "MHz", "SM clock",
                   number=float(data['sm_clock_mhz']))


def utilization(data: dict) -> Reading:
    return Reading(f"{data['gpu_util_pct']:.0f}", "%", "GPU busy",
                   number=float(data['gpu_util_pct']))


def vram(data: dict) -> Reading:
    used, total = data['vram_used_mb'], data['vram_total_mb']
    pct = 100 * used / total if total else 0.0
    return Reading(f"{used / 1024:.1f}", "GiB", f"of {total / 1024:.1f} · {pct:.0f}%",
                   _above(pct, VRAM_WARN_PCT, VRAM_CRIT_PCT), float(used))


def fan(data: dict) -> Reading:
    pct, rpm = data.get('fan_speed_pct'), data.get('fan_rpm')
    if pct is None:
        return Reading(NA, caption="passively cooled")
    return Reading(f"{pct:.0f}", "%", f"{rpm} RPM" if rpm else "", number=float(pct))


def pcie(data: dict) -> Reading:
    gen, width = data.get('pcie_gen'), data.get('pcie_width')
    if gen is None or width is None:
        return Reading(NA)
    gen_max, width_max = data.get('pcie_gen_max'), data.get('pcie_width_max')
    # Only worth showing when the link is downshifted — at idle it drops to a
    # lower generation to save power, which is normal, not a broken slot.
    caption = ""
    if gen_max and width_max and (gen, width) != (gen_max, width_max):
        caption = f"max Gen{gen_max} x{width_max}"
    return Reading(f"Gen{gen} x{width}", caption=caption)


def throttle(data: dict) -> Reading:
    mask = data.get('throttle_reasons')
    if mask is None:
        return Reading(NA, caption="driver does not say")
    reasons = decode_throttle_reasons(mask)
    if not reasons:
        return Reading("None", caption="running free")
    return Reading(", ".join(reasons), caption="holding the clock down", state=WARN)


# Readings with a natural full scale get an arc gauge: (key, title, unit).
GAUGES = (
    ("temp", "Temperature", "°C"),
    ("power", "Power", "W"),
    ("vram", "VRAM", "GiB"),
    ("util", "Utilization", "%"),
)


def gauge_readings(data: dict, limit_w: float | None = None) -> dict[str, Reading]:
    """The four gauge readings by key. Power needs the card's limit to judge."""
    return {
        "temp": temperature(data),
        "power": power(data, limit_w),
        "vram": vram(data),
        "util": utilization(data),
    }

# Full scales known up front; power and VRAM come from the card itself.
TEMP_SCALE_C = 100.0
UTIL_SCALE_PCT = 100.0

# The rest are tiles, in dashboard order. The numeric ones carry a
# one-minute trend line: (key, title, reading, trend).
TILES = (
    ("headroom", "Thermal Headroom", headroom, True),
    ("mem_temp", "Memory Temp", memory_temperature, True),
    ("clock", "Clock", clock, True),
    ("fan", "Fan", fan, True),
    ("pcie", "PCIe Link", pcie, False),
    ("throttle", "Throttle", throttle, False),
)
