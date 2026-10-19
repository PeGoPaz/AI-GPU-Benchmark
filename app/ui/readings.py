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


def temperature(data: dict) -> Reading:
    temp = data['temp_gpu']
    mem = data.get('temp_memory')
    return Reading(f"{temp:.0f}", "°C", f"memory {mem:.0f} °C" if mem is not None else "",
                   _above(temp, TEMP_WARN_C, TEMP_CRIT_C), float(temp))


def headroom(data: dict) -> Reading:
    room = data.get('temp_headroom_c')
    if room is None:
        return Reading(NA, caption="not reported by this card")
    state = CRIT if room <= HEADROOM_CRIT_C else WARN if room <= HEADROOM_WARN_C else NORMAL
    return Reading(f"{room:.0f}", "°C", "before thermal slowdown", state, float(room))


def power(data: dict) -> Reading:
    # The poll's reading is smoothed; the peak, when the native sampler
    # supplies it, is the highest instant reading since the last poll.
    peak = data.get('power_peak_w')
    caption = f"peak {peak:.0f} W" if peak is not None else "250 ms average"
    return Reading(f"{data['power_w']:.0f}", "W", caption, number=float(data['power_w']))


def clock(data: dict) -> Reading:
    return Reading(f"{data['sm_clock_mhz']:.0f}", "MHz", "SM clock",
                   number=float(data['sm_clock_mhz']))


def utilization(data: dict) -> Reading:
    return Reading(f"{data['gpu_util_pct']:.0f}", "%", "GPU busy",
                   number=float(data['gpu_util_pct']))


def vram(data: dict) -> Reading:
    used, total = data['vram_used_mb'], data['vram_total_mb']
    pct = 100 * used / total if total else 0.0
    return Reading(f"{used / 1024:.1f}", f"/ {total / 1024:.1f} GiB", f"{pct:.0f}% in use",
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


# Tile order on the dashboard: (key, title, reading function).
TILES = (
    ("temp", "GPU Temperature", temperature),
    ("headroom", "Thermal Headroom", headroom),
    ("power", "Power", power),
    ("util", "Utilization", utilization),
    ("clock", "Clock", clock),
    ("vram", "VRAM", vram),
    ("fan", "Fan", fan),
    ("pcie", "PCIe Link", pcie),
    ("throttle", "Throttle", throttle),
)
