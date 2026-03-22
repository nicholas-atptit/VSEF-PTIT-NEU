"""Vietnam timezone utilities and trading hour helpers."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Vietnamese public holidays (fixed dates) — extend as needed
# Format: (month, day)
VN_FIXED_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1),   # New Year
    (4, 30),  # Reunification Day
    (5, 1),   # International Labor Day
    (9, 2),   # National Day
]


def now_vn() -> dt.datetime:
    """Return current datetime in Vietnam timezone."""
    return dt.datetime.now(tz=VN_TZ)


def today_vn() -> dt.date:
    """Return current date in Vietnam timezone."""
    return now_vn().date()


def to_vn_time(utc_dt: dt.datetime) -> dt.datetime:
    """Convert a UTC datetime to Vietnam timezone."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=dt.timezone.utc)
    return utc_dt.astimezone(VN_TZ)


def parse_time(time_str: str) -> dt.time:
    """Parse HH:MM string to time object."""
    h, m = time_str.split(":")
    return dt.time(int(h), int(m), tzinfo=VN_TZ)


def is_weekday(date: dt.date | None = None) -> bool:
    """Check if date is a weekday (Mon=0 to Fri=4)."""
    d = date or today_vn()
    return d.weekday() < 5


def is_vn_holiday(date: dt.date | None = None) -> bool:
    """Check if date is a known Vietnamese fixed public holiday.

    Note: Lunar-based holidays (Tet) must be updated annually.
    """
    d = date or today_vn()
    return (d.month, d.day) in VN_FIXED_HOLIDAYS


def is_trading_day(date: dt.date | None = None) -> bool:
    """Check if date is a potential trading day (weekday + not a holiday)."""
    d = date or today_vn()
    return is_weekday(d) and not is_vn_holiday(d)


def is_within_session(
    current_time: dt.time | None = None,
    morning_open: str = "08:45",
    morning_close: str = "11:30",
    afternoon_open: str = "12:45",
    afternoon_close: str = "15:00",
) -> bool:
    """Check if current time falls within a VN trading session window."""
    t = current_time or now_vn().time()
    # Strip tzinfo for comparison
    t_naive = t.replace(tzinfo=None)

    mo = dt.time(*map(int, morning_open.split(":")))
    mc = dt.time(*map(int, morning_close.split(":")))
    ao = dt.time(*map(int, afternoon_open.split(":")))
    ac = dt.time(*map(int, afternoon_close.split(":")))

    return (mo <= t_naive <= mc) or (ao <= t_naive <= ac)


def trading_session_label(current_time: dt.time | None = None) -> str | None:
    """Return 'morning', 'afternoon', or None based on current VN time."""
    t = current_time or now_vn().time()
    t_naive = t.replace(tzinfo=None)

    if dt.time(8, 45) <= t_naive <= dt.time(11, 30):
        return "morning"
    elif dt.time(12, 45) <= t_naive <= dt.time(15, 0):
        return "afternoon"
    return None
