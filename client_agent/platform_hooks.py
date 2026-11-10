
# client_agent/platform_hooks.py

import sys
import time
from typing import Dict, Tuple

import psutil


# Safe platform-specific imports
_PLATFORM = sys.platform

win32gui = None
NSWorkspace = None
X = None
display = None

if _PLATFORM.startswith("win"):
    try:
        import win32gui
    except ImportError:
        pass

elif _PLATFORM == "darwin":
    try:
        from AppKit import NSWorkspace
    except ImportError:
        pass

elif _PLATFORM.startswith("linux"):
    try:
        from Xlib import X, display
    except ImportError:
        pass


def get_active_window_title() -> str:
    """
    Retrieves active window metadata.

    Windows: Foreground window title.
    macOS: Frontmost application name (not window title).
    Linux: Active X11 window title.

    Does not capture screenshots, keystrokes or browser history.
    """

    try:
        if _PLATFORM.startswith("win"):

            if win32gui is None:
                return "Unknown Application"

            hwnd = win32gui.GetForegroundWindow()

            if not hwnd:
                return "Unknown Window"

            title = win32gui.GetWindowText(hwnd)

            return title.strip() if title else "Desktop"

        elif _PLATFORM == "darwin":

            if NSWorkspace is None:
                return "Unknown Application"

            active_app = (
                NSWorkspace.sharedWorkspace()
                .frontmostApplication()
            )

            if active_app:
                name = active_app.localizedName()

                if name:
                    return name.strip()

            return "macOS Desktop"

        elif _PLATFORM.startswith("linux"):

            if display is None:
                return "Unknown Application"

            connection = None

            try:
                connection = display.Display()

                root = connection.screen().root

                atom = connection.intern_atom(
                    "_NET_ACTIVE_WINDOW"
                )

                prop = root.get_full_property(
                    atom,
                    X.AnyPropertyType
                )

                if prop is None or not len(prop.value):
                    return "Linux Desktop"

                window_id = prop.value[0]

                if not window_id:
                    return "Linux Desktop"

                window = connection.create_resource_object(
                    "window",
                    window_id
                )

                title = window.get_wm_name()

                return title.strip() if title else "Linux Desktop"

            finally:
                if connection is not None:
                    connection.close()

    except Exception:
        return "Unknown Application"

    return "Unknown Platform"


class NetworkSampler:
    """
    Stateful aggregate network sampler.

    Calculates byte and packet throughput using psutil
    interface counters.
    """

    def __init__(self):

        initial = psutil.net_io_counters()

        if initial is None:
            raise RuntimeError("Network counters unavailable")

        self.prev_bytes = (
            initial.bytes_recv + initial.bytes_sent
        )

        self.prev_packets = (
            initial.packets_recv + initial.packets_sent
        )

        self.prev_time = time.monotonic()

    def sample_delta(self) -> Tuple[float, float]:
        """
        Returns:
            network_speed_kb: KB/s
            packet_frequency: packets/s
        """

        current = psutil.net_io_counters()

        current_time = time.monotonic()

        if current is None:
            raise RuntimeError("Network counters unavailable")

        curr_bytes = (
            current.bytes_recv + current.bytes_sent
        )

        curr_packets = (
            current.packets_recv + current.packets_sent
        )

        elapsed = current_time - self.prev_time

        if elapsed <= 0:
            raise RuntimeError("Invalid sampling interval")

        # Handle counter resets.
        delta_bytes = max(
            0,
            curr_bytes - self.prev_bytes
        )

        delta_packets = max(
            0,
            curr_packets - self.prev_packets
        )

        # Update historical counters.
        self.prev_bytes = curr_bytes
        self.prev_packets = curr_packets
        self.prev_time = current_time

        network_speed_kb = (
            delta_bytes / 1024.0
        ) / elapsed

        packet_frequency = delta_packets / elapsed

        return (
            round(network_speed_kb, 4),
            round(packet_frequency, 4)
        )


def harvest_telemetry_slice(
    sampler: NetworkSampler
) -> Dict[str, float | str]:
    """
    Collects one telemetry sample.

    Exactly three attributes are returned.
    """

    speed_kb, packets = sampler.sample_delta()

    window = get_active_window_title()

    return {
        "window_title": window,
        "network_speed_kb": float(speed_kb),
        "packet_frequency": float(packets),
    }
