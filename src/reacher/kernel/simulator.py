"""Simulated serial port for testing without physical Arduino hardware.

Provides SimulatedSerial (drop-in for serial.Serial) and FirmwareSimulator
(generates paradigm-aware firmware output).

The simulated board is a Python port of the firmware's scheduling logic
(Scheduler.cpp, Trigger.h, each sketch's Config.h / ReconfigureChain, and
PavlovianScheduler.cpp), configured by the same serial commands the real board
accepts. What it does not script is the animal: ``SimulatedSubject`` presses and
licks stochastically, and every consequence — classification, ratio counting,
availability windows, absence timers, timeouts, reward chains, Pavlovian trial
structure — is computed by the ported logic from whatever the session
configured. Change a setting and the outcome changes the way the firmware's
would. Runs are reproducible per ``seed``.
"""

import heapq
import itertools
import json
import logging
import math
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .commands import LITE_CAPABLE_PARADIGMS
from .simulated_subject import SimulatedSubject, SubjectProfile

logger = logging.getLogger(__name__)

PARADIGM_TO_SCHEDULE = {
    "fr": "FIXED_RATIO",
    "pr": "PROGRESSIVE_RATIO",
    "vi": "VARIABLE_INTERVAL",
    "omission": "OMISSION",
    "pavlovian": "PAVLOVIAN",
}
# A "_lite" board runs the same schedule as its base paradigm — only the
# two-photon hardware is absent — so mirror the base entry rather than
# restating it per variant.
PARADIGM_TO_SCHEDULE.update({f"{p}_lite": PARADIGM_TO_SCHEDULE[p] for p in LITE_CAPABLE_PARADIGMS})

SCHEDULE_TO_SKETCH = {
    "FIXED_RATIO": "fr.ino",
    "PROGRESSIVE_RATIO": "pr.ino",
    "VARIABLE_INTERVAL": "vi.ino",
    "OMISSION": "omission.ino",
    "PAVLOVIAN": "pavlovian.ino",
}


# Scheduler.h's timeout-mode constants, mirrored host-side.
TIMEOUT_MODE_EVERY_PRESS = 0
TIMEOUT_MODE_REWARD_ONLY = 1

# Schedules whose sketches implement the lever timeout at all. Omission forces
# SetTimeoutInterval(0) at setup and handles neither 1074/1374 nor 1077/1377;
# pavlovian is non-operant and has no lever timeout.
_TIMEOUT_SCHEDULES = ("FIXED_RATIO", "PROGRESSIVE_RATIO", "VARIABLE_INTERVAL")

# Boot-time values: each sketch's Config.h DEFAULT_* constants and .ino globals.
# FR ships everything at zero ("until explicitly configured"); the other
# operant sketches ship usable defaults. A "_lite" twin has a byte-identical Config.h.
_OPERANT_DEFAULTS = {
    "fr": dict(cue_frequency=0, cue_duration=0, pump_duration=0, laser_frequency=0, laser_duration=0,
               timeout_interval=0),
    "pr": dict(cue_frequency=8000, cue_duration=1600, pump_duration=2000, laser_frequency=40,
               laser_duration=5000, timeout_interval=20000),
    "vi": dict(cue_frequency=8000, cue_duration=1600, pump_duration=2000, laser_frequency=40,
               laser_duration=5000, timeout_interval=20000),
    "omission": dict(cue_frequency=8000, cue_duration=1600, pump_duration=2000, laser_frequency=40,
                     laser_duration=5000, timeout_interval=0),
}

# Pins.h defaults — what the `pin` field of a real event carries until a
# *_SET_PIN command moves the device.
_DEFAULT_PINS = {
    "LEVER_RH": 10, "LEVER_LH": 13, "LICK": 5, "CUE": 3, "CUE_2": 7,
    "PUMP": 4, "PUMP_2": 8, "LASER": 6, "MICROSCOPE": 2,
}
_PIN_COMMANDS = {
    376: "CUE", 386: "CUE_2", 476: "PUMP", 486: "PUMP_2", 576: "LICK", 676: "LASER",
    1076: "LEVER_RH", 1376: "LEVER_LH",
}
_ARM_ATTR = {
    "CUE": "cue_armed", "CUE_2": "cue2_armed", "PUMP": "pump_armed", "PUMP_2": "pump2_armed",
    "LICK": "lick_armed", "LASER": "laser_armed", "LEVER_RH": "lever_rh_armed",
    "LEVER_LH": "lever_lh_armed", "MICROSCOPE": "microscope_armed",
}
_ARM_COMMANDS = {
    300: ("CUE", False), 301: ("CUE", True), 310: ("CUE_2", False), 311: ("CUE_2", True),
    400: ("PUMP", False), 401: ("PUMP", True), 410: ("PUMP_2", False), 411: ("PUMP_2", True),
    500: ("LICK", False), 501: ("LICK", True), 600: ("LASER", False), 601: ("LASER", True),
    900: ("MICROSCOPE", False), 901: ("MICROSCOPE", True),
    1000: ("LEVER_RH", False), 1001: ("LEVER_RH", True),
    1300: ("LEVER_LH", False), 1301: ("LEVER_LH", True),
}

MAX_PAVLOV_TRIALS = 128  # PavlovianScheduler.h
FRAME_INTERVAL_MS = 100  # microscope frame cadence while armed (a real scope runs faster)


class TimeoutWindow:
    """Pure host-side model of the firmware lever-timeout window.

    Mirrors three pieces of ``Scheduler``/``SwitchLever`` behavior, with the
    simulated timestamp passed in rather than read from a clock, so it can be
    unit-tested on its own:

    * ``SwitchLever::InTimeout`` is ``ts <= timeoutEnd``, and
      ``Scheduler::ClassifyPress`` turns an in-window press on the reinforced
      lever into ``TIMEOUT``. A ``TIMEOUT`` press is logged but never reaches
      ``Trigger::OnInputEvent``, so it does not advance the ratio.
    * mode 0 (legacy default) arms the window on *every* ACTIVE press,
    * mode 1 arms it only on an ACTIVE press that fired a reward chain.

    An interval of 0 arms nothing: firmware would write ``ts + 0``, which only
    re-classifies a press landing in the very same millisecond.
    """

    def __init__(self, interval: int = 0, mode: int = TIMEOUT_MODE_EVERY_PRESS):
        self.interval = interval
        self.mode = mode
        self._end: dict = {}

    def reset(self) -> None:
        """Clear both levers' windows (mirrors Scheduler::StartSession)."""
        self._end.clear()

    def in_timeout(self, orientation: str, now: int) -> bool:
        end = self._end.get(orientation)
        return end is not None and now <= end

    def classify(self, orientation: str, now: int) -> str:
        """ACTIVE, or TIMEOUT when the lever's window is still open."""
        return "TIMEOUT" if self.in_timeout(orientation, now) else "ACTIVE"

    def note_active_press(self, orientation: str, now: int, rewarded: bool) -> None:
        """Arm the window for an ACTIVE press, if this mode calls for it."""
        if self.interval <= 0:
            return
        if self.mode == TIMEOUT_MODE_EVERY_PRESS or rewarded:
            self._end[orientation] = now + self.interval

    def set_end(self, orientation: str, end: int) -> None:
        """A chain's SET_TIMEOUT step: write the lever's window end directly."""
        self._end[orientation] = end


@dataclass
class _Trigger:
    """Trigger.h, reduced to the fields the three operant schedules use.

    ``absence_start is None`` stands for firmware's ``absenceStart == 0`` — the
    timer is off, because ``OnTick`` gates on ``absenceStart > 0``.
    """

    kind: str = "count"  # count (FR/PR) | absence (Omission) | window (VI)
    chain: int = 0
    enabled: bool = False
    threshold: int = 1
    initial_threshold: int = 1
    press_count: int = 0
    pr_step: int = 0
    absence_ms: int = 0
    absence_start: Optional[int] = None
    window_start: int = 0
    window_end: int = 0
    interval: int = 0
    fired_in_window: bool = False
    source_filter: Optional[str] = None

    def reset(self) -> None:
        """Trigger::Reset — runtime state only, never configuration."""
        self.press_count = 0
        self.absence_start = None
        self.window_start = 0
        self.window_end = 0
        self.fired_in_window = False
        if self.pr_step > 0:
            self.threshold = self.initial_threshold

    def start_window(self, now: int, rng: random.Random) -> None:
        """Scheduler::StartSession / Trigger::OnTick: place a fresh availability window."""
        self.window_start = now + (rng.randrange(self.interval) if self.interval > 0 else 0)
        self.window_end = now + self.interval
        self.fired_in_window = False

    def on_input(self, source: str, now: int, rng: random.Random, note: Callable[[str], None]) -> bool:
        """Trigger::OnInputEvent — True when the trigger fires its chain."""
        if not self.enabled:
            return False
        if self.source_filter is not None and self.source_filter != source:
            return False
        if self.kind == "count":
            self.press_count += 1
            if self.press_count >= self.threshold:
                self.press_count = 0
                if self.pr_step > 0:
                    if self.threshold <= 255 - self.pr_step:
                        self.threshold += self.pr_step
                    else:
                        note(f"PR threshold capped at {self.threshold}")
                return True
            return False
        if self.kind == "absence":
            self.absence_start = now
            return False
        if self.kind == "window":
            # OnTick re-places the window the first loop pass after it ends.
            while self.window_end > 0 and self.interval > 0 and now >= self.window_end:
                self.start_window(self.window_end, rng)
            if not self.fired_in_window and self.window_start <= now < self.window_end:
                self.fired_in_window = True
                return True
        return False


@dataclass(frozen=True)
class _Step:
    """One Action of a Chain (Action.h)."""

    kind: str  # activate | timeout | none
    target: str  # CUE | CUE_2 | PUMP | PUMP_2 | LASER (activate); RH | LH (timeout)
    source_filter: Optional[str] = None
    offset: int = 0
    param: int = 0


def _clamp(value, lo: int, hi: int) -> int:
    """ReacherHelpers' clampParam, minus the missing-key default."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = lo
    return max(lo, min(hi, value))


class FirmwareSimulator:
    """Generates firmware-protocol-compliant JSON output for a simulated session."""

    def __init__(
        self,
        tx_queue: queue.Queue,
        paradigm: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        realtime: bool = True,
        speed: Optional[float] = None,
        subject_profile: Optional[SubjectProfile] = None,
    ):
        """
        Args:
            tx_queue: queue the generated firmware lines are written to.
            paradigm: which sketch to impersonate. A real board's paradigm is
                fixed by the hex flashed onto it, and ``connect`` reads it back
                from IDENTIFY; the simulator has no hex, so the session that
                created it has to say. ``None`` keeps the historical ``fr``
                default, which is what the bare ``FirmwareSimulator(q)`` calls
                throughout the test-suite rely on.
            seed: seeds the animal, ITI draws, VI windows and trial order, so a
                run is reproducible. ``REACHER_SIM_SEED`` supplies it when unset.
            realtime: pace events against the wall clock on a worker thread (what
                the kernel needs). ``False`` runs no thread: the caller drives
                time with :meth:`run_until`, which is how the tests stay fast.
            speed: simulated seconds per wall second when ``realtime``
                (``REACHER_SIM_SPEED``, default 1).
            subject_profile: tune the animal; see ``SubjectProfile``.
        """
        self._tx = tx_queue
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if seed is None and os.environ.get("REACHER_SIM_SEED"):
            seed = int(os.environ["REACHER_SIM_SEED"])
        self._rng = random.Random(seed)
        self.subject = SimulatedSubject(self._rng, subject_profile or SubjectProfile())
        self._realtime = realtime
        if speed is None:
            speed = float(os.environ.get("REACHER_SIM_SPEED", "1") or 1)
        self._speed = speed if speed > 0 else 1.0

        # Configuration state (updated by incoming commands).
        # An unknown name falls back to "fr" rather than raising: this is a test
        # double, and a bad paradigm should not be able to break a connect.
        self.paradigm = paradigm if paradigm in PARADIGM_TO_SCHEDULE else "fr"
        self.schedule = PARADIGM_TO_SCHEDULE[self.paradigm]
        base = self._base
        defaults = _OPERANT_DEFAULTS.get(base)
        # Pavlovian's own boot values (pavlovian.ino / PavlovianScheduler.cpp).
        if defaults is None:
            defaults = dict(cue_frequency=8000, cue_duration=2000, pump_duration=2000, laser_frequency=40,
                            laser_duration=5000, timeout_interval=0)
        self.ratio = 1
        self.pr_step = 1
        self.vi_interval = 15000  # ms
        self.omission_interval = 20000  # ms
        self.cue_frequency = defaults["cue_frequency"]
        self.cue_duration = defaults["cue_duration"]
        self.pump_duration = defaults["pump_duration"]
        # A board powers up with every device disarmed and IDENTIFY disarms them
        # again; the UI only sends ARM for what the user armed, so an armed-by-
        # default simulator fires devices the session never asked for.
        self.lever_rh_armed = False
        self.lever_lh_armed = False
        self.cue_armed = False
        self.pump_armed = False
        self.microscope_armed = False
        self.lick_armed = False
        self.laser_armed = False
        # Operant sketches reinforce RH only; Pavlovian reinforces both.
        self.lever_rh_active = True
        self.lever_lh_active = base == "pavlovian"
        self._active_lever = "RH"  # fr.ino's `activeLever`: set only by 1081/1381
        self.laser_frequency = defaults["laser_frequency"]
        self.laser_duration = defaults["laser_duration"]  # ms
        self.laser_mode = "CONTINGENT"
        self.laser_rh_only = False
        self._laser_lever_filter = "RH"
        self.laser_onset_delay = 0
        self.cue2_armed = False
        self.cue2_frequency = defaults["cue_frequency"]
        self.cue2_duration = defaults["cue_duration"]
        self.pump2_armed = False
        self.pump2_duration = defaults["pump_duration"]
        self.pump2_active = False
        self.cue_onset_delay = 0
        self.pump_onset_delay = 0
        self.pump2_onset_delay = 0
        self._filters: Dict[str, Optional[str]] = {"CUE": None, "CUE_2": None, "PUMP": None, "PUMP_2": None}
        self._pins = dict(_DEFAULT_PINS)
        # Scheduler-wide, not per-lever — 1074/1374 write one firmware interval
        # and 1077/1377 write one mode, so the simulator holds one of each.
        self.timeout_interval = defaults["timeout_interval"]
        self.lever_timeout_mode = TIMEOUT_MODE_EVERY_PRESS
        self._timeout = TimeoutWindow()
        self._sync_timeout_config()

        # Pavlovian parameters (pavlovian.ino globals)
        self.pav_cs_plus_count = 50
        self.pav_cs_minus_count = 50
        self.pav_cs_plus_freq = 12000
        self.pav_cs_minus_freq = 3000
        self.pav_cs_plus_prob = 100
        self.pav_cs_minus_prob = 0
        self.pav_counterbalance = False
        self.pav_cue_duration = 2000
        self.pav_trace_interval = 1000
        self.pav_consumption = 3000
        self.pav_iti_mean = 30000
        self.pav_iti_min = 10000
        self.pav_iti_max = 90000
        self.pav_laser_filter = "CS_BOTH"
        self.pav_laser_phase = "REWARD"
        if base == "pavlovian":
            self.cue_frequency = self.pav_cs_plus_freq
            self.cue2_frequency = self.pav_cs_minus_freq

        # Runtime clock (ms since session start)
        self._clock = 0
        self._heap: list = []
        self._seq = itertools.count()
        self._anchor = 0.0  # monotonic() at session ms 0, pushed forward by pauses
        self._pause_wall = 0.0
        self._saved_arm_state: Optional[dict] = None
        self._end_ms: Optional[int] = None  # set when the board ends the session itself
        self._last_source: Optional[str] = None  # Scheduler::_lastInputSource
        self._pav_types: List[bool] = []
        self._pav_index = 0
        self._pav_iti_ms = 0
        self._pav_reward = False

        # Boot: setup() runs configureXxx() with ratio 1 / the sketch's intervals.
        self._triggers = [_Trigger(), _Trigger()]
        self._chains: List[List[_Step]] = [[], []]
        self._reconfigure()

    # --- Derived state ---

    @property
    def _base(self) -> str:
        return self.paradigm.removesuffix("_lite")

    @property
    def _is_pav(self) -> bool:
        return self._base == "pavlovian"

    @property
    def lever_rh_timeout(self) -> int:
        return self.timeout_interval

    @lever_rh_timeout.setter
    def lever_rh_timeout(self, value: int) -> None:
        self.timeout_interval = value

    @property
    def lever_lh_timeout(self) -> int:
        return self.timeout_interval

    @lever_lh_timeout.setter
    def lever_lh_timeout(self, value: int) -> None:
        self.timeout_interval = value

    def _send(self, msg: dict):
        self._tx.put(json.dumps(msg).encode() + b"\n")

    def _firmware_error(self, desc: str, **extra) -> None:
        self._send({"level": "006", "device": "CONTROLLER", "desc": desc, **extra})

    def handle_command(self, cmd_data: dict):
        cmd = cmd_data.get("cmd")
        if cmd is None:
            return

        base = self._base
        operant = base != "pavlovian"

        if cmd == 102:  # IDENTIFY
            if not self._running:
                # fr.ino:IDENTIFY — a reconnect disarms the board, so whatever the
                # host wants live has to be re-armed after it.
                self._disarm_all()
            self._send_identification()
        elif cmd == 101:  # SESSION_START
            self.start()
        elif cmd == 100:  # SESSION_END
            self.stop()
        elif cmd == 105:  # SESSION_PAUSE
            self._set_paused(bool(cmd_data.get("paused")))
        elif cmd == 103:  # TEST_CHAIN
            self._send_test_chain()
        elif cmd == 104:  # TEST_MODE
            pass
        # Device tests
        elif cmd == 303:  # CUE_TEST
            self._send_device_test("CUE", self._pins["CUE"], "TONE", self.cue_duration)
        elif cmd == 403:  # PUMP_TEST
            self._send_device_test("PUMP", self._pins["PUMP"], "INFUSION", self.pump_duration)
        elif cmd == 903:  # MICROSCOPE_TEST
            self._send_device_test("MICROSCOPE", 10, "TIMESTAMP", 100)
        elif cmd == 603:  # LASER_TEST
            self._send_device_test("LASER", self._pins["LASER"], "PULSE", 500)
        # Arm/disarm
        elif cmd in _ARM_COMMANDS:
            device, arm = _ARM_COMMANDS[cmd]
            setattr(self, _ARM_ATTR[device], arm)
            self._saved_arm_state = None  # a deliberate arm change beats a restart's restore
        elif cmd in _PIN_COMMANDS:
            device = _PIN_COMMANDS[cmd]
            self._pins[device] = _clamp(cmd_data.get("pin"), 2, 53)
            # Device::SetPin disarms an armed device while it moves the pin.
            setattr(self, _ARM_ATTR[device], False)
            self._saved_arm_state = None
        elif cmd == 1081:
            self.lever_rh_active = True
            self._active_lever = "RH"
        elif cmd == 1080:
            self.lever_rh_active = False
        elif cmd == 1381:
            self.lever_lh_active = True
            self._active_lever = "LH"
        elif cmd == 1380:
            self.lever_lh_active = False
        # Parameter setters
        elif cmd == 201:
            self.ratio = cmd_data.get("ratio", self.ratio)
            self._set_ratio()
            self._ack_ratio("CONTROLLER")
        elif cmd == 202:
            paradigm_val = cmd_data.get("paradigm")
            if isinstance(paradigm_val, str):
                self.schedule = PARADIGM_TO_SCHEDULE.get(paradigm_val, self.schedule)
                self.paradigm = paradigm_val
                self._reconfigure()
        elif cmd == 203:
            self.omission_interval = cmd_data.get("interval", self.omission_interval)
            self._reconfigure()
        elif cmd == 204:
            self.vi_interval = cmd_data.get("interval", self.vi_interval)
            self._reconfigure()
        elif cmd == 205:
            self.pr_step = _clamp(cmd_data.get("step", self.pr_step), 0, 255)
            if base == "pr":
                self._triggers[0].pr_step = self.pr_step
        elif cmd == 371:
            self.cue_frequency = _clamp(cmd_data.get("frequency"), 1, 65535)
            if not operant:
                self.pav_cs_plus_freq = self.cue_frequency
        elif cmd == 372:
            self.cue_duration = _clamp(cmd_data.get("duration"), 1, 600000)
            if operant:
                self._reconfigure()
            else:
                self.pav_cue_duration = self.cue_duration
        elif cmd == 472:
            self.pump_duration = _clamp(cmd_data.get("duration"), 1, 600000)
            self._reconfigure()
        elif cmd == 381:
            self.cue2_frequency = _clamp(cmd_data.get("frequency"), 1, 65535)
            if not operant:
                self.pav_cs_minus_freq = self.cue2_frequency
        elif cmd == 382:
            self.cue2_duration = _clamp(cmd_data.get("duration"), 1, 600000)
            self._reconfigure()
        elif cmd == 313:  # CUE2_TEST
            self._send_device_test("CUE", self._pins["CUE_2"], "TONE", self.cue2_duration)
        elif cmd == 413:  # PUMP2_TEST
            self._send({"level": "000", "device": "PUMP2", "param": "test", "value": "FIRED"})
        elif cmd == 482:
            self.pump2_duration = _clamp(cmd_data.get("duration"), 1, 600000)
            self._reconfigure()
        elif cmd == 221:  # SET_ACTIVE_PUMP
            self.pump2_active = bool(cmd_data.get("pump2", False))
            self._reconfigure()
        elif cmd == 671:
            self.laser_frequency = _clamp(cmd_data.get("frequency"), 1, 65535)
        elif cmd == 672:
            self.laser_duration = _clamp(cmd_data.get("duration"), 1, 600000)
            self._reconfigure()
        elif cmd == 673:  # LASER_SET_ONSET_DELAY
            self.laser_onset_delay = self._clamp_delay(cmd_data.get("delay"))
            # vi/omission only rebuild the chain here when the laser is lever-contingent.
            if base in ("fr", "pr") or self.laser_rh_only:
                self._reconfigure()
        elif cmd in (377, 477, 487):
            attr = {377: "cue_onset_delay", 477: "pump_onset_delay", 487: "pump2_onset_delay"}[cmd]
            setattr(self, attr, self._clamp_delay(cmd_data.get("delay")))
            self._reconfigure()
        elif cmd in (378, 388, 478, 488):  # per-device lever filters
            device = {378: "CUE", 388: "CUE_2", 478: "PUMP", 488: "PUMP_2"}[cmd]
            src = {1: "RH", 2: "LH"}.get(_clamp(cmd_data.get("filter", 0), 0, 255))
            self._filters[device] = src
            if operant:
                self._reconfigure()
                if src == "LH":
                    self.lever_lh_active = True
                elif src == "RH":
                    self.lever_rh_active = True
        elif cmd == 681:
            self.laser_mode = "CONTINGENT"
            self.laser_rh_only = False
            self._reconfigure()
        elif cmd == 682:
            self.laser_mode = "INDEPENDENT"
        elif cmd == 684:
            self.laser_rh_only = True
            self._laser_lever_filter = "RH"
            self._reconfigure()
        elif cmd == 685:
            if base in ("fr", "pr"):
                self.laser_rh_only = True
                self._laser_lever_filter = "LH"
                self._reconfigure()
            else:
                # vi/omission never handled LH-only (schema.KNOWN_FIRMWARE_GAPS):
                # the code falls through to the sketch's default case.
                self._send({"level": "006", "desc": "Command not found", "command": cmd})
        elif cmd == 1074 or cmd == 1374:
            if base in ("fr", "pr", "vi"):
                self.timeout_interval = cmd_data.get("timeout", self.timeout_interval)
                self._sync_timeout_config()
        elif cmd == 1075:
            self.ratio = cmd_data.get("ratio", self.ratio)
            self._set_ratio()
            self._ack_ratio("LEVER_RH")
        elif cmd == 1375:
            self.ratio = cmd_data.get("ratio", self.ratio)
            self._set_ratio()
            self._ack_ratio("LEVER_LH")
        elif cmd in (1077, 1377):  # LEVER_{RH,LH}_SET_TIMEOUT_MODE
            # Both codes write the single scheduler-wide flag; the firmware
            # clamps anything above 1 (Scheduler::SetTimeoutMode).
            mode = cmd_data.get("timeout_mode", self.lever_timeout_mode)
            self.lever_timeout_mode = min(int(mode), TIMEOUT_MODE_REWARD_ONLY)
            self._sync_timeout_config()
        # Pavlovian parameters
        elif cmd == 206:
            self.pav_cs_plus_prob = _clamp(cmd_data.get("probability"), 0, 100)
        elif cmd == 207:
            self.pav_cs_minus_prob = _clamp(cmd_data.get("probability"), 0, 100)
        elif cmd in (208, 209, 212):
            names = {208: "CS+ count", 209: "CS- count", 212: "counterbalance"}
            if self._running:
                self._firmware_error(f"Cannot change {names[cmd]} during active session")
            elif cmd == 208:
                self.pav_cs_plus_count = _clamp(cmd_data.get("count"), 0, 255)
            elif cmd == 209:
                self.pav_cs_minus_count = _clamp(cmd_data.get("count"), 0, 255)
            else:
                # pavlovian.ino reads the key "counterbalance"; the host's
                # CommandSpec for 212 sends "enabled", which the board reads as
                # absent (false). Mirrored as-is so the gap shows up here too.
                self.pav_counterbalance = bool(cmd_data.get("counterbalance", False))
        elif cmd == 210:
            self.pav_cs_plus_freq = cmd_data.get("frequency", self.pav_cs_plus_freq)
            self.cue_frequency = self.pav_cs_plus_freq
        elif cmd == 211:
            self.pav_cs_minus_freq = cmd_data.get("frequency", self.pav_cs_minus_freq)
            self.cue2_frequency = self.pav_cs_minus_freq
        elif cmd == 213:
            self.pav_cue_duration = cmd_data.get("duration", self.pav_cue_duration)
            self.cue_duration = self.pav_cue_duration
        elif cmd == 214:
            self.pav_trace_interval = cmd_data.get("interval", self.pav_trace_interval)
        elif cmd == 215:
            self.pav_consumption = cmd_data.get("duration", self.pav_consumption)
        elif cmd == 216:
            self.pav_iti_mean = cmd_data.get("iti_mean", self.pav_iti_mean)
        elif cmd == 217:
            self.pav_iti_min = cmd_data.get("iti_min", self.pav_iti_min)
        elif cmd == 218:
            self.pav_iti_max = cmd_data.get("iti_max", self.pav_iti_max)
        elif cmd in (691, 692, 693):
            self.pav_laser_filter = {691: "CS_PLUS", 692: "CS_MINUS", 693: "CS_BOTH"}[cmd]
        elif cmd in (694, 695):
            self.pav_laser_phase = {694: "REWARD", 695: "CUE"}[cmd]

    def _clamp_delay(self, value) -> int:
        # vi.ino caps onset delays at 60 s; the other operant sketches at 600 s.
        return _clamp(value, 0, 60000 if self._base == "vi" else 600000)

    def _ack_ratio(self, device: str) -> None:
        """Echo the firmware's ``logParamChange`` for a ratio setter.

        Only the fr/pr sketches (and their lite twins) handle these codes, and
        the device differs per code: 201 answers as CONTROLLER, 1075/1375 as
        the lever the command is named for.
        """
        if self.paradigm.removesuffix("_lite") in ("fr", "pr"):
            self._send({"level": "000", "device": device, "param": "ratio", "value": self.ratio})

    def _set_ratio(self) -> None:
        """Scheduler::SetRatio — writes the live threshold, *not* the PR reset value.

        ``initial_threshold`` only follows at the next ReconfigureChain(), so on
        PR a ratio sent before session start is reset to the old initial value by
        ``Trigger::Reset()`` unless some command that reconfigures the chain
        (a duration, a filter, SET_ACTIVE_PUMP, ...) arrives in between.
        """
        if self._base in ("fr", "pr"):
            trig = self._triggers[0]
            trig.threshold = int(self.ratio) & 0xFF
            trig.press_count = 0

    # --- Board lifecycle ---

    def _disarm_all(self) -> None:
        for attr in _ARM_ATTR.values():
            setattr(self, attr, False)
        self._saved_arm_state = None

    def _set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        if paused:
            self._pause_wall = time.monotonic()
        else:
            # The session clock stood still for as long as the board was paused.
            self._anchor += time.monotonic() - self._pause_wall
        self._paused = paused

    def _send_identification(self):
        # "_lite" paradigms share a schedule with their full counterpart
        # (fr_lite and fr both report FIXED_RATIO), so the sketch name has
        # to come from the paradigm itself, not the schedule, to be accurate.
        if self.paradigm.endswith("_lite"):
            sketch = f"{self.paradigm}.ino"
        else:
            sketch = SCHEDULE_TO_SKETCH.get(self.schedule, "fr.ino")
        self._send({
            "level": "000", "device": "CONTROLLER", "sketch": sketch,
            "version": "v2.0.0-sim", "baud_rate": 115200,
            "schedule": self.schedule,
        })
        self._send_device_configs()
    def _send_device_configs(self) -> None:
        """reportDeviceConfig / reportDeviceLever rows (level 000), as dumped at session start."""
        pav = self._is_pav
        self._send({
            "level": "000", "device": "CUE", "armed": self.cue_armed,
            "frequency": self.pav_cs_plus_freq if pav else self.cue_frequency,
            "duration": self.pav_cue_duration if pav else self.cue_duration,
        })
        self._send({
            "level": "000", "device": "CUE2", "armed": self.cue2_armed,
            "frequency": self.pav_cs_minus_freq if pav else self.cue2_frequency,
            "duration": self.cue2_duration,
        })
        self._send({"level": "000", "device": "PUMP", "armed": self.pump_armed, "duration": self.pump_duration})
        self._send({"level": "000", "device": "PUMP2", "armed": self.pump2_armed, "duration": self.pump2_duration})
        self._send({
            "level": "000", "device": "LASER", "armed": self.laser_armed,
            "frequency": self.laser_frequency, "duration": self.laser_duration,
        })
        # Level 000 spells the lick circuit LICK; only level 007 uses LICK_CIRCUIT.
        self._send({"level": "000", "device": "LICK", "armed": self.lick_armed})
        if not self.paradigm.endswith("_lite"):
            self._send({"level": "000", "device": "MICROSCOPE", "armed": self.microscope_armed})
        # Level 000 spells levers LEVER_RH / LEVER_LH (reportDeviceLever).
        self._send({
            "level": "000", "device": "LEVER_RH", "armed": self.lever_rh_armed, "reinforced": self.lever_rh_active,
        })
        self._send({
            "level": "000", "device": "LEVER_LH", "armed": self.lever_lh_armed, "reinforced": self.lever_lh_active,
        })

    def _send_device_test(self, device: str, pin: int, event: str, duration: int):
        ts = self._clock
        self._send({
            "level": "007", "device": device, "pin": pin,
            "event": event, "start_timestamp": ts,
            "end_timestamp": ts + duration,
        })

    def _send_test_chain(self):
        """Scheduler::TestChain: fire chain 0 once, outside any session.

        Offsets are stamped rather than waited out — nothing is ticking between
        sessions — but the armed, lever-filter, duration and onset-delay rules
        are the real ones. pavlovian.ino has no TEST_CHAIN; it keeps the plain
        cue-then-pump stand-in.
        """
        if self._running:
            return
        if self._is_pav:
            self._send_device_test("CUE", self._pins["CUE"], "TONE", self.cue_duration)
            self._send_device_test("PUMP", self._pins["PUMP"], "INFUSION", self.pump_duration)
            return
        for step in self._chains[0]:
            if step.source_filter is not None and step.source_filter != self._last_source:
                continue
            self._execute(step, step.offset, schedule_licks=False)

    def _capture_arm_state(self) -> dict:
        """Snapshot all device arm states."""
        return {
            "lever_rh_armed": self.lever_rh_armed,
            "lever_lh_armed": self.lever_lh_armed,
            "cue_armed": self.cue_armed,
            "cue2_armed": self.cue2_armed,
            "pump_armed": self.pump_armed,
            "pump2_armed": self.pump2_armed,
            "lick_armed": self.lick_armed,
            "laser_armed": self.laser_armed,
            "microscope_armed": self.microscope_armed,
        }

    def _restore_arm_state(self, snap: dict):
        """Restore previously captured arm states."""
        self.lever_rh_armed = snap.get("lever_rh_armed", self.lever_rh_armed)
        self.lever_lh_armed = snap.get("lever_lh_armed", self.lever_lh_armed)
        self.cue_armed = snap.get("cue_armed", self.cue_armed)
        self.cue2_armed = snap.get("cue2_armed", self.cue2_armed)
        self.pump_armed = snap.get("pump_armed", self.pump_armed)
        self.pump2_armed = snap.get("pump2_armed", self.pump2_armed)
        self.lick_armed = snap.get("lick_armed", self.lick_armed)
        self.laser_armed = snap.get("laser_armed", self.laser_armed)
        self.microscope_armed = snap.get("microscope_armed", self.microscope_armed)

    # --- Timeout ---

    def _timeout_applies(self) -> bool:
        """True for the schedules whose sketches implement a lever timeout."""
        return self.schedule in _TIMEOUT_SCHEDULES

    def _active_orientation(self) -> str:
        return self._active_lever

    def _sync_timeout_config(self):
        """Push interval/mode into the window model.

        Called at session start and from every command that changes either, so
        a mid-session edit takes effect on the next press — the firmware
        contract for 1074/1374, and now for 1077/1377 too.
        """
        self._timeout.interval = self.timeout_interval if self._timeout_applies() else 0
        self._timeout.mode = self.lever_timeout_mode

    def _send_session_config(self):
        """Level-000 dump printed at session start (each sketch's StartSession()).

        The timeout fields appear only on fr/pr/vi, whose sketches carry them.
        The sketches' device-less extras (``pr_step``, ``variable_interval``,
        ``omission_interval`` and Pavlovian's second row) are left out; the
        kernel only folds them into ``firmware_information``, so they add
        nothing a simulated run needs.
        """
        controller = {"level": "000", "device": "CONTROLLER", "paradigm": self.schedule}
        if self._is_pav:
            controller.update(
                cs_plus_count=self.pav_cs_plus_count, cs_minus_count=self.pav_cs_minus_count,
                cs_plus_prob=self.pav_cs_plus_prob, cs_minus_prob=self.pav_cs_minus_prob,
                counterbalance=self.pav_counterbalance,
            )
        else:
            if self._timeout_applies():
                controller.update(timeout=self.timeout_interval, timeout_mode=self.lever_timeout_mode)
            controller["active_lever"] = self._active_orientation()
        self._send(controller)
        self._send_device_configs()

    # --- Session ---

    def _session_ms(self) -> int:
        """Session time now: the wall clock against the anchor, or the stepped clock."""
        if self._realtime and self._running:
            wall = self._paused and self._pause_wall or time.monotonic()
            return max(self._clock, int((wall - self._anchor) * 1000 * self._speed))
        return self._clock

    def _begin_run(self):
        if self._saved_arm_state is not None:
            self._restore_arm_state(self._saved_arm_state)
            self._saved_arm_state = None
        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._clock = 0
        self._heap = []
        self._end_ms = None
        self._anchor = time.monotonic()
        self._scheduler_start()
        self._reconfigure()  # fr.ino StartSession(): ReconfigureChain() after Scheduler::StartSession()
        self._send_session_config()
        self._start_activity()
        if self._realtime:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def start(self):
        if self._running:
            return
        self._send({"level": "007", "device": "CONTROLLER", "event": "START", "timestamp": 0, "source": "software"})
        self._begin_run()

    def stop(self):
        # Capture arm states before disarming (mirrors firmware behavior)
        self._saved_arm_state = self._capture_arm_state()
        end_ts = self._end_ms if self._end_ms is not None else self._session_ms()
        was_running = self._running
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None
        self._paused = False
        if was_running:
            self._scheduler_end()
        # Mirrors firmware's EndSession() (e.g. fr.ino:225-233), which prints the
        # CONTROLLER END event synchronously in the cmd-100 handler. Without this,
        # stop_program()'s _controller_end_received.wait() (reacher.py) always hits
        # its 8s timeout against the simulator.
        self._send({
            "level": "007", "device": "CONTROLLER",
            "event": "END", "timestamp": end_ts,
        })

    def run_until(self, session_ms: int) -> None:
        """Advance a ``realtime=False`` session to *session_ms*, running every event due.

        Events are processed in timestamp order and the output queue is filled
        exactly as the worker thread would fill it, minus the waiting.
        """
        while self._running and not self._paused and self._step(session_ms):
            pass
        if self._running and not self._paused:
            self._clock = max(self._clock, session_ms)

    # --- Event engine ---

    def _at(self, t_ms: int, fn: Callable[[], None]) -> None:
        heapq.heappush(self._heap, (t_ms, next(self._seq), fn))

    def _run_loop(self):
        while self._step(None):
            pass

    def _step(self, until: Optional[int]) -> bool:
        """Run the next due event. False once stopped, drained, or past *until*."""
        if self._stop_event.is_set() or not self._heap:
            return False
        t = self._heap[0][0]
        if until is not None and t > until:
            return False
        if not self._wait_until(t):
            return False
        _, _, fn = heapq.heappop(self._heap)
        self._clock = max(self._clock, t)
        fn()
        return True

    def _wait_until(self, t_ms: int) -> bool:
        """Block until session time *t_ms* (False if stopped first). Instant unless realtime."""
        if not self._realtime:
            return not self._stop_event.is_set()
        while not self._stop_event.is_set():
            if self._paused:
                self._stop_event.wait(0.05)
                continue
            delay = self._anchor + t_ms / 1000.0 / self._speed - time.monotonic()
            if delay <= 0:
                return True
            self._stop_event.wait(min(delay, 0.1))
        return False

    def _start_activity(self) -> None:
        """Seed the recurring processes a freshly started session runs."""
        self._at(self.subject.press_gap_ms(self._is_pav), self._subject_press)
        if self.microscope_armed and not self.paradigm.endswith("_lite"):
            self._at(FRAME_INTERVAL_MS, self._frame_tick)
        if self.laser_armed and self.laser_mode == "INDEPENDENT":
            self._at(0, self._laser_cycle)
        if self._is_pav:
            self._pav_begin()
        else:
            self._absence_rearm()

    # --- Scheduler (Scheduler.cpp / Trigger.h) ---

    def _scheduler_start(self) -> None:
        """Scheduler::StartSession — reset triggers, seed windows, clear lever timeouts."""
        for trig in self._triggers:
            trig.reset()
        for trig in self._triggers:
            if not trig.enabled:
                continue
            if trig.kind == "absence":
                trig.absence_start = 0
            elif trig.kind == "window":
                trig.start_window(0, self._rng)
        self._timeout.reset()
        self._sync_timeout_config()

    def _scheduler_end(self) -> None:
        """Scheduler::EndSession — triggers back to their start state (PR threshold included)."""
        for trig in self._triggers:
            trig.reset()
        self._heap = []

    def _reconfigure(self) -> None:
        """ReconfigureChain() + the sketch's configureXxx(): rebuild triggers and chains.

        Called from the same commands the firmware calls it from, because it is
        not idempotent against live state: it re-bases the PR reset ratio on the
        current threshold, and zeroes the omission timer (``absenceStart = 0``).
        """
        base = self._base
        if base == "pavlovian":
            return
        t0, t1 = self._triggers
        if base in ("fr", "pr"):
            current = t0.threshold
            t0.kind, t0.chain, t0.enabled = "count", 0, True
            t0.threshold = t0.initial_threshold = current
            t0.press_count = 0
            t0.pr_step = self.pr_step if base == "pr" else 0
            t0.source_filter = None
        elif base == "vi":
            t0.kind, t0.chain, t0.enabled = "window", 0, True
            t0.interval = int(self.vi_interval)
            t0.source_filter = None
        else:  # omission
            t0.kind, t0.chain, t0.enabled = "absence", 0, True
            t0.absence_ms = int(self.omission_interval)
            t0.absence_start = None
            t0.source_filter = None

        pump_target = "PUMP_2" if self.pump2_active else "PUMP"
        steps = [
            _Step("activate", "CUE", self._filters["CUE"], self.cue_onset_delay, self.cue_duration),
            _Step("activate", "CUE_2", self._filters["CUE_2"], self.cue_onset_delay, self.cue2_duration),
            _Step(
                "activate", pump_target, self._filters[pump_target],
                self.pump2_onset_delay if self.pump2_active else self.pump_onset_delay,
                self.pump2_duration if self.pump2_active else self.pump_duration,
            ),
            _Step("none" if self.laser_rh_only else "activate", "LASER", None, self.laser_onset_delay,
                  self.laser_duration),
        ]
        if base != "omission":
            steps.append(_Step("timeout", self._active_orientation(), None, 0, self.timeout_interval))
        self._chains[0] = steps

        if self.laser_rh_only:
            t1.kind, t1.chain, t1.enabled = "count", 1, True
            t1.threshold = t1.initial_threshold = 1
            t1.press_count = 0
            t1.pr_step = 0
            t1.source_filter = self._laser_lever_filter if base in ("fr", "pr") else "RH"
            self._chains[1] = [_Step("activate", "LASER", None, self.laser_onset_delay, self.laser_duration)]
        else:
            t1.enabled = False
            self._chains[1] = []

    def _chain_applies_timeout(self, chain: int) -> bool:
        return any(step.kind == "timeout" for step in self._chains[chain])

    def _fire_chain(self, chain: int, now: int) -> None:
        for step in self._chains[chain]:
            if step.kind == "none":
                continue
            if step.source_filter is not None and step.source_filter != self._last_source:
                continue
            if step.offset == 0:
                self._execute(step, now)
            else:
                self._at(now + step.offset, lambda s=step: self._execute(s, self._clock))

    def _execute(self, step: _Step, now: int, schedule_licks: bool = True) -> None:
        """Scheduler::ExecuteAction."""
        if step.kind == "timeout":
            # SetTimeoutInterval() rewrites every SET_TIMEOUT step, so a mid-session
            # 1074/1374 reaches the very next reward without a ReconfigureChain().
            self._timeout.set_end(step.target, now + self.timeout_interval)
            return
        if step.kind != "activate":
            return
        target, end = step.target, now + step.param
        if target == "CUE":
            if self.cue_armed:
                self._log_activation("CUE", now, end)
        elif target == "CUE_2":
            if self.cue2_armed:
                self._log_activation("CUE_2", now, end)
        elif target == "PUMP":
            if self.pump_armed:
                self._log_activation("PUMP", now, end)
                if schedule_licks:
                    self._drink(now, step.param)
        elif target == "PUMP_2":
            if self.pump2_armed:
                self._log_activation("PUMP_2", now, end)
                if schedule_licks:
                    self._drink(now, step.param)
        elif target == "LASER":
            if self.laser_armed and self.laser_mode == "CONTINGENT":
                self._log_activation("LASER", now, end)

    def _on_active_press(self, orientation: str, now: int) -> str:
        """Scheduler::OnInputEvent for a press that classified ACTIVE.

        Returns the class that gets logged: on an interval schedule only the press
        that collects the availability window stays ACTIVE.
        """
        self._last_source = orientation
        reward_fired = interval_fired = interval_schedule = False
        for trig in self._triggers:
            if trig.enabled and trig.kind == "window":
                interval_schedule = True
            if trig.on_input(orientation, now, self._rng, self._note_info):
                if self._chain_applies_timeout(trig.chain):
                    reward_fired = True
                if trig.kind == "window":
                    interval_fired = True
                self._fire_chain(trig.chain, now)
        self._absence_rearm()
        if interval_schedule and not interval_fired:
            return "INACTIVE"
        self._timeout.note_active_press(orientation, now, reward_fired)
        return "ACTIVE"

    def _note_info(self, desc: str) -> None:
        self._send({"level": "001", "device": "CONTROLLER", "desc": desc})

    def _absence_rearm(self) -> None:
        """Queue the Omission timer's expiry (Trigger::OnTick) for its current start."""
        trig = self._triggers[0]
        if not (trig.enabled and trig.kind == "absence") or trig.absence_start is None:
            return
        token = trig.absence_start
        self._at(token + trig.absence_ms, lambda: self._absence_expired(token))

    def _absence_expired(self, token: int) -> None:
        trig = self._triggers[0]
        if not trig.enabled or trig.absence_start != token:
            return  # a press (or a reconfigure) restarted or cancelled this timer
        trig.absence_start = self._clock
        self._fire_chain(trig.chain, self._clock)
        self._absence_rearm()

    # --- The animal ---

    def _reinforced(self, orientation: str) -> bool:
        return self.lever_rh_active if orientation == "RH" else self.lever_lh_active

    def _subject_press(self) -> None:
        reinforced = [o for o in ("RH", "LH") if self._reinforced(o)]
        other = [o for o in ("RH", "LH") if not self._reinforced(o)]
        orientation = self.subject.pick_lever(reinforced, other)
        duration = self.subject.press_duration_ms()
        start = self._clock
        self._lever_press(orientation, start, duration)
        self._at(start + duration + self.subject.press_gap_ms(self._is_pav), self._subject_press)

    def _lever_press(self, orientation: str, start: int, duration: int) -> None:
        """SwitchLever::Monitor + Scheduler: a disarmed lever is invisible; otherwise
        the press is classified on the way down and logged on the way up."""
        armed = self.lever_rh_armed if orientation == "RH" else self.lever_lh_armed
        if not armed:
            return
        if self._is_pav:
            # PavlovianScheduler::ClassifyPress: both levers reinforced, no timeout.
            press_class = "ACTIVE"
        elif not self._reinforced(orientation):
            press_class = "INACTIVE"
        else:
            press_class = self._timeout.classify(orientation, start)
            if press_class == "ACTIVE":
                press_class = self._on_active_press(orientation, start)
        end = start + duration
        self._at(end, lambda: self._log_press(orientation, press_class, start, end))

    def _drink(self, start: int, infusion_ms: int) -> None:
        """Licking that follows a delivered infusion."""
        for lick_start, lick_end in self.subject.lick_bout(start, infusion_ms):
            self._at(lick_end, lambda s=lick_start, e=lick_end: self._log_lick(s, e))

    def _frame_tick(self) -> None:
        if self.microscope_armed:
            self._send({
                "level": "008", "device": "MICROSCOPE", "pin": self._pins["MICROSCOPE"],
                "event": "TIMESTAMP", "timestamp": self._clock, "missed": 0,
            })
        self._at(self._clock + FRAME_INTERVAL_MS, self._frame_tick)

    def _laser_cycle(self) -> None:
        """Laser::Cycle in INDEPENDENT mode: ON for `duration`, OFF for `duration`, repeat."""
        if not (self.laser_armed and self.laser_mode == "INDEPENDENT") or self.laser_duration <= 0:
            return
        self._log_activation("LASER", self._clock, self._clock + self.laser_duration)
        self._at(self._clock + 2 * self.laser_duration, self._laser_cycle)

    # --- Pavlovian (PavlovianScheduler.cpp) ---

    def _pav_begin(self) -> None:
        """PavlovianScheduler::StartSession — arm-gate the counts, shuffle, queue the first ITI."""
        plus_cue_armed = self.cue2_armed if self.pav_counterbalance else self.cue_armed
        minus_cue_armed = self.cue_armed if self.pav_counterbalance else self.cue2_armed
        cs_plus = min(self.pav_cs_plus_count, MAX_PAVLOV_TRIALS)
        cs_minus = min(self.pav_cs_minus_count, MAX_PAVLOV_TRIALS - cs_plus)
        cs_plus = cs_plus if plus_cue_armed else 0
        cs_minus = cs_minus if minus_cue_armed else 0
        total = min(cs_plus + cs_minus, MAX_PAVLOV_TRIALS)
        self._pav_index = 0
        if total == 0:
            self._pav_types = []
            self._end_ms = 0  # IsComplete() right away: the sketch ends the session itself
            return
        cs_plus = min(cs_plus, total)
        types = [False] * cs_plus + [True] * (total - cs_plus)
        for _ in range(50):
            for i in range(total - 1, 0, -1):
                j = self._rng.randrange(i + 1)
                types[i], types[j] = types[j], types[i]
            run = longest = 1
            for i in range(1, total):
                run = run + 1 if types[i] == types[i - 1] else 1
                longest = max(longest, run)
            if longest <= 3:
                break
        self._pav_types = types
        self._pav_schedule_iti(0)

    def _pav_effective(self):
        """Configure()'s clamping, applied where the value is read."""
        iti_min = int(self.pav_iti_min)
        iti_max = max(int(self.pav_iti_max), iti_min)
        iti_mean = min(max(int(self.pav_iti_mean), iti_min), iti_max)
        return iti_min, iti_max, iti_mean

    def _pav_schedule_iti(self, from_ms: int) -> None:
        iti_min, iti_max, iti_mean = self._pav_effective()
        # PavSampleIti: an exponential draw (mean = ITI mean) clipped to [min, max].
        draw = -math.log(1.0 - self._rng.random()) * iti_mean
        self._pav_iti_ms = int(min(max(draw, iti_min), iti_max))
        self._at(from_ms + self._pav_iti_ms, self._pav_start_trial)

    def _pav_trial_event(self, event: str, **extra) -> None:
        self._send({
            "level": "007", "device": "PAVLOV", "event": event, "trial": self._pav_index,
            **extra, "timestamp": self._clock,
        })

    def _pav_laser(self, window_ms: int, is_cs_minus: bool, phase: str) -> None:
        if not (self.laser_armed and self.laser_mode == "CONTINGENT" and self.pav_laser_phase == phase):
            return
        wanted = self.pav_laser_filter
        if not (wanted == "CS_BOTH" or (wanted == "CS_PLUS" and not is_cs_minus)
                or (wanted == "CS_MINUS" and is_cs_minus)):
            return
        delay = self.laser_onset_delay
        if delay >= window_ms:
            delay = window_ms - 1 if window_ms > 1 else 0
        self._log_activation("LASER", self._clock + delay, self._clock + delay + self.laser_duration)

    def _pav_start_trial(self) -> None:
        now = self._clock
        is_cs_minus = self._pav_types[self._pav_index]
        prob = self.pav_cs_minus_prob if is_cs_minus else self.pav_cs_plus_prob
        self._pav_reward = self._rng.randrange(100) < min(int(prob), 100)
        cue_ms = max(int(self.pav_cue_duration), 1)
        use_cue2 = is_cs_minus != self.pav_counterbalance
        if (self.cue2_armed if use_cue2 else self.cue_armed):
            self._log_activation("CUE_2" if use_cue2 else "CUE", now, now + cue_ms)
        self._pav_laser(cue_ms, is_cs_minus, "CUE")
        self._pav_trial_event(
            "TRIAL_START", trial_type="CS_MINUS" if is_cs_minus else "CS_PLUS",
            reward_scheduled=self._pav_reward, iti_ms=self._pav_iti_ms,
        )
        self._at(now + cue_ms, self._pav_trace)

    def _pav_trace(self) -> None:
        self._pav_trial_event("TRACE_START")
        self._at(self._clock + int(self.pav_trace_interval), self._pav_reward_phase)

    def _pav_reward_phase(self) -> None:
        now = self._clock
        is_cs_minus = self._pav_types[self._pav_index]
        if self._pav_reward:
            use_pump2 = is_cs_minus != self.pav_counterbalance
            pump_armed = self.pump2_armed if use_pump2 else self.pump_armed
            duration = self.pump2_duration if use_pump2 else self.pump_duration
            if pump_armed:
                self._log_activation("PUMP_2" if use_pump2 else "PUMP", now, now + duration)
                self._drink(now, duration)
            self._pav_trial_event("REWARD_DELIVERED")
        else:
            self._pav_trial_event("REWARD_OMITTED")
        self._pav_laser(int(self.pav_consumption), is_cs_minus, "REWARD")
        self._at(now + int(self.pav_consumption), self._pav_trial_done)

    def _pav_trial_done(self) -> None:
        self._pav_index += 1
        if self._pav_index >= len(self._pav_types):
            self._pav_index -= 1  # the event names the trial that just finished
            self._pav_trial_event("ALL_TRIALS_COMPLETE")
            self._pav_index += 1
            self._end_ms = self._clock
        else:
            self._pav_schedule_iti(self._clock)

    # --- Event emitters ---

    def _device_name(self, target: str) -> str:
        """Level-007 spelling: the operant Scheduler says CUE_1/PUMP_1, the Pavlovian one CUE/PUMP."""
        if self._is_pav:
            return target
        return {"CUE": "CUE_1", "PUMP": "PUMP_1"}.get(target, target)

    def _log_activation(self, target: str, start: int, end: int) -> None:
        """Scheduler::LogDeviceActivation / PavlovianScheduler::LogDeviceActivation."""
        if target == "LASER":
            event = "STIMULATION" if self._is_pav else "STIM"
        else:
            event = "TONE" if target.startswith("CUE") else "INFUSION"
        self._send({
            "level": "007", "device": self._device_name(target), "pin": self._pins[target],
            "event": event, "start_timestamp": start, "end_timestamp": end,
        })

    def _log_press(self, orientation: str, press_class: str, start: int, end: int) -> None:
        pin = self._pins[f"LEVER_{orientation}"]
        msg = {
            "level": "007", "device": "SWITCH_LEVER" if self._is_pav else f"LEVER_{orientation}", "pin": pin,
            "event": "PRESS", "class": press_class,
            "start_timestamp": start, "end_timestamp": end, "orientation": orientation,
        }
        self._send(msg)

    def _log_lick(self, start: int, end: int) -> None:
        if not self.lick_armed:
            return
        self._send({
            "level": "007", "device": "LICK_CIRCUIT", "pin": self._pins["LICK"],
            "event": "LICK", "start_timestamp": start, "end_timestamp": end,
        })


class SimulatedSerial:
    """Drop-in replacement for serial.Serial used by the REACHER kernel.

    Implements only the interface subset that read_serial() and
    send_serial_command() rely on. Data flows through an internal queue
    bridging the FirmwareSimulator output thread to REACHER's serial reader.
    """

    def __init__(
        self,
        baudrate: int = 115200,
        timeout: float = 1,
        paradigm: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.port: Optional[str] = "SIMULATOR"
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = False
        self.paradigm = paradigm  # what this "board" is flashed with; None = fr
        self._rx_queue: queue.Queue = queue.Queue()
        self._simulator = FirmwareSimulator(self._rx_queue, paradigm=paradigm, seed=seed)

    def open(self):
        self.is_open = True
        # Opening the port reboots the board, and the firmware's active pump is RAM-only.
        self._simulator.pump2_active = False
        self._simulator._reconfigure()
        logger.info("SimulatedSerial opened")

    def close(self):
        self._simulator.stop()
        self.is_open = False
        logger.info("SimulatedSerial closed")

    def readline(self) -> bytes:
        try:
            return self._rx_queue.get(timeout=self.timeout)
        except queue.Empty:
            return b""

    def write(self, data: bytes):
        try:
            text = data.decode("utf-8").strip()
            cmd_data = json.loads(text)
            self._simulator.handle_command(cmd_data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("SimulatedSerial: could not parse write data: %r", data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        while not self._rx_queue.empty():
            try:
                self._rx_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def in_waiting(self) -> int:
        return 1 if not self._rx_queue.empty() else 0
