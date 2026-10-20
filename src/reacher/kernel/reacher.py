import bisect
import csv
import serial
import queue
import threading
import time
import json
import io
import os
import logging
from typing import Callable, Dict, List, Optional, Union
from serial.tools import list_ports

from ..uploader.boards import SIMULATOR_PORT, is_simulator_port
from .commands import build_command_payload, CommandCode, SCHEDULE_TO_PARADIGM
# Leaf submodules rather than the `diagnostics` package, so importing the kernel
# never re-enters a partially initialised `reacher/__init__`.
from ..diagnostics.schema import TIER_WIRE as _TIER_WIRE
from ..diagnostics.setup import log as _diag_log


def _package_version() -> str:
    """This package's version, resolved lazily.

    Deferred rather than imported at module scope because ``reacher/__init__``
    imports this module, so the name does not exist yet at import time.
    """
    try:
        from .. import __version__

        return __version__
    except Exception:
        return "unknown"


#: The firmware's serial input buffer.  Every sketch silently drains commands
#: longer than this, so the backend detects and reports the truncation.
_FIRMWARE_RX_BUFFER = 128

_USE_VALUE = object()  # sentinel: use the `value` arg from send_command()
# Queue items drained at restart: (mark, epoch, line). The epoch is the one
# in force when the line was still queued, so the handler logs it but does
# not append it to the session the swap installed.
_REPLAY_EPOCH = object()

_PUMP_KEYS = ("PUMP_1", "PUMP_2")
# Pavlovian firmware logs pump 1 as "PUMP"; operant firmware logs it as "PUMP_1".
_PUMP_DEVICE_KEYS = {"PUMP": "PUMP_1", "PUMP_1": "PUMP_1", "PUMP_2": "PUMP_2"}

# Maps command codes to (device_name, field_name, value_or_sentinel)
# _USE_VALUE means the value is taken from the `value` arg passed to send_command().
#
# device_name MUST be spelled the way firmware spells it at log level 000
# (reportDeviceConfig / reportDeviceLever), not level 007. This path and the
# firmware-config path both append into `hardware_settings` and dedup on plain
# string equality, so a name firmware never emits at 000 will not collide with
# the firmware's own row — it silently creates a second, permanent entry for the
# same device with disjoint fields, and both reach the browser.
#
# The lick circuit is the trap: firmware calls it LICK at 000 and LICK_CIRCUIT at
# 007. LICK is canonical here, matching the normalization the 007 handler already
# performs below. tests/test_device_names.py::test_l8_* enforces this.
_COMMAND_STATE_MAP: dict[int, tuple[str, str, object]] = {
    # --- Arm/Disarm ---
    300: ("CUE", "armed", False),
    301: ("CUE", "armed", True),
    310: ("CUE2", "armed", False),
    311: ("CUE2", "armed", True),
    400: ("PUMP", "armed", False),
    401: ("PUMP", "armed", True),
    410: ("PUMP2", "armed", False),
    411: ("PUMP2", "armed", True),
    500: ("LICK", "armed", False),
    501: ("LICK", "armed", True),
    600: ("LASER", "armed", False),
    601: ("LASER", "armed", True),
    900: ("MICROSCOPE", "armed", False),
    901: ("MICROSCOPE", "armed", True),
    1000: ("LEVER_RH", "armed", False),
    1001: ("LEVER_RH", "armed", True),
    1300: ("LEVER_LH", "armed", False),
    1301: ("LEVER_LH", "armed", True),
    # --- Cue parameters ---
    371: ("CUE", "frequency", _USE_VALUE),
    372: ("CUE", "duration", _USE_VALUE),
    381: ("CUE2", "frequency", _USE_VALUE),
    382: ("CUE2", "duration", _USE_VALUE),
    # --- Cue pulse parameters ---
    374: ("CUE", "pulse_on", _USE_VALUE),
    375: ("CUE", "pulse_off", _USE_VALUE),
    384: ("CUE2", "pulse_on", _USE_VALUE),
    385: ("CUE2", "pulse_off", _USE_VALUE),
    # --- Pump parameters ---
    472: ("PUMP", "duration", _USE_VALUE),
    482: ("PUMP2", "duration", _USE_VALUE),
    221: ("PUMP2", "active", _USE_VALUE),
    # --- Laser parameters ---
    671: ("LASER", "frequency", _USE_VALUE),
    672: ("LASER", "duration", _USE_VALUE),
    673: ("LASER", "onset_delay", _USE_VALUE),
    681: ("LASER", "mode", "contingent"),
    682: ("LASER", "mode", "independent"),
    684: ("LASER", "mode", "rh_lever"),
    # --- Pavlovian laser ---
    691: ("LASER", "trial_filter", "cs_plus"),
    692: ("LASER", "trial_filter", "cs_minus"),
    693: ("LASER", "trial_filter", "cs_both"),
    694: ("LASER", "phase", "reward"),
    695: ("LASER", "phase", "cue"),
    # --- Reinforcement ratio (scheduler-wide, not per-lever) ---
    # 201, 1075 and 1375 are three names for one write. Every sketch that
    # handles them calls the same `scheduler.SetRatio()`, which sets
    # `triggers[0].threshold` on the sketch's single Scheduler instance
    # (firmware Scheduler.cpp:377-383) — the per-lever cases differ only in the
    # string they pass to logParamChange. Modelling them as two independent
    # LEVER_RH/LEVER_LH fields recorded a per-lever ratio no layer below
    # implements, and left 201 — the code the session-start path actually
    # sends — unmodelled entirely. One scheduler-scoped field, exactly as
    # 1074/1374 share one timeout interval and 1077/1377 share one timeout mode.
    201: ("CONTROLLER", "ratio", _USE_VALUE),
    1075: ("CONTROLLER", "ratio", _USE_VALUE),
    1375: ("CONTROLLER", "ratio", _USE_VALUE),
    # --- Lever timeout (scheduler-wide, not per-lever) ---
    # 1074/1374 both write the one TIMEOUT_INTERVAL / Scheduler::SetTimeoutInterval
    # (fr.ino:315-317, 335-337) and 1077/1377 the one Scheduler::SetTimeoutMode
    # flag, so the last write wins on the board. Modelling them as LEVER_RH and
    # LEVER_LH fields recorded two values the board cannot hold — a GUI that sent
    # RH then LH showed RH = N while the board ran LH's value. One
    # scheduler-scoped field each, exactly as 201/1075/1375 share CONTROLLER.ratio.
    # 1077/1377 were previously unmodelled on the host entirely.
    1074: ("CONTROLLER", "timeout", _USE_VALUE),
    1374: ("CONTROLLER", "timeout", _USE_VALUE),
    1077: ("CONTROLLER", "timeout_mode", _USE_VALUE),
    1377: ("CONTROLLER", "timeout_mode", _USE_VALUE),
    # --- Pin reassignment (suffix x76) ---
    376: ("CUE", "pin", _USE_VALUE),
    386: ("CUE2", "pin", _USE_VALUE),
    476: ("PUMP", "pin", _USE_VALUE),
    486: ("PUMP2", "pin", _USE_VALUE),
    576: ("LICK", "pin", _USE_VALUE),
    676: ("LASER", "pin", _USE_VALUE),
    976: ("MICROSCOPE", "trigger_pin", _USE_VALUE),
    1076: ("LEVER_RH", "pin", _USE_VALUE),
    1376: ("LEVER_LH", "pin", _USE_VALUE),
    # --- SLM ---
    1100: ("SLM", "armed", False),
    1101: ("SLM", "armed", True),
    # Field names match the "frequency"/"duration" keys reportDeviceConfig()'s
    # 4-arg overload emits at SESSION_START, so the optimistic command-send
    # update and the firmware's own config report land on the same keys.
    1102: ("SLM", "frequency", _USE_VALUE),
    1103: ("SLM", "duration", _USE_VALUE),
    1176: ("SLM", "pin", _USE_VALUE),
}

class REACHER:
    """A class to manage serial communication and data collection for REACHER experiments."""

    # Class default so an instance built without __init__ still reads the follow-reward mode.
    _counted_pumps: Optional[List[str]] = None

    def __init__(
        self,
        session_id: Optional[str] = None,
        event_callback: Optional[Callable[[str, str, dict], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize a REACHER instance.

        Args:
            session_id: Unique identifier for this session (assigned by SessionManager).
            event_callback: Optional callback ``(session_id, event_type, data)`` for
                broadcasting events over WebSocket.
            on_stop: Optional callback invoked after ``stop_program()`` completes,
                used by SessionManager to broadcast the "stopped" state.
        """

        self.session_id: Optional[str] = session_id
        self.event_callback = event_callback
        self._on_stop = on_stop

        # Serial variables
        self._is_simulated: bool = False
        # Name of the simulated port (SIMULATOR, or SIMn when the session manager
        # gave this session its own instance); kept across reset/reconnect.
        self._simulated_port: str = SIMULATOR_PORT
        # Which sketch the simulated board impersonates. Meaningful only when
        # _is_simulated; a real board's paradigm comes from its flashed hex and
        # is read back from IDENTIFY, never from here.
        self._simulated_paradigm: Optional[str] = None
        self.ser: serial.Serial = serial.Serial(baudrate=115200, timeout=1)
        self.queue: queue.Queue = queue.Queue(maxsize=5000)
        # Bytes from a readline() that returned before the newline. Connect
        # drains the bootloader with reset_input_buffer(), which slices the
        # identification line setup() prints; the piece is not a frame.
        self._rx_fragment: bytes = b""
        # Fix: F-003 — Configurable serial reconnection parameters
        self._SERIAL_RECONNECT_RETRIES: int = 3
        self._SERIAL_RECONNECT_DELAY: int = 3  # seconds between retry attempts
        # Fix 2.6: count queue-overflow drops and throttle WS-warning emission.
        # Without visibility, an overflowed queue silently drops serial lines
        # while the frontend still shows "connected". The counter is cumulative;
        # the throttle keeps an overflow storm from flooding the WS queue in
        # turn (which would itself need to drop). -inf ensures the first
        # overflow always emits.
        self._queue_overflow_count: int = 0
        self._last_queue_overflow_emit: float = float("-inf")

        # Fix: #33 — Surface log-write failures to the operator via WS warning.
        # Throttle per log_type (1 s) and accumulate a consecutive-failure count
        # so a ENOSPC storm emits one event/sec with the run total, not one per write.
        self._last_log_failure_emit: Dict[str, float] = {}
        self._log_failure_counts: Dict[str, int] = {"event_log": 0, "controller_log": 0}

        # Fix 7.4: track cumulative event_callback failures so the API layer
        # can surface degraded WS/observer health without risking recursion
        # (a broken callback is the one channel we can't use to self-report).
        self._emit_failure_count: int = 0

        # Thread variables
        # Fix: PY-007 — Threads created here but started lazily in open_serial()
        # Fix 7.1: wrap each target in a resilient shim so an unhandled
        # exception in read_serial/handle_queue/monitor_time_limit does not
        # silently kill the daemon. See _resilient().
        self.serial_thread: threading.Thread = self._make_thread(self.read_serial, "read_serial")
        self.queue_thread: threading.Thread = self._make_thread(self.handle_queue, "handle_queue")
        self.time_check_thread: threading.Thread = self._make_thread(self.monitor_time_limit, "monitor_time_limit")
        self.thread_lock: threading.Lock = threading.Lock()
        # Bumped under thread_lock when restart_program swaps the session
        # buffers. handle_queue samples it at dequeue so a handler already
        # running cannot append into the buffers the swap just installed.
        self._record_epoch: int = 0
        self._dequeued_epoch = threading.local()
        # Close must not take thread_lock: send_serial_command holds it across
        # its write, and a caller that already owns it would deadlock. This
        # lock only covers the post-pause flush+close.
        self._serial_close_lock: threading.Lock = threading.Lock()
        self.serial_flag: threading.Event = threading.Event()
        self.program_flag: threading.Event = threading.Event()
        self.program_running: bool = False
        self.time_check_flag: threading.Event = threading.Event()
        self.serial_flag.set()
        self.program_flag.set()
        self.time_check_flag.set()
        # Fix: LAZ-001 — Event gate for firmware readiness (IDENTIFY response)
        # Set when firmware_information is populated (IDENTIFY ack received),
        # used to delay "connected" state transition until bootloader exits and
        # firmware is truly ready to process commands.
        self._firmware_ready: threading.Event = threading.Event()
        # Gate for the firmware's CONTROLLER END event (level 007, device CONTROLLER, event END).
        # stop_program() waits on this before exporting so the END row lands in behavior_data.
        self._controller_end_received: threading.Event = threading.Event()

        # Data process variables (guarded by thread_lock for cross-thread access)
        self.behavior_data: List[Dict[str, Union[str, int]]] = []
        self.frame_data: List[int] = []
        self.slm_data: List[int] = []
        # Config/state change trail for the session summary (issue #12): every
        # level-000 (config) and level-001 (arm/disarm, state) event, in
        # arrival order. Host-stamped with `received_at` because firmware
        # carries no onboard clock for these frames (unlike level-007 events,
        # which come with their own start/end millis).
        self._session_changes: List[Dict] = []
        self._infusion_count: int = 0  # Atomic counter — avoids O(n) rescan in check_limit_met
        # Per-pump tallies. _infusion_count counts only infusions from the reward pump
        # active at arrival; these also record the other pump for the session totals.
        self._infusions_by_pump: Dict[str, int] = {"PUMP_1": 0, "PUMP_2": 0}
        self._reward_pump: str = "PUMP_1"  # Firmware power-on target; SET_ACTIVE_PUMP (221) moves it
        # None follows the reward pump; an explicit list pins which pumps' infusions count.
        self._counted_pumps: Optional[List[str]] = None
        # Counted off the serial stream for the same reason as infusions: an
        # export must not depend on a browser's tally of what it happened to
        # receive. Presses include every lever contact (ACTIVE/INACTIVE/TIMEOUT),
        # matching what the UI displays.
        self._press_count: int = 0
        self._trial_count: int = 0
        # Fix: F-002 — Memory warning thresholds for unbounded data lists
        self._DATA_WARNING_THRESHOLD = 100_000  # Warn when lists exceed this size
        self._data_warning_emitted: bool = False

        # Segmentation state
        self._segment_number: int = 0
        self._cumulative_infusion_count: int = 0
        self._cumulative_infusions_by_pump: Dict[str, int] = {"PUMP_1": 0, "PUMP_2": 0}
        self._cumulative_press_count: int = 0
        self._cumulative_trial_count: int = 0
        self._segment_exports: List[str] = []
        self._segment_event_counts: List[int] = []
        # Cumulative recorded-event tally keyed "DEVICE.EVENT". Unlike behavior_data it is
        # NOT cleared by split_segment(), so the summary counts every segment.
        self._event_type_counts: Dict[str, int] = {}

        # Program variables
        self.program_start_time: Optional[float] = None
        self.program_end_time: Optional[float] = None
        self.paused_time: float = 0
        self.paused_start_time: Optional[float] = None
        self.limit_type: Optional[str] = None
        self.infusion_limit: Optional[int] = None
        self.time_limit: Optional[int] = None
        self.stop_delay: int = 0
        self.last_infusion_time: Optional[float] = None

        # Configuration variables
        self.box_name: Optional[str] = None
        self.firmware_information: Dict = {
            "sketch": None,
            "version": None,
            "baud_rate": None,
            "desc": None
        }
        self.hardware_settings: List = []
        # Distinct from REACHER_LOG_DIR, which roots the diagnostics sink (LOG/runs).
        _session_log_root = os.environ.get("REACHER_SESSION_LOG_DIR") or os.path.expanduser("~/REACHER/LOG")
        # The directory name is a one-second timestamp, so two sessions created in
        # the same second (multi-session runs, e.g. SIM1 + SIM2) would otherwise
        # share one directory and interleave their event/controller/interface
        # logs. exist_ok=False makes the claim atomic; later arrivals get _2, _3...
        _base = os.path.join(_session_log_root, self.get_time())
        self.reacher_log_path = _base
        _n = 1
        while True:
            try:
                os.makedirs(self.reacher_log_path, exist_ok=False)
                break
            except FileExistsError:
                _n += 1
                self.reacher_log_path = f"{_base}_{_n}"
        self.controller_log: str = os.path.join(self.reacher_log_path, "controller_log.json")
        self.interface_log: str = os.path.join(self.reacher_log_path, "interface_log.log")
        self.logger = logging.getLogger(f"reacher.{session_id or 'default'}")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
        fh = logging.FileHandler(self.interface_log)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        if not self.logger.handlers or not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        # Fix: F-010 — Persistent file handle for event log (avoids open/close per event)
        self._event_log_path = os.path.join(self.reacher_log_path, "event_log.jsonl")
        self._event_log_file: Optional[io.TextIOWrapper] = None
        self._event_log_write_count: int = 0
        self._EVENT_LOG_FSYNC_INTERVAL: int = 50  # fsync every N writes

        # Fix 4.9: mirror the event-log pattern — keep controller_log.json open
        # across writes and batch fsyncs instead of open/close/fsync per line.
        self._controller_log_file: Optional[io.TextIOWrapper] = None
        self._controller_log_write_count: int = 0
        self._CONTROLLER_LOG_FSYNC_INTERVAL: int = 10

        self.data_destination: Optional[str] = None
        self.behavior_filename: Optional[str] = None
        self.code_dict: Dict = {
            "000": self.update_firmware_information,
            "001": self.logger.info,
            "006": self.handle_firmware_error,
            "007": self.update_behavioral_events,
            "008": self.update_frame_events,
            "009": self.update_slm_events
        }

        self.logger.info("REACHER instance created")

    def reset(self) -> None:
        """Reset the REACHER instance to its initial state.

        **Description:**
        - Stops any active program and clears the data queue.
        - Closes the serial connection and reinitializes all variables.
        - Restarts serial and queue handling threads.
        - Ensures a clean slate for a new experiment session.
        """
        
        self.logger.info("Resetting REACHER instance")

        if not self.program_flag.is_set():
            self.stop_program()

        self.clear_queue()
        self.close_serial()
        self.time_check_flag.clear()  # Stop the time check thread

        self.behavior_data = []
        self._event_type_counts = {}
        self.frame_data = []
        self.slm_data = []
        self._infusion_count = 0
        self._infusions_by_pump = {"PUMP_1": 0, "PUMP_2": 0}
        self._data_warning_emitted = False  # Fix: F-002 — Reset warning flag
        self._event_log_write_count = 0  # Fix: F-010 — Reset write counter
        self._controller_log_write_count = 0  # Fix 4.9 — Reset write counter
        # reacher#12 fix: _session_changes is NOT cleared here — session_summary.json
        # is documented (see _write_session_summary) to span the full log-directory
        # lifetime, same as controller_log.json, which stays open across reset().
        # Clearing this list here silently dropped pre-reset history from the summary
        # while the raw controller_log.json kept it, so the two diverged.

        self.program_start_time = None
        self.program_end_time = None
        self.paused_time = 0
        self.paused_start_time = None
        self.last_infusion_time = None

        self.firmware_information = {}
        self.controller_log: str = os.path.join(self.reacher_log_path, "controller_log.json")
        self.interface_log: str = os.path.join(self.reacher_log_path, "interface_log.log")
        self.data_destination = None
        self.behavior_filename = None

        if self._is_simulated:
            from .simulator import SimulatedSerial
            # Carry the paradigm across the rebuild — a reset that dropped it
            # would silently revert the board to fr mid-session, which is the
            # bug this parameter exists to prevent (P1-BUG-1).
            self.ser = SimulatedSerial(baudrate=115200, timeout=1, paradigm=self._simulated_paradigm)
            self.ser.port = self._simulated_port
        else:
            self.ser = serial.Serial(baudrate=115200, timeout=1)
        self.queue = queue.Queue(maxsize=5000)  # Fix: F-006 — Match __init__ maxsize
        self._rx_fragment = b""

        # Old threads must be gone before the flags flip back: a surviving read_serial
        # would otherwise keep reading the rebuilt serial object.
        self._join_worker_threads(timeout=5.0)
        self.serial_flag.clear()
        self.program_flag.set()
        self.time_check_flag.set()
        self.serial_thread = self._make_thread(self.read_serial, "read_serial")
        self.queue_thread = self._make_thread(self.handle_queue, "handle_queue")
        self.time_check_thread = self._make_thread(self.monitor_time_limit, "monitor_time_limit")
        self.serial_thread.start()
        self.queue_thread.start()
        self.time_check_thread.start()

        self.logger.info("REACHER instance reset")

    def get_COM_ports(self) -> List[str]:
        """Retrieve a list of available COM ports.

        **Description:**
        - Scans the system for connected serial devices.
        - Returns a list of COM port names (e.g., "COM1", "/dev/ttyUSB0").
        - If no ports are found, returns ["No available ports"].

        **Returns:**
        - `List[str]`: Available COM port names or a placeholder if none are detected.
        """
        
        self.logger.info("Accessing available COM ports")
        
        available_ports = [p.device for p in list_ports.comports() if p.vid and p.pid]
        available_ports.append("SIMULATOR")

        self.logger.info("COM ports successfully accessed")

        return available_ports
    
    def set_COM_port(self, port: str, paradigm: Optional[str] = None) -> None:
        """Set the COM port for serial communication.

        **Description:**
        - Configures the serial port to the specified name (e.g., "COM3").
        - Raises ValueError if port is not valid or not found in the system.

        **Args:**
        - `port (str)`: The name of the COM port to use.
        - `paradigm (str | None)`: **Simulator only.** Which sketch the
          simulated board should impersonate, normally the session's paradigm.
          A real board is ignored here on purpose — its paradigm is whatever
          hex is flashed on it, and `connect` reads that back from IDENTIFY
          (`api/routers/serial.py`). Without this the simulator always
          identified as `fr.ino`, so connecting *overwrote* the session's
          paradigm with "fr" and every non-FR command then 400'd at the
          hardware router's paradigm gate (P1-BUG-1).

        **Raises:**
        - `ValueError`: If the port is not available.
        """

        self.logger.info("Setting COM port")

        if is_simulator_port(port):
            from .simulator import SimulatedSerial
            self._simulated_paradigm = paradigm
            self._simulated_port = port
            self.ser = SimulatedSerial(baudrate=115200, timeout=1, paradigm=paradigm)
            self.ser.port = port
            self._is_simulated = True
        elif port in [p.device for p in list_ports.comports() if p.vid and p.pid]:
            self.ser.port = port
            self._is_simulated = False
            self._simulated_paradigm = None
        else:
            # Fix: SER-003 — Raise on invalid port instead of silently ignoring
            raise ValueError(f"Port {port!r} is not available")

        self.logger.info(f"Set COM port to {port}")

    def open_serial(self) -> None:
        """Open the serial connection and start communication threads.

        **Description:**
        - Establishes a serial connection using the configured port.
        - Starts threads for reading serial data and processing the queue.
        - Sends a "LINK" command to initiate communication with the microcontroller.
        """
        
        self.logger.info("Opening serial connection")
        
        if self.ser.is_open:
            self.ser.close()
            time.sleep(1)
        self.ser.open()
        # Opening the port reboots the board, which falls back to pump 1.
        self._reward_pump = "PUMP_1"
        if self.serial_flag.is_set():
            self.serial_flag.clear()
        if not self.serial_thread.is_alive():
            self.logger.info("--> Starting serial thread")
            self.serial_thread = self._make_thread(self.read_serial, "read_serial")
            self.serial_thread.start()
        if not self.queue_thread.is_alive():
            self.logger.info("--> Starting queue thread")
            self.queue_thread = self._make_thread(self.handle_queue, "handle_queue")
            self.queue_thread.start()
        # Fix: PY-007 — Start time-check thread on connection (deferred from __init__)
        if not self.time_check_thread.is_alive():
            self.logger.info("--> Starting time check thread")
            self.time_check_thread = self._make_thread(
                self.monitor_time_limit, "monitor_time_limit"
            )
            self.time_check_thread.start()
        time.sleep(2)
        self.ser.reset_input_buffer()
        # Drop a short read taken before this drain. The bytes still in the
        # driver buffer were just discarded, so that fragment must not be
        # glued onto the next frame.
        self._rx_fragment = b""

        self.logger.info("--> Serial connection opened")

    def clear_queue(self) -> None:
        """Clear the data queue and wait for processing to complete.

        **Description:**
        - Empties the queue by adding a sentinel value (None).
        - Waits for all queued items to be processed before proceeding.
        - Ensures no residual data remains in the queue.
        """
        
        self.logger.info("Clearing queue")
        
        self.logger.info("---> Sending sentinel...")
        self.queue.put_nowait(None)
        self.logger.info("---> Waiting for queue to be processed")
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.logger.info("---> Waiting for queue thread to terminate...")
        self.queue.join()
        
        self.logger.info("Queue terminated")

    def close_serial(self) -> None:
        """Close the serial connection and terminate related threads.

        **Description:**
        - Sends an "UNLINK" command to the microcontroller.
        - Closes the serial port and stops the serial thread.
        - Handles any errors during closure with detailed logging.
        """
        
        self.logger.info("Closing serial connection")
        
        try:
            self.serial_flag.set()
            self.logger.info("---> Serial flag set to terminate threads")
            if self.ser.is_open:
                # Left unlocked. destroy_session and a second close_serial can
                # both close this port during the drain pause; holding the
                # close lock across it deadlocks that nested close.
                time.sleep(0.5)
            # flush() raises PortNotOpenError if the port was closed during the
            # pause. Re-check under the close lock so overlapping closes flush
            # at most once.
            with self._serial_close_lock:
                if self.ser.is_open:
                    self.ser.flush()
                    self.ser.close()
                    self.logger.info("---> Serial port closed")
            self.logger.info("---> Waiting for serial thread to terminate...")
            self.serial_thread.join(timeout=5)
            self.logger.info("---> Serial thread terminated")
        except Exception as e:
            self.logger.error(f"Error during closure: {e}")
        finally:
            self._close_event_log()
            self._close_controller_log()  # Fix 4.9
            self.logger.info("--> Cleanup complete")

    def shutdown(self, timeout: float = 20.0) -> None:
        """Stop this instance's threads and release its log handlers.

        **Description:**
        - Idempotent and never raises.
        - Does not stop a running program; the caller must do that first.
        - Signals the serial, queue and time-check threads to exit, closes the
          serial port, and joins the threads under one shared deadline.
        - Removes and closes the log handlers, and drops the session logger so
          it does not persist in ``logging.Logger.manager.loggerDict``.
        """
        try:
            self.time_check_flag.clear()
            self.serial_flag.set()
            self.program_flag.set()
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            if self.ser.is_open or self.serial_thread.is_alive():
                self.close_serial()
            self._join_worker_threads(timeout)
        except Exception:
            self.logger.exception("Error during shutdown")
        finally:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
                handler.close()
            if self.logger.name != "reacher.default":
                logging.Logger.manager.loggerDict.pop(self.logger.name, None)

    def _join_worker_threads(self, timeout: float) -> None:
        """Join the serial, queue and time-check threads under one shared deadline.

        Skips threads that are not alive and the calling thread, so it is safe
        to call from one of the kernel's own threads. Warns about any still
        alive after the deadline rather than raising.
        """
        deadline = time.monotonic() + timeout
        threads = (self.serial_thread, self.queue_thread, self.time_check_thread)
        for thread in threads:
            if thread is threading.current_thread() or not thread.is_alive():
                continue
            thread.join(timeout=max(0, deadline - time.monotonic()))
        still_alive = [t.name for t in threads if t.is_alive() and t is not threading.current_thread()]
        if still_alive:
            self.logger.warning("Threads still alive after %.1fs: %s", timeout, still_alive)

    def _resilient(self, target, name: str):
        """Fix 7.1: wrap a thread body so an unhandled exception does not
        silently kill the daemon.

        Catches exceptions from ``target``, logs a traceback, emits a
        ``warning`` WS event with ``reason="thread_crash"``, and retries
        after a one-second back-off. Gives up after ten consecutive
        failures so a fundamentally broken target does not spin forever.
        A clean return from ``target`` (the normal stop path — serial_flag
        or time_check_flag is set) exits the shim immediately.
        """
        _MAX_RESTARTS = 10

        def wrapped():
            restarts = 0
            while True:
                try:
                    target()
                    return  # target exited cleanly via its own stop flag
                except Exception:
                    restarts += 1
                    self.logger.exception(
                        "Thread %s crashed (restart %d/%d)",
                        name, restarts, _MAX_RESTARTS,
                    )
                    try:
                        self._emit("warning", {
                            "reason": "thread_crash",
                            "thread": name,
                        })
                    except Exception:
                        # Protect against a failing event_callback — bug 7.4
                        # handles the counting; do not recurse here.
                        pass
                    if restarts >= _MAX_RESTARTS:
                        self.logger.error(
                            "Thread %s exceeded restart budget; giving up", name,
                        )
                        return
                    time.sleep(1.0)

        wrapped.__name__ = f"resilient_{name}"
        return wrapped

    def _make_thread(self, target, name: str) -> threading.Thread:
        """Fix 7.1: build a resilient daemon thread using ``_resilient``."""
        return threading.Thread(
            target=self._resilient(target, name),
            daemon=True,
            name=name,
        )

    def _assemble_rx_frame(self, data: bytes) -> Optional[bytes]:
        """Return one newline-terminated frame, or None if ``data`` is not one yet.

        ``readline()`` returns on ``\\n`` or when the 1s timeout expires.
        ``open_serial`` waits out the bootloader and then calls
        ``reset_input_buffer()`` while this thread may already be inside
        ``readline()``. ``setup()`` prints ``SendIdentification()`` in that
        window, so the read comes back as the tail of that line (through the
        newline) or as the head with no newline. A held head is dropped when
        the next read itself starts an object, so it cannot be glued onto the
        IDENTIFY reply that connect waits for.
        """
        if not data:
            return None
        if self._rx_fragment and data.lstrip().startswith(b"{"):
            self.logger.debug("Dropping sliced serial fragment (%d bytes)", len(self._rx_fragment))
            self._rx_fragment = b""
        if not data.endswith(b"\n"):
            self._rx_fragment += data
            self.logger.debug("Holding incomplete serial line (%d bytes)", len(self._rx_fragment))
            return None
        frame = self._rx_fragment + data
        self._rx_fragment = b""
        return frame

    def read_serial(self) -> None:
        """Read data from the serial port and queue it for processing.

        **Description:**
        - Runs in a thread to continuously monitor the serial port.
        - Reads incoming data when available and adds it to the queue.
        - Uses a lock to ensure thread-safe operations.
        """
        while not self.serial_flag.is_set():
            try:
                if self.ser.is_open and self.ser.in_waiting > 0:
                    # Fix: SER-004 — Read outside lock, only queue inside lock
                    frame = self._assemble_rx_frame(self.ser.readline())
                    if frame is None:
                        continue
                    # Fix: SER-001 — Strict UTF-8 decode; discard corrupt lines
                    try:
                        decoded = frame.decode(encoding='utf-8', errors='strict').strip()
                    except UnicodeDecodeError:
                        self.logger.warning("Corrupt serial data (non-UTF-8), discarding: %s", frame.hex())
                        _diag_log("serial.rx_corrupt", tier=_TIER_WIRE, lvl="warn",
                                  src="reacher.kernel", session_id=self.session_id,
                                  hex=frame.hex()[:512])
                        continue
                    # Firmware events are JSON objects. The tail of the boot
                    # identification line (or a blank line) is not one, and
                    # json.loads on it is the connect-time parse error.
                    if not decoded.startswith("{"):
                        self.logger.debug("Discarding non-JSON serial line: %r", decoded)
                        continue
                    self.logger.info(f"Serial data received: {decoded}")
                    # Verbatim wire record.  Carries session_id but no corr_id:
                    # this runs on a long-lived daemon thread with no request
                    # context, and the firmware cannot echo one back.
                    _diag_log(
                        "serial.rx",
                        tier=_TIER_WIRE,
                        lvl="debug",
                        src="reacher.kernel",
                        session_id=self.session_id,
                        line=decoded,
                    )
                    # Fix: F-003 — Discard if queue is full; prevents OOM on I/O lag
                    try:
                        self.queue.put_nowait(decoded)
                    except queue.Full:
                        # Fix 2.6: surface overflow to the frontend so a silent
                        # data-loss window is visible. Throttle to one emission
                        # per second so an overflow storm doesn't itself flood
                        # the WS queue.
                        self._queue_overflow_count += 1
                        self.logger.warning("Serial queue full — dropping line")
                        _now = time.monotonic()
                        if _now - self._last_queue_overflow_emit > 1.0:
                            self._emit("warning", {
                                "reason": "queue_overflow",
                                "count": self._queue_overflow_count,
                            })
                            self._last_queue_overflow_emit = _now
                else:
                    time.sleep(0.1)
            except (serial.SerialException, OSError) as e:
                # Fix: F-003 — Attempt serial reconnection before giving up
                self.logger.error("Serial disconnect detected: %s", e)
                self._emit("disconnect", {"reason": str(e), "reconnecting": True})

                reconnected = False
                for attempt in range(1, self._SERIAL_RECONNECT_RETRIES + 1):
                    self.logger.info(
                        "Reconnection attempt %d/%d in %ds...",
                        attempt, self._SERIAL_RECONNECT_RETRIES, self._SERIAL_RECONNECT_DELAY,
                    )
                    time.sleep(self._SERIAL_RECONNECT_DELAY)
                    try:
                        if self.ser.is_open:
                            self.ser.close()
                        self.ser.open()
                        self.ser.reset_input_buffer()
                        self._rx_fragment = b""
                        self.logger.info("Serial reconnected on attempt %d", attempt)
                        self._emit("reconnected", {"attempt": attempt})
                        reconnected = True
                        break
                    except (serial.SerialException, OSError) as retry_err:
                        self.logger.warning("Reconnection attempt %d failed: %s", attempt, retry_err)

                if not reconnected:
                    self.logger.error("All reconnection attempts exhausted — serial permanently lost")
                    self._emit("disconnect", {"reason": str(e), "reconnecting": False})
                    self.serial_flag.set()
                    if self.program_running:
                        try:
                            self.stop_program()
                        except Exception:
                            self.logger.warning("Failed to stop program after disconnect", exc_info=True)
                    break

    def handle_queue(self) -> None:
        """Process data from the queue.

        **Description:**
        - Runs in a thread to handle queued serial data.
        - Processes each item as configuration or event data.
        - Terminates when a sentinel value (None) is received or serial flag is set.
        """
        while True:
            try:
                line = self.queue.get(timeout=1)
                self.queue.task_done()
                if line is None:
                    self.logger.info("Sentinel received. Exiting queue thread.")
                    break
                if isinstance(line, tuple) and len(line) == 3 and line[0] is _REPLAY_EPOCH:
                    # Drained at restart. Keep the epoch from before that swap.
                    epoch, line = line[1], line[2]
                else:
                    # Sampled before handle_data so the event-log write stays
                    # ahead of thread_lock. The append re-checks under the lock.
                    epoch = self._record_epoch
                self.logger.info(f"--> Data in queue: {line}")
                self._dequeued_epoch.value = epoch
                try:
                    self.handle_data(line)
                finally:
                    self._dequeued_epoch.value = None
            except queue.Empty:
                if self.serial_flag.is_set():
                    break
                continue

    def handle_data(self, line: str) -> None:
        """Process a line of data from the queue.

        **Description:**
        - Interprets a line of serial data.
        - Attempts to parse as JSON for configuration or as comma-separated event data.
        - Delegates to specific handlers based on data format.

        **Args:**
        - `line (str)`: The raw data line to process.
        """

        try:
            self.logger.info(f"--> Processing data: {line}")

            data = json.loads(line)

            self._write_controller_log(data)

            # Fix: F-004 — Use .get() so a missing 'level' key doesn't KeyError
            level = data.get('level')
            if level is None:
                self.logger.warning("Firmware event missing 'level': %s", data)
                return

            if level in ("000", "001"):
                with self.thread_lock:  # Fix: F-009 — guard cross-thread list access
                    self._session_changes.append({"received_at": time.time(), **data})
            handler = self.code_dict.get(level)
            if handler is not None:
                handler(data)
            else:
                self.logger.warning(f"Unknown event level: {level}. Data: {data}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}. Raw data: {line}")
        except Exception as e:
            # Fix: F-006 — Notify frontend of processing failures so events aren't silently dropped
            self.logger.error(f"Error processing data: {e}. Raw line: {line}")
            self._emit("kernel_error", {"reason": str(e), "raw": line})

    def handle_firmware_error(self, event: dict) -> None:
        """Handle firmware error events (level 006).

        Args:
            event: Parsed JSON event dict from firmware containing 'desc' key.
        """
        # Fix: XL-002 — Log error_code when present
        error_code = event.get("error_code", "UNKNOWN")
        desc = event.get("desc", "Unknown")
        # Firmware names the offending code on an unrecognised command. Resolve
        # it here: paradigms differ in what they implement, so "Command not
        # found" alone cannot tell an operator which control did nothing.
        if (command := event.get("command")) is not None:
            try:
                desc = f"{desc} ({CommandCode(int(command)).name}={command})"
            except (ValueError, TypeError):
                desc = f"{desc} (unregistered code {command})"
        self.logger.error(f"Firmware error [{error_code}]: {desc}")
        self._emit("error", event)

    # Fix: XL-001 — Minimum firmware version check
    MIN_FIRMWARE_VERSION = "v2.0.0"

    @staticmethod
    def _parse_version(version_str: str) -> tuple:
        """Parse a version string like 'v2.0.0' into a comparable tuple."""
        clean = version_str.lstrip("vV")
        parts = []
        for p in clean.split("."):
            try:
                parts.append(int(p))
            except ValueError:
                parts.append(0)
        return tuple(parts)

    @staticmethod
    def _normalise_param_change(event: dict) -> Optional[dict]:
        """Fold a firmware ``logParamChange`` record into the host row shape.

        ``{"device": D, "param": P, "value": V}`` becomes ``{"device": D, P: V}``,
        which is what ``_update_hardware_setting`` writes for a host-side send.
        IDENTIFY also carries a ``device`` but has a ``sketch``, so it is not one.
        """
        if "param" not in event or "value" not in event or "sketch" in event:
            return None
        return {**{k: v for k, v in event.items() if k not in ("param", "value")}, event["param"]: event["value"]}

    def update_firmware_information(self, event: dict) -> None:
        if "device" not in event:
            # Session-parameter extras of the StartSession dump (pr_step,
            # variable_interval, omission_interval, Pavlovian timing). Not a
            # device row, and not emitted as config: the web store appends a
            # fresh hardware row for every device-less config record.
            with self.thread_lock:
                self.firmware_information.update(event)
            self.logger.info("--> Updated session parameters: %s", event)
            return
        param_row = self._normalise_param_change(event)
        if event["device"] == "CONTROLLER":
            with self.thread_lock:  # Fix: F-009 — guard cross-thread dict access
                self.firmware_information.update(event)
                if param_row is not None:
                    self._merge_hardware_row(param_row)
            self.logger.info("--> Updated arduino configuration")
            # Fix: LAZ-001 — Signal firmware readiness (IDENTIFY ack received)
            # Unblock any waiting connect/post-upload flow so state transitions to "connected"
            self._firmware_ready.set()
            # Fix: XL-001 — Warn on firmware version mismatch
            fw_version = event.get("version", "")
            if fw_version:
                try:
                    if self._parse_version(fw_version) < self._parse_version(self.MIN_FIRMWARE_VERSION):
                        self.logger.warning(
                            "Firmware version %s is below minimum %s",
                            fw_version, self.MIN_FIRMWARE_VERSION,
                        )
                        self._emit("error", {
                            "level": "006",
                            "error_code": "E_FW_VERSION",
                            "desc": f"Firmware {fw_version} is below minimum {self.MIN_FIRMWARE_VERSION}",
                            "device": "CONTROLLER",
                        })
                except Exception:
                    self.logger.debug("Could not parse firmware version %r", fw_version)

            # The floor check above only catches firmware older than the minimum.
            # Hex that is above the floor but not what this package ships is the
            # signature of a stale manual flash, and otherwise passes silently.
            _pkg_version = _package_version()
            _mismatch = bool(fw_version) and _pkg_version not in ("", "unknown") and (
                fw_version.lstrip("v") != _pkg_version
            )
            _diag_log(
                "firmware.identified",
                tier=_TIER_WIRE,
                lvl="warn" if _mismatch else "info",
                msg=(
                    f"Firmware {fw_version} does not match packaged {_pkg_version}"
                    if _mismatch
                    else f"Firmware identified: {fw_version}"
                ),
                src="reacher.kernel",
                session_id=self.session_id,
                firmware_version=fw_version,
                package_version=_pkg_version,
                sketch=event.get("sketch"),
                baud_rate=event.get("baud_rate"),
                version_mismatch=_mismatch,
            )
            # Fix: SER-002 — Verify baud rate matches after firmware identification
            fw_baud = event.get("baud_rate")
            if fw_baud is not None and self.ser.is_open:
                if int(fw_baud) != self.ser.baudrate:
                    self.logger.warning(
                        "Baud rate mismatch: firmware reports %s, serial configured %s",
                        fw_baud, self.ser.baudrate,
                    )
                    self._emit("error", {
                        "level": "006",
                        "error_code": "E_BAUD_MISMATCH",
                        "desc": f"Baud mismatch: firmware={fw_baud}, host={self.ser.baudrate}",
                        "device": "CONTROLLER",
                    })
            self._emit("config", event)
        else:
            device = event.get("device")
            with self.thread_lock:  # Fix: F-009 — guard cross-thread list access
                if param_row is not None:
                    self._merge_hardware_row(param_row)
                else:
                    for i, entry in enumerate(self.hardware_settings):
                        if entry.get("device") == device:
                            self.hardware_settings[i] = event
                            break
                    else:
                        self.hardware_settings.append(event)
            self.logger.info("--> Updated hardware defaults list")
            self._emit("config", event)

    def _merge_hardware_row(self, row: dict) -> None:
        """Merge *row* into its device's ``hardware_settings`` entry, appending if absent.

        Caller holds ``thread_lock``.
        """
        for entry in self.hardware_settings:
            if entry.get("device") == row.get("device"):
                entry.update(row)
                return
        self.hardware_settings.append(dict(row))

    def _update_hardware_setting(self, device: str, updates: dict) -> None:
        """Update a device entry in hardware_settings in-place and emit a config event."""
        with self.thread_lock:  # Fix: F-009 — guard cross-thread list access
            for entry in self.hardware_settings:
                if entry.get("device") == device:
                    entry.update(updates)
                    emit_data = dict(entry)
                    break
            else:
                new_entry = {"device": device, **updates}
                self.hardware_settings.append(new_entry)
                emit_data = dict(new_entry)
        self._emit("config", emit_data)

    def _dequeued_epoch_is_current(self) -> bool:
        """Caller holds ``thread_lock``.

        A line taken off the queue before ``restart_program`` swaps the session
        buffers carries the epoch sampled at dequeue. It is still written to
        the event log, but it must not re-seed the new session. Direct callers
        (no sample) belong to whatever session is current.
        """
        dequeued = getattr(self._dequeued_epoch, "value", None)
        if dequeued is None:
            return True
        return dequeued == self._record_epoch

    def _take_queued_lines(self) -> tuple:
        """Pull every line still on the queue. ``None`` is reported, not returned.

        Replay tuples keep the epoch they were tagged with. Anything else is
        ``(None, line)`` and the caller stamps the pre-swap epoch.
        """
        pending = []
        saw_sentinel = False
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            self.queue.task_done()
            if item is None:
                saw_sentinel = True
                continue
            if isinstance(item, tuple) and len(item) == 3 and item[0] is _REPLAY_EPOCH:
                pending.append((item[1], item[2]))
            else:
                pending.append((None, item))
        return pending, saw_sentinel

    def _requeue_for_epoch(self, pending: list, old_epoch: int, saw_sentinel: bool) -> None:
        """Put drained lines back tagged with the epoch from before the swap.

        They go in front of lines that arrived after the drain, so the event
        log keeps pre-restart lines ahead of the new session. The queue thread
        logs them and the epoch fence keeps them out of the new buffers. A
        stopped queue thread, or a queue with no room, cannot do that, so log
        those lines here.
        """
        tagged = [(_REPLAY_EPOCH, old_epoch if epoch is None else epoch, line) for epoch, line in pending]
        if not self.queue_thread.is_alive():
            self._log_replay_lines(tagged)
            self._restore_sentinel(saw_sentinel)
            return
        if not self._put_replay_ahead(tagged):
            # Queueing a suffix and logging the rest would split their order.
            # Write them all here. The epoch tag still fences the new session.
            self.logger.error(
                "Restart could not requeue %d pre-swap line(s); writing them to the event log",
                len(tagged),
            )
            self._log_replay_lines(tagged)
        self._restore_sentinel(saw_sentinel)

    def _put_replay_ahead(self, tagged: list) -> bool:
        """Insert *tagged* in front of whatever was queued after the drain.

        ``queue.Queue.put`` only appends, which logs those later lines first.
        Returns False when they do not all fit; nothing is queued in that case.
        """
        if not tagged:
            return True
        with self.queue.mutex:
            if self.queue.maxsize > 0 and self.queue._qsize() + len(tagged) > self.queue.maxsize:
                return False
            # Same bookkeeping as Queue.put. The deque is the queue's storage;
            # appendleft is the only way to keep the drained lines ahead.
            for item in reversed(tagged):
                self.queue.queue.appendleft(item)
                self.queue.unfinished_tasks += 1
            self.queue.not_empty.notify()
        return True

    def _restore_sentinel(self, saw_sentinel: bool) -> None:
        if not saw_sentinel:
            return
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            self.logger.warning("Queue full; restart could not restore the queue sentinel")

    def _log_replay_lines(self, tagged: list) -> None:
        """Log drained lines on this thread.

        Used when the queue thread is not running, and when the queue has no
        room to take the lines back ahead of the new session. ``_dequeued_epoch``
        is thread-local, so this sample does not clobber the queue thread's.
        """
        for _, epoch, line in tagged:
            self._dequeued_epoch.value = epoch
            try:
                self.handle_data(line)
            finally:
                self._dequeued_epoch.value = None

    def update_behavioral_events(self, event: dict) -> None:
        entry_dict: Dict[str, Union[str, int]] = {}
        
        match event.get('device'):
            case "LEVER_RH" | "LEVER_LH":
                entry_dict['device'] = event.get('device')
                entry_dict['event'] = f"{event.get('class')}_{event.get('event')}"
                entry_dict['start_timestamp'] = event.get('start_timestamp')
                entry_dict['end_timestamp'] = event.get('end_timestamp')
            case "SWITCH_LEVER":
                # Legacy firmware (pre-v2.4.x): orientation field holds "RH" or "LH"
                entry_dict['device'] = "LEVER_" + event.get('orientation', '')
                entry_dict['event'] = f"{event.get('class')}_{event.get('event')}"
                entry_dict['start_timestamp'] = event.get('start_timestamp')
                entry_dict['end_timestamp'] = event.get('end_timestamp')
            case "LICK_CIRCUIT":
                entry_dict['device'] = "LICK"
                entry_dict['event'] = event.get('event')
                entry_dict['start_timestamp'] = event.get('start_timestamp')
                entry_dict['end_timestamp'] = event.get('end_timestamp')
            case "CONTROLLER":
                entry_dict['device'] = event.get('device')
                entry_dict['event'] = event.get('event')
                entry_dict['start_timestamp'] = event.get('timestamp')
                entry_dict['end_timestamp'] = event.get('timestamp')
                if event.get('event') == 'END':
                    self._controller_end_received.set()
            case "PAVLOV":
                entry_dict['device'] = event.get('device')
                entry_dict['event'] = event.get('event')
                entry_dict['start_timestamp'] = event.get('start_timestamp') or event.get('timestamp')
                entry_dict['end_timestamp'] = event.get('end_timestamp') or event.get('timestamp')
                if (trial_type := event.get('trial_type')) is not None:
                    entry_dict['trial_type'] = trial_type
            case _:
                entry_dict['device'] = event.get('device')
                entry_dict['event'] = event.get('event')
                entry_dict['start_timestamp'] = event.get('start_timestamp')
                entry_dict['end_timestamp'] = event.get('end_timestamp')  
                
        # Append-only event log (written before program_running guard)
        self._write_event_log({"type": "behavior", **entry_dict})

        # Always emit for real-time UI feedback (e.g. hardware testing)
        self._emit("event", entry_dict)

        # Persist to dataset when actively recording, and always for CONTROLLER
        # events (START/END markers must survive the program_running=False guard
        # that stop_program() sets before the END event arrives from firmware).
        # A line dequeued before the restart swap is already in the event log
        # above; it must not re-seed the new session's buffers or counters.
        stale = False
        if (self.program_running and not self.program_flag.is_set()) \
                or entry_dict.get('device') == 'CONTROLLER':
            infusion_pump = None
            with self.thread_lock:
                if not self._dequeued_epoch_is_current():
                    stale = True
                else:
                    self.behavior_data.append(entry_dict)
                    _type_key = f"{entry_dict.get('device')}.{entry_dict.get('event')}"
                    self._event_type_counts[_type_key] = self._event_type_counts.get(_type_key, 0) + 1
                    pump = _PUMP_DEVICE_KEYS.get(entry_dict.get('device'))
                    if pump and entry_dict.get('event') == 'INFUSION':
                        self._infusions_by_pump[pump] += 1
                        if pump == self._reward_pump:
                            self._infusion_count += 1
                        infusion_pump = pump
                    elif entry_dict.get('device') in ('LEVER_RH', 'LEVER_LH') \
                            and str(entry_dict.get('event', '')).endswith('PRESS'):
                        self._press_count += 1
                    elif entry_dict.get('device') == 'PAVLOV' and entry_dict.get('event') == 'TRIAL_START':
                        self._trial_count += 1
                    # Fix: F-002 — Warn when data lists grow dangerously large
                    total = len(self.behavior_data) + len(self.frame_data) + len(self.slm_data)
            if not stale:
                if total >= self._DATA_WARNING_THRESHOLD and not self._data_warning_emitted:
                    self._data_warning_emitted = True
                    self.logger.warning(
                        "Data list size (%d) exceeds threshold (%d) — risk of OOM. "
                        "event_log.jsonl is the durable backup.",
                        total, self._DATA_WARNING_THRESHOLD,
                    )
                    self._emit("warning", {
                        "message": "In-memory data lists are large. Data is safely logged to disk.",
                        "total_entries": total,
                        "threshold": self._DATA_WARNING_THRESHOLD,
                    })
                self.logger.info("--> Updated behavioral data")
                if infusion_pump:
                    self._emit_infusion_count()

        # Auto-stop when firmware signals all Pavlovian trials are complete.
        # A completion that was already dequeued when restart swapped the
        # buffers belongs to the previous run.
        if (not stale
                and entry_dict.get('device') == 'PAVLOV'
                and entry_dict.get('event') == 'ALL_TRIALS_COMPLETE'
                and self.program_running):
            self.logger.info("All Pavlovian trials complete, stopping program")
            threading.Thread(target=self.stop_program, daemon=True).start()
                
    def update_frame_events(self, event: dict) -> None:
        if self.program_flag.is_set():
            return
        ts = int(event.get('timestamp'))
        missed = int(event.get('missed', 0))
        self._write_event_log({"type": "frame", "timestamp": ts, "missed": missed})
        with self.thread_lock:
            if self._dequeued_epoch_is_current():
                self.frame_data.append(ts)
        if missed > 0:
            self.logger.warning(f"--> Dropped {missed} microscope frame(s) before timestamp {ts}")
        else:
            self.logger.info("--> Updated frame data")
        self._emit("frame", {"timestamp": ts, "missed": missed})

    def update_slm_events(self, event: dict) -> None:
        if self.program_flag.is_set():
            return
        ts = int(event.get('timestamp'))
        self._write_event_log({"type": "slm", "timestamp": ts})
        with self.thread_lock:
            if self._dequeued_epoch_is_current():
                self.slm_data.append(ts)
        self.logger.info("--> Updated SLM timestamp data")
        self._emit("event", {
            "device": "SLM",
            "event": "TIMESTAMP",
            "start_timestamp": ts,
            "end_timestamp": ts,
        })

    def send_serial_command(self, command: dict) -> None:
        """Send a command to the Arduino via serial.

        **Description:**
        - Transmits a string command to the connected microcontroller.
        - Ensures the serial port is open before sending.

        **Args:**
        - `command (str)`: The command to send (e.g., "START-PROGRAM").

        **Raises:**
        - `Exception`: If the serial port is not open.
        """
        with self.thread_lock:
            if not self.ser.is_open:
                raise Exception("Serial port is not open.")
            send = json.dumps(command).encode() + b'\n'
            self.logger.info(f"Sending command '{send}' to Arduino.")
            # The firmware reads into a 128-byte buffer and *silently* drains
            # anything longer (see ParseCommands in each sketch), so an
            # oversized command is lost with no error on either side.  Surface
            # it here — this is the only place it can be detected.
            if len(send) > _FIRMWARE_RX_BUFFER:
                _diag_log(
                    "serial.tx_oversize",
                    tier=_TIER_WIRE,
                    lvl="warn",
                    msg=f"Command exceeds firmware {_FIRMWARE_RX_BUFFER}-byte buffer; it will be truncated",
                    src="reacher.kernel",
                    session_id=self.session_id,
                    length=len(send),
                    limit=_FIRMWARE_RX_BUFFER,
                    line=send.decode("utf-8", "replace"),
                )
            _diag_log(
                "serial.tx",
                tier=_TIER_WIRE,
                lvl="debug",
                src="reacher.kernel",
                session_id=self.session_id,
                line=send.decode("utf-8", "replace").rstrip("\n"),
            )
            self.ser.write(send)
            self.ser.flush()
            time.sleep(0.05)

    def send_command(self, code: int, value=None) -> None:
        """Send any command from the command registry over serial.

        Uses ``build_command_payload`` to construct the proper JSON payload
        (e.g. ``{"cmd": 371, "frequency": 8000}``) and transmits it.

        Args:
            code: Command code from CommandCode / COMMAND_REGISTRY.
            value: Optional payload value (int or bool depending on command spec).
        """
        payload = build_command_payload(code, value)
        self.send_serial_command(payload)
        if code == int(CommandCode.SET_ACTIVE_PUMP):
            with self.thread_lock:
                self._reward_pump = "PUMP_2" if value else "PUMP_1"
        if code in _COMMAND_STATE_MAP:
            device, field, mapped = _COMMAND_STATE_MAP[code]
            effective = value if mapped is _USE_VALUE else mapped
            self._update_hardware_setting(device, {field: effective})

    def _emit(self, event_type: str, data: dict) -> None:
        """Broadcast an event via the registered callback (if any).

        Fix 7.4: increment a failure counter rather than re-emitting a warning
        through the same callback (which would recurse if the callback is the
        broken channel). The counter is surfaced via /api/sessions/{sid}.
        """
        if self.event_callback and self.session_id:
            try:
                self.event_callback(self.session_id, event_type, data)
            except Exception:
                self._emit_failure_count += 1
                self.logger.warning(
                    "Event callback failed (total=%d)",
                    self._emit_failure_count, exc_info=True,
                )

    @property
    def emit_failure_count(self) -> int:
        """Cumulative count of event_callback invocations that raised."""
        return self._emit_failure_count

    def _write_event_log(self, entry: dict) -> None:
        """Append a JSON line to the append-only event log.

        Fix: F-010 — Keeps the file handle open for the session lifetime and
        batches fsync calls (every _EVENT_LOG_FSYNC_INTERVAL writes) to reduce
        I/O overhead at high event rates while maintaining durability.
        """
        try:
            if self._event_log_file is None or self._event_log_file.closed:
                self._event_log_file = open(self._event_log_path, "a")
            self._event_log_file.write(json.dumps(entry) + "\n")
            self._event_log_file.flush()
            self._event_log_write_count += 1
            if self._event_log_write_count >= self._EVENT_LOG_FSYNC_INTERVAL:
                os.fsync(self._event_log_file.fileno())
                self._event_log_write_count = 0
        except Exception:
            self.logger.warning("Failed to write event log entry", exc_info=True)
            self._log_failure_counts["event_log"] += 1
            _now = time.monotonic()
            if _now - self._last_log_failure_emit.get("event_log", float("-inf")) > 1.0:
                self._last_log_failure_emit["event_log"] = _now
                self._emit("warning", {
                    "reason": "log_write_failure",
                    "log_type": "event_log",
                    "failures_since_last_emit": self._log_failure_counts["event_log"],
                })
                self._log_failure_counts["event_log"] = 0

    def _close_event_log(self) -> None:
        """Flush and close the persistent event log file handle."""
        if self._event_log_file is not None and not self._event_log_file.closed:
            try:
                self._event_log_file.flush()
                os.fsync(self._event_log_file.fileno())
                self._event_log_file.close()
            except Exception:
                self.logger.warning("Failed to close event log", exc_info=True)
                self._emit("warning", {"reason": "log_close_failure", "log_type": "event_log"})

    def _write_controller_log(self, data: dict) -> None:
        """Append a JSON line to controller_log.json (Fix 4.9).

        Mirrors _write_event_log: persistent handle, flush every write, fsync
        every _CONTROLLER_LOG_FSYNC_INTERVAL writes. Avoids the syscall storm
        of open/close/fsync per firmware line at high event rates.
        """
        try:
            if self._controller_log_file is None or self._controller_log_file.closed:
                self._controller_log_file = open(self.controller_log, "a", newline="")
            self._controller_log_file.write(json.dumps(data) + "\n")
            self._controller_log_file.flush()
            self._controller_log_write_count += 1
            if self._controller_log_write_count >= self._CONTROLLER_LOG_FSYNC_INTERVAL:
                os.fsync(self._controller_log_file.fileno())
                self._controller_log_write_count = 0
        except Exception:
            self.logger.warning("Failed to write controller log", exc_info=True)
            self._log_failure_counts["controller_log"] += 1
            _now = time.monotonic()
            if _now - self._last_log_failure_emit.get("controller_log", float("-inf")) > 1.0:
                self._last_log_failure_emit["controller_log"] = _now
                self._emit("warning", {
                    "reason": "log_write_failure",
                    "log_type": "controller_log",
                    "failures_since_last_emit": self._log_failure_counts["controller_log"],
                })
                self._log_failure_counts["controller_log"] = 0

    def _close_controller_log(self) -> None:
        """Flush and close the persistent controller log file handle."""
        if self._controller_log_file is not None and not self._controller_log_file.closed:
            try:
                self._controller_log_file.flush()
                os.fsync(self._controller_log_file.fileno())
                self._controller_log_file.close()
            except Exception:
                self.logger.warning("Failed to close controller log", exc_info=True)
                self._emit("warning", {"reason": "log_close_failure", "log_type": "controller_log"})

    def _export_segment(self, behavior: list, suffix: str = "") -> str:
        """Write behavior_events{suffix}.csv to the session log directory.

        Uses the full frame_data list for frame-index binary search so that
        microscope frame indices remain continuous across splits.

        Args:
            behavior: Snapshot of behavior_data rows to export.
            suffix: Optional filename suffix (e.g. "_001").
        Returns:
            Path to the written CSV file.
        """
        frame_data = self.get_frame_data()
        frame_timestamps = sorted(int(ts) for ts in frame_data if ts)

        csv_buf = io.StringIO()
        fieldnames = ["device", "event", "start_timestamp", "end_timestamp", "start_frame_index", "end_frame_index"]
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in behavior:
            start_ts = row.get("start_timestamp")
            end_ts = row.get("end_timestamp")
            start_fi = self._find_frame_index(frame_timestamps, int(start_ts)) if start_ts not in (None, "") else None
            end_fi = self._find_frame_index(frame_timestamps, int(end_ts)) if end_ts not in (None, "") else None
            out = {k: row.get(k, "") for k in ("device", "event", "start_timestamp", "end_timestamp")}
            out["start_frame_index"] = start_fi if start_fi is not None else ""
            out["end_frame_index"] = end_fi if end_fi is not None else ""
            writer.writerow(out)
        path = os.path.join(self.reacher_log_path, f"behavior_events{suffix}.csv")
        with open(path, "w", newline="") as f:
            f.write(csv_buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        return path

    def _auto_export(self) -> None:
        """Write behavior_events.csv and frame_timestamps.csv to the session log directory.

        Called automatically on stop_program(). Failure never blocks shutdown.
        If segments were split during the session, the remaining data is exported
        with the next segment number suffix.
        """
        try:
            behavior = self.get_behavior_data()
            suffix = f"_{self._segment_number + 1:03d}" if self._segment_number > 0 else ""
            self._export_segment(behavior, suffix)

            # frame_timestamps.csv — only when microscope data was captured
            frame_data = self.get_frame_data()
            frame_timestamps = sorted(int(ts) for ts in frame_data if ts)
            if frame_timestamps:
                ft_buf = io.StringIO()
                ft_writer = csv.DictWriter(ft_buf, fieldnames=["frame_index", "timestamp_ms"])
                ft_writer.writeheader()
                for i, ts in enumerate(frame_timestamps):
                    ft_writer.writerow({"frame_index": i, "timestamp_ms": ts})
                path = os.path.join(self.reacher_log_path, "frame_timestamps.csv")
                with open(path, "w", newline="") as f:
                    f.write(ft_buf.getvalue())
                    f.flush()
                    os.fsync(f.fileno())

            # slm_timestamps.csv — only when SLM data was captured
            slm_data = self.get_slm_data()
            slm_timestamps = sorted(int(ts) for ts in slm_data if ts)
            if slm_timestamps:
                st_buf = io.StringIO()
                st_writer = csv.DictWriter(st_buf, fieldnames=["event_index", "timestamp_ms"])
                st_writer.writeheader()
                for i, ts in enumerate(slm_timestamps):
                    st_writer.writerow({"event_index": i, "timestamp_ms": ts})
                path = os.path.join(self.reacher_log_path, "slm_timestamps.csv")
                with open(path, "w", newline="") as f:
                    f.write(st_buf.getvalue())
                    f.flush()
                    os.fsync(f.fileno())

            export_parts = [f"behavior_events{suffix}.csv"]
            if frame_timestamps:
                export_parts.append("frame_timestamps.csv")
            if slm_timestamps:
                export_parts.append("slm_timestamps.csv")
            self.logger.info("Auto-export complete: %s", " + ".join(export_parts))
        except Exception as e:
            # Fix: F-005 — Surface export failures to the frontend in real time
            self.logger.warning("Auto-export failed", exc_info=True)
            self._emit("export_failed", {"reason": str(e)})

    def _build_session_summary(self) -> Dict:
        """Distill level-000/001 events into initial config, ordered changes, and final state.

        reacher#12: config/arm-disarm frames carry no onboard clock (unlike
        level-007 events, which come with their own start/end millis), so
        `received_at` on each change is the host's wall-clock time at serial
        receipt, not a firmware timestamp. `final_state` reuses
        `firmware_information`/`hardware_settings` directly rather than
        replaying `changes`, since those dicts are already maintained as the
        latest-value-per-device view by `update_firmware_information`.
        """
        with self.thread_lock:  # Fix: F-009 — snapshot under lock
            changes = [dict(entry) for entry in self._session_changes]
            firmware_information = dict(self.firmware_information)
            hardware_settings = [dict(entry) for entry in self.hardware_settings]
            event_type_counts = dict(self._event_type_counts)

        initial_firmware = None
        initial_devices: Dict[str, Dict] = {}
        for entry in changes:
            if entry.get("level") != "000":
                continue
            device = entry.get("device")
            if device == "CONTROLLER":
                if initial_firmware is None:
                    initial_firmware = entry
                continue
            if device and device not in initial_devices:
                initial_devices[device] = entry

        return {
            "session_id": self.session_id,
            "generated_at": time.time(),
            "note": (
                "received_at is the host's wall-clock time when the frame was read "
                "off serial, not a firmware timestamp — level-000/001 frames carry "
                "none. Treat it as accurate to serial-read latency, not to the "
                "on-device clock that level-007 behavioral events use."
            ),
            # Count of every recorded behavior event, by type — the same events that land
            # in behavior_events*.csv, across all segments of the run.
            "event_counts": {
                "total": sum(event_type_counts.values()),
                "by_type": dict(sorted(event_type_counts.items())),
            },
            "initial_configuration": {
                "firmware": initial_firmware,
                "devices": initial_devices,
            },
            "changes": changes,
            "final_state": {
                "firmware": firmware_information,
                "devices": {
                    entry["device"]: entry for entry in hardware_settings if entry.get("device")
                },
            },
        }

    def get_session_summary(self) -> Dict:
        """Thread-safe accessor for the distilled session change-tracking summary (reacher#12)."""
        return self._build_session_summary()

    def _write_session_summary(self) -> None:
        """Write session_summary.json to the session log directory.

        Called automatically on stop_program(), mirroring _auto_export(). Spans
        the full connected lifetime of this log directory (like
        controller_log.json), not just the most recent program run. Failure
        never blocks shutdown.
        """
        try:
            summary = self._build_session_summary()
            path = os.path.join(self.reacher_log_path, "session_summary.json")
            with open(path, "w") as f:
                f.write(json.dumps(summary, indent=2))
                f.flush()
                os.fsync(f.fileno())
            self.logger.info("Session summary written: %s", path)
        except Exception as e:
            self.logger.warning("Failed to write session_summary.json", exc_info=True)
            self._emit("export_failed", {"reason": str(e), "artifact": "session_summary.json"})

    @staticmethod
    def _find_frame_index(frame_timestamps: list, event_ts: int):
        """Return the index of the last frame at or before *event_ts*, or None."""
        if not frame_timestamps:
            return None
        idx = bisect.bisect_right(frame_timestamps, event_ts) - 1
        if idx < 0:
            return None
        return idx

    def _join_queue_with_timeout(self, timeout: float = 5.0) -> None:
        """Wait for all queued items to finish, with a timeout to avoid hanging."""
        with self.queue.all_tasks_done:
            while self.queue.unfinished_tasks:
                if not self.queue.all_tasks_done.wait(timeout=timeout):
                    self.logger.warning("Queue drain timed out after %.1fs", timeout)
                    break

    def get_detected_paradigm(self) -> Optional[str]:
        """Return the paradigm detected from firmware identification, or None.

        "_lite" sketches (e.g. fr_lite.ino) report the same ``schedule`` as
        their full counterpart (fr.ino both report FIXED_RATIO), so the
        schedule alone can't distinguish them — check the ``sketch`` field too.
        """
        schedule = self.firmware_information.get("schedule")
        if not schedule or not isinstance(schedule, str):
            return None
        base = SCHEDULE_TO_PARADIGM.get(schedule)
        if base is None:
            return None
        sketch = self.firmware_information.get("sketch")
        if isinstance(sketch, str) and sketch.endswith("_lite.ino"):
            return f"{base}_lite"
        return base

    def set_limit_type(self, limit_type: str) -> None:
        """Set the type of limit for program execution.

        **Description:**
        - Defines the stopping condition for the experiment.
        - Valid options: "Time", "Infusion", or "Both".

        **Args:**
        - `limit_type (str)`: The type of limit to enforce.
        """
        if limit_type in ['Time', 'Infusion', 'Both', 'Trials']:
            self.limit_type = limit_type
            self.logger.info(f"Limit type set to: {limit_type}")
        else:
            self.logger.warning(f"Invalid limit type: {limit_type}")

    def set_infusion_limit(self, limit: int) -> None:
        """Set the maximum number of infusions allowed.

        **Description:**
        - Specifies the maximum number of infusions before stopping.

        **Args:**
        - `limit (int)`: The infusion limit.
        """
        self.infusion_limit = limit

    def set_time_limit(self, limit: int) -> None:
        """Set the maximum time allowed in seconds.

        **Description:**
        - Specifies the maximum duration of the experiment in seconds.

        **Args:**
        - `limit (int)`: The time limit in seconds.
        """
        self.time_limit = limit

    def set_stop_delay(self, delay: int) -> None:
        """Set the delay after last infusion before stopping.

        **Description:**
        - Sets a delay in seconds after the last infusion before program termination.

        **Args:**
        - `delay (int)`: The delay in seconds; None means no delay.
        """
        self.stop_delay = delay or 0

    def _reset_session_buffers(self) -> None:
        """Clear all per-session data buffers and counters."""
        self.behavior_data = []
        self._event_type_counts = {}
        self.frame_data = []
        self.slm_data = []
        self._infusion_count = 0
        self._infusions_by_pump = {"PUMP_1": 0, "PUMP_2": 0}
        self._press_count = 0
        self._trial_count = 0
        self._segment_number = 0
        self._cumulative_infusion_count = 0
        self._cumulative_infusions_by_pump = {"PUMP_1": 0, "PUMP_2": 0}
        self._cumulative_press_count = 0
        self._cumulative_trial_count = 0
        self._segment_exports = []
        self._segment_event_counts = []
        self.paused_time = 0
        self.last_infusion_time = None

    def start_program(self) -> None:
        """Start the experimental program.

        **Description:**
        - Initiates the experiment by sending "START-PROGRAM" to the microcontroller.
        - Records the start time for limit checking.
        """
        # Fail before destroying anything: a start that can't reach the
        # firmware anyway must not wipe buffered behavior data first.
        if not self.ser.is_open:
            raise Exception("Serial port is not open.")
        self._reset_session_buffers()
        if self.program_flag.is_set():
            self.program_flag.clear()
        self.program_running = True
        # Cleared here, not in stop_program: firmware can end the session on
        # its own (an unarmed Pavlovian run ends on the first loop tick), and
        # that END must still count when Stop is pressed later.
        self._controller_end_received.clear()
        try:
            self.send_serial_command({"cmd": 101})
        except Exception:
            # Backstop for a port that closes between the check above and
            # here — send_serial_command raises before writing anything in
            # that case too, so the firmware never saw a start command.
            self.program_running = False
            raise
        self.program_start_time = time.time()
        self._write_event_log({"type": "SESSION_START", "timestamp": self.program_start_time})
        self.logger.info(f"Program started at {self.get_time()}")

    def stop_program(self) -> None:
        """Stop the experimental program.

        **Description:**
        - Terminates the experiment by sending "END-PROGRAM".
        - Drains remaining queued events before closing serial.
        - Cleans up resources and records the end time.
        - Invokes the ``on_stop`` callback so the session manager can
          broadcast the "stopped" state to the frontend.
        """
        with self.thread_lock:
            if not self.program_running:
                return
            self.program_running = False  # Guard against re-entrance
        self.logger.info("Ending program...")

        # Notify frontend IMMEDIATELY — before any blocking I/O
        if self._on_stop:
            try:
                self._on_stop()
            except Exception:
                self.logger.warning("on_stop callback failed", exc_info=True)

        # Then do cleanup (send firmware END, close serial, etc.)
        port_open = getattr(self.ser, "is_open", False)
        if port_open:
            try:
                self.send_serial_command({"cmd": 100})
            except Exception:
                self.logger.warning("END-PROGRAM send failed; treating serial port as closed", exc_info=True)
                port_open = False
        if port_open:
            # Wait for the firmware's CONTROLLER END event (~200ms after cmd 100).
            # Fallback to 2s sleep if it doesn't arrive within 8s.
            received = self._controller_end_received.wait(timeout=8.0)
            if not received:
                self.logger.warning("Timed out waiting for CONTROLLER END — export may be incomplete")
                time.sleep(2)
        else:
            self.logger.warning("Serial port closed; skipping END handshake, exporting buffered data")
        self.program_flag.set()
        # Drain remaining events with timeout to avoid indefinite hang
        self._join_queue_with_timeout(5.0)
        try:
            self.close_serial()
        except Exception:
            self.logger.warning("close_serial failed; continuing with export", exc_info=True)
        self.program_end_time = time.time()
        self._write_event_log({"type": "SESSION_END", "timestamp": self.program_end_time})
        self._close_event_log()  # Fix: F-010 — Ensure all events flushed before export
        self._auto_export()
        self._write_session_summary()  # reacher#12
        self.logger.info(f"Program ended at {self.get_time()}")

    def pause_program(self) -> None:
        """Pause the experimental program.

        **Description:**
        - Temporarily halts the experiment and records the pause start time.
        - Sends SESSION_PAUSE command to firmware to gate input processing.
        """
        self.program_flag.set()
        self.paused_start_time = time.time()
        self.send_serial_command({"cmd": 105, "paused": True})

    def resume_program(self) -> None:
        """Resume the experimental program.

        **Description:**
        - Resumes the experiment and calculates total paused time.
        - Sends SESSION_PAUSE(false) to firmware to resume processing.
        """
        # Fix: F-007 — Guard against calling resume when not paused (paused_start_time is None)
        if self.paused_start_time is None:
            self.logger.warning("resume_program() called but paused_start_time is None — ignoring")
            return
        if self.program_flag.is_set():
            self.program_flag.clear()
        self.paused_time += time.time() - self.paused_start_time
        self.paused_start_time = None
        self.send_serial_command({"cmd": 105, "paused": False})

    def get_program_running(self) -> bool:
        """Check if the program is currently running (not paused or stopped).

        **Returns:**
        - `bool`: True if running and not paused, False otherwise.
        """
        return self.program_running and not self.program_flag.is_set()

    def split_segment(self) -> dict:
        """Export the current segment and reset buffers for the next segment.

        Snapshots behavior_data, exports it as a numbered CSV, clears the
        in-memory buffer and per-segment counters, and increments the segment
        counter.  frame_data is NOT cleared — microscope frame indices remain
        continuous across splits.

        Returns:
            Dict with segment_number and export_path.
        """
        with self.thread_lock:
            snapshot = list(self.behavior_data)
            segment_infusions = self._infusion_count
            self._cumulative_infusion_count += segment_infusions
            for p in _PUMP_KEYS:
                self._cumulative_infusions_by_pump[p] += self._infusions_by_pump[p]
            self._cumulative_press_count += self._press_count
            self._cumulative_trial_count += self._trial_count
            self.behavior_data = []
            self._infusion_count = 0
            self._infusions_by_pump = {"PUMP_1": 0, "PUMP_2": 0}
            self._press_count = 0
            self._trial_count = 0
            self._data_warning_emitted = False
            self._segment_number += 1

        if not snapshot:
            self.logger.warning("split_segment: no behavioral data in segment %d", self._segment_number)

        export_path = self._export_segment(snapshot, f"_{self._segment_number:03d}")
        self._segment_exports.append(export_path)
        self._segment_event_counts.append(len(snapshot))

        self._write_event_log({
            "type": "SEGMENT_SPLIT",
            "segment": self._segment_number,
            "events_exported": len(snapshot),
            "timestamp": time.time(),
        })
        self._emit("split", {
            "segment_number": self._segment_number,
            "export_path": export_path,
        })
        self.logger.info("Segment %d exported (%d events): %s", self._segment_number, len(snapshot), export_path)
        return {"segment_number": self._segment_number, "export_path": export_path}

    def restart_program(self) -> None:
        """Stop and re-start the Arduino without destroying the session.

        Sends SESSION_END then SESSION_START to the firmware, clears all
        in-memory data and counters, and resets timing.  The serial connection
        and hardware configuration are preserved.  program_running stays True
        throughout to avoid triggering stop_program's re-entrance guard.

        Lines already off the queue, and lines still sitting on it, are written
        to the event log but do not re-enter the cleared buffers or counters.
        """
        self.logger.info("Restarting program...")
        self._controller_end_received.clear()
        self.send_serial_command({"cmd": 100})
        time.sleep(1)

        # Take lines the queue thread has not dequeued yet, then bump the
        # epoch in the same critical section as the buffer swap. A handler
        # that already sampled the old epoch (or a replay of a drained line)
        # then loses the append. clear_queue() cannot be used here: it stops
        # the queue thread, and the new session still needs it.
        pending, saw_sentinel = self._take_queued_lines()
        with self.thread_lock:
            old_epoch = self._record_epoch
            self._record_epoch += 1
            self.behavior_data = []
            self._event_type_counts = {}
            self.frame_data = []
            self.slm_data = []
            self._infusion_count = 0
            self._infusions_by_pump = {"PUMP_1": 0, "PUMP_2": 0}
            self._segment_number = 0
            self._cumulative_infusion_count = 0
            self._cumulative_infusions_by_pump = {"PUMP_1": 0, "PUMP_2": 0}
            self._segment_exports = []
            self._segment_event_counts = []
            self._data_warning_emitted = False

        self._requeue_for_epoch(pending, old_epoch, saw_sentinel)

        self.paused_time = 0
        self.paused_start_time = None
        self.last_infusion_time = None
        if self.program_flag.is_set():
            self.program_flag.clear()

        # The END that cmd 100 produced belongs to the run being replaced. Let
        # the queue thread handle it, then clear the flag, so the next Stop
        # waits for this run's own END. No new END can arrive before cmd 101.
        self._controller_end_received.wait(timeout=1.0)
        self._controller_end_received.clear()
        self.send_serial_command({"cmd": 101})
        self.program_start_time = time.time()

        self._write_event_log({"type": "SESSION_RESTART", "timestamp": self.program_start_time})
        self._emit("restart", {})
        self.logger.info("Program restarted at %s", self.get_time())

    def get_segment_number(self) -> int:
        """Return the current segment number (0 means no splits have occurred)."""
        return self._segment_number

    def get_total_infusion_count(self) -> int:
        """Return infusions across every segment of this session.

        This is the count the kernel maintained from the serial stream itself,
        which is what ``check_limit_met`` enforces against. It is authoritative:
        unlike a count reconstructed in a browser, it cannot be disturbed by a
        WebSocket reconnect, a segment split, or a tab lifecycle.

        Spans all segments because that is what an export archive contains —
        ``_infusion_count`` alone covers only the current one.
        """
        with self.thread_lock:
            return self._counted_total_locked()

    def _counted_total_locked(self) -> int:
        """Caller holds ``thread_lock``."""
        if self._counted_pumps is None:
            return self._cumulative_infusion_count + self._infusion_count
        return sum(self._cumulative_infusions_by_pump[p] + self._infusions_by_pump[p] for p in self._counted_pumps)

    def set_counted_pumps(self, pumps: Optional[List[str]]) -> None:
        """Choose which pump's infusions count toward an Infusion limit.

        **Args:**
        - `pumps (list | None)`: Pump keys to count, or None to follow the reward pump.

        **Raises:**
        - `ValueError`: If a pump key is not ``PUMP_1`` or ``PUMP_2``.
        """
        if pumps is None:
            counted = None
        else:
            unknown = [p for p in pumps if p not in _PUMP_KEYS]
            if unknown:
                raise ValueError(f"Unknown pump key(s): {unknown}")
            counted = list(dict.fromkeys(pumps))
        with self.thread_lock:
            self._counted_pumps = counted
        self.logger.info("Counted pumps set to: %s", counted if counted is not None else "reward pump")
        self._emit_infusion_count()

    def get_counted_pumps(self) -> List[str]:
        """Return the pumps whose infusions count, following the reward pump by default."""
        with self.thread_lock:
            if self._counted_pumps is None:
                return [self._reward_pump]
            return list(self._counted_pumps)

    def get_infusions_by_pump(self) -> Dict[str, int]:
        """Return infusions per pump across every segment of this session."""
        with self.thread_lock:
            return {p: self._cumulative_infusions_by_pump[p] + self._infusions_by_pump[p] for p in _PUMP_KEYS}

    def _emit_infusion_count(self) -> None:
        with self.thread_lock:
            payload = {
                "counted": self._counted_total_locked(),
                "by_pump": {p: self._cumulative_infusions_by_pump[p] + self._infusions_by_pump[p] for p in _PUMP_KEYS},
                "counted_pumps": list(self._counted_pumps) if self._counted_pumps is not None else [self._reward_pump],
                "follows_reward_pump": self._counted_pumps is None,
            }
        self._emit("infusion_count", payload)

    def get_total_press_count(self) -> int:
        """Return lever presses across every segment of this session.

        Counts every contact — ACTIVE, INACTIVE and TIMEOUT — on either lever,
        which is the figure the UI shows. Authoritative and cross-segment for
        the same reasons as :meth:`get_total_infusion_count`.
        """
        with self.thread_lock:
            return self._cumulative_press_count + self._press_count

    def get_total_trial_count(self) -> int:
        """Return Pavlovian trials started across every segment of this session.

        Zero on operant paradigms, which emit no PAVLOV events.
        """
        with self.thread_lock:
            return self._cumulative_trial_count + self._trial_count

    def get_segment_exports(self) -> List[str]:
        """Return a snapshot of split-segment CSV paths (does not include the final in-memory segment)."""
        with self.thread_lock:
            return list(self._segment_exports)

    def get_segment_event_counts(self) -> List[int]:
        """Return per-split behavior event counts, parallel to get_segment_exports()."""
        with self.thread_lock:
            return list(self._segment_event_counts)

    def get_event_log_path(self) -> str:
        """Return the absolute path to this session's event_log.jsonl."""
        return self._event_log_path

    def flush_event_log(self) -> None:
        """Force-flush the event log file handle. Safe to call when the handle is closed or None."""
        f = self._event_log_file
        if f is not None and not f.closed:
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                self.logger.warning("flush_event_log failed", exc_info=True)

    def monitor_time_limit(self) -> None:
        """Continuously monitor the time limit in a separate thread.

        This method runs in a dedicated thread and checks if program limits are met,
        ensuring timely stopping even when no serial data is received.
        """
        while self.time_check_flag.is_set():
            if not self.program_flag.is_set():  # Program is running
                self.check_limit_met()
            time.sleep(0.1)  # Check every 100ms for responsiveness

    def check_limit_met(self) -> None:
        """Check if program limits have been met and stop if necessary.

        **Description:**
        - Evaluates time and/or infusion limits based on `limit_type`.
        - Automatically stops the program if limits are exceeded.
        """
        current_time = time.time()
        if self.program_start_time is None or self.limit_type is None:
            return  # No limits to check if program hasn't started or type unset

        elapsed_time = current_time - self.program_start_time - self.paused_time
        # Whole-session total: a /split zeroes _infusion_count, which would restart the limit from 0
        infusion_count = self.get_total_infusion_count()
        stop_delay = self.stop_delay or 0
        infusion_met = self.infusion_limit is not None and infusion_count >= self.infusion_limit
        time_met = self.time_limit is not None and elapsed_time >= self.time_limit
        self.logger.debug(f"Checking limits: elapsed_time={elapsed_time:.2f}, time_limit={self.time_limit}, infusion_count={infusion_count}, infusion_limit={self.infusion_limit}")

        if self.limit_type == "Time":
            if time_met:
                self.logger.info("Time limit met, stopping program")
                self.stop_program()
        elif self.limit_type == "Infusion":
            if infusion_met:
                if self.last_infusion_time is None:
                    self.last_infusion_time = current_time
                if self.last_infusion_time and (current_time - self.last_infusion_time >= stop_delay):
                    self.logger.info("Infusion limit met and stop delay elapsed, stopping program")
                    self.stop_program()
        elif self.limit_type == "Both":
            if infusion_met:
                if self.last_infusion_time is None:
                    self.last_infusion_time = current_time
            if (self.last_infusion_time and (current_time - self.last_infusion_time) >= stop_delay) or time_met:
                self.logger.info("Either infusion limit with stop delay or time limit met, stopping program")
                self.stop_program()
        elif self.limit_type == "Trials":
            pass  # Firmware signals ALL_TRIALS_COMPLETE; handled in update_behavioral_events

    def set_data_destination(self, folder: str) -> None:
        """Set the destination folder for data files.

        **Description:**
        - Specifies where data files (e.g., CSV logs) will be saved.

        **Args:**
        - `folder (str)`: The folder path.
        """
        self.data_destination = folder

    def set_filename(self, filename: str) -> None:
        """Set the filename for behavioral data.

        **Description:**
        - Defines the name of the behavioral data file.
        - Automatically appends ".csv" if not provided.

        **Args:**
        - `filename (str)`: The desired filename.
        """
        self.behavior_filename = filename

    def make_destination_folder(self, destination: Optional[str] = None) -> str:
        """Create a destination folder for data files.

        **Description:**
        - Creates a unique folder for storing data based on filename and timestamp.
        - Uses `destination` if given, else the configured destination, else falls
          back to `~/Downloads`. The fallback is resolved locally and never written
          to `data_destination` — `get_data_destination()` stays unset until the
          caller explicitly saves one.

        **Args:**
        - `destination (Optional[str])`: Folder to use instead of the configured
          destination, without persisting it.

        **Returns:**
        - `str`: The path to the created folder.
        """
        resolved_destination = destination or self.data_destination or os.path.expanduser("~/Downloads")
        if not self.behavior_filename:
            self.behavior_filename = f"{self.get_time()}"
        containing_folder = os.path.join(resolved_destination, self.behavior_filename.split('.')[0])
        if os.path.exists(containing_folder):
            data_folder_path = os.path.join(resolved_destination, f"{self.behavior_filename.split('.')[0]}-{time.time_ns()}")
        else:
            data_folder_path = containing_folder
        os.makedirs(data_folder_path, exist_ok=True)
        return data_folder_path

    def set_box_name(self, box_name: str) -> None:
        """Set the name of the box for data organization.

        **Description:**
        - Specifies the name of the box for data storage and organization.

        **Args:**
        - `box_name (str)`: The name of the box.
        """
        self.box_name = box_name

    def get_data_destination(self) -> Optional[str]:
        """Get the current data destination folder.

        **Description:**
        - Retrieves the folder where data files are stored.

        **Returns:**
        - `Optional[str]`: The folder path or None if not set.
        """
        return self.data_destination
    
    def get_filename(self) -> Optional[str]:
        """Get the current behavioral data filename.

        **Description:**
        - Retrieves the filename used for behavioral data logging.

        **Returns:**
        - `Optional[str]`: The filename or None if not set.
        """
        return self.behavior_filename

    def get_behavior_data(self) -> List[Dict[str, Union[str, int]]]:
        """Get a snapshot of the collected behavioral data.

        **Returns:**
        - `List[Dict[str, Union[str, int]]]`: List of event dictionaries.
        """
        with self.thread_lock:
            return list(self.behavior_data)

    def get_frame_data(self) -> List[int]:
        """Get a snapshot of the collected frame data.

        **Returns:**
        - `List[int]`: List of frame timestamps in milliseconds.
        """
        with self.thread_lock:
            return list(self.frame_data)

    def get_frame_timestamps_count(self) -> int:
        with self.thread_lock:
            return len(self.frame_data)

    def get_slm_data(self) -> List[int]:
        """Get a snapshot of the collected SLM timestamp data.

        **Returns:**
        - `List[int]`: List of SLM event timestamps in milliseconds.
        """
        with self.thread_lock:
            return list(self.slm_data)
    
    def get_firmware_information(self) -> Dict:
        """Get the current Arduino configuration.

        **Description:**
        - Retrieves the configuration data received from the microcontroller.
        - Thread-safe access to the configuration dictionary.

        **Returns:**
        - `Dict`: The configuration dictionary.
        """
        with self.thread_lock:  # Fix: F-009 — snapshot under lock
            return dict(self.firmware_information)

    def get_hardware_settings(self) -> List:
        with self.thread_lock:  # Fix: F-009 — snapshot under lock
            return list(self.hardware_settings)
    
    def get_box_name(self) -> Optional[str]:
        """Get the name of the box.

        **Description:**
        - Retrieves the name of the box for data organization.

        **Returns:**
        - `Optional[str]`: The box name or None if not set.
        """
        return self.box_name
    
    def get_start_time(self) -> Optional[float]:
        """Get the program start time.

        **Description:**
        - Returns the timestamp when the experiment began.

        **Returns:**
        - `Optional[float]`: Start time in seconds since epoch, or None if not started.
        """
        return self.program_start_time
    
    def get_end_time(self) -> Optional[float]:
        """Get the program end time.

        **Description:**
        - Returns the timestamp when the experiment ended.

        **Returns:**
        - `Optional[float]`: End time in seconds since epoch, or None if not ended.
        """
        return self.program_end_time
    
    def get_time(self) -> str:
        """Get the current time as a formatted string.

        **Description:**
        - Provides the current local time in a readable format.

        **Returns:**
        - `str`: Time in "YYYY-MM-DD_HH-MM-SS" format.
        """
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        return formatted_time

    @staticmethod
    def prune_logs(days: int = 30) -> int:
        """Remove log directories older than *days* from ~/REACHER/LOG.

        Fix: PY-006 — CLI maintenance command support.

        Returns:
            Number of directories removed.
        """
        import shutil
        log_root = os.path.expanduser("~/REACHER/LOG")
        if not os.path.isdir(log_root):
            return 0
        cutoff = time.time() - (days * 86400)
        removed = 0
        for entry in os.listdir(log_root):
            path = os.path.join(log_root, entry)
            if os.path.isdir(path) and os.stat(path).st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    """
    **Contact:**
    - For inquiries or support, please email: [thejoshbq@proton.me](mailto:thejoshbq@proton.me).
    """