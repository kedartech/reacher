"""A pre-restart infusion already off the kernel queue must not survive restart.

``restart_program`` replaces ``behavior_data`` and zeroes the infusion counters
under ``thread_lock``, then leaves ``program_running`` set. ``handle_queue`` can
already have taken the line off ``self.queue`` and be inside
``update_behavioral_events`` before that append. The append then lands in the
new session and increments the counters the clear just zeroed.

The barrier below only widens that gap. The record path after it is the real
``update_behavioral_events``.
"""

import json
import threading
import time
from unittest.mock import Mock

from reacher.kernel.reacher import REACHER


class _NoSleep:
    """Stand-in for the kernel's ``time`` module whose ``sleep`` returns immediately."""

    def sleep(self, _seconds):
        return None

    def __getattr__(self, name):
        return getattr(time, name)


# Timestamp 2499 is the infusion the flaky HTTP test reports. Here it is queued
# before restart, not emitted by the session that restart starts.
_IN_FLIGHT = {
    "level": "007",
    "device": "PUMP_1",
    "pin": 4,
    "event": "INFUSION",
    "start_timestamp": 2499,
    "end_timestamp": 2499,
}

_ALREADY_RECORDED = {
    "device": "PUMP_1",
    "event": "INFUSION",
    "start_timestamp": 721,
    "end_timestamp": 721,
}


def test_restart_drops_infusion_dequeued_before_clear(monkeypatch):
    """After restart, no pre-restart infusion remains in behavior_data or the counters.

    One infusion is already in ``behavior_data``. A second has been dequeued by
    the queue thread and is held inside ``update_behavioral_events`` until
    ``restart_program`` has cleared state (the send of cmd 101 is the first
    call after that clear). Releasing the thread then runs the real append.
    """
    monkeypatch.setattr("reacher.kernel.reacher.time", _NoSleep())

    kernel = REACHER(session_id="restart-inflight")
    kernel.ser = Mock(is_open=True, baudrate=115200)
    kernel.program_running = True
    kernel.program_flag.clear()
    kernel.serial_flag.clear()
    kernel.behavior_data = [dict(_ALREADY_RECORDED)]
    kernel._infusion_count = 1
    kernel._infusions_by_pump = {"PUMP_1": 1, "PUMP_2": 0}

    entered = threading.Event()
    release = threading.Event()
    appended = threading.Event()
    original = kernel.update_behavioral_events

    def _hold_then_record(event):
        in_flight = event.get("device") == "PUMP_1" and event.get("start_timestamp") == 2499
        if in_flight:
            entered.set()
            assert release.wait(timeout=5), "restart did not reach the post-clear send"
        original(event)
        if in_flight:
            appended.set()

    # handle_data dispatches through this dict, not the method attribute.
    kernel.code_dict["007"] = _hold_then_record

    real_send = kernel.send_serial_command

    def _send(command):
        if command.get("cmd") == 101:
            release.set()
            assert appended.wait(timeout=5), "dequeued infusion was not delivered"
        return real_send(command)

    kernel.send_serial_command = _send
    kernel.queue.put(json.dumps(_IN_FLIGHT))
    kernel.queue_thread.start()
    try:
        assert entered.wait(timeout=5), "queue thread did not dequeue the infusion"
        kernel.restart_program()

        infusions = [row for row in kernel.get_behavior_data() if row.get("event") == "INFUSION"]
        recorded_timestamps = [row.get("start_timestamp") for row in infusions]
        count = kernel.get_total_infusion_count()
        by_pump = kernel.get_infusions_by_pump()
        # The clear did run: the row that was already buffered is gone.
        assert 721 not in recorded_timestamps
        assert not infusions, (
            f"pre-restart infusion still in behavior_data: {infusions!r}; count={count}; by_pump={by_pump}"
        )
        assert count == 0
        assert by_pump == {"PUMP_1": 0, "PUMP_2": 0}
    finally:
        release.set()
        kernel.serial_flag.set()
        kernel.queue.put(None)
        if kernel.queue_thread.is_alive():
            kernel.queue_thread.join(timeout=2)
