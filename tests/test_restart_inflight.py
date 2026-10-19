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
import queue
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


def _behavior_rows(kernel):
    return [row for row in _event_rows(kernel) if row.get("type") == "behavior"]


def _event_rows(kernel):
    kernel.flush_event_log()
    rows = []
    with open(kernel.get_event_log_path()) as handle:
        for line in handle:
            if not line.strip():
                continue
            # A poll can observe a line the queue thread has not finished writing.
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _await_behavior_timestamps(kernel, expected, timeout=5):
    deadline = time.monotonic() + timeout
    last = []
    while time.monotonic() < deadline:
        last = [row.get("start_timestamp") for row in _behavior_rows(kernel)]
        if all(timestamp in last for timestamp in expected):
            return last
        time.sleep(0.01)
    return last


def _start_queue_thread_held(kernel):
    """Start the queue thread and hold its first ``get`` until the event is set.

    Restart drains with ``get_nowait``, so a line put after this returns stays
    queued. The thread is alive, which is the path that requeues instead of
    logging on the restart thread.
    """
    release = threading.Event()
    entered = threading.Event()
    real_get = kernel.queue.get

    def _get(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            release.wait(timeout=5)
        return real_get(*args, **kwargs)

    kernel.queue.get = _get
    kernel.queue_thread.start()
    assert entered.wait(timeout=5), "queue thread did not reach get"
    return release


def _stop_queue_thread(kernel, release):
    release.set()
    kernel.serial_flag.set()
    try:
        kernel.queue.put_nowait(None)
    except queue.Full:
        pass
    if kernel.queue_thread.is_alive():
        kernel.queue_thread.join(timeout=2)


# A line still on the queue when restart drains it. 2500 is the second
# pre-restart infusion from the original race (2499 is the one already dequeued).
_QUEUED_BEFORE_RESTART = {
    "level": "007",
    "device": "PUMP_1",
    "pin": 4,
    "event": "INFUSION",
    "start_timestamp": 2500,
    "end_timestamp": 2500,
}

# Occupies the only free slot after the drain, so the requeue cannot put 2500 back.
_RESTART_FILLER = {
    "level": "007",
    "device": "PUMP_1",
    "pin": 4,
    "event": "INFUSION",
    "start_timestamp": 3001,
    "end_timestamp": 3001,
}


def test_restart_full_requeue_keeps_pre_restart_line_in_event_log(monkeypatch):
    """A pre-restart line that cannot be requeued is still written to event_log.jsonl.

    The queue is drained, then filled to maxsize before the requeue. ``put_nowait``
    raises ``queue.Full`` and the line is only logged at error. It must still reach
    the event log, and it must not re-enter the cleared session.
    """
    monkeypatch.setattr("reacher.kernel.reacher.time", _NoSleep())

    kernel = REACHER(session_id="restart-requeue-full")
    kernel.ser = Mock(is_open=True, baudrate=115200)
    kernel.program_running = True
    kernel.program_flag.clear()
    kernel.serial_flag.clear()
    kernel.queue = queue.Queue(maxsize=1)

    release = _start_queue_thread_held(kernel)
    try:
        kernel.queue.put_nowait(json.dumps(_QUEUED_BEFORE_RESTART))
        real_take = kernel._take_queued_lines

        def _fill_after_drain():
            pending, saw_sentinel = real_take()
            assert pending, "pre-restart line was not waiting on the queue"
            kernel.queue.put_nowait(json.dumps(_RESTART_FILLER))
            return pending, saw_sentinel

        kernel._take_queued_lines = _fill_after_drain
        kernel.restart_program()

        timestamps = [row.get("start_timestamp") for row in _behavior_rows(kernel)]
        assert 2500 in timestamps, f"pre-restart line missing from event log: {timestamps}"
        buffered = [row.get("start_timestamp") for row in kernel.get_behavior_data()]
        assert 2500 not in buffered
        assert kernel.get_total_infusion_count() == 0
        assert kernel.get_infusions_by_pump() == {"PUMP_1": 0, "PUMP_2": 0}
    finally:
        _stop_queue_thread(kernel, release)


def _infusion(timestamp):
    return {
        "level": "007",
        "device": "PUMP_1",
        "pin": 4,
        "event": "INFUSION",
        "start_timestamp": timestamp,
        "end_timestamp": timestamp,
    }


def test_restart_requeue_logs_pre_restart_lines_before_new_session(monkeypatch):
    """Pre-restart lines stay ahead of a line that arrives between drain and requeue.

    Requeue appends, so that new-session line is logged first and event_log.jsonl
    is out of order around the restart. The drained lines have to be recorded
    first. They still must not re-enter the cleared session.
    """
    monkeypatch.setattr("reacher.kernel.reacher.time", _NoSleep())

    kernel = REACHER(session_id="restart-requeue-order")
    kernel.ser = Mock(is_open=True, baudrate=115200)
    kernel.program_running = True
    kernel.program_flag.clear()
    kernel.serial_flag.clear()

    release = _start_queue_thread_held(kernel)
    try:
        kernel.queue.put_nowait(json.dumps(_infusion(2500)))
        kernel.queue.put_nowait(json.dumps(_infusion(2501)))
        real_take = kernel._take_queued_lines

        def _insert_new_session_line():
            pending, saw_sentinel = real_take()
            queued = [json.loads(line)["start_timestamp"] for _epoch, line in pending]
            assert queued == [2500, 2501], queued
            kernel.queue.put_nowait(json.dumps(_infusion(3001)))
            return pending, saw_sentinel

        kernel._take_queued_lines = _insert_new_session_line
        kernel.restart_program()
        release.set()

        timestamps = _await_behavior_timestamps(kernel, (2500, 2501, 3001))
        assert timestamps == [2500, 2501, 3001], timestamps
        buffered = [row.get("start_timestamp") for row in kernel.get_behavior_data()]
        assert 2500 not in buffered
        assert 2501 not in buffered
        assert 3001 in buffered
        assert kernel.get_total_infusion_count() == 1
        assert kernel.get_infusions_by_pump() == {"PUMP_1": 1, "PUMP_2": 0}
    finally:
        _stop_queue_thread(kernel, release)
