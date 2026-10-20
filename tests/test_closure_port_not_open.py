"""Mid-teardown close must not flush a port that is already closed.

Bench logs contained ``Error during closure: Attempting to use a port that is
not open``. The only site that emits that line is ``REACHER.close_serial``:
it observes ``ser.is_open``, pauses, then calls ``flush()``. pyserial's
``flush()`` raises ``PortNotOpenError`` ("Attempting to use a port that is not
open") when the port was closed during the pause, and the handler logs the
exception text.

``SessionManager.destroy_session`` closes that same port on its way out
(``close_serial``, then a force ``ser.close()``). A close that overlaps that
teardown must not touch the closed port or log the error.
"""

import logging
import threading
import time

from serial.serialutil import PortNotOpenError

from reacher.session_manager import SessionManager


class _HardwareSerial:
    """pyserial's closed-port contract for the methods the kernel calls.

    ``close()`` is idempotent. ``flush`` / ``write`` / ``readline`` /
    ``reset_input_buffer`` raise ``PortNotOpenError`` once the port is closed,
    which is what a real ``serial.Serial`` does on the bench. ``SimulatedSerial``
    does not, so it cannot stand in for this failure.
    """

    def __init__(self):
        self._open = False
        self.touched_while_closed: list[str] = []

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def _use(self, op: str) -> None:
        if not self._open:
            self.touched_while_closed.append(op)
            raise PortNotOpenError()

    def flush(self) -> None:
        self._use("flush")

    def write(self, data: bytes) -> int:
        self._use("write")
        return len(data)

    def readline(self) -> bytes:
        self._use("readline")
        return b""

    def reset_input_buffer(self) -> None:
        self._use("reset_input_buffer")

    @property
    def in_waiting(self) -> int:
        # serialposix.in_waiting does not check is_open; a closed fd raises OSError.
        if not self._open:
            raise OSError(9, "Bad file descriptor")
        return 0


def _finished_thread(name: str) -> threading.Thread:
    """A thread ``close_serial`` can join without raising.

    ``Thread.join`` on a thread that was never started raises, and that
    exception is also reported as ``Error during closure``. The port bug is
    the failure under test, so the worker threads are started and already
    finished — the same state they are in after a connected session's reader
    has been told to exit.
    """
    thread = threading.Thread(target=lambda: None, name=name, daemon=True)
    thread.start()
    thread.join()
    return thread


def test_destroy_during_close_does_not_touch_closed_port(monkeypatch, caplog):
    """destroy_session closing the port during close_serial's pause must be quiet.

    Disconnect, firmware upload, and stop_program all call ``close_serial``.
    Delete calls ``destroy_session``, which closes the port again. The pause
    after the ``is_open`` check is where the second close lands on the bench.
    """
    sm = SessionManager()
    sid = sm.create_session("SIMULATOR", "fr")
    instance = sm.get_instance(sid)
    port = _HardwareSerial()
    port.open()
    instance.ser = port
    instance.serial_thread = _finished_thread("read_serial")
    instance.queue_thread = _finished_thread("handle_queue")
    instance.time_check_thread = _finished_thread("monitor_time_limit")

    in_pause = threading.Event()
    destroyed = threading.Event()
    racer_error: list[BaseException] = []
    first_pause = True
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        nonlocal first_pause
        # close_serial's pre-flush pause (reacher.py). The first caller is the
        # close under test; destroy_session's own close_serial hits the same
        # pause and must not wait on itself.
        if seconds == 0.5 and first_pause:
            first_pause = False
            in_pause.set()
            assert destroyed.wait(timeout=5), "destroy_session did not finish during the close pause"
            return
        if seconds == 0.5:
            return
        real_sleep(seconds)

    def destroy() -> None:
        try:
            assert in_pause.wait(timeout=5), "close_serial never reached its pre-flush pause"
            sm.destroy_session(sid)
        except BaseException as exc:  # pragma: no cover - harness failure
            racer_error.append(exc)
        finally:
            destroyed.set()

    monkeypatch.setattr(time, "sleep", sleep)
    racer = threading.Thread(target=destroy, name="destroy-session", daemon=True)
    racer.start()
    try:
        with caplog.at_level(logging.ERROR):
            instance.close_serial()
        racer.join(timeout=5)
    finally:
        if sm._sessions.get(sid) is not None:
            sm.destroy_session(sid)
        instance.shutdown()

    assert racer_error == []
    assert not racer.is_alive()
    closure = [
        record.getMessage()
        for record in caplog.records
        if "Error during closure" in record.getMessage()
        or "Attempting to use a port that is not open" in record.getMessage()
    ]
    assert closure == [], (
        "closing a session whose port teardown already closed must not touch the "
        f"closed port or log the closed-port error; got {closure}; "
        f"ops on closed port: {port.touched_while_closed}"
    )
    assert port.touched_while_closed == []


def _wired_session():
    """A session whose port follows pyserial's closed-port contract."""
    sm = SessionManager()
    sid = sm.create_session("SIMULATOR", "fr")
    instance = sm.get_instance(sid)
    port = _HardwareSerial()
    port.open()
    instance.ser = port
    instance.serial_thread = _finished_thread("read_serial")
    instance.queue_thread = _finished_thread("handle_queue")
    instance.time_check_thread = _finished_thread("monitor_time_limit")
    return sm, sid, instance, port


def _release_session(sm, sid, instance) -> None:
    if sm._sessions.get(sid) is not None:
        sm.destroy_session(sid)
    instance.shutdown()


def test_close_serial_completes_while_thread_lock_held():
    """close_serial must finish even if thread_lock is held or contended.

    The close path does not take ``thread_lock``. A caller that already owns
    it, and a close that runs while another thread owns it, both have to
    return. Blocking on that lock deadlocks the first and stalls the second
    for as long as ``send_serial_command`` holds it.
    """
    sm, sid, instance, port = _wired_session()
    try:
        release = threading.Event()
        held = threading.Event()

        def hold() -> None:
            with instance.thread_lock:
                held.set()
                assert release.wait(timeout=5)

        holder = threading.Thread(target=hold, name="hold-thread-lock", daemon=True)
        holder.start()
        assert held.wait(timeout=5), "holder did not acquire thread_lock"

        done = threading.Event()

        def close_while_contended() -> None:
            instance.close_serial()
            done.set()

        closer = threading.Thread(target=close_while_contended, name="close-contended", daemon=True)
        closer.start()
        # The pre-flush pause is 0.5s. Anything past that is waiting on thread_lock.
        assert done.wait(timeout=2), "close_serial blocked while another thread held thread_lock"
        closer.join(timeout=1)
        release.set()
        holder.join(timeout=2)

        assert not port.is_open
        assert port.touched_while_closed == []
        assert not holder.is_alive()

        port.open()

        def close_while_holding() -> None:
            with instance.thread_lock:
                instance.close_serial()

        same = threading.Thread(target=close_while_holding, name="close-holding-lock", daemon=True)
        same.start()
        same.join(timeout=2)
        assert not same.is_alive(), "close_serial deadlocked on a thread_lock the caller already held"
        assert not port.is_open
        assert port.touched_while_closed == []
    finally:
        _release_session(sm, sid, instance)
