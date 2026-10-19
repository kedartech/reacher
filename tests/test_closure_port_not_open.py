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
