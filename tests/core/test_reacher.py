import logging
import os
import queue
import json
import threading
import time

import pytest
import serial
from unittest.mock import Mock, patch

from reacher.kernel.reacher import REACHER
from reacher.kernel.commands import CommandCode


@pytest.fixture
def mock_serial():
    """Fixture providing a mocked serial connection with predefined COM ports."""
    with patch("serial.Serial") as mock_serial_class, patch("serial.tools.list_ports.comports") as mock_comports:
        mock_serial_instance = Mock()
        mock_serial_instance.baudrate = 115200
        mock_serial_class.return_value = mock_serial_instance
        mock_comports.return_value = [Mock(device="COM1", vid=1, pid=1)]
        yield mock_serial_instance


@pytest.fixture
def reacher(mock_serial):
    """Fixture providing a REACHER instance with mocked threading and logging."""
    with (
        patch("threading.Thread"),
        patch("os.makedirs"),
        patch("logging.basicConfig"),
        patch.object(logging.FileHandler, "_open", return_value=Mock()),
    ):
        reacher_instance = REACHER()
        return reacher_instance


def test_reacher_init(reacher, mock_serial):
    """Test that REACHER initializes with correct serial, queue, and thread attributes."""
    assert isinstance(reacher.ser, Mock)
    assert reacher.ser.baudrate == 115200
    assert isinstance(reacher.queue, queue.Queue)
    assert hasattr(reacher, "serial_thread")
    assert hasattr(reacher, "queue_thread")
    assert hasattr(reacher, "time_check_thread")
    assert hasattr(reacher, "thread_lock")
    assert callable(reacher.thread_lock.acquire)
    assert reacher.serial_flag.is_set()
    assert reacher.program_flag.is_set()
    assert reacher.time_check_flag.is_set()
    assert reacher.behavior_data == []
    assert reacher.frame_data == []
    assert reacher.program_start_time is None
    assert reacher.firmware_information["sketch"] is None


def test_init_with_session_id(mock_serial):
    """Test REACHER initializes with session_id and event_callback."""
    with (
        patch("threading.Thread"),
        patch("os.makedirs"),
        patch("logging.basicConfig"),
        patch.object(logging.FileHandler, "_open", return_value=Mock()),
    ):
        cb = Mock()
        r = REACHER(session_id="abc123", event_callback=cb)
        assert r.session_id == "abc123"
        assert r.event_callback is cb


def test_resilient_wrapper_restarts_after_exception(reacher, mocker):
    """Fix 7.1: unhandled exception in a thread body is caught, a warning is
    emitted, and the target is retried."""
    mocker.patch("time.sleep")  # skip the 1s back-off
    cb = mocker.Mock()
    reacher.event_callback = cb
    reacher.session_id = "sid12345"

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        # Second call returns cleanly, simulating a recovered target.

    wrapped = reacher._resilient(flaky, "flaky")
    wrapped()

    assert calls["n"] == 2  # raised once, restarted once, clean return
    # Exactly one warning event emitted with reason=thread_crash.
    warnings = [
        c.args for c in cb.call_args_list
        if c.args[1] == "warning" and c.args[2].get("reason") == "thread_crash"
    ]
    assert len(warnings) == 1
    assert warnings[0][2]["thread"] == "flaky"


def test_resilient_wrapper_gives_up_after_max_restarts(reacher, mocker):
    """Fix 7.1: a target that fails forever is abandoned after the budget."""
    mocker.patch("time.sleep")
    cb = mocker.Mock()
    reacher.event_callback = cb
    reacher.session_id = "sid12345"

    def always_raises():
        raise RuntimeError("persistent")

    wrapped = reacher._resilient(always_raises, "always_raises")
    wrapped()  # Must return; must not hang.

    # The budget is 10 restarts; we should see exactly 10 warning emissions.
    warnings = [c for c in cb.call_args_list if c.args[1] == "warning"]
    assert len(warnings) == 10


def test_resilient_wrapper_tolerates_broken_callback(reacher, mocker):
    """Fix 7.1: a failing event_callback must not crash the shim itself."""
    mocker.patch("time.sleep")
    reacher.event_callback = Mock(side_effect=RuntimeError("cb broken"))
    reacher.session_id = "sid12345"

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    wrapped = reacher._resilient(flaky, "flaky")
    wrapped()  # Should not raise.
    assert calls["n"] == 2


def _simulate_queue_overflow(reacher_instance, now: float) -> None:
    """Drive the overflow branch once at the given monotonic time."""
    reacher_instance._queue_overflow_count += 1
    if now - reacher_instance._last_queue_overflow_emit > 1.0:
        reacher_instance._emit("warning", {
            "reason": "queue_overflow",
            "count": reacher_instance._queue_overflow_count,
        })
        reacher_instance._last_queue_overflow_emit = now


def test_queue_overflow_emits_first_warning_immediately(reacher, mocker):
    """Fix 2.6: the first overflow always emits a warning."""
    cb = mocker.Mock()
    reacher.event_callback = cb
    reacher.session_id = "sid12345"

    _simulate_queue_overflow(reacher, now=0.0)

    warnings = [c for c in cb.call_args_list
                if c.args[1] == "warning" and c.args[2].get("reason") == "queue_overflow"]
    assert len(warnings) == 1
    assert warnings[0].args[2]["count"] == 1


def test_queue_overflow_throttles_repeat_warnings(reacher, mocker):
    """Fix 2.6: overflows inside a 1s window do not double-emit."""
    cb = mocker.Mock()
    reacher.event_callback = cb
    reacher.session_id = "sid12345"

    _simulate_queue_overflow(reacher, now=0.0)
    _simulate_queue_overflow(reacher, now=0.3)
    _simulate_queue_overflow(reacher, now=0.9)

    warnings = [c for c in cb.call_args_list
                if c.args[1] == "warning" and c.args[2].get("reason") == "queue_overflow"]
    assert len(warnings) == 1
    # Counter advances even when warnings are suppressed.
    assert reacher._queue_overflow_count == 3


def test_queue_overflow_emits_again_after_throttle_window(reacher, mocker):
    """Fix 2.6: once the throttle window elapses, a new overflow re-emits."""
    cb = mocker.Mock()
    reacher.event_callback = cb
    reacher.session_id = "sid12345"

    _simulate_queue_overflow(reacher, now=0.0)
    _simulate_queue_overflow(reacher, now=1.1)

    warnings = [c for c in cb.call_args_list
                if c.args[1] == "warning" and c.args[2].get("reason") == "queue_overflow"]
    assert len(warnings) == 2


def test_make_thread_returns_daemon_thread(reacher):
    """Fix 7.1: helper builds a daemon thread with the right name."""
    import threading as _threading
    t = reacher._make_thread(lambda: None, "myname")
    assert isinstance(t, _threading.Thread)
    assert t.daemon is True
    assert t.name == "myname"


def test_reset(reacher, mocker):
    """Test that reset clears data, stops the program, and resets flags."""
    mocker.patch.object(reacher, "stop_program")
    mocker.patch.object(reacher, "clear_queue")
    mocker.patch.object(reacher, "close_serial")
    mocker.patch("threading.Thread")

    reacher.program_flag.clear()
    reacher.reset()

    assert reacher.behavior_data == []
    assert reacher.frame_data == []
    assert reacher.program_start_time is None
    assert reacher.program_flag.is_set()
    assert not reacher.serial_flag.is_set()
    reacher.stop_program.assert_called_once()
    reacher.clear_queue.assert_called_once()
    reacher.close_serial.assert_called_once()


def test_get_COM_ports(reacher):
    """Test that get_COM_ports returns available ports including SIMULATOR."""
    ports = reacher.get_COM_ports()
    assert ports == ["COM1", "SIMULATOR"]

    with patch("serial.tools.list_ports.comports", return_value=[]):
        ports = reacher.get_COM_ports()
        assert ports == ["SIMULATOR"]


def test_set_COM_port(reacher):
    """Test that set_COM_port updates the serial port only if valid."""
    reacher.set_COM_port("COM1")
    assert reacher.ser.port == "COM1"

    with pytest.raises(ValueError, match="not available"):
        reacher.set_COM_port("COM2")


def test_open_serial(reacher, mock_serial):
    """Test that open_serial opens the port."""
    reacher.serial_flag.clear()
    reacher.open_serial()
    mock_serial.open.assert_called_once()
    mock_serial.reset_input_buffer.assert_called_once()


def test_close_serial(reacher, mock_serial):
    """Test that close_serial closes the port."""
    reacher.ser.is_open = True
    reacher.close_serial()
    mock_serial.close.assert_called_once()
    assert reacher.serial_flag.is_set()


def test_send_serial_command(reacher, mock_serial):
    """Test that send_serial_command sends JSON when port is open, raises error when closed."""
    reacher.ser.is_open = True
    reacher.send_serial_command({"cmd": 101})
    expected = json.dumps({"cmd": 101}).encode() + b"\n"
    mock_serial.write.assert_called_with(expected)
    mock_serial.flush.assert_called_once()

    reacher.ser.is_open = False
    with pytest.raises(Exception, match="Serial port is not open"):
        reacher.send_serial_command({"cmd": 100})


def test_send_command(reacher, mock_serial):
    """Test that send_command uses the command registry to build payloads."""
    reacher.ser.is_open = True
    reacher.send_command(371, 8000)
    expected = json.dumps({"cmd": 371, "frequency": 8000}).encode() + b"\n"
    mock_serial.write.assert_called_with(expected)


def test_send_command_no_value(reacher, mock_serial):
    """Test send_command without a value."""
    reacher.ser.is_open = True
    reacher.send_command(101)
    expected = json.dumps({"cmd": 101}).encode() + b"\n"
    mock_serial.write.assert_called_with(expected)


def test_handle_data_json_config(reacher, mocker):
    """Test that handle_data processes JSON firmware configuration."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    config = {"level": "000", "device": "CONTROLLER", "sketch": "fr", "version": "v2.0.0"}
    reacher.handle_data(json.dumps(config))
    assert config.items() <= reacher.firmware_information.items()


def test_handle_data_hardware_settings(reacher, mocker):
    """Test that handle_data appends hardware settings for non-controller devices."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    hw = {"level": "000", "device": "CUE", "frequency": 2900}
    reacher.handle_data(json.dumps(hw))
    assert hw in reacher.hardware_settings


def test_handle_data_records_session_change_with_host_timestamp(reacher, mocker):
    """reacher#12: level-000/001 frames are appended to _session_changes with a host-side received_at."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    config = {"level": "000", "device": "CUE", "frequency": 2900}
    with patch("time.time", return_value=1700000000.0):
        reacher.handle_data(json.dumps(config))
    assert len(reacher._session_changes) == 1
    assert reacher._session_changes[0] == {**config, "received_at": 1700000000.0}


def test_handle_data_ignores_non_config_levels_for_session_changes(reacher, mocker):
    """Only level 000/001 are change-tracked — behavioral (007) events are not config."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    reacher.program_running = True
    reacher.program_flag.clear()
    event = {
        "level": "007",
        "device": "PUMP",
        "event": "INFUSION",
        "start_timestamp": 12345,
        "end_timestamp": 12346,
    }
    reacher.handle_data(json.dumps(event))
    assert reacher._session_changes == []


def test_build_session_summary_distills_initial_changes_final_state(reacher, mocker):
    """reacher#12: summary separates first-seen config (initial), the full ordered trail
    (changes), and the latest-known state (final_state, reused from firmware_information /
    hardware_settings rather than replayed).

    Real time.time() is left unpatched here: handle_data's own logger.info() calls also
    consume time.time() internally for log formatting, so pinning a short side_effect list
    to just the four session-change appends is not reliable — instead each change's own
    recorded `received_at` is used as the expected value.
    """
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")

    identify = {"level": "000", "device": "CONTROLLER", "sketch": "fr", "version": "v2.0.0"}
    arm = {"level": "001", "device": "PUMP", "pin": 7, "desc": "ARMED"}
    cue_initial = {"level": "000", "device": "CUE", "frequency": 2900}
    cue_updated = {"level": "000", "device": "CUE", "frequency": 8000}

    for payload in (identify, arm, cue_initial, cue_updated):
        reacher.handle_data(json.dumps(payload))

    summary = reacher.get_session_summary()
    changes = summary["changes"]
    assert len(changes) == 4
    assert all(isinstance(c["received_at"], float) for c in changes)
    for change, expected in zip(changes, (identify, arm, cue_initial, cue_updated)):
        assert expected.items() <= change.items()

    # Initial configuration: first-seen CONTROLLER identify + first-seen CUE config only —
    # the later CUE update must not overwrite it.
    assert summary["initial_configuration"]["firmware"] == changes[0]
    assert summary["initial_configuration"]["devices"] == {"CUE": changes[2]}

    # Final state: latest-known value per device, reused from hardware_settings/firmware_information.
    assert summary["final_state"]["devices"]["CUE"] == cue_updated
    assert identify.items() <= summary["final_state"]["firmware"].items()


def test_write_session_summary_writes_json_to_log_dir(reacher, mocker):
    """_write_session_summary() persists session_summary.json into reacher_log_path."""
    # Isolate the open() call under test from _write_controller_log's own open() —
    # mock_open() shares one file handle across every open() target, so a real
    # controller-log write would otherwise pollute the write() calls asserted below.
    mocker.patch.object(reacher, "_write_controller_log")
    hw = {"level": "000", "device": "CUE", "frequency": 2900}
    reacher.handle_data(json.dumps(hw))

    m_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    reacher._write_session_summary()

    expected_path = os.path.join(reacher.reacher_log_path, "session_summary.json")
    m_open.assert_called_once_with(expected_path, "w")
    written = "".join(call.args[0] for call in m_open().write.call_args_list)
    payload = json.loads(written)
    assert payload["final_state"]["devices"]["CUE"] == hw


def test_stop_program_writes_session_summary(reacher, mocker, mock_serial):
    """stop_program() persists the session summary alongside the CSV auto-export."""
    reacher.ser.is_open = True
    reacher.program_running = True
    mocker.patch.object(reacher, "_join_queue_with_timeout")
    mocker.patch.object(reacher, "close_serial")
    mocker.patch.object(reacher, "_write_event_log")
    mocker.patch.object(reacher, "_close_event_log")
    mocker.patch.object(reacher, "_auto_export")
    mocker.patch.object(reacher, "_write_session_summary")
    reacher.stop_program()
    reacher._write_session_summary.assert_called_once()


def test_reset_preserves_session_changes_for_summary(reacher, mocker):
    """reacher#12 regression: reset() must not clear _session_changes.

    session_summary.json is documented to span the full log-directory lifetime,
    same as controller_log.json (which reset() reuses, not rotates). A run ->
    stop -> reset -> run -> stop sequence must produce a summary covering BOTH
    runs, not just the post-reset one — otherwise the summary silently disagrees
    with the append-only controller_log.json in the same directory.
    """
    mocker.patch.object(reacher, "stop_program")
    mocker.patch.object(reacher, "clear_queue")
    mocker.patch.object(reacher, "close_serial")
    mocker.patch.object(reacher, "_write_controller_log")
    mocker.patch("threading.Thread")

    pre_reset = {"level": "000", "device": "CONTROLLER", "sketch": "fr", "version": "v2.0.0"}
    reacher.handle_data(json.dumps(pre_reset))
    assert len(reacher._session_changes) == 1

    reacher.reset()

    post_reset = {"level": "001", "device": "PUMP", "pin": 7, "desc": "ARMED"}
    reacher.handle_data(json.dumps(post_reset))

    summary = reacher.get_session_summary()
    changes = summary["changes"]
    assert len(changes) == 2
    assert pre_reset.items() <= changes[0].items()
    assert post_reset.items() <= changes[1].items()


def test_handle_behavioral_events(reacher, mocker):
    """Test that handle_data processes behavioral event data into behavior_data."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    reacher.program_running = True
    reacher.program_flag.clear()
    event = {
        "level": "007",
        "device": "PUMP",
        "event": "INFUSION",
        "start_timestamp": 12345,
        "end_timestamp": 12346,
    }
    reacher.handle_data(json.dumps(event))
    assert len(reacher.behavior_data) == 1
    assert reacher.behavior_data[0]["device"] == "PUMP"
    assert reacher.behavior_data[0]["event"] == "INFUSION"


def test_update_behavioral_events_persists_pavlov(reacher, mocker):
    """PAVLOV rows must land in behavior_data so they reach behavior_events.csv."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    reacher.program_running = True
    reacher.program_flag.clear()
    event = {
        "level": "007",
        "device": "PAVLOV",
        "event": "TRIAL_START",
        "timestamp": 42000,
        "trial_type": "rewarded",
    }
    reacher.update_behavioral_events(event)
    assert len(reacher.behavior_data) == 1
    row = reacher.behavior_data[0]
    assert row["device"] == "PAVLOV"
    assert row["event"] == "TRIAL_START"
    assert row["start_timestamp"] == 42000
    assert row["end_timestamp"] == 42000
    assert row["trial_type"] == "rewarded"


def test_handle_frame_events(reacher, mocker):
    """Test that handle_data processes frame event data into frame_data during active session."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("os.fsync")
    reacher.program_flag.clear()
    event = {"level": "008", "timestamp": 54321}
    reacher.handle_data(json.dumps(event))
    assert reacher.frame_data == [54321]


def test_start_program(reacher, mock_serial):
    """Test that start_program sets flags, sends command, and records start time."""
    reacher.ser.is_open = True
    with patch("time.time", return_value=1000.0):
        reacher.start_program()
    assert not reacher.program_flag.is_set()
    expected = json.dumps({"cmd": 101}).encode() + b"\n"
    mock_serial.write.assert_called_with(expected)
    assert reacher.program_start_time == 1000.0


def test_stop_program(reacher, mocker, mock_serial):
    """Test that stop_program resets flags, sends command, and records end time."""
    reacher.ser.is_open = True
    reacher.program_running = True
    mocker.patch.object(reacher, "_join_queue_with_timeout")
    mocker.patch.object(reacher, "close_serial")
    mocker.patch.object(reacher, "_write_event_log")
    mocker.patch.object(reacher, "_close_event_log")
    mocker.patch.object(reacher, "_auto_export")
    with patch("time.time", return_value=2000.0):
        reacher.stop_program()
    assert reacher.program_flag.is_set()
    expected = json.dumps({"cmd": 100}).encode() + b"\n"
    mock_serial.write.assert_called_with(expected)
    reacher._join_queue_with_timeout.assert_called_once()
    reacher.close_serial.assert_called_once()
    assert reacher.program_end_time == 2000.0


def test_check_limit_met_time(reacher, mocker):
    """Test that check_limit_met stops the program when time limit is exceeded."""
    mocker.patch.object(reacher, "stop_program")
    reacher.limit_type = "Time"
    reacher.time_limit = 10
    reacher.program_start_time = 1000.0
    with patch("time.time", return_value=1015.0):
        reacher.check_limit_met()
    reacher.stop_program.assert_called_once()


def test_check_limit_met_infusion(reacher, mocker):
    """Test that check_limit_met stops the program after infusion limit with delay."""
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch.object(reacher, "stop_program")

    reacher.limit_type = "Infusion"
    reacher.infusion_limit = 2
    reacher.stop_delay = 5
    reacher.program_flag.clear()
    reacher.program_start_time = 1000.0

    # Directly populate behavior_data with pump events
    reacher.behavior_data.append(
        {"device": "PUMP", "event": "INFUSION", "start_timestamp": 1005000, "end_timestamp": 1006000}
    )
    reacher.behavior_data.append(
        {"device": "PUMP", "event": "INFUSION", "start_timestamp": 1010000, "end_timestamp": 1011000}
    )
    reacher._infusion_count = 2

    with patch("time.time", return_value=1010.0):
        reacher.check_limit_met()
        assert reacher.last_infusion_time == 1010.0

    with patch("time.time", return_value=1016.0):
        reacher.check_limit_met()
    reacher.stop_program.assert_called_once()


def test_set_data_destination_and_filename(reacher):
    """Test that set_data_destination and set_filename update paths correctly."""
    reacher.set_data_destination("/path/to/data")
    assert reacher.data_destination == "/path/to/data"

    reacher.set_filename("test")
    assert reacher.behavior_filename == "test"

    reacher.set_filename("test.csv")
    assert reacher.behavior_filename == "test.csv"


def test_make_destination_folder(reacher, mocker):
    """Test that make_destination_folder creates and returns the correct folder path."""
    mocker.patch("os.makedirs")
    mocker.patch("os.path.exists", return_value=False)
    with patch.object(reacher, "get_time", return_value="2023-01-01_12-00-00"):
        reacher.data_destination = "/data"
        reacher.behavior_filename = "experiment.csv"
        folder = reacher.make_destination_folder()
        assert folder == "/data/experiment"


def test_make_destination_folder_explicit_override_not_persisted(reacher, mocker):
    """An explicit destination argument wins over the configured one and is never persisted."""
    mocker.patch("os.makedirs")
    mocker.patch("os.path.exists", return_value=False)
    with patch.object(reacher, "get_time", return_value="2023-01-01_12-00-00"):
        reacher.data_destination = None
        reacher.behavior_filename = "experiment"
        folder = reacher.make_destination_folder("/explicit/path")
        assert folder == "/explicit/path/experiment"
        assert reacher.data_destination is None


def test_make_destination_folder_partial_none(reacher, mocker):
    """make_destination_folder must not raise when only data_destination is None,
    and must not persist the ~/Downloads fallback into data_destination."""
    mocker.patch("os.makedirs")
    mocker.patch("os.path.exists", return_value=False)
    with patch.object(reacher, "get_time", return_value="2023-01-01_12-00-00"):
        reacher.data_destination = None
        reacher.behavior_filename = "experiment"
        folder = reacher.make_destination_folder()
        import os
        assert folder.startswith(os.path.expanduser("~/Downloads"))
        assert reacher.data_destination is None


def test_event_callback_fires(mock_serial):
    """Test that _emit invokes the event_callback."""
    with (
        patch("threading.Thread"),
        patch("os.makedirs"),
        patch("logging.basicConfig"),
        patch.object(logging.FileHandler, "_open", return_value=Mock()),
    ):
        cb = Mock()
        r = REACHER(session_id="sess1", event_callback=cb)
        r._emit("event", {"foo": "bar"})
        cb.assert_called_once_with("sess1", "event", {"foo": "bar"})


def test_emit_failure_counter_increments_on_callback_error(reacher):
    """Fix 7.4: a raising callback must bump emit_failure_count, not propagate."""
    reacher.event_callback = Mock(side_effect=RuntimeError("broken cb"))
    reacher.session_id = "sid12345"

    assert reacher.emit_failure_count == 0
    reacher._emit("event", {"x": 1})
    reacher._emit("event", {"x": 2})
    reacher._emit("event", {"x": 3})
    assert reacher.emit_failure_count == 3


def test_emit_failure_counter_stays_zero_on_success(reacher):
    """A working callback must not touch the failure counter."""
    reacher.event_callback = Mock()
    reacher.session_id = "sid12345"
    reacher._emit("event", {"x": 1})
    reacher._emit("event", {"x": 2})
    assert reacher.emit_failure_count == 0


def test_controller_log_uses_persistent_handle(reacher, mocker):
    """Fix 4.9: _write_controller_log opens the file once across N writes."""
    m_open = mocker.mock_open()
    m_open.return_value.closed = False
    mocker.patch("builtins.open", m_open)
    mocker.patch("os.fsync")

    for i in range(5):
        reacher._write_controller_log({"level": "007", "i": i})

    # Exactly one open() for controller_log across the 5 writes.
    controller_opens = [
        c for c in m_open.call_args_list if c.args and c.args[0] == reacher.controller_log
    ]
    assert len(controller_opens) == 1


def test_controller_log_fsync_cadence(reacher, mocker):
    """Fix 4.9: fsync fires every _CONTROLLER_LOG_FSYNC_INTERVAL writes."""
    m_open = mocker.mock_open()
    m_open.return_value.closed = False
    mocker.patch("builtins.open", m_open)
    fsync = mocker.patch("os.fsync")
    reacher._CONTROLLER_LOG_FSYNC_INTERVAL = 3

    for i in range(7):
        reacher._write_controller_log({"i": i})

    # 7 writes, interval 3 → fsync at write 3 and 6 → exactly 2 calls.
    assert fsync.call_count == 2


def test_controller_log_close_flushes_and_closes(reacher, mocker):
    """_close_controller_log must flush, fsync, and close the handle."""
    handle = mocker.MagicMock()
    handle.closed = False
    reacher._controller_log_file = handle
    fsync = mocker.patch("os.fsync")

    reacher._close_controller_log()

    handle.flush.assert_called_once()
    fsync.assert_called_once_with(handle.fileno.return_value)
    handle.close.assert_called_once()


def test_get_detected_paradigm(reacher):
    """Test paradigm detection from firmware info."""
    assert reacher.get_detected_paradigm() is None
    reacher.firmware_information = {"schedule": "FIXED_RATIO", "device": "CONTROLLER"}
    assert reacher.get_detected_paradigm() == "fr"


@pytest.mark.parametrize("schedule,sketch,expected", [
    ("FIXED_RATIO", "fr_lite.ino", "fr_lite"),
    ("PROGRESSIVE_RATIO", "pr_lite.ino", "pr_lite"),
    ("VARIABLE_INTERVAL", "vi_lite.ino", "vi_lite"),
    ("OMISSION", "omission_lite.ino", "omission_lite"),
])
def test_get_detected_paradigm_lite_sketch(reacher, schedule, sketch, expected):
    """A lite build reports the same schedule as its base — only the sketch name distinguishes it."""
    reacher.firmware_information = {
        "schedule": schedule, "device": "CONTROLLER", "sketch": sketch,
    }
    assert reacher.get_detected_paradigm() == expected


def test_get_detected_paradigm_full_sketch_unaffected(reacher):
    """A full (non-lite) sketch name still resolves to the bare paradigm."""
    reacher.firmware_information = {
        "schedule": "FIXED_RATIO", "device": "CONTROLLER", "sketch": "fr.ino",
    }
    assert reacher.get_detected_paradigm() == "fr"


# ===== New tests for security & reliability audit fixes =====


class TestF006BoundedQueue:
    """F-006: reset() creates a bounded queue."""

    def test_reset_creates_bounded_queue(self, reacher, mocker):
        mocker.patch.object(reacher, "stop_program")
        mocker.patch.object(reacher, "clear_queue")
        mocker.patch.object(reacher, "close_serial")
        mocker.patch("threading.Thread")

        reacher.program_flag.clear()
        reacher.reset()

        assert reacher.queue.maxsize == 5000


class TestF002DataWarning:
    """F-002: Warning emitted when in-memory data crosses threshold."""

    def test_data_warning_emitted_at_threshold(self, reacher, mocker):
        mocker.patch("builtins.open", mocker.mock_open())
        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "test-sess"
        reacher.program_running = True
        reacher.program_flag.clear()

        # Pre-populate with 99,999 entries
        reacher.behavior_data = [{"device": "X", "event": "Y"}] * 99_999
        reacher.frame_data = []
        reacher._DATA_WARNING_THRESHOLD = 100_000
        reacher._data_warning_emitted = False

        # The 100,000th entry triggers the warning
        event = {
            "level": "007",
            "device": "PUMP",
            "event": "INFUSION",
            "start_timestamp": 1000,
            "end_timestamp": 1001,
        }
        reacher.update_behavioral_events(event)

        # Find the warning call
        warning_calls = [c for c in cb.call_args_list if c[0][1] == "warning"]
        assert len(warning_calls) == 1
        assert warning_calls[0][0][2]["threshold"] == 100_000
        assert reacher._data_warning_emitted is True

        # Second call should NOT emit another warning
        cb.reset_mock()
        event2 = {
            "level": "007",
            "device": "PUMP",
            "event": "INFUSION",
            "start_timestamp": 2000,
            "end_timestamp": 2001,
        }
        reacher.update_behavioral_events(event2)
        warning_calls2 = [c for c in cb.call_args_list if c[0][1] == "warning"]
        assert len(warning_calls2) == 0


class TestF003SerialReconnect:
    """F-003: Serial reconnection on disconnect."""

    def test_serial_reconnect_success(self, reacher, mocker):
        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "test-sess"
        reacher.serial_flag.clear()
        reacher._SERIAL_RECONNECT_DELAY = 0
        reacher._SERIAL_RECONNECT_RETRIES = 3

        mock_ser = reacher.ser
        mock_ser.is_open = True

        # First in_waiting raises SerialException; after reconnect flag exits loop
        call_count = [0]

        def in_waiting_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise serial.SerialException("USB detached")
            return 0

        type(mock_ser).in_waiting = property(lambda self: in_waiting_side_effect())

        # open() succeeds on reconnect; reset_input_buffer sets serial_flag to exit
        mock_ser.open.return_value = None
        mock_ser.reset_input_buffer.side_effect = lambda: reacher.serial_flag.set()

        mocker.patch("time.sleep")
        reacher.read_serial()

        disconnect_calls = [c for c in cb.call_args_list if c[0][1] == "disconnect"]
        reconnected_calls = [c for c in cb.call_args_list if c[0][1] == "reconnected"]
        assert len(disconnect_calls) >= 1
        assert disconnect_calls[0][0][2]["reconnecting"] is True
        assert len(reconnected_calls) == 1
        assert reconnected_calls[0][0][2]["attempt"] == 1

    def test_serial_reconnect_all_retries_exhausted(self, reacher, mocker):
        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "test-sess"
        reacher.serial_flag.clear()
        reacher._SERIAL_RECONNECT_DELAY = 0
        reacher._SERIAL_RECONNECT_RETRIES = 3
        reacher.program_running = False

        mock_ser = reacher.ser
        mock_ser.is_open = True

        # in_waiting always raises SerialException
        type(mock_ser).in_waiting = property(lambda self: (_ for _ in ()).throw(serial.SerialException("USB gone")))
        # open() always fails on reconnect
        mock_ser.open.side_effect = serial.SerialException("No device")

        mocker.patch("time.sleep")
        reacher.read_serial()

        disconnect_calls = [c for c in cb.call_args_list if c[0][1] == "disconnect"]
        final = disconnect_calls[-1]
        assert final[0][2]["reconnecting"] is False
        assert reacher.serial_flag.is_set()


class TestF010EventLogFsync:
    """F-010: Event log fsync batching and cleanup on close_serial."""

    def test_event_log_fsync_batching(self, reacher, mocker):
        reacher._EVENT_LOG_FSYNC_INTERVAL = 3

        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99

        opener = mocker.patch("builtins.open", return_value=mock_file)
        fsync = mocker.patch("os.fsync")

        # Reset write counter
        reacher._event_log_file = None
        reacher._event_log_write_count = 0

        for i in range(4):
            reacher._write_event_log({"i": i})

        # File opened once and reused
        opener.assert_called_once()
        # fsync at write 3 (counter hits interval), counter resets, write 4 doesn't trigger
        assert fsync.call_count == 1

    def test_close_serial_closes_event_log(self, reacher, mocker, mock_serial):
        mock_serial.is_open = True
        mock_close_log = mocker.patch.object(reacher, "_close_event_log")
        reacher.close_serial()
        mock_close_log.assert_called_once()


class TestIssue33LogWriteFailureSurfacing:
    """Issue #33: kernel log-write failure surfacing — throttled WS warnings."""

    def test_write_event_log_failure_emits_warning(self, reacher, mocker):
        """Write failure increments count and emits throttled warning to WS."""
        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99
        mock_file.write.side_effect = OSError("No space left on device")
        mocker.patch("builtins.open", return_value=mock_file)
        mocker.patch("os.fsync")

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._write_event_log({"x": 1})

        assert cb.call_count == 1
        call_args = cb.call_args[0]
        assert call_args[0] == "sid"
        assert call_args[1] == "warning"
        data = call_args[2]
        assert data["reason"] == "log_write_failure"
        assert data["log_type"] == "event_log"
        assert data["failures_since_last_emit"] == 1

    def test_write_event_log_failure_throttled(self, reacher, mocker):
        """Multiple write failures within 1s throttle window emit only once."""
        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99
        mock_file.write.side_effect = OSError("No space left on device")
        mocker.patch("builtins.open", return_value=mock_file)
        mocker.patch("os.fsync")

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._write_event_log({"x": 1})
        reacher._write_event_log({"x": 2})

        assert cb.call_count == 1

    def test_write_event_log_failure_resets_count_after_emit(self, reacher, mocker):
        """Failure count resets to 0 after emit, then increments on next failure."""
        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99
        mock_file.write.side_effect = OSError("No space left on device")
        mocker.patch("builtins.open", return_value=mock_file)
        mocker.patch("os.fsync")

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._write_event_log({"x": 1})
        assert cb.call_count == 1

        reacher._write_event_log({"x": 2})
        assert cb.call_count == 1
        assert reacher._log_failure_counts["event_log"] == 1

    def test_write_controller_log_failure_emits_warning(self, reacher, mocker):
        """Controller log write failure emits warning with correct log_type."""
        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99
        mock_file.write.side_effect = OSError("No space left on device")
        mocker.patch("builtins.open", return_value=mock_file)
        mocker.patch("os.fsync")

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._write_controller_log({"x": 1})

        assert cb.call_count == 1
        call_args = cb.call_args[0]
        assert call_args[0] == "sid"
        assert call_args[1] == "warning"
        data = call_args[2]
        assert data["reason"] == "log_write_failure"
        assert data["log_type"] == "controller_log"
        assert data["failures_since_last_emit"] == 1

    def test_close_event_log_failure_emits_warning(self, reacher, mocker):
        """Close failure emits warning (not throttled) immediately."""
        handle = mocker.MagicMock()
        handle.closed = False
        handle.flush.side_effect = OSError("disk error")
        reacher._event_log_file = handle
        mocker.patch("os.fsync")

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._close_event_log()

        assert cb.call_count == 1
        call_args = cb.call_args[0]
        assert call_args[0] == "sid"
        assert call_args[1] == "warning"
        data = call_args[2]
        assert data["reason"] == "log_close_failure"
        assert data["log_type"] == "event_log"

    def test_close_controller_log_failure_emits_warning(self, reacher, mocker):
        """Controller log close failure emits warning with correct log_type."""
        handle = mocker.MagicMock()
        handle.closed = False
        handle.flush.side_effect = OSError("disk error")
        reacher._controller_log_file = handle
        mocker.patch("os.fsync")

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._close_controller_log()

        assert cb.call_count == 1
        call_args = cb.call_args[0]
        assert call_args[0] == "sid"
        assert call_args[1] == "warning"
        data = call_args[2]
        assert data["reason"] == "log_close_failure"
        assert data["log_type"] == "controller_log"

    def test_write_event_log_failure_no_exception_escapes(self, reacher, mocker):
        """Write failure is silently caught; broken callback increments emit_failure_count."""
        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99
        mock_file.write.side_effect = OSError("No space left on device")
        mocker.patch("builtins.open", return_value=mock_file)
        mocker.patch("os.fsync")

        cb = Mock(side_effect=Exception("Callback error"))
        reacher.event_callback = cb
        reacher.session_id = "sid"

        try:
            reacher._write_event_log({"x": 1})
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

        assert reacher.emit_failure_count == 1

    def test_close_event_log_failure_no_emit_when_no_callback(self, reacher, mocker):
        """Close failure with no callback is a no-op on _emit — no exception, no WS call."""
        handle = mocker.MagicMock()
        handle.closed = False
        handle.flush.side_effect = OSError("disk error")
        reacher._event_log_file = handle
        mocker.patch("os.fsync")

        reacher.event_callback = None

        try:
            reacher._close_event_log()
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

        assert reacher.emit_failure_count == 0

    def test_write_event_log_fsync_failure_emits_warning(self, reacher, mocker):
        """fsync failure in write is caught and emitted as log_write_failure."""
        mock_file = Mock()
        mock_file.closed = False
        mock_file.fileno.return_value = 99
        mocker.patch("builtins.open", return_value=mock_file)
        fsync_mock = mocker.patch("os.fsync")
        fsync_mock.side_effect = OSError("fsync error")

        reacher._EVENT_LOG_FSYNC_INTERVAL = 1

        cb = Mock()
        reacher.event_callback = cb
        reacher.session_id = "sid"

        reacher._write_event_log({"x": 1})

        assert cb.call_count == 1
        call_args = cb.call_args[0]
        assert call_args[0] == "sid"
        assert call_args[1] == "warning"
        data = call_args[2]
        assert data["reason"] == "log_write_failure"
        assert data["log_type"] == "event_log"


class TestFrameTimestampStopSemantics:
    """Fix FW-002: frame timestamps must be gated by program_flag, not discarded silently."""

    def test_frame_event_logged_while_program_flag_clear(self, reacher, mocker):
        """Level-008 events during active session land in frame_data."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        reacher.program_flag.clear()
        reacher.update_frame_events({"timestamp": 10000})
        assert reacher.frame_data == [10000]

    def test_frame_event_blocked_after_program_flag_set(self, reacher, mocker):
        """Level-008 events after drain window (program_flag set) are discarded."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        reacher.program_flag.set()
        reacher.update_frame_events({"timestamp": 99999})
        assert reacher.frame_data == []

    def test_frame_events_straddle_stop(self, reacher, mocker):
        """Frames before program_flag.set() persist; frames after are blocked."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        reacher.program_flag.clear()
        reacher.update_frame_events({"timestamp": 1000})
        reacher.update_frame_events({"timestamp": 2000})
        reacher.program_flag.set()
        reacher.update_frame_events({"timestamp": 3000})
        assert reacher.frame_data == [1000, 2000]

    def test_frame_data_non_empty_after_session(self, reacher, mocker):
        """frame_data is populated after a simulated session with frame events."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        reacher.program_flag.clear()
        for ts in range(0, 5000, 100):
            reacher.update_frame_events({"timestamp": ts})
        reacher.program_flag.set()
        assert len(reacher.frame_data) == 50

    def test_behavioral_events_unaffected_by_program_flag_guard(self, reacher, mocker):
        """program_flag guard on frames does not interfere with behavioral event recording."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        reacher.program_running = True
        reacher.program_flag.clear()
        event = {
            "level": "007",
            "device": "PUMP",
            "event": "INFUSION",
            "start_timestamp": 5000,
            "end_timestamp": 5100,
        }
        reacher.update_behavioral_events(event)
        reacher.update_frame_events({"timestamp": 5050})
        assert len(reacher.behavior_data) == 1
        assert reacher.frame_data == [5050]

    # Fix: FW-003 — missed-frame counter tests
    def test_frame_event_clean_emits_missed_zero(self, reacher, mocker):
        """Normal frame (missed=0) is emitted with missed:0; no warning logged."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        emit_calls = []
        reacher._emit = lambda event, data: emit_calls.append((event, data))
        warn_calls = []
        # patch.object so pytest-mock restores it: REACHER builds its logger as
        # logging.getLogger(f"reacher.{session_id or 'default'}"), so every
        # session_id=None instance shares one logger object and a bare
        # assignment here leaks the stub into every later test.
        mocker.patch.object(
            reacher.logger, "warning",
            side_effect=lambda msg, *a, **kw: warn_calls.append(msg),
        )
        reacher.program_flag.clear()
        reacher.update_frame_events({"timestamp": 1000, "missed": 0})
        assert reacher.frame_data == [1000]
        assert emit_calls == [("frame", {"timestamp": 1000, "missed": 0})]
        assert warn_calls == []

    def test_frame_event_with_drops_warns_and_emits(self, reacher, mocker):
        """Frame with missed>0 logs a warning and emits missed count; frame_data still updated."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        emit_calls = []
        reacher._emit = lambda event, data: emit_calls.append((event, data))
        warn_calls = []
        # patch.object so pytest-mock restores it: REACHER builds its logger as
        # logging.getLogger(f"reacher.{session_id or 'default'}"), so every
        # session_id=None instance shares one logger object and a bare
        # assignment here leaks the stub into every later test.
        mocker.patch.object(
            reacher.logger, "warning",
            side_effect=lambda msg, *a, **kw: warn_calls.append(msg),
        )
        reacher.program_flag.clear()
        reacher.update_frame_events({"timestamp": 2000, "missed": 2})
        assert reacher.frame_data == [2000]
        assert emit_calls == [("frame", {"timestamp": 2000, "missed": 2})]
        assert len(warn_calls) == 1
        assert "Dropped 2" in warn_calls[0] and "2000" in warn_calls[0]

    def test_frame_event_backward_compat_no_missed_key(self, reacher, mocker):
        """Old firmware events without 'missed' key default to missed=0 without error."""
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("os.fsync")
        emit_calls = []
        reacher._emit = lambda event, data: emit_calls.append((event, data))
        reacher.program_flag.clear()
        reacher.update_frame_events({"timestamp": 9999})
        assert reacher.frame_data == [9999]
        assert emit_calls == [("frame", {"timestamp": 9999, "missed": 0})]


class TestKernelCounters:
    """The tallies the export summary reports: infusions, presses, trials.

    A browser-side count is reconstructed from a refetched event list after a
    WebSocket reconnect, and can silently disagree with reality. The kernel
    counts off the serial stream as events arrive, so it has no equivalent
    failure mode — provided it recognises every spelling firmware uses.
    """

    @pytest.fixture
    def counter(self, reacher):
        """A REACHER wired only for counting.

        The shared `reacher` fixture is reused across this module, so its
        logger and event-log path may carry state from earlier tests. Counting
        is the only behaviour under test here, so the durable-write and logging
        side paths are neutralised rather than exercised.
        """
        reacher._write_event_log = lambda *a, **k: None
        reacher.logger = logging.getLogger("test.counters")
        reacher.program_running = True
        reacher.program_flag.clear()
        reacher._infusion_count = 0
        reacher._cumulative_infusion_count = 0
        reacher._press_count = 0
        reacher._cumulative_press_count = 0
        reacher._trial_count = 0
        reacher._cumulative_trial_count = 0
        reacher.behavior_data = []
        return reacher

    def _infusion(self, device):
        return {
            "level": "007", "device": device, "event": "INFUSION",
            "pin": 4, "start_timestamp": 100, "end_timestamp": 200,
        }

    def test_counts_both_operant_and_pavlovian_spellings(self, counter):
        """The operant scheduler emits PUMP_1; the Pavlovian one emits PUMP.

        Recognising only one silently zeroes the count for eight of the nine
        paradigms, or for the ninth.
        """
        for device in ("PUMP_1", "PUMP", "PUMP_1"):
            counter.update_behavioral_events(self._infusion(device))
        assert counter.get_total_infusion_count() == 3

    def test_ignores_other_devices_and_events(self, counter):
        counter.update_behavioral_events(self._infusion("PUMP_2"))
        counter.update_behavioral_events(
            {"level": "007", "device": "PUMP_1", "event": "TONE",
             "pin": 4, "start_timestamp": 1, "end_timestamp": 2}
        )
        assert counter.get_total_infusion_count() == 0

    def test_total_spans_segments(self):
        """An export archive contains every segment, so its count must too.

        `_infusion_count` alone resets at each split; a summary built from it
        would describe only the final segment while the ZIP holds them all.
        """
        import threading

        instance = REACHER.__new__(REACHER)
        instance._infusion_count = 4
        instance._cumulative_infusion_count = 11
        instance.thread_lock = threading.Lock()
        assert instance.get_total_infusion_count() == 15

    def test_count_survives_a_refetch_of_the_event_list(self, counter):
        """The kernel has no recompute path, which is why it cannot drift.

        Replacing behavior_data wholesale — the server-side analogue of the
        frontend's reconnect recovery — must not disturb the tally, because the
        tally was never derived from that list.
        """
        for _ in range(5):
            counter.update_behavioral_events(self._infusion("PUMP_1"))
        assert counter.get_total_infusion_count() == 5

        with counter.thread_lock:
            counter.behavior_data = []
        assert counter.get_total_infusion_count() == 5

    def _press(self, device, cls="ACTIVE"):
        return {
            "level": "007", "device": device, "event": "PRESS", "class": cls,
            "pin": 2, "start_timestamp": 100, "end_timestamp": 200,
        }

    @pytest.mark.parametrize("cls", ["ACTIVE", "INACTIVE", "TIMEOUT"])
    def test_counts_every_press_class(self, counter, cls):
        """The exported figure means "lever contacts", matching what the UI shows.

        Firmware sends event=PRESS with a separate class field; the kernel
        rewrites it to <CLASS>_PRESS. Counting only ACTIVE would silently change
        what the number means relative to every export already on disk.
        """
        counter.update_behavioral_events(self._press("LEVER_RH", cls))
        assert counter.get_total_press_count() == 1

    def test_counts_both_levers(self, counter):
        counter.update_behavioral_events(self._press("LEVER_RH"))
        counter.update_behavioral_events(self._press("LEVER_LH", "TIMEOUT"))
        assert counter.get_total_press_count() == 2

    def test_counts_legacy_switch_lever_spelling(self, counter):
        """Pre-v2.4.x firmware sends SWITCH_LEVER + orientation.

        The kernel rewrites it to LEVER_RH before the counter sees it, so the
        counter needs no second spelling — but that rewrite is load-bearing and
        a regression in it would zero the count on legacy rigs.
        """
        counter.update_behavioral_events({
            "level": "007", "device": "SWITCH_LEVER", "event": "PRESS",
            "class": "ACTIVE", "orientation": "RH",
            "start_timestamp": 1, "end_timestamp": 2,
        })
        assert counter.get_total_press_count() == 1

    def test_press_ignores_non_lever_and_non_press(self, counter):
        counter.update_behavioral_events(self._press("PUMP_1"))
        counter.update_behavioral_events({
            "level": "007", "device": "LEVER_RH", "event": "RELEASE",
            "start_timestamp": 1, "end_timestamp": 2,
        })
        assert counter.get_total_press_count() == 0

    def test_counts_pavlovian_trials(self, counter):
        for _ in range(3):
            counter.update_behavioral_events({
                "level": "007", "device": "PAVLOV", "event": "TRIAL_START",
                "trial_type": "CS_PLUS", "timestamp": 5,
            })
        counter.update_behavioral_events({
            "level": "007", "device": "PAVLOV", "event": "ALL_TRIALS_COMPLETE",
            "timestamp": 9,
        })
        assert counter.get_total_trial_count() == 3

    def test_counters_do_not_cross_contaminate(self, counter):
        """One event must advance exactly one tally.

        The increments share a locked block; an if/elif slip there would let a
        press bump the infusion count, which check_limit_met enforces against.
        """
        counter.update_behavioral_events(self._infusion("PUMP_1"))
        counter.update_behavioral_events(self._press("LEVER_RH"))
        counter.update_behavioral_events({
            "level": "007", "device": "PAVLOV", "event": "TRIAL_START",
            "timestamp": 5,
        })
        assert counter.get_total_infusion_count() == 1
        assert counter.get_total_press_count() == 1
        assert counter.get_total_trial_count() == 1

    def test_split_segment_accumulates_every_counter(self, counter, tmp_path):
        """The cross-segment total is only correct if the split folds all three in.

        This is the mechanism the export summary depends on: per-segment
        counters reset here, so anything not accumulated is lost silently and
        the archive under-reports.
        """
        counter._export_segment = lambda behavior, suffix="": str(tmp_path / f"seg{suffix}.csv")
        counter._emit = lambda *a, **k: None

        counter.update_behavioral_events(self._infusion("PUMP_1"))
        counter.update_behavioral_events(self._press("LEVER_RH"))
        counter.update_behavioral_events(self._press("LEVER_LH"))
        counter.update_behavioral_events({
            "level": "007", "device": "PAVLOV", "event": "TRIAL_START", "timestamp": 1,
        })
        counter.split_segment()

        assert counter._infusion_count == 0
        assert counter._press_count == 0
        assert counter._trial_count == 0

        counter.update_behavioral_events(self._press("LEVER_RH"))
        assert counter.get_total_infusion_count() == 1
        assert counter.get_total_press_count() == 3
        assert counter.get_total_trial_count() == 1

    def test_totals_span_segments(self):
        """An export archive contains every segment, so its counts must too."""
        import threading

        instance = REACHER.__new__(REACHER)
        instance.thread_lock = threading.Lock()
        instance._press_count, instance._cumulative_press_count = 6, 20
        instance._trial_count, instance._cumulative_trial_count = 1, 9
        assert instance.get_total_press_count() == 26
        assert instance.get_total_trial_count() == 10

    def test_session_summary_counts_every_event_type(self, counter):
        """session_summary.json carries a flat per-type tally so users need not re-count the CSV."""
        counter.update_behavioral_events(self._infusion("PUMP_1"))
        counter.update_behavioral_events(self._infusion("PUMP_1"))
        counter.update_behavioral_events(self._press("LEVER_RH"))
        counter.update_behavioral_events(self._press("LEVER_RH", "INACTIVE"))

        counts = counter.get_session_summary()["event_counts"]
        assert counts["total"] == 4
        assert counts["by_type"] == {
            "LEVER_RH.ACTIVE_PRESS": 1,
            "LEVER_RH.INACTIVE_PRESS": 1,
            "PUMP_1.INFUSION": 2,
        }
        assert counts["total"] == sum(counts["by_type"].values())

    def test_event_counts_span_segments_and_clear_on_reset(self, counter, tmp_path):
        """Splits empty behavior_data, so the tally must live outside it or the summary under-counts."""
        counter._export_segment = lambda behavior, suffix="": str(tmp_path / f"seg{suffix}.csv")
        counter._emit = lambda *a, **k: None

        counter.update_behavioral_events(self._press("LEVER_RH"))
        counter.update_behavioral_events(self._press("LEVER_RH"))
        counter.split_segment()
        counter.update_behavioral_events(self._press("LEVER_RH"))

        counts = counter.get_session_summary()["event_counts"]
        assert counts["by_type"] == {"LEVER_RH.ACTIVE_PRESS": 3}
        assert counts["total"] == 3

        counter._reset_session_buffers()
        assert counter.get_session_summary()["event_counts"] == {"total": 0, "by_type": {}}


class TestFirmwareErrorReporting:
    """Level-006 errors, the only signal that a command was dropped.

    Paradigms implement different command subsets, so a control the UI offers
    can be silently ignored by the firmware behind it — see
    schema.KNOWN_FIRMWARE_GAPS. The 006 record is the operator's only evidence
    that happened, which makes naming the offending command load-bearing.
    """

    @pytest.fixture
    def instance(self, reacher):
        reacher.logger = logging.getLogger("test.fwerror")
        reacher._emit = lambda *a, **k: None
        return reacher

    def test_resolves_a_known_command_code_to_its_name(self, instance, caplog):
        with caplog.at_level(logging.ERROR, logger="test.fwerror"):
            instance.handle_firmware_error({
                "level": "006", "desc": "Command not found",
                "command": int(CommandCode.LASER_TRIGGER_LH_ONLY),
            })
        assert "LASER_TRIGGER_LH_ONLY" in caplog.text
        assert "685" in caplog.text

    def test_labels_an_unregistered_code_without_raising(self, instance, caplog):
        """Firmware can outrun the registry; a diagnostic must not crash on it."""
        with caplog.at_level(logging.ERROR, logger="test.fwerror"):
            instance.handle_firmware_error({
                "level": "006", "desc": "Command not found", "command": 99999,
            })
        assert "unregistered code 99999" in caplog.text

    def test_errors_without_a_command_field_are_unchanged(self, instance, caplog):
        """Most 006 records carry no command; they must not gain noise."""
        with caplog.at_level(logging.ERROR, logger="test.fwerror"):
            instance.handle_firmware_error({"level": "006", "desc": "Pump jam"})
        assert "Pump jam" in caplog.text
        assert "(" not in caplog.text.split("Pump jam")[1]

    def test_the_raw_event_still_reaches_the_frontend(self, instance):
        """Resolution is for the log; the browser gets the event untouched."""
        seen = []
        instance._emit = lambda kind, payload: seen.append((kind, payload))
        event = {"level": "006", "desc": "Command not found", "command": 685}
        instance.handle_firmware_error(event)
        assert seen == [("error", event)]


class TestSimulatorControllerEnd:
    """reacher#8 follow-up: simulator.py never sent the CONTROLLER END event
    that stop_program() blocks on, so every simulated stop burned the full
    8s _controller_end_received timeout (+2s fallback sleep) for nothing.
    Uses a real REACHER instance with real threads against SimulatedSerial —
    the `reacher` fixture's mocked serial/threads can't exercise this path.
    """

    @pytest.fixture
    def sim_reacher(self):
        instance = REACHER(session_id="test-sim-controller-end")
        instance.set_COM_port("SIMULATOR")
        instance.open_serial()
        yield instance
        if instance.ser.is_open:
            instance.close_serial()

    def test_stop_program_resolves_promptly_via_simulator(self, sim_reacher):
        """Previously this took ~10s (8s wait + 2s fallback sleep) because the
        simulator never acked CONTROLLER END; now it should resolve in well
        under a second once the simulator's run-loop thread has joined."""
        sim_reacher.start_program()
        time.sleep(0.2)  # let the simulator's run-loop thread get going

        start = time.monotonic()
        sim_reacher.stop_program()
        elapsed = time.monotonic() - start

        assert elapsed < 3.0, f"stop_program() took {elapsed:.2f}s — CONTROLLER END likely timed out"
        assert sim_reacher._controller_end_received.is_set()
        assert sim_reacher.program_flag.is_set()


class TestDevicelessParamRecords:
    """CLI-3: pr/vi/omission/pavlovian print level-000 session-parameter rows
    with no ``device`` key at StartSession. They used to KeyError in
    update_firmware_information and surface as an ERROR log + kernel_error.
    """

    RECORDS = [
        {"level": "000", "pr_step": 1},
        {"level": "000", "variable_interval": 15000},
        {"level": "000", "omission_interval": 20000},
        {"level": "000", "cs_plus_freq": 6000, "cs_minus_freq": 12000, "cue_duration": 2000,
         "trace_interval": 1000, "consumption_ms": 3000, "iti_mean": 30000,
         "iti_min": 20000, "iti_max": 40000},
    ]

    @pytest.fixture
    def instance(self, reacher, mocker):
        reacher.logger = logging.getLogger("test.paramrecord")
        mocker.patch.object(reacher, "_write_controller_log")
        reacher._emit = Mock()
        return reacher

    @pytest.mark.parametrize("record", RECORDS, ids=["pr", "vi", "omission", "pavlovian"])
    def test_record_is_stored_without_error(self, instance, record, caplog):
        hardware_before = list(instance.hardware_settings)

        with caplog.at_level(logging.DEBUG, logger="test.paramrecord"):
            instance.handle_data(json.dumps(record))

        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        emitted = [c.args[0] for c in instance._emit.call_args_list]
        assert "kernel_error" not in emitted
        assert "config" not in emitted
        assert instance.hardware_settings == hardware_before
        for key, value in record.items():
            if key != "level":
                assert instance.firmware_information[key] == value
        assert not instance._firmware_ready.is_set()


class TestStopAfterFirmwareEnd:
    """CLI-4: an unarmed Pavlovian session ends itself on the first loop tick,
    so the CONTROLLER END lands before Stop. The later cmd 100 hits the
    firmware's "already ended" early return, so no second END is sent.
    """

    END = json.dumps({"level": "007", "device": "CONTROLLER", "event": "END", "timestamp": 87})

    @pytest.fixture
    def instance(self, reacher, mocker, mock_serial):
        reacher.ser.is_open = True
        reacher.logger = logging.getLogger("test.stopafterend")
        for name in ("_join_queue_with_timeout", "close_serial", "_write_event_log",
                     "_close_event_log", "_auto_export", "_write_session_summary",
                     "_write_controller_log"):
            mocker.patch.object(reacher, name)
        return reacher

    def test_stop_after_end_is_prompt_and_quiet(self, instance, caplog):
        instance.start_program()
        instance.handle_data(self.END)

        with patch("reacher.kernel.reacher.time.sleep") as sleep, \
                caplog.at_level(logging.DEBUG, logger="test.stopafterend"):
            start = time.monotonic()
            instance.stop_program()
            elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"stop_program() took {elapsed:.2f}s"
        assert "Timed out waiting for CONTROLLER END" not in caplog.text
        assert "export may be incomplete" not in caplog.text
        assert (2,) not in [c.args for c in sleep.call_args_list]

    def test_end_from_previous_run_does_not_satisfy_next_stop(self, instance, mocker, caplog):
        instance.start_program()
        instance.handle_data(self.END)
        instance.stop_program()
        instance.ser.is_open = True

        instance.start_program()
        assert not instance._controller_end_received.is_set()

        ev = instance._controller_end_received
        wait = mocker.patch.object(ev, "wait", side_effect=lambda timeout=None: ev.is_set())
        with patch("reacher.kernel.reacher.time.sleep"), \
                caplog.at_level(logging.WARNING, logger="test.stopafterend"):
            instance.stop_program()

        wait.assert_called_once()
        assert "Timed out waiting for CONTROLLER END" in caplog.text

    def test_end_from_restart_does_not_satisfy_next_stop(self, instance, mocker, caplog):
        # Restart's own cmd 100 makes the firmware send END for the run it replaces.
        instance.start_program()

        def firmware(command):
            if command.get("cmd") == 100:
                instance.handle_data(self.END)

        with patch.object(instance, "send_serial_command", side_effect=firmware), \
                patch("reacher.kernel.reacher.time.sleep"):
            instance.restart_program()
        assert not instance._controller_end_received.is_set()

        ev = instance._controller_end_received
        wait = mocker.patch.object(ev, "wait", side_effect=lambda timeout=None: ev.is_set())
        with patch("reacher.kernel.reacher.time.sleep"), \
                caplog.at_level(logging.WARNING, logger="test.stopafterend"):
            instance.stop_program()

        wait.assert_called_once()
        assert "Timed out waiting for CONTROLLER END" in caplog.text

    def test_late_end_from_restart_is_consumed_before_new_run(self, instance):
        # The replaced run's END can be handled after restart's sleep (a slow
        # queue, or lines requeued at the buffer swap). It must not carry over.
        instance.start_program()
        timer = threading.Timer(0.05, instance.handle_data, (self.END,))

        def firmware(command):
            if command.get("cmd") == 100:
                timer.start()

        with patch.object(instance, "send_serial_command", side_effect=firmware), \
                patch("reacher.kernel.reacher.time.sleep"):
            instance.restart_program()
        timer.join()

        assert not instance._controller_end_received.is_set()
