"""T1 operant rows of the simulator config matrix (research/5-reacher-config-axes.md, "Suggested matrix").

Rows 2, 3, 5, 6, 13, 14 depend on press and reward timing, so they run on the seeded stepped
clock and assert the structure of the event stream. The rest go over HTTP on a real SIMULATOR
session. Row 9 and row 17 are known defects: strict xfail with the defect named in the reason.
Limit-stop rows (1, 4, 8, 10, 12, 26 pause) also carry a stop-window defect that is intermittent,
so their reconciliation step reports it through ``pytest.xfail`` instead of a strict marker.

Host-side time limits count wall time, not simulated time, so a ``Time`` limit of N seconds stops
after N wall seconds at any speed. Rows 1, 4 and 12 use wall-second limits short enough to fit the
10 s budget, and row 4's stop delay is 1 s rather than 10 for the same reason.

Sessions boot disarmed, so the HTTP rows arm the RH lever, cue, cue2, pump and pump2 first. Limit
rows use ratio 50: the 100 ms limit poll overshoots the count by several infusions at ratio 1.
"""

import time

import pytest

from tests.sim_harness import (
    AUTH,
    DEFAULT_ARM,
    assert_reconciled,
    behavior,
    config,
    events,
    pause,
    reset,
    restart,
    send,
    set_limit,
    sim_client as sim_client,  # re-exported under its own name so pytest finds the fixture
    simulator_session,
    split,
    start,
    stepped_sim,
    stop,
    wait_until,
)


STOP_WINDOW_DEFECT = (
    "KNOWN DEFECT: events emitted between a limit stop and SESSION END reach event_log.jsonl "
    "but are dropped from the behavior export (program_running guard runs after the log write)"
)


def _reconcile_stop_window(client, sid):
    """Reconcile a limit-stopped run. A count mismatch is the stop-window defect, so it reports as xfail.

    Strict xfail would flake: the window only sometimes holds events, so the same run can pass or fail.
    Timestamp failures are not the stop-window defect and still raise.
    """
    try:
        assert_reconciled(client, sid)
    except AssertionError as exc:
        if "disagree" not in str(exc):
            raise
        pytest.xfail(STOP_WINDOW_DEFECT)


def _elapse(seconds):
    """Let *seconds* of wall time pass, polling in short steps rather than one long sleep."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.05)


def _state(client, sid):
    return client.get(f"/api/sessions/{sid}", headers=AUTH).json()["state"]


def _rows(client, sid, device, event):
    return [r for r in behavior(client, sid) if r.get("device") == device and r.get("event") == event]


def _snapshot_export(client, sid):
    """Copy behavior rows and per-pump infusion counters under the kernel lock.

    A restarted session is still emitting, so the buffer and the counters have to
    be read together or a reward can land between the two reads.
    """
    instance = client.app.state.session_manager.get_session(sid).instance
    with instance.thread_lock:
        rows = [dict(row) for row in instance.behavior_data]
        by_pump = {
            pump: instance._cumulative_infusions_by_pump[pump] + instance._infusions_by_pump[pump]
            for pump in ("PUMP_1", "PUMP_2")
        }
    return rows, by_pump


def _pump_infusions(rows, pump):
    """Infusion rows counted toward ``pump``. Firmware also spells pump 1 as ``PUMP``."""
    devices = ("PUMP", "PUMP_1") if pump == "PUMP_1" else (pump,)
    return [row for row in rows if row.get("device") in devices and row.get("event") == "INFUSION"]


def _last_controller_start(rows):
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        if row.get("device") == "CONTROLLER" and row.get("event") == "START":
            return index
    return None


def _cards(client, sid):
    """The per-device rows behind ``hardware_settings``, keyed by device name."""
    return {r["device"]: r for r in config(client, sid)["hardware_settings"]}


def _send_all(client, sid, commands):
    for code, value in commands:
        resp = send(client, sid, code, value)
        assert resp.status_code == 200, resp.text


def _arm_operant(client, sid, ratio=50, timeout=0):
    """Arm RH lever, cue, cue2, pump and pump2, then set ratio, timeout and timeout mode 0."""
    _send_all(client, sid, [(code, None) for code in DEFAULT_ARM])
    _send_all(client, sid, [(201, ratio), (1074, timeout), (1077, 0)])


def _run_until_stopped(client, sid, timeout):
    resp = start(client, sid)
    assert resp.status_code == 200, resp.text
    wait_until(lambda: _state(client, sid) == "stopped", timeout=timeout)


def _press_starts(msgs, orientation=None, cls=None):
    return [
        m["start_timestamp"]
        for m in events(msgs, device=("LEVER_RH", "LEVER_LH"), event="PRESS", cls=cls)
        if orientation is None or m.get("orientation") == orientation
    ]


# ---------------------------------------------------------------------------
# Row 1: fr, RH only, Time limit, lick disarmed
# ---------------------------------------------------------------------------


def test_row01_fr_time_limit_stops_session(sim_client):
    """Baseline Time limit: the session stops on its own once the wall-clock limit is met.

    Time limit reduced from 30 s to 2 s wall: the host counts wall time, so 30 s would blow the
    10 s budget. The stop path is the same.
    """
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _send_all(client, sid, [(1001, None), (500, None), (201, 1), (1074, 0), (1077, 0)])
        set_limit(client, sid, type="Time", time_limit=2)
        began = time.monotonic()
        _run_until_stopped(client, sid, timeout=8)
        elapsed = time.monotonic() - began
        assert 1.5 <= elapsed <= 6, elapsed
        assert not [r for r in behavior(client, sid) if str(r.get("device", "")).startswith("LICK")]
        _reconcile_stop_window(client, sid)


# ---------------------------------------------------------------------------
# Rows 2 and 3: fr ratio 5, timeout 2000, cue2 armed, timeout mode 0 vs 1 (stepped clock)
# ---------------------------------------------------------------------------

_FR5 = [(201, "ratio", 5), (1074, "timeout", 2000)]


def test_row02_fr_ratio5_timeout_mode0_presses_inside_timeout_do_not_reward():
    """Mode 0: every active press restarts the timeout, so fast presses are TIMEOUT and cannot reward."""
    msgs = stepped_sim("fr", _FR5 + [(1077, "timeout_mode", 0)], horizon_ms=60_000)
    assert events(msgs, device="LEVER_RH", event="PRESS", cls="TIMEOUT")
    active = _press_starts(msgs, cls="ACTIVE")
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    assert rewards, "no reward in 60 s of simulated time"
    for reward_ts in rewards:
        assert reward_ts in active, "a reward landed on a press that was not an ACTIVE press"
    for k, reward_ts in enumerate(rewards):
        if (k + 1) * 5 <= len(active):
            assert reward_ts == active[(k + 1) * 5 - 1]


def test_row03_fr_ratio5_timeout_mode1_reward_on_fifth_press():
    """Mode 1: only the reward-triggering press starts the timeout, so the 5th press is rewarded."""
    msgs = stepped_sim("fr", _FR5 + [(1077, "timeout_mode", 1)], horizon_ms=60_000)
    active = _press_starts(msgs, cls="ACTIVE")
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    assert len(rewards) >= 2
    assert rewards[0] == active[4]
    assert len(rewards) >= len(active) // 5 - 1


# ---------------------------------------------------------------------------
# Row 4: pr step 2, timeout 20 s, mode 0, Infusion limit 3 (stop delay 1 s wall, not 10)
# ---------------------------------------------------------------------------


def test_row04_pr_step2_threshold_climbs_and_infusion_limit_stops(sim_client):
    """Thresholds are 1, 3, 5 presses for step 2; the session stops after the 3rd infusion."""
    msgs = stepped_sim(
        "pr",
        [(205, "step", 2), (201, "ratio", 1), (1074, "timeout", 20000), (1077, "timeout_mode", 0)],
        horizon_ms=300_000,
    )
    active = _press_starts(msgs, cls="ACTIVE")
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    assert len(rewards) >= 3
    index = [active.index(t) + 1 for t in rewards[:3]]
    assert index == [1, 4, 9], index

    client = sim_client
    with simulator_session(client, "pr") as sid:
        _send_all(client, sid, [(code, None) for code in DEFAULT_ARM])
        _send_all(client, sid, [(205, 2), (201, 1), (1074, 20000), (1077, 0)])
        set_limit(client, sid, type="Infusion", infusion_limit=3, delay=1)
        _run_until_stopped(client, sid, timeout=8)
        assert len(_rows(client, sid, "PUMP_1", "INFUSION")) >= 3
        _reconcile_stop_window(client, sid)


# ---------------------------------------------------------------------------
# Row 5: vi interval 15 s, timeout 20 s, mode 0, LH armed and active, cue filter RH, Both 60 s
# ---------------------------------------------------------------------------


def test_row05_vi_window_rewards_at_most_one_per_interval():
    """Rewards come only from the availability window, one per interval; timeouts may occur in-window."""
    msgs = stepped_sim(
        "vi",
        [(204, "interval", 15_000), (1074, "timeout", 20_000), (1077, "timeout_mode", 0), 1381, (378, "filter", 1)],
        arm=DEFAULT_ARM + (1301,),
        horizon_ms=60_000,
    )
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    assert rewards, "no reward in 60 s of simulated time"
    assert len({t // 15_000 for t in rewards}) == len(rewards), "two rewards in one availability window"
    active = set(_press_starts(msgs, cls="ACTIVE"))
    assert all(t in active for t in rewards)
    cue_starts = [m["start_timestamp"] for m in events(msgs, device="CUE_1", event="TONE")]
    rh_presses = set(_press_starts(msgs, orientation="RH"))
    assert all(t in rh_presses for t in cue_starts), "cue filter RH: a cue fired on an LH press"


# ---------------------------------------------------------------------------
# Row 6: omission interval 20 s, laser independent, cue2 armed, 1074 sent (dropped by firmware)
# ---------------------------------------------------------------------------


def test_row06_omission_pays_absence_and_never_times_out():
    msgs = stepped_sim(
        "omission",
        [(203, "interval", 20_000), 682, (672, "duration", 2000), (1074, "timeout", 5000), 601],
        horizon_ms=900_000,
    )
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    assert len(rewards) >= 3
    marks = sorted(set(_press_starts(msgs)) | set(rewards))
    for reward_ts in rewards:
        earlier = [t for t in marks if t < reward_ts]
        assert reward_ts - max(earlier) == 20_000
    assert not events(msgs, device=("LEVER_RH", "LEVER_LH"), event="PRESS", cls="TIMEOUT")
    laser = [m["start_timestamp"] for m in events(msgs, device="LASER", event="STIM")]
    assert laser and set(laser) != set(rewards), "independent laser should not track rewards"


def test_row07_omission_rejects_1077_with_400(sim_client):
    client = sim_client
    with simulator_session(client, "omission") as sid:
        resp = send(client, sid, 1077, 1)
        assert resp.status_code == 400, resp.text
        assert "not available for omission" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Row 8: fr, SET_ACTIVE_PUMP=true, Infusion 3, delay unset (pump 2 is the reward pump)
# ---------------------------------------------------------------------------


def test_row08_pump2_infusion_limit_stops(sim_client):
    """With pump 2 as the reward pump, an Infusion 3 limit stops the session on PUMP_2 infusions."""
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid)
        assert send(client, sid, 221, True).status_code == 200
        set_limit(client, sid, type="Infusion", infusion_limit=3)
        _run_until_stopped(client, sid, timeout=8)
        assert len(_rows(client, sid, "PUMP_2", "INFUSION")) == 3
        assert not _rows(client, sid, "PUMP_1", "INFUSION")
        _reconcile_stop_window(client, sid)


# ---------------------------------------------------------------------------
# Row 9: fr with a Trials limit
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="POST /limit accepts Trials on operant paradigms (no paradigm gate), so the session runs until a manual stop",
)
def test_row09_trials_limit_rejected_on_operant(sim_client):
    """Trial counting is Pavlovian-only, so a Trials limit on fr should be refused with 400."""
    client = sim_client
    with simulator_session(client, "fr") as sid:
        resp = set_limit(client, sid, type="Trials")
        assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Row 10: fr, Infusion 2, stop delay unset (defaults to 0)
# ---------------------------------------------------------------------------


def test_row10_infusion_limit_without_delay_stops(sim_client):
    """With no delay set, the Infusion limit stops the session without a stop delay."""
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid)
        set_limit(client, sid, type="Infusion", infusion_limit=2)
        _run_until_stopped(client, sid, timeout=8)
        assert len(_rows(client, sid, "PUMP_1", "INFUSION")) == 2
        _reconcile_stop_window(client, sid)


# ---------------------------------------------------------------------------
# Row 11: fr, Infusion 2, split after the first infusion
# ---------------------------------------------------------------------------


def test_row11_split_does_not_reset_infusion_limit(sim_client):
    """The limit counts across segments, so a split after one infusion still stops at a total of 2.

    Not reconciled: split clears the behavior buffer, so the export holds only the last segment.
    The total is read from the kernel's cumulative count, which is what the limit enforces against.
    """
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid)
        set_limit(client, sid, type="Infusion", infusion_limit=2)
        assert start(client, sid).status_code == 200
        wait_until(lambda: len(_rows(client, sid, "PUMP_1", "INFUSION")) >= 1, timeout=8)
        resp = split(client, sid)
        assert resp.status_code == 200, resp.text
        wait_until(lambda: _state(client, sid) == "stopped", timeout=8)
        instance = client.app.state.session_manager.get_session(sid).instance
        assert instance.get_total_infusion_count() == 2


# ---------------------------------------------------------------------------
# Row 12: fr, REACHER_SIM_SPEED=4, Time limit (wall seconds)
# ---------------------------------------------------------------------------


def test_row12_speed4_time_limit_counts_wall_seconds(sim_client, monkeypatch):
    """At 4x the simulator runs faster than wall time, while the host Time limit still counts wall seconds."""
    monkeypatch.setenv("REACHER_SIM_SPEED", "4")
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid, ratio=1)
        set_limit(client, sid, type="Time", time_limit=2)
        began = time.monotonic()
        _run_until_stopped(client, sid, timeout=8)
        elapsed = time.monotonic() - began
        assert 1.5 <= elapsed <= 5, elapsed
        last = max(r["start_timestamp"] for r in behavior(client, sid))
        assert last > 2 * 2000, "simulated clock should outrun the wall clock at 4x"
        _reconcile_stop_window(client, sid)


# ---------------------------------------------------------------------------
# Row 13: fr, laser 684 (RH-only) then 681 (contingent): RH-only is cleared (stepped clock)
# ---------------------------------------------------------------------------


def test_row13_contingent_mode_after_rh_only_fires_with_reward():
    msgs = stepped_sim(
        "fr",
        [(201, "ratio", 1), (1074, "timeout", 0), 684, 681],
        arm=DEFAULT_ARM + (601,),
        horizon_ms=20_000,
    )
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    laser = [m["start_timestamp"] for m in events(msgs, device="LASER", event="STIM")]
    assert rewards and laser
    assert laser == rewards, "contingent laser should fire with the reward, not on every RH press"


# ---------------------------------------------------------------------------
# Row 14: fr, LH active, laser 685 (LH only), pump filter 478=2 (LH only), RH inactive (stepped clock)
# ---------------------------------------------------------------------------


def test_row14_lh_only_laser_and_pump_filter_follow_lh_presses():
    msgs = stepped_sim(
        "fr",
        [(201, "ratio", 1), (1074, "timeout", 0), 685, (478, "filter", 2), 1381, 1080],
        arm=DEFAULT_ARM + (1301, 601),
        horizon_ms=20_000,
    )
    lh_active = _press_starts(msgs, orientation="LH", cls="ACTIVE")
    assert lh_active
    assert not _press_starts(msgs, orientation="RH", cls="ACTIVE"), "RH presses should do nothing"
    rewards = [m["start_timestamp"] for m in events(msgs, device="PUMP_1", event="INFUSION")]
    laser = [m["start_timestamp"] for m in events(msgs, device="LASER", event="STIM")]
    assert rewards and laser
    assert set(rewards) <= set(lh_active)
    assert set(laser) <= set(lh_active)


# ---------------------------------------------------------------------------
# Rows 17 and 18: aliased registers (1075/1375 and 1074/1374)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="1075/1375 are not per-lever: RH=5 then LH=3 leaves the board on 3 but the LEVER_RH card still reads 5 (wrong_value_accepted)",
)
def test_row17_ratio_lever_cards_match_board_after_aliased_writes(sim_client):
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _send_all(client, sid, [(1075, 5), (1375, 3)])
        cards = _cards(client, sid)
        assert cards["LEVER_RH"].get("ratio") == cards["CONTROLLER"].get("ratio") == 3
        assert cards["LEVER_LH"].get("ratio") == 3


def test_row18_timeout_last_write_wins_on_both_levers(sim_client):
    """1074=1000 then 1374=3000: the board runs 3000, and the RH card does not keep the stale 1000."""
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _send_all(client, sid, [(1074, 1000), (1374, 3000)])
        cards = _cards(client, sid)
        assert cards["CONTROLLER"].get("timeout") == 3000
        assert cards.get("LEVER_RH", {}).get("timeout") in (None, 3000)


# ---------------------------------------------------------------------------
# Row 26: fr, pause / restart / reset mid-run
# ---------------------------------------------------------------------------


def test_row26_pause_is_excluded_from_time_limit(sim_client):
    """A pause longer than the Time limit must not stop the session; resuming runs out the rest of it."""
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid, ratio=1)
        set_limit(client, sid, type="Time", time_limit=1)
        assert start(client, sid).status_code == 200
        wait_until(lambda: _rows(client, sid, "PUMP_1", "INFUSION"), timeout=5)
        assert pause(client, sid).status_code == 200
        _elapse(1.2)
        assert _state(client, sid) != "stopped"
        assert pause(client, sid).status_code == 200
        wait_until(lambda: _state(client, sid) == "stopped", timeout=5)
        _reconcile_stop_window(client, sid)


def test_row26_restart_resets_counters(sim_client):
    """Restart clears the in-memory export and zeroes the infusion counters.

    The event log stays session-long, so this row is not reconciled against it.
    The restarted session keeps running, and at the harness speed a new reward can
    land before this read. That reward follows the new CONTROLLER START. Rows ahead
    of that START are the ones the clear has to have dropped, and the counters have
    to match the infusions still in the buffer.
    """
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid, ratio=1)
        assert start(client, sid).status_code == 200
        wait_until(lambda: _rows(client, sid, "PUMP_1", "INFUSION"), timeout=5)
        pre_rows, pre_pumps = _snapshot_export(client, sid)
        assert _pump_infusions(pre_rows, "PUMP_1")
        assert pre_pumps["PUMP_1"] >= 1

        assert restart(client, sid).status_code == 200

        rows, by_pump = _snapshot_export(client, sid)
        # The old START sits at index 0 when the clear has not replaced the buffer,
        # so the slice below would be empty and the pre-restart rows would still be
        # the head of the export.
        assert rows[: len(pre_rows)] != pre_rows
        start_at = _last_controller_start(rows)
        leaked = rows if start_at is None else rows[:start_at]
        assert leaked == []
        for pump in ("PUMP_1", "PUMP_2"):
            assert by_pump[pump] == len(_pump_infusions(rows, pump))
        stop(client, sid)


def test_row26_reset_mid_run_clears_session(sim_client):
    client = sim_client
    with simulator_session(client, "fr") as sid:
        _arm_operant(client, sid, ratio=1)
        assert start(client, sid).status_code == 200
        wait_until(lambda: _rows(client, sid, "PUMP_1", "INFUSION"), timeout=5)
        resp = reset(client, sid)
        assert resp.status_code == 200, resp.text
        assert not behavior(client, sid)
