"""Tests for PID file management."""

from __future__ import annotations

import os

from qracer.pidfile import acquire, is_running, release


class TestPidFile:
    def test_acquire_creates_file(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        assert acquire(pid_path) is True
        assert pid_path.exists()
        assert pid_path.read_text().strip() == str(os.getpid())

    def test_acquire_fails_if_running(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        # Write current PID — process is running
        pid_path.write_text(str(os.getpid()))
        assert acquire(pid_path) is False

    def test_acquire_succeeds_with_stale_pid(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        # Write a PID that almost certainly doesn't exist
        pid_path.write_text("999999999")
        assert acquire(pid_path) is True

    def test_release_removes_file(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("12345")
        release(pid_path)
        assert not pid_path.exists()

    def test_release_missing_file_ok(self, tmp_path) -> None:
        pid_path = tmp_path / "nonexistent.pid"
        release(pid_path)  # Should not raise

    def test_is_running_current_process(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        pid_path.write_text(str(os.getpid()))
        assert is_running(pid_path) is True

    def test_is_running_stale_pid(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("999999999")
        assert is_running(pid_path) is False
        # Stale PID file should be cleaned up
        assert not pid_path.exists()

    def test_is_running_no_file(self, tmp_path) -> None:
        pid_path = tmp_path / "nonexistent.pid"
        assert is_running(pid_path) is False

    def test_is_running_corrupt_file(self, tmp_path) -> None:
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("not-a-number")
        assert is_running(pid_path) is False
