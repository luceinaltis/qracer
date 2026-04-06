"""Tests for Watchlist."""

from __future__ import annotations

from qracer.watchlist import Watchlist


class TestWatchlist:
    def test_add_ticker(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        assert wl.add("AAPL") is True
        assert wl.tickers == ["AAPL"]

    def test_add_duplicate(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        wl.add("AAPL")
        assert wl.add("AAPL") is False
        assert wl.tickers == ["AAPL"]

    def test_add_case_insensitive(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        wl.add("aapl")
        assert wl.tickers == ["AAPL"]
        assert wl.add("AAPL") is False

    def test_remove_ticker(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        wl.add("AAPL")
        wl.add("TSLA")
        assert wl.remove("AAPL") is True
        assert wl.tickers == ["TSLA"]

    def test_remove_nonexistent(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        assert wl.remove("AAPL") is False

    def test_contains(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        wl.add("AAPL")
        assert wl.contains("AAPL") is True
        assert wl.contains("TSLA") is False
        assert wl.contains("aapl") is True

    def test_clear(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        wl.add("AAPL")
        wl.add("TSLA")
        wl.clear()
        assert wl.tickers == []
        assert len(wl) == 0

    def test_persistence(self, tmp_path) -> None:
        path = tmp_path / "wl.json"
        wl1 = Watchlist(path)
        wl1.add("AAPL")
        wl1.add("MSFT")

        # Load from same file
        wl2 = Watchlist(path)
        assert wl2.tickers == ["AAPL", "MSFT"]

    def test_load_empty_file(self, tmp_path) -> None:
        path = tmp_path / "wl.json"
        wl = Watchlist(path)
        assert wl.tickers == []

    def test_load_corrupt_file(self, tmp_path) -> None:
        path = tmp_path / "wl.json"
        path.write_text("not json", encoding="utf-8")
        wl = Watchlist(path)
        assert wl.tickers == []

    def test_insertion_order(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        wl.add("TSLA")
        wl.add("AAPL")
        wl.add("NVDA")
        assert wl.tickers == ["TSLA", "AAPL", "NVDA"]

    def test_len(self, tmp_path) -> None:
        wl = Watchlist(tmp_path / "wl.json")
        assert len(wl) == 0
        wl.add("AAPL")
        wl.add("TSLA")
        assert len(wl) == 2
