"""Tests for Alert model and AlertStore."""

from __future__ import annotations

from qracer.alerts import Alert, AlertCondition, AlertStore


class TestAlertEvaluate:
    def _make_alert(self, condition: AlertCondition, threshold: float) -> Alert:
        return Alert(
            id="test1",
            ticker="AAPL",
            condition=condition,
            threshold=threshold,
            created_at="2026-01-01T00:00:00+00:00",
        )

    def test_above_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.ABOVE, 200.0)
        assert alert.evaluate(201.0) is True

    def test_above_not_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.ABOVE, 200.0)
        assert alert.evaluate(199.0) is False

    def test_above_exact_not_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.ABOVE, 200.0)
        assert alert.evaluate(200.0) is False

    def test_below_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.BELOW, 150.0)
        assert alert.evaluate(149.0) is True

    def test_below_not_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.BELOW, 150.0)
        assert alert.evaluate(151.0) is False

    def test_below_exact_not_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.BELOW, 150.0)
        assert alert.evaluate(150.0) is False

    def test_change_pct_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.CHANGE_PCT, 5.0)
        # 10% increase from 100 -> 110
        assert alert.evaluate(110.0, reference_price=100.0) is True

    def test_change_pct_not_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.CHANGE_PCT, 5.0)
        # 2% increase from 100 -> 102
        assert alert.evaluate(102.0, reference_price=100.0) is False

    def test_change_pct_negative_triggered(self) -> None:
        alert = self._make_alert(AlertCondition.CHANGE_PCT, 5.0)
        # -6% from 100 -> 94
        assert alert.evaluate(94.0, reference_price=100.0) is True

    def test_change_pct_no_reference(self) -> None:
        alert = self._make_alert(AlertCondition.CHANGE_PCT, 5.0)
        assert alert.evaluate(110.0) is False

    def test_change_pct_zero_reference(self) -> None:
        alert = self._make_alert(AlertCondition.CHANGE_PCT, 5.0)
        assert alert.evaluate(110.0, reference_price=0.0) is False


class TestAlertDescribe:
    def test_above(self) -> None:
        alert = Alert(
            id="x",
            ticker="TSLA",
            condition=AlertCondition.ABOVE,
            threshold=300.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert alert.describe() == "TSLA goes above 300.0"

    def test_below(self) -> None:
        alert = Alert(
            id="x",
            ticker="TSLA",
            condition=AlertCondition.BELOW,
            threshold=200.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert alert.describe() == "TSLA goes below 200.0"

    def test_change_pct(self) -> None:
        alert = Alert(
            id="x",
            ticker="TSLA",
            condition=AlertCondition.CHANGE_PCT,
            threshold=5.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert alert.describe() == "TSLA changes by 5.0%"


class TestAlertStore:
    def test_create_alert(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        alert = store.create("AAPL", AlertCondition.ABOVE, 200.0)
        assert alert.ticker == "AAPL"
        assert alert.condition == AlertCondition.ABOVE
        assert alert.threshold == 200.0
        assert alert.active is True
        assert len(store) == 1

    def test_create_uppercases_ticker(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        alert = store.create("aapl", AlertCondition.ABOVE, 200.0)
        assert alert.ticker == "AAPL"

    def test_get_active(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        a1 = store.create("AAPL", AlertCondition.ABOVE, 200.0)
        store.create("TSLA", AlertCondition.BELOW, 150.0)
        store.mark_triggered(a1.id)
        active = store.get_active()
        assert len(active) == 1
        assert active[0].ticker == "TSLA"

    def test_get_by_ticker(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        store.create("AAPL", AlertCondition.ABOVE, 200.0)
        store.create("TSLA", AlertCondition.BELOW, 150.0)
        store.create("AAPL", AlertCondition.BELOW, 180.0)
        assert len(store.get_by_ticker("AAPL")) == 2
        assert len(store.get_by_ticker("TSLA")) == 1
        assert len(store.get_by_ticker("NVDA")) == 0

    def test_mark_triggered(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        alert = store.create("AAPL", AlertCondition.ABOVE, 200.0)
        assert store.mark_triggered(alert.id) is True
        assert store.alerts[0].active is False
        assert store.alerts[0].triggered_at is not None

    def test_mark_triggered_nonexistent(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        assert store.mark_triggered("nonexistent") is False

    def test_mark_triggered_already_triggered(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        alert = store.create("AAPL", AlertCondition.ABOVE, 200.0)
        store.mark_triggered(alert.id)
        assert store.mark_triggered(alert.id) is False

    def test_remove(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        alert = store.create("AAPL", AlertCondition.ABOVE, 200.0)
        assert store.remove(alert.id) is True
        assert len(store) == 0

    def test_remove_nonexistent(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        assert store.remove("nonexistent") is False

    def test_clear(self, tmp_path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        store.create("AAPL", AlertCondition.ABOVE, 200.0)
        store.create("TSLA", AlertCondition.BELOW, 150.0)
        store.clear()
        assert len(store) == 0

    def test_persistence(self, tmp_path) -> None:
        path = tmp_path / "alerts.json"
        store1 = AlertStore(path)
        store1.create("AAPL", AlertCondition.ABOVE, 200.0)
        store1.create("TSLA", AlertCondition.BELOW, 150.0)

        store2 = AlertStore(path)
        assert len(store2) == 2
        assert store2.alerts[0].ticker == "AAPL"
        assert store2.alerts[1].ticker == "TSLA"

    def test_persistence_after_trigger(self, tmp_path) -> None:
        path = tmp_path / "alerts.json"
        store1 = AlertStore(path)
        alert = store1.create("AAPL", AlertCondition.ABOVE, 200.0)
        store1.mark_triggered(alert.id)

        store2 = AlertStore(path)
        assert store2.alerts[0].active is False
        assert store2.alerts[0].triggered_at is not None

    def test_load_empty_file(self, tmp_path) -> None:
        path = tmp_path / "alerts.json"
        store = AlertStore(path)
        assert len(store) == 0

    def test_load_corrupt_file(self, tmp_path) -> None:
        path = tmp_path / "alerts.json"
        path.write_text("not json", encoding="utf-8")
        store = AlertStore(path)
        assert len(store) == 0
