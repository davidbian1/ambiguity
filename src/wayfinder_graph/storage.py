"""JSON-backed ChartStore — the only tracker backend implemented so far
(see SPEC.md's swap points). One file per ticket so parallel sessions
working different tickets never contend; chart.json is only touched at
claim-time and resolution-time, guarded by a plain exclusive-create lock
file (no new dependency needed for a prototype this small).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Protocol

from .schema import Chart, Ticket


class ChartStore(Protocol):
    def load_chart(self) -> Chart: ...
    def save_chart(self, chart: Chart) -> None: ...
    def load_ticket(self, ticket_id: str) -> Ticket: ...
    def save_ticket(self, ticket: Ticket) -> None: ...
    def list_tickets(self) -> List[Ticket]: ...
    def append_event(self, event: Dict) -> None: ...
    def read_events(self) -> List[Dict]: ...


@contextmanager
def _lock(lock_path: Path, timeout: float = 5.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for lock {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


class JsonChartStore:
    def __init__(self, chart_dir: Path):
        self.chart_dir = Path(chart_dir)
        self.tickets_dir = self.chart_dir / "tickets"
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        self.chart_path = self.chart_dir / "chart.json"
        self.chart_lock_path = self.chart_dir / "chart.json.lock"
        self.events_path = self.chart_dir / "events.jsonl"

    def load_chart(self) -> Chart:
        return Chart.from_dict(json.loads(self.chart_path.read_text(encoding="utf-8")))

    def save_chart(self, chart: Chart) -> None:
        with _lock(self.chart_lock_path):
            self.chart_path.write_text(json.dumps(chart.to_dict(), indent=2), encoding="utf-8")

    def _ticket_path(self, ticket_id: str) -> Path:
        return self.tickets_dir / f"{ticket_id}.json"

    def load_ticket(self, ticket_id: str) -> Ticket:
        return Ticket.from_dict(json.loads(self._ticket_path(ticket_id).read_text(encoding="utf-8")))

    def save_ticket(self, ticket: Ticket) -> None:
        self._ticket_path(ticket.id).write_text(json.dumps(ticket.to_dict(), indent=2), encoding="utf-8")

    def list_tickets(self) -> List[Ticket]:
        return [Ticket.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(self.tickets_dir.glob("*.json"))]

    def append_event(self, event: Dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def read_events(self) -> List[Dict]:
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
