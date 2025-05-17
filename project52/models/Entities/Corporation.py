from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Unicode blocks for sparkline
_SPARK_BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def sparkline(values: List[float]) -> str:
    """Generate a sparkline string from a list of numeric values."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    extent = mx - mn
    if extent == 0:
        return _SPARK_BLOCKS[0] * len(values)
    result = []
    for v in values:
        idx = int((v - mn) / extent * (len(_SPARK_BLOCKS) - 1))
        result.append(_SPARK_BLOCKS[idx])
    return "".join(result)


class Corporation:
    """
    Bloomberg-style terminal dashboard for corporate entities,
    with enhanced branding for The 254 Company.
    """

    BRAND_NAME = "THE 254 COMPANY"
    BRAND_TAGLINE = "Empowering Market Intelligence"
    BRAND_URL = "https://the254company.com"

    def __init__(
        self,
        name: str,
        ticker: str,
        industry: str,
        exchange: str,
        country: str,
        founded: date,
        shares_outstanding: float,
        share_price: float,
        cik: Optional[str] = None,
        **data: Any,
    ):
        # Input validation
        if not name or not ticker:
            raise ValueError("`name` and `ticker` are required.")
        if shares_outstanding < 0 or share_price < 0:
            raise ValueError("Shares and price must be non-negative.")

        # Core attributes
        self.name = name
        self.ticker = ticker.upper()
        self.industry = industry
        self.exchange = exchange.upper()
        self.country = country
        self.founded = founded
        self.shares_outstanding = shares_outstanding
        self.share_price = share_price
        self.cik = cik

        # Additional dynamic data
        self.data = data
        self.price_history: List[tuple[datetime, float]] = [
            (datetime.utcnow(), share_price)
        ]
        self._metrics: Dict[str, Callable[[Corporation], Any]] = {}

    def market_cap(self) -> float:
        return self.shares_outstanding * self.share_price

    def update_share_price(self, new_price: float) -> None:
        if new_price < 0:
            raise ValueError("Price must be non-negative.")
        self.share_price = new_price
        self.price_history.append((datetime.utcnow(), new_price))

    def register_metric(self, name: str, func: Callable[[Corporation], Any]) -> None:
        self._metrics[name] = func

    def compute_metric(self, name: str) -> Any:
        if name not in self._metrics:
            raise KeyError(f"Metric not found: {name}")
        return self._metrics[name](self)

    def _create_header(self) -> Panel:
        title = Text(self.BRAND_NAME, style="bold white on gold1", justify="center")
        subtitle = Text(
            self.BRAND_TAGLINE, style="italic white on gold1", justify="center"
        )
        stock = Text(
            f"{self.ticker} — {self.name}", style="bold yellow", justify="center"
        )
        header = Align.center(
            Text.assemble(title, "\n", subtitle, "\n", stock), vertical="middle"
        )
        return Panel(header, box=box.HEAVY, padding=(1, 2), border_style="gold1")

    def _create_summary_panel(self) -> Panel:
        table = Table.grid(padding=0)
        table.add_column(style="bold cyan", justify="right", ratio=2)
        table.add_column(style="white", justify="left", ratio=3)
        values = [
            ("Industry", self.industry),
            ("Exchange", self.exchange),
            ("Country", self.country),
            ("Founded", self.founded.isoformat()),
            ("Shares O/S", f"{self.shares_outstanding:,.0f}"),
            ("Price", f"${self.share_price:.2f}"),
            ("Market Cap", f"${self.market_cap():,.2f}"),
        ]
        for label, val in values:
            table.add_row(label, str(val))
        # Add sparkline trend
        prices = [p for _, p in self.price_history[-10:]]
        table.add_row("Trend", sparkline(prices))
        return Panel(
            table, title="[bold]Summary[/bold]", box=box.ROUNDED, border_style="cyan"
        )

    def _create_metrics_panel(self) -> Panel:
        table = Table(
            show_header=True, header_style="bold magenta", box=box.MINIMAL_DOUBLE_HEAD
        )
        table.add_column("Metric", style="bright_green")
        table.add_column("Value", justify="right", style="bright_yellow")
        for name, func in self._metrics.items():
            try:
                val = func(self)
            except Exception:
                val = "Error"
            table.add_row(name, str(val))
        return Panel(
            table, title="[bold]Metrics[/bold]", box=box.ROUNDED, border_style="magenta"
        )

    def _create_history_panel(self, limit: int = 10) -> Panel:
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        table.add_column("Time", style="dim")
        table.add_column("Price", justify="right", style="bold green")
        for ts, price in self.price_history[-limit:]:
            table.add_row(ts.strftime("%H:%M:%S"), f"${price:.2f}")
        return Panel(
            table,
            title="[bold]Recent Prices[/bold]",
            box=box.ROUNDED,
            border_style="blue",
        )

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="body", ratio=4),
            Layout(name="footer", size=2),
        )
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(
            Layout(name="summary", ratio=3), Layout(name="history", ratio=2)
        )
        layout["header"].update(self._create_header())
        layout["summary"].update(self._create_summary_panel())
        layout["history"].update(self._create_history_panel())
        layout["right"].update(self._create_metrics_panel())
        footer_text = Text.assemble(
            ("Powered by ", "dim"),
            (self.BRAND_NAME, "bold gold1"),
            (f" | {self.BRAND_URL}", "dim cyan"),
        )
        layout["footer"].update(Panel(Align.center(footer_text), box=box.MINIMAL))
        return layout

    def display(self) -> None:
        console.clear()
        console.print(self._build_layout())

    def to_dict(self) -> Dict[str, Any]:
        d = {
            k: (v.isoformat() if isinstance(v, (date, datetime)) else v)
            for k, v in vars(self).items()
            if not k.startswith("_")
        }
        d["metrics"] = list(self._metrics.keys())
        d["data"] = self.data
        d["price_history"] = [(ts.isoformat(), p) for ts, p in self.price_history]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        return f"<Corporation {self.ticker} | {self.name}>"


if __name__ == "__main__":
    from datetime import date

    corp = Corporation(
        name="Dynamic Corp",
        ticker="DYN",
        industry="Technology",
        exchange="NASDAQ",
        country="USA",
        founded=date(2000, 1, 1),
        shares_outstanding=50_000_000,
        share_price=120.5,
        headquarters="NYC",
        employees=5000,
    )
    corp.register_metric("PE Ratio", lambda c: round(c.share_price / 5, 2))
    corp.display()
    corp.update_share_price(125.0)
    corp.display()
