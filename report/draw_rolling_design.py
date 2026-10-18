"""Draw the report's illustrative rolling forecast timeline (no data input)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parents[1] / "output" / "figures"
BLUE, GREEN, INK, MUTED = "#2a78d6", "#1b8a67", "#202124", "#62666b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.0, 5.5), facecolor="white")
ax = fig.add_axes([0.02, 0.03, 0.96, 0.95])
ax.set(xlim=(0, 14), ylim=(0, 10.6))
ax.axis("off")


def label(x, y, text, **kwargs):
    ax.text(x, y, text, color=INK, va="center", **kwargs)


label(0, 10.2, "1  Roll the 12-quarter window forward", weight="bold", fontsize=11)
label(0, 9.65, "Blue: quarters eligible for estimation     Green: quarter being forecast", fontsize=8.5)
left, step, width = 0.15, 0.98, 0.87
quarters = [(2017 + (1 + i) // 4, (1 + i) % 4 + 1) for i in range(14)]
for year in range(2017, 2021):
    positions = [i for i, (y, _) in enumerate(quarters) if y == year]
    center = left + (min(positions) + max(positions)) / 2 * step + width / 2
    label(center, 9.05, str(year), ha="center", weight="bold")
for row, (start, target) in enumerate([(0, 12), (1, 13)]):
    y = 7.95 - row * 1.55
    for i, (_, quarter) in enumerate(quarters):
        active = start <= i < target
        color = BLUE if active else GREEN if i == target else "#f0f1f2"
        ax.add_patch(Rectangle((left + i * step, y), width, 0.55,
                               facecolor=color, edgecolor="white"))
        ax.text(left + i * step + width / 2, y + 0.275, f"Q{quarter}",
                ha="center", va="center", fontsize=8,
                color="white" if active or i == target else "#888888")
    label(left + start * step, y - 0.32,
          "2017Q2–2020Q1: 12 quarters" if row == 0 else "2017Q3–2020Q2: 12 quarters",
          fontsize=8.5)
    label(13.8, y - 0.32, "Forecast 2020Q2" if row == 0 else "Forecast 2020Q3",
          ha="right", fontsize=8.5, weight="bold")
label(0, 5.48, "Only outcomes published before that forecast month enter estimation.", fontsize=8.5)

label(0, 4.65, "2  Within one forecast: two different information cutoffs", weight="bold", fontsize=10.5)
label(0, 4.08, "Example: a firm's 2020Q2 EPS is announced on 28 August 2020.", fontsize=8.5)
cards = [
    (0.05, "1 AUGUST", "Estimate the models", "Use eligible training outcomes\npublished before 1 August.\nThen hold parameters fixed.", BLUE),
    (4.80, "27 AUGUST", "Make the forecast", "Use EPS known by this day,\nincluding new August inputs.\nSave the EPS prediction.", BLUE),
    (9.55, "28 AUGUST", "Observe and score", "The target EPS is released.\nCompare it with the saved\nforecast to measure error.", GREEN),
]
for x, date, title, body, color in cards:
    ax.add_patch(FancyBboxPatch((x, 1.35), 4.3, 2.25,
                               boxstyle="round,pad=0.035,rounding_size=0.08",
                               facecolor="#f6f8fa", edgecolor="#d8dde3", linewidth=0.7))
    ax.text(x + 0.2, 3.23, date, color=color, fontsize=9, weight="bold", va="center")
    label(x + 0.2, 2.78, title, fontsize=9, weight="bold")
    label(x + 0.2, 2.02, body, fontsize=8, linespacing=1.5)
for x in [4.4, 9.15]:
    ax.annotate("", xy=(x + 0.35, 2.50), xytext=(x, 2.50),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
label(0, 0.70, "Out of sample: the target outcome is hidden until after its forecast is saved.",
      fontsize=9, weight="bold")
label(0, 0.20, "Repeat for each announcement; update estimation monthly and slide the window with q.",
      fontsize=8.5)
OUT.mkdir(parents=True, exist_ok=True)
for extension in ("pdf", "png"):
    fig.savefig(OUT / f"fig_rolling_oos_design.{extension}", dpi=200)
plt.close(fig)
