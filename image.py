import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
TERMS_PER_ROW   = 1      # top-level terms rendered per row (lower = taller but clearer)
FONT_SIZE       = 11     # pt  – increase if you want bigger glyphs
ROW_HEIGHT_IN   = 3.5    # inches per row
FIG_WIDTH_IN    = 36     # inches wide
DPI             = 200
OUTPUT_FILE     = "output.png"

# ── Read file ──────────────────────────────────────────────────────────────────
with open("output.txt", "r", encoding="utf-8") as f:
    latex_content = f.read().strip()


# ── Split at top-level + / - (respects brace / \left-\right depth) ────────────
def split_top_level(s: str) -> list[str]:
    """
    Split a LaTeX math string at + and - signs that are at the outermost
    nesting level (depth == 0), keeping the sign attached to the term that
    follows it.
    """
    depth   = 0
    parts   = []
    current = []
    i       = 0

    while i < len(s):
        # \left and \right each shift depth by 1
        if s[i:i+5] == r"\left":
            depth += 1
            current.append(s[i:i+5])
            i += 5
            continue
        if s[i:i+6] == r"\right":
            depth -= 1
            current.append(s[i:i+6])
            i += 6
            continue

        c = s[i]

        if c in "{(":
            depth += 1
            current.append(c)
        elif c in "})":
            depth -= 1
            current.append(c)
        elif c in "+-" and depth == 0 and current:
            # Flush current term, start a new one beginning with this sign
            parts.append("".join(current).strip())
            current = [c]
        else:
            current.append(c)

        i += 1

    if current:
        parts.append("".join(current).strip())

    return [p for p in parts if p]


parts = split_top_level(latex_content)
print(f"Expression split into {len(parts)} top-level term(s).")

# ── Group terms into rows ──────────────────────────────────────────────────────
rows = []
for i in range(0, len(parts), TERMS_PER_ROW):
    group = " ".join(parts[i : i + TERMS_PER_ROW])
    rows.append(r"$" + group + r"$")

n_rows     = len(rows)
fig_height = ROW_HEIGHT_IN * n_rows

print(f"Rendering {n_rows} row(s) → {FIG_WIDTH_IN}\" × {fig_height:.1f}\" @ {DPI} DPI")

# ── Draw ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
fig.patch.set_facecolor("white")

for idx, row_latex in enumerate(rows):
    # Evenly space rows from top to bottom
    y = 1.0 - (idx + 0.5) / n_rows
    fig.text(
        0.5, y,
        row_latex,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        transform=fig.transFigure,
    )

plt.savefig(
    OUTPUT_FILE,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.6,
)
print(f"Saved → {OUTPUT_FILE}")
plt.show()