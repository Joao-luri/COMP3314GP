"""Build COMP3314_Spectral_Slides_v3.pptx: nine questions, one worked six-point example.

Usage:  .venv/bin/python slides/build_deck_v3.py

Shares drawing helpers with build_deck.py (v2). Animated steps are separate
slides joined by Morph; matrix cells and points are named shapes ("!!...") so
Morph animates their numbers, colours and positions.
"""

from __future__ import annotations

import math as pymath
from pathlib import Path

import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import build_deck as B
import figures as F
from build_deck import (CARD, GREY, LIGHT, NAVY, ORANGE, RING, SLATE, TEAL, WARM, WHITE, Deck, Panel, card,
                        dot, formula, line, notes, picture, polyline, rect, ring, roadmap, takeaway, text)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "slides" / "assets"
OUT = ROOT / "COMP3314_Spectral_Slides_v3.pptx"

LIGHT_TEAL = (0xA8, 0xD8, 0xDC)


# ------------------------------------------------------------------ helpers


def _mix(c1, c2, t):
    return B.RGBColor(*(int(round(a + (b - a) * t)) for a, b in zip(c1, c2)))


def heat(v, vmax):
    t = min(max(v / vmax, 0.0), 1.0) if vmax > 0 else 0.0
    w, lt, tl, nv = (255, 255, 255), LIGHT_TEAL, (0x08, 0x7F, 0x8C), (0x15, 0x2C, 0x43)
    if t < 0.35:
        return _mix(w, lt, t / 0.35), False
    if t < 0.8:
        return _mix(lt, tl, (t - 0.35) / 0.45), t > 0.6
    return _mix(tl, nv, (t - 0.8) / 0.2), True


def diverge(v, vmax):
    t = min(abs(v) / vmax, 1.0) * 0.8
    end = (0x08, 0x7F, 0x8C) if v >= 0 else (0xE0, 0x7A, 0x1F)
    return _mix((255, 255, 255), end, t), t > 0.55


def cell(s, x, y, w, h, label, fill, dark_text, name, size=11, bold=False):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = LIGHT
    shp.line.width = Pt(0.5)
    shp.shadow.inherit = False
    shp.name = name
    tf = shp.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold = B.FONT, Pt(size), bold
    r.font.color.rgb = WHITE if dark_text else NAVY
    return shp


def matrix(s, M, x, y, cw, ch, key, show=None, mode="heat", vmax=None, row_labels=True, col_labels=None,
           size=11, fmt="{:.2f}"):
    """Draw M as a grid of named cells. show: boolean mask of cells that display a value."""
    M = np.atleast_2d(M)
    n, m = M.shape
    vmax = vmax or (np.abs(M).max() or 1.0)
    for i in range(n):
        for j in range(m):
            vis = True if show is None else bool(show[i, j])
            v = M[i, j]
            if vis:
                fill, dark = heat(v, vmax) if mode == "heat" else diverge(v, vmax)
                lab = fmt.format(0.0 if abs(v) < 0.005 else v)
            else:
                fill, dark, lab = WHITE, False, ""
            cell(s, x + j * cw, y + i * ch, cw, ch, lab, fill, dark, f"!!{key}_{i}{j}", size=size)
    if row_labels:
        for i in range(n):
            text(s, x - 0.32, y + i * ch + ch / 2 - 0.14, 0.26, 0.28, str(i + 1), size=10, color=SLATE,
                 bold=True, align=PP_ALIGN.RIGHT, name=f"!!{key}_r{i}")
    if col_labels:
        for j, lab in enumerate(col_labels):
            text(s, x + j * cw, y - 0.3, cw, 0.28, lab, size=10, color=SLATE, bold=True,
                 align=PP_ALIGN.CENTER, name=f"!!{key}_c{j}")


def cell_box(s, x, y, cw, ch, i0, j0, i1, j1, name, color=ORANGE, width=2.25):
    return rect(s, x + j0 * cw - 0.02, y + i0 * ch - 0.02, (j1 - j0 + 1) * cw + 0.04, (i1 - i0 + 1) * ch + 0.04,
                fill=None, line=color, radius=False, name=name, line_w=width)


class Mapper:
    """Equal-aspect map from data limits onto a slide box (centred)."""

    def __init__(self, x, y, w, h, xlim, ylim):
        sx, sy = w / (xlim[1] - xlim[0]), h / (ylim[1] - ylim[0])
        self.k = min(sx, sy)
        self.x0 = x + (w - self.k * (xlim[1] - xlim[0])) / 2 - self.k * xlim[0]
        self.y0 = y + (h - self.k * (ylim[1] - ylim[0])) / 2 + self.k * ylim[1]

    def xy(self, px, py):
        return self.x0 + self.k * px, self.y0 - self.k * py


def num_dot(s, cx, cy, label, color, name, d=0.3):
    shp = dot(s, cx, cy, d, color, name=name, line=WHITE)
    tf = shp.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold = B.FONT, Pt(10), True
    r.font.color.rgb = WHITE
    return shp


def arc90(s, pan, Y, groups, name="!!arc", label=True):
    angs = sorted(pymath.degrees(pymath.atan2(*Y[groups == c].mean(0)[::-1])) for c in (0, 1))
    rr = 0.3
    pts = [pan.xy(rr * np.cos(np.radians(t)), rr * np.sin(np.radians(t))) for t in np.linspace(*angs, 24)]
    polyline(s, pts, color=NAVY, width=1.5, name=name)
    if label:
        mid = np.radians(sum(angs) / 2)
        tx, ty = pan.xy(0.52 * np.cos(mid), 0.52 * np.sin(mid))
        diff = angs[1] - angs[0]
        text(s, tx - 0.35, ty - 0.17, 0.7, 0.34, f"{diff:.0f}°", size=13, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER, name=f"{name}l")


def unit_panel(s, x, y, size, lim=1.2, circle=True):
    pan = Panel(x, y, size, lim)
    pan.axes(s)
    pan.unit_circle(s, show=circle)
    return pan


# ------------------------------------------------------------------ slides


def s_title(deck):
    s = deck.prs.slides.add_slide(deck.layout)
    deck.page += 1
    text(s, 0.67, 0.9, 7.0, 0.8, "On Spectral Clustering", size=36, bold=True)
    text(s, 0.67, 1.65, 7.0, 0.6, "Analysis and an Algorithm", size=24)
    text(s, 0.67, 2.55, 6.5, 0.9, ["Andrew Y. Ng, Michael I. Jordan, Yair Weiss", "NIPS 2001"], size=15,
         color=SLATE)
    text(s, 0.67, 5.6, 6.5, 0.9, [[("COMP3314 Group Project", {"b": True, "c": TEAL})],
                                  "Group members:  ________________"], size=14, color=SLATE)
    picture(s, F.GEN / "moons_truth_km.png", 6.9, 1.3, w=5.9)
    text(s, 6.9, 3.75, 5.9, 0.35, "Our data: two moons. Left: the groups we want. Right: K-means.", size=10.5,
         color=SLATE, align=PP_ALIGN.CENTER)
    notes(s, "[~10 s] Title. Ng, Jordan and Weiss, NIPS 2001. We start straight from the example on the right.")


def s_q1(deck, D):
    m = D["moons_motivation"]
    s = deck.slide("Why does K-means get this wrong?", "Two moons, 400 points, k = 2. Same data in both panels.",
                   section=1)
    picture(s, D["img_moons_motivation"], 0.45, 1.65, w=8.3)
    card(s, 8.95, 1.75, 3.72, 2.35, "What K-means does",
         ["Each point joins the nearest centroid (✕).", "Two centroids give one straight boundary (dashed)."],
         size=13)
    card(s, 8.95, 4.25, 3.72, 1.95, "Why that fails",
         ["The moons interlock. Any straight line cuts through both of them."], size=13, fill=WARM,
         head_color=ORANGE)
    takeaway(s, f"K-means ARI = {F.fmt2(m['km'])} (1 = perfect, 0 = chance). These groups follow connectivity, "
                "not distance to a centre.")
    notes(s, """
[45 s] Here are two moons, 400 points we generated. On the left are the groups we would expect. On the right is
K-means with k = 2. K-means puts every point with its nearest centroid - the two crosses - so the boundary between the
two clusters is a straight line, the dashed one. The moons interlock, so any straight line cuts through both.
ARI is 0.27; ARI 1 would mean identical to the expected groups. The problem: these groups are defined by which points
are connected to which, not by distance to a centre.
""")


def s_q2(deck, D):
    s = deck.slide("What does the new method change?", "It changes the coordinates, not the clustering step.",
                   section=2)
    roadmap(s, -1)
    picture(s, D["img_moons_overview"], 0.55, 2.25, w=12.2)
    xs = [0.67 + i * 3.04 for i in range(4)]
    body = [("Raw points", "S:  n × 2"), ("Who is close to whom", "A:  n × n"),
            ("One row per point", "Y:  n × k"), ("Same K-means as before", "labels")]
    for x, (a, b) in zip(xs, body):
        rect(s, x, 5.2, 2.9, 0.75, fill=CARD)
        text(s, x, 5.25, 2.9, 0.7, [[(a, {"b": True, "s": 12.5})], [(b, {"s": 11.5, "c": SLATE})]],
             align=PP_ALIGN.CENTER, space_after=0)
    notes(s, """
[25 s] The method keeps K-means as its last step. What it changes is the representation. From the points we build a
similarity graph: who is close to whom. From the graph we compute new coordinates, one row per point. In those
coordinates each moon collapses to almost a single point, and K-means is trivial. The next three slides open up these
steps on a tiny example with six points, so that every number is visible.
""")


SIX_MAP = (0.67, 2.35, 5.0, 2.3, (-0.35, 3.55), (-0.3, 1.3))
A_BOX = (6.4, 2.6, 0.66, 0.5)


def six_scatter(s, six, edges="none", hl=None, colors=None):
    mp = Mapper(*SIX_MAP)
    P, A = six["P"], six["A"]
    if edges != "none":
        for i in range(6):
            for j in range(i + 1, 6):
                cross = six["group"][i] != six["group"][j]
                if A[i, j] < (0.03 if cross else 0.005) or (edges == "one" and (i, j) != hl):
                    continue
                col = ORANGE if cross else (TEAL if (i, j) == hl else SLATE)
                ln = line(s, *mp.xy(*P[i]), *mp.xy(*P[j]), color=col, width=0.75 + 9 * A[i, j], name=f"!!e{i}{j}")
                if (i, j) == hl:
                    ln.line.width = Pt(4)
    for i, (px, py) in enumerate(P):
        num_dot(s, *mp.xy(px, py), str(i + 1), colors[i] if colors else NAVY, name=f"!!q{i}")
    return mp


def s_q3(deck, D):
    six = D["six"]
    A = six["A"]
    x, y, cw, ch = A_BOX
    shows = [np.eye(6, dtype=bool), np.eye(6, dtype=bool), np.ones((6, 6), bool)]
    shows[1][0, 1] = shows[1][1, 0] = True
    side = [
        ["Six points, σ = 0.7.", "Entry Aᵢⱼ compares point i with point j.",
         "The diagonal is 0: a point is not its own neighbour."],
        [[("Points 1 and 2", {"b": True, "c": TEAL}), (f" are {six['dist12']:.2f} apart, so A₁₂ = "
                                                        f"{A[0, 1]:.2f}.", {})],
         "A is symmetric: A₂₁ = A₁₂."],
        ["Two blocks: {1, 2, 3} and {4, 5, 6}.",
         [("One weak link across: ", {}), (f"A₂₄ = {A[1, 3]:.2f}", {"b": True, "c": ORANGE}), (".", {})],
         "Line width in the graph = weight."],
    ]
    for f in range(3):
        s = deck.slide("How do points become a matrix?",
                       "Every pair of points gets a similarity weight between 0 and 1.",
                       section=2, new_page=f == 0, transition=True)
        roadmap(s, 1)
        six_scatter(s, six, edges=["none", "one", "all"][f], hl=(0, 1) if f == 1 else None)
        matrix(s, A, x, y, cw, ch, "M", show=shows[f], vmax=0.6,
               col_labels=[str(j + 1) for j in range(6)])
        text(s, x, y + 6 * ch + 0.05, 6 * cw, 0.3, "A  (6 × 6)", size=12, bold=True, align=PP_ALIGN.CENTER,
             name="!!Mname")
        if f == 1:
            cell_box(s, x, y, cw, ch, 0, 1, 0, 1, "!!hl1", color=TEAL)
            cell_box(s, x, y, cw, ch, 1, 0, 1, 0, "!!hl2", color=TEAL)
        if f == 2:
            cell_box(s, x, y, cw, ch, 1, 3, 1, 3, "!!hl1")
            cell_box(s, x, y, cw, ch, 3, 1, 3, 1, "!!hl2")
        if f == 1:
            formula(s, rf"$A_{{12}}=\exp\left(-\frac{{{six['dist12']:.2f}^2}}{{2\cdot 0.7^2}}\right)={A[0, 1]:.2f}$",
                    "q3b", 0.85, 5.05, size=19, name="!!f3")
        else:
            formula(s, r"$A_{ij}=\exp\left(-\frac{\|s_i-s_j\|^2}{2\sigma^2}\right),\ \ A_{ii}=0$", "q3a", 0.85, 5.05,
                    size=19, name="!!f3")
        text(s, 10.75, 2.55, 1.95, 3.6, side[f], size=12, space_after=9, name="!!side")
        if f == 2:
            takeaway(s, "Rows are numbered by group only so the blocks are easy to see. NJW never uses that order.")
        notes(s, [
            """
[65 s, two clicks] To see every number, take six points, sigma 0.7. A is a 6 by 6 table: entry i, j says how similar
point i and point j are. The diagonal is zero.
""",
            """
(click) Points 1 and 2 are 0.85 apart. Plug into the Gaussian: exp of minus 0.85 squared over 2 sigma squared gives
0.47. A is symmetric, so the same number goes in row 2, column 1.
""",
            """
(click) Fill in every pair. Within each group the weights are around 0.4 to 0.6; between the groups they are almost zero,
except one weak link, 0.07 between points 2 and 4. So A has two blocks. That block structure is what the rest of the
algorithm turns into coordinates.
""",
        ][f])


def s_q4(deck, D):
    six = D["six"]
    A, L, d, X, ev = six["A"], six["L"], six["d"], six["X"], six["evals"]
    mx, my, cw, ch = 1.05, 2.6, 0.66, 0.5
    for f in range(3):
        s = deck.slide("How do we get new coordinates?", "A → D → L → X, on the same six points.",
                       section=2, new_page=f == 0, transition=True)
        roadmap(s, 2 if f < 2 else 3)
        M = A if f == 0 else L
        matrix(s, M, mx, my, cw, ch, "M", vmax=0.6, col_labels=[str(j + 1) for j in range(6)])
        text(s, mx, my + 6 * ch + 0.05, 6 * cw, 0.3, "A  (6 × 6)" if f == 0 else "L  (6 × 6)", size=12, bold=True,
             align=PP_ALIGN.CENTER, name="!!Mname")
        if f < 2:
            dx = mx + 6 * cw + 0.3
            matrix(s, d[:, None], dx, my, 0.7, ch, "d", vmax=1.3, row_labels=False, col_labels=["d"])
            text(s, dx - 0.2, my + 6 * ch + 0.05, 1.1, 0.3, "degrees", size=11, color=SLATE, align=PP_ALIGN.CENTER,
                 name="!!dname")
        tx = 6.3
        if f == 0:
            formula(s, r"$d_i=\sum_j A_{ij}$", "q4a", tx, 2.35, size=20, name="!!f4a")
            text(s, tx, 3.15, 6.3, 2.5,
                 ["dᵢ is the row sum of A: how strongly point i is connected overall.",
                  f"Point 2 has the largest degree, d₂ = {d[1]:.2f}: it also touches the other group.",
                  "D is the diagonal matrix with d on the diagonal."], size=13, space_after=9, name="!!body")
        elif f == 1:
            formula(s, r"$L=D^{-1/2}AD^{-1/2},\qquad L_{ij}=\frac{A_{ij}}{\sqrt{d_i\,d_j}}$", "q4b", tx, 2.35,
                    size=20, name="!!f4a")
            formula(s, rf"$L_{{12}}=\frac{{{A[0, 1]:.2f}}}{{\sqrt{{{d[0]:.2f}\cdot{d[1]:.2f}}}}}={L[0, 1]:.2f}$",
                    "q4c", tx, 3.25, size=18, name="!!f4c")
            text(s, tx, 4.25, 6.3, 1.9,
                 ["Each weight is divided by how connected its two endpoints are.",
                  "L is still 6 × 6, symmetric, with the same two blocks."], size=13, space_after=9, name="!!body")
        else:
            text(s, tx, 2.25, 4.6, 0.3, "Eigenvalues of L", size=12, bold=True, color=SLATE, name="!!evl")
            for j, v in enumerate(ev):
                keep = j < 2
                cell(s, tx + j * 0.74, 2.6, 0.7, 0.5, f"{v:.2f}", TEAL if keep else CARD, keep, f"!!ev{j}", size=11,
                     bold=keep)
            text(s, tx, 3.15, 4.5, 0.3, "keep the k = 2 largest  →  their eigenvectors v₁, v₂", size=11,
                 color=TEAL, name="!!evk")
            xx = 11.35
            matrix(s, X, xx, my, 0.62, ch, "X", mode="div", vmax=0.6, col_labels=["v₁", "v₂"])
            text(s, xx, my + 6 * ch + 0.05, 1.24, 0.3, "X  (6 × 2)", size=12, bold=True, align=PP_ALIGN.CENTER,
                 name="!!Xname")
            cell_box(s, xx, my, 0.62, ch, 1, 0, 1, 1, "!!xhl")
            text(s, tx, 3.75, 4.75, 2.4,
                 [[("Row i of X = new coordinates of point i.", {"b": True})],
                  [("Point 2 → ", {}), (f"({X[1, 0]:.2f}, {X[1, 1]:.2f})", {"b": True, "c": ORANGE}),
                   (".", {})],
                  "v₂ is positive on {1, 2, 3} and negative on {4, 5, 6}: it already separates the groups."],
                 size=13, space_after=9, name="!!body")
        notes(s, [
            """
[70 s, two clicks] Step two. The degree d_i is the row sum of A: how connected point i is overall. Point 2 has the
largest degree because it also touches the other group.
""",
            """
(click) L divides each weight by the square root of the two degrees. For example L12 = 0.47 over the square root of
0.88 times 1.09, which is 0.49. L is still 6 by 6 and keeps the two blocks.
""",
            """
(click) Now the eigenvalues of L: 1.00 and 0.94, then a big drop to negative values. We keep the k = 2 largest and put
their eigenvectors side by side: that is X, 6 by 2. Read it by rows: row i is the new coordinate of point i. Point 2 is
now (0.42, 0.42). Look at the second column: positive on the first group, negative on the second. The groups are already
separated, but the rows still have different lengths.
""",
        ][f])


def s_q5(deck, D):
    six = D["six"]
    X, nrm, Y, g, lab = six["X"], six["norms"], six["Y"], six["group"], six["labels"]
    ty, ch, cw = 2.6, 0.5, 0.72
    xX, xN, xY = 1.05, 2.75, 3.85
    caps = [
        "Rows of X as points: group {1, 2, 3} and group {4, 5, 6} point in two directions, at different lengths.",
        "‖xᵢ‖ is the length of row i. It differs inside a group, so it carries no group information.",
        "yᵢ = xᵢ / ‖xᵢ‖: keep the direction, set the length to 1. Each group collapses to one point, 90° apart.",
        "K-means on the rows of Y now has an easy job: {1, 2, 3} and {4, 5, 6}.",
    ]
    for f in range(4):
        s = deck.slide("Why normalize the rows?", "X → Y. The numbers on the left and the points on the right "
                       "change together.", section=2, new_page=f == 0, transition=True)
        roadmap(s, 4 if f < 3 else 5)
        matrix(s, X, xX, ty, cw, ch, "X", mode="div", vmax=0.6, col_labels=["x₁", "x₂"])
        matrix(s, nrm[:, None], xN, ty, cw, ch, "N", show=np.full((6, 1), f >= 1), vmax=1.0, row_labels=False,
               col_labels=["‖xᵢ‖"])
        matrix(s, Y, xY, ty, cw, ch, "Y", mode="div", vmax=1.0, show=np.full((6, 2), f >= 2), row_labels=False,
               col_labels=["y₁", "y₂"])
        for name, x0, w in [("X", xX, 2 * cw), ("÷", xN, cw), ("= Y", xY, 2 * cw)]:
            text(s, x0, ty + 6 * ch + 0.05, w, 0.3, name, size=12, bold=True, align=PP_ALIGN.CENTER,
                 name=f"!!tn{name}")
        if f == 3:
            for i in range(6):
                rect(s, xX - 0.42, ty + i * ch + 0.08, 0.14, ch - 0.16, fill=RING[lab[i]], radius=False,
                     name=f"!!rowc{i}")
        text(s, 0.67, 6.05, 6.0, 0.6, caps[f], size=12, name="!!cap")
        pan = unit_panel(s, 7.05, 2.3, 3.9, lim=1.15)
        pts = X if f < 2 else Y
        if f >= 1:
            B.rays(s, pan, [Y[g == c].mean(0) for c in (0, 1)])
        if f >= 2:
            arc90(s, pan, Y, g)
        for i, (px, py) in enumerate(pts):
            num_dot(s, *pan.xy(px, py), str(i + 1), RING[lab[i]] if f == 3 else NAVY, name=f"!!q{i}", d=0.28)
        side = [
            ["True coordinates, no rescaling.", "Unit circle for reference."],
            ["Dashed: the two directions."],
            ["Points slide along their direction onto the circle."],
            ["K-means labels:", [("teal = {1, 2, 3}", {"c": TEAL, "b": True})],
             [("orange = {4, 5, 6}", {"c": ORANGE, "b": True})]],
        ][f]
        text(s, 11.1, 2.5, 1.6, 3.5, side, size=11.5, space_after=8, name="!!side")
        notes(s, [
            """
[85 s, three clicks] Why the extra normalization? On the left is X; on the right each row of X is drawn as a point,
with the true coordinates. The two groups already point in two different directions, but at different lengths.
""",
            """
(click) Compute the length of each row. The lengths differ inside a group - 0.56 to 0.60 here, and much more on real
data - so the length says nothing about the group. It only adds noise for K-means.
""",
            """
(click) Divide each row by its length. The numbers become Y, and every point slides along its direction onto the unit
circle. Each group collapses to one point, and the two points are 90 degrees apart. The next slide explains why 90.
""",
            """
(click) K-means on these rows is trivial: two far-apart dots. The labels go back to the six original points.
""",
        ][f])


def s_q6(deck, D):
    six = D["six"]
    g = six["group"]
    frames = six["cross_frames"]
    mx, my, cw, ch = 1.05, 2.6, 0.6, 0.46
    caps = [
        "Ideal: no links across groups. L is block diagonal, λ₁ = λ₂ = 1, and Y is exactly two orthogonal points.",
        "Our actual weak link (A₂₄ = 0.07): λ₂ drops to {l2:.2f}. Y barely moves.",
        "Links ×10 (A₂₄ = {a24:.2f}, as strong as inside a group): λ₂ = {l2:.2f}. Points 2 and 4 are pulled toward "
        "each other.",
    ]
    ideal = frames[0]["Y"]
    for f, fr in enumerate(frames):
        s = deck.slide("Why does this recover the clusters?", "Start from a disconnected graph, then add links "
                       "across the groups.", section=2, new_page=f == 0, transition=True)
        matrix(s, fr["A"], mx, my, cw, ch, "M", vmax=0.6, col_labels=[str(j + 1) for j in range(6)], size=10)
        text(s, mx, my + 6 * ch + 0.05, 6 * cw, 0.3, "A", size=12, bold=True, align=PP_ALIGN.CENTER, name="!!Mname")
        cell_box(s, mx, my, cw, ch, 0, 3, 2, 5, "!!xb1")
        cell_box(s, mx, my, cw, ch, 3, 0, 5, 2, "!!xb2")
        ev = fr["evals"]
        text(s, 5.0, 2.6, 1.9, 2.2,
             [[("λ₁ = ", {}), (f"{ev[0]:.2f}", {"b": True})],
              [("λ₂ = ", {}), (f"{ev[1]:.2f}", {"b": True, "c": TEAL if ev[1] > 0.99 else ORANGE})],
              [("λ₃ = ", {}), (f"{ev[2]:.2f}", {"b": True})],
              [("gap λ₂ − λ₃ = ", {"s": 11, "c": SLATE}), (f"{ev[1] - ev[2]:.2f}", {"s": 11, "b": True})]],
             size=14, space_after=8, name="!!ev")
        pan = unit_panel(s, 7.2, 2.3, 3.7, lim=1.15)
        B.rays(s, pan, [ideal[g == c].mean(0) for c in (0, 1)])
        Y = fr["Y"]
        for i, (px, py) in enumerate(Y):
            num_dot(s, *pan.xy(px, py), str(i + 1), RING[g[i]], name=f"!!q{i}", d=0.28)
        text(s, 11.1, 2.5, 1.6, 2.5, ["Dashed: the ideal directions.", "Colour = group."], size=11.5,
             space_after=8, name="!!side")
        text(s, 0.67, 5.78, 6.3, 0.6, caps[f].format(l2=ev[1], a24=fr["A"][1, 3]), size=11.5, name="!!cap")
        takeaway(s, "Theorem 2 (paper): if the gap is large compared with the links across groups, Y stays close to "
                    "k orthogonal points.")
        notes(s, [
            """
[50 s, two clicks] Why does this work? Start from the ideal case: delete every link between the groups. Then L is block
diagonal, the top two eigenvalues are both exactly 1, and each group lands on exactly one point - two orthogonal points.
""",
            """
(click) Now our actual data, with the weak link of 0.07. Lambda 2 drops slightly to 0.94, and Y barely moves.
""",
            """
(click) Make the cross links ten times stronger - as strong as the links inside a group. Lambda 2 falls, the gap
shrinks, and points 2 and 4, which carry the link, are pulled toward each other. This is the paper's Theorem 2 in a
picture: small links and a large gap keep Y near the ideal; large links push it away. This is also why sigma matters.
""",
        ][f])


def s_q7(deck, D):
    pairs = D["pairs"]
    st, pl = D["stage1"], D["paper_like"]
    s = deck.slide("Did we see the same effect?", "Original paper (pictures only) next to our implementation "
                   "on data we rebuilt.", section=3)
    text(s, 2.0, 1.55, 3.6, 0.3, "Original paper, Figure 1", size=12.5, bold=True, color=SLATE,
         align=PP_ALIGN.CENTER)
    text(s, 5.95, 1.55, 3.6, 0.3, "Our reproduction (ARI)", size=12.5, bold=True, color=TEAL,
         align=PP_ALIGN.CENTER)
    line(s, 5.78, 1.6, 5.78, 6.6, color=LIGHT, width=1.0)
    w = 1.7
    text(s, 0.67, 2.6, 1.3, 0.8, [[("Two circles", {"b": True})], "k = 2"], size=12.5)
    picture(s, ASSETS / "paper_fig1e_njw.png", 2.0, 1.95, w=w)
    picture(s, ASSETS / "paper_fig1i_kmeans.png", 3.85, 1.95, w=w)
    picture(s, pairs["ours_tc_njw"][0], 6.0, 2.0, w=w)
    picture(s, pairs["ours_tc_km"][0], 7.85, 2.0, w=w)
    for x, lab, col in [(2.0, "NJW", NAVY), (3.85, "K-means", NAVY),
                        (6.0, f"NJW  {F.fmt2(pairs['ours_tc_njw'][1])}", TEAL),
                        (7.85, f"K-means  {F.fmt2(pairs['ours_tc_km'][1])}", ORANGE)]:
        text(s, x, 3.85, w, 0.3, lab, size=11.5, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, 0.67, 4.85, 1.3, 0.8, [[("Joined rings", {"b": True})], "k = 3"], size=12.5)
    picture(s, ASSETS / "paper_fig1_left6.png", 2.0, 4.3, w=w)
    picture(s, ASSETS / "paper_fig1_right6.png", 3.85, 4.3, w=w)
    picture(s, pairs["ours_joined_njw"][0], 6.0, 4.35, w=w)
    joined_ari = pairs["ours_joined_njw"][1]
    for x, lab, col in [(2.0, "NJW", NAVY), (3.85, "Components", NAVY),
                        (6.0, f"NJW  {F.fmt2(joined_ari)}", TEAL if joined_ari > 0.99 else ORANGE)]:
        text(s, x, 6.2, w, 0.3, lab, size=11.5, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, 7.85, 4.7, 1.75, 1.2, "Three rings recovered. K-means still fails on this shape.", size=12,
         color=TEAL)
    card(s, 9.85, 1.6, 2.82, 4.75, "All our numbers",
         [[("K-means → NJW", {"s": 11, "c": SLATE})],
          f"Circles  {F.fmt2(st['circles']['km'])} → {F.fmt2(st['circles']['njw'])}",
          f"Moons  {F.fmt2(st['moons']['km'])} → {F.fmt2(st['moons']['njw'])}",
          f"Blobs  {F.fmt2(st['blobs']['km'])} → {F.fmt2(st['blobs']['njw'])}",
          [("Figure 1 settings: ", {"b": True}),
           (f"NJW 1.00 on {sum(v['njw'] > 0.99 for v in pl.values())} of {len(pl)}.", {})],
          [("The paper reports no metric and released no data.", {"s": 11, "c": SLATE})]],
         size=12)
    notes(s, """
[65 s] Did we see the same thing? Left, the paper's Figure 1; right, our own implementation on data we rebuilt - the
paper released neither the points nor sigma, so this is a qualitative comparison, and the ARI numbers are ours.
Top row, two circles: the paper shows NJW succeeding and K-means failing; we get exactly that, 1.00 against 0.00.
Bottom row, three rings joined by short bridges: NJW recovers the three rings here too. One sentence, then move on.
Across everything we ran: NJW beats K-means on circles and moons, ties on blobs, and gets 1.00 on the rebuilt
Figure 1 settings.
""")


def s_q8(deck, D):
    sw = D["moons_sweep"]
    P, keep = sw["P"], sw["keep"]
    br, wk = D["bridge"], D["wrong_k"]
    mp = Mapper(0.67, 2.0, 5.6, 2.45, (-1.25, 2.25), (-0.75, 1.25))
    gx, gy, gw, gh = 1.15, 4.95, 4.7, 1.05

    def g(r, a):
        lx = (pymath.log(r) - pymath.log(0.25)) / (pymath.log(8) - pymath.log(0.25))
        return gx + lx * gw, gy + (1.0 - (a + 0.05) / 1.1) * gh

    expl = {1.0: "Our default σ₀.", 2.8: "Bigger σ = stronger links across the moons (previous slide).",
            8.0: "Links everywhere: the graph no longer shows two groups."}
    for f, fr in enumerate(sw["frames"]):
        s = deck.slide("When does it fail, and why?", "Scale σ, noise, and a wrong-looking answer with the "
                       "right k.", section=4, new_page=f == 0, transition=True)
        ari = fr["ari"]
        text(s, 0.67, 1.55, 5.6, 0.4,
             [[(f"Moons, σ = {fr['r']:g} σ₀   ", {"b": True}),
               (f"ARI = {F.fmt2(ari)}", {"b": True, "c": TEAL if ari > 0.99 else ORANGE})]],
             size=14, name="!!sv")
        for i in keep:
            dot(s, *mp.xy(*P[i]), 0.075, RING[fr["labels"][i]], name=f"!!m{i:03d}")
        for name, col, wdt in [("circles", GREY, 1.25), ("moons", TEAL, 2.25)]:
            cv = polyline(s, [g(r, a) for r, a in sw["curve"][name]], color=col, width=wdt, name=f"!!cv_{name}")
            if name == "circles":
                cv.line.dash_style = B.MSO_LINE_DASH_STYLE.DASH
        line(s, gx, gy + gh, gx + gw, gy + gh, color=SLATE, width=0.75, name="!!gx")
        line(s, gx, gy, gx, gy + gh, color=SLATE, width=0.75, name="!!gy")
        for r in (0.25, 1, 4):
            px, _ = g(r, 0)
            text(s, px - 0.3, gy + gh + 0.02, 0.6, 0.25, f"{r:g}", size=9, color=SLATE, align=PP_ALIGN.CENTER,
                 name=f"!!gt{r:g}")
        text(s, gx + gw - 1.6, gy + gh + 0.02, 1.6, 0.25, "σ / σ₀", size=9, color=SLATE, align=PP_ALIGN.RIGHT,
             name="!!gxl")
        text(s, gx - 0.75, gy - 0.05, 0.7, 0.25, "ARI 1", size=9, color=SLATE, align=PP_ALIGN.RIGHT, name="!!gy1")
        text(s, gx + 0.1, gy - 0.3, 2.5, 0.25, "moons (solid), circles (dashed)", size=9, color=SLATE,
             name="!!gleg")
        dot(s, *g(fr["r"], ari), 0.2, ORANGE, name="!!mark", line=WHITE)
        text(s, 0.67, 6.3, 5.6, 0.3, expl[fr["r"]], size=11.5, color=NAVY, name="!!expl")
        rect(s, 6.6, 1.6, 6.07, 2.35, fill=CARD)
        text(s, 6.78, 1.68, 5.8, 0.35, "Noise: a bridge of 80 points between the moons", size=13, bold=True,
             color=TEAL)
        picture(s, D["img_bridge"], 6.75, 2.08, w=4.2)
        text(s, 11.0, 2.2, 1.6, 1.7, [f"NJW {F.fmt2(br['njw'])}", f"Components {F.fmt2(br['cc'])}",
                                      [("Weights, not a hard threshold.", {"s": 10.5, "c": SLATE})]],
             size=11.5, space_after=6)
        rect(s, 6.6, 4.1, 6.07, 2.3, fill=WARM)
        text(s, 6.78, 4.18, 5.8, 0.35, f"A failure: three rings, true k = 3 → ARI {F.fmt2(wk[(3, 1.0)])} at σ₀", size=13, bold=True, color=ORANGE)
        picture(s, D["img_wrong_k"], 6.72, 4.55, w=5.85)
        notes(s, [
            """
[95 s, two clicks] When does it fail? First, the scale sigma. Here are the moons at our default sigma: perfect.
""",
            """
(click) Make sigma 2.8 times larger. That is exactly the stronger cross links from the previous slide: the moons start to
mix and ARI drops to 0.81. On circles - the dashed curve - it collapses to zero already at 2 sigma-zero.
""",
            """
(click) At 8 times, links are everywhere, the graph no longer has two groups, and ARI is 0.30. One global sigma, and
nothing in the method tells you the safe window.
Top right, noise: an 80-point bridge between the moons. NJW still scores 1.00, while connected components, a hard
threshold, falls to 0 - one chain of points is enough to merge them. So it is robust to sparse noise.
Bottom right, a failure even with the right k: three rings, k = 3, ARI 0.52 at our default sigma - the outer rings mix.
At 0.7 sigma-zero it reaches 0.90. k must be given by hand, and even the right k does not fix a wrong sigma.
""",
        ][f])


def s_q9(deck, D):
    m = D["moons_final"]
    asm = D["assumptions"]["moons"]
    s = deck.slide("What to remember", "Back to the opening example: same points, same k.", section=5)
    picture(s, D["img_moons_final"], 0.45, 1.55, w=7.9)
    card(s, 8.6, 1.6, 4.07, 2.25, "Contribution",
         ["A simple algorithm: graph → top-k eigenvectors → unit rows → K-means.",
          "An explanation: groups become orthogonal directions; weak links only perturb them."], size=12)
    card(s, 8.6, 4.0, 4.07, 2.25, "Limits",
         ["One global σ, and k given by hand.",
          f"Theorem 2's condition fails on these moons (gap {float(asm['delta']):.4f} < "
          f"{float(asm['required_delta_(2+sqrt2)eps']):.4f}), yet NJW is perfect: the theory is sufficient, "
          "not necessary."], size=12, fill=WARM, head_color=ORANGE)
    takeaway(s, f"K-means {F.fmt2(m['km'])}  →  NJW {F.fmt2(m['njw'])}. Not a new clustering step: a better "
                "representation for the old one.")
    notes(s, """
[40 s] Back to where we started: the same moons, the same k. K-means 0.27, NJW 1.00. What to remember: NJW is not a new
clustering step, it is a better representation for the old one. Its contribution is a simple algorithm and an
explanation: each group becomes one direction, and weak links only perturb it. Its limits: one global sigma, k chosen by
hand, and a theory whose condition does not even hold on these moons - it is sufficient, not necessary. Thank you.
""")


def s_backup_theory(deck, D):
    asm = D["assumptions"]
    s = deck.slide("Backup · Theorem 2 on our runs", "True clusters used only as the blocks in the proof.", section=0)
    rows = [("Dataset", "gap δ", "needs > (2+√2)ε", "Condition", "NJW ARI", "Bound vs observed")]
    for name in ("circles", "moons", "blobs"):
        r = asm[name]
        holds = r["theorem2_condition_holds"] == "True"
        bound = (f"{float(r['theorem2_bound']):.1e} vs {float(r['observed_mean_sq_dist_to_orthogonal_centers']):.1e}"
                 if holds else "-")
        need = float(r["required_delta_(2+sqrt2)eps"])
        rows.append((name, f"{float(r['delta']):.4f}", f"{need:.4f}" if need >= 1e-3 else f"{need:.1e}",
                     "holds" if holds else "fails", "1.00", bound))
    B._table(s, rows, 0.67, 1.6, [1.7, 1.5, 2.3, 1.6, 1.6, 3.3], size=12.5)
    text(s, 0.67, 4.0, 12.0, 1.5,
         ["Thin shapes (rings, moons) mix slowly inside a cluster, so their own λ₂ is close to 1 and the gap is tiny.",
          "Where the condition holds (blobs), the bound is about ten orders of magnitude loose."], size=13,
         space_after=8)


def main():
    D = F.build_v3()
    deck = Deck()
    s_title(deck)
    s_q1(deck, D)
    s_q2(deck, D)
    s_q3(deck, D)
    s_q4(deck, D)
    s_q5(deck, D)
    s_q6(deck, D)
    s_q7(deck, D)
    s_q8(deck, D)
    s_q9(deck, D)
    s_backup_theory(deck, D)
    B.s_backup(deck, D)
    deck.prs.save(OUT)
    print(f"wrote {OUT}  ({len(deck.prs.slides._sldIdLst)} slides)")
    six = D["six"]
    print("six evals", np.round(six["evals"], 3), "cross frames",
          [(fr["factor"], round(float(fr["evals"][1]), 3)) for fr in six["cross_frames"]])


if __name__ == "__main__":
    main()
