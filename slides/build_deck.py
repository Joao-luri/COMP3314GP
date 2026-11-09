"""Build COMP3314_Spectral_Slides_v2.pptx.

Usage:  .venv/bin/python slides/build_deck.py

Animated steps are separate slides joined by PowerPoint's Morph transition.
Shapes that should move between two slides share a name starting with "!!",
which tells Morph to pair them. Morph needs PowerPoint 2019 / Microsoft 365;
older viewers fall back to a fade.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

import figures as F

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "slides" / "assets"
OUT = ROOT / "COMP3314_Spectral_Slides_v2.pptx"

FONT = "Verdana"
NAVY = RGBColor(0x15, 0x2C, 0x43)
TEAL = RGBColor(0x08, 0x7F, 0x8C)
SLATE = RGBColor(0x52, 0x62, 0x73)
ORANGE = RGBColor(0xE0, 0x7A, 0x1F)
GREY = RGBColor(0x9A, 0xA6, 0xB2)
LIGHT = RGBColor(0xC9, 0xD2, 0xDA)
CARD = RGBColor(0xE8, 0xF0, 0xF3)
WARM = RGBColor(0xF7, 0xF1, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RING = [TEAL, ORANGE, RGBColor(0x7A, 0x51, 0x95)]

SECTIONS = {
    1: "1  MOTIVATION",
    2: "2  THE ALGORITHM",
    3: "3  OUR REPRODUCTION",
    4: "4  CRITICAL ANALYSIS",
    5: "5  SUMMARY",
    0: "BACKUP",
}


# ------------------------------------------------------------------ primitives


def _rgb(c):
    if isinstance(c, RGBColor):
        return c
    c = c.lstrip("#")
    return RGBColor(int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def text(slide, x, y, w, h, paras, size=16, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, spacing=1.1, space_after=4):
    """paras: list of paragraphs; a paragraph is a str or a list of (text, style) runs.

    style keys: b (bold), i (italic), c (color), s (size).
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.02))
    if isinstance(paras, str):
        paras = [paras]
    for k, para in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, st in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(st.get("s", size))
            f.bold = st.get("b", bold)
            f.italic = st.get("i", False)
            f.color.rgb = _rgb(st.get("c", color))
    return tb


def rect(slide, x, y, w, h, fill=CARD, line=None, radius=True, name=None, line_w=1.0):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shp.adjustments[0] = 0.08
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = _rgb(line)
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def dot(slide, cx, cy, d, color, name=None, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2), Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(color)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = _rgb(line)
        shp.line.width = Pt(1.25)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def ring(slide, cx, cy, r, color=GREY, name=None, width=1.0, dash=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    shp.fill.background()
    shp.line.color.rgb = _rgb(color)
    shp.line.width = Pt(width)
    if dash:
        shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def line(slide, x1, y1, x2, y2, color=GREY, width=1.0, name=None, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = _rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if name:
        c.name = name
    return c


def polyline(slide, pts_in, color=NAVY, width=1.5, name=None):
    fb = slide.shapes.build_freeform(Inches(pts_in[0][0]), Inches(pts_in[0][1]), scale=1.0)
    fb.add_line_segments([(Inches(x), Inches(y)) for x, y in pts_in[1:]], close=False)
    shp = fb.convert_to_shape()
    shp.fill.background()
    shp.line.color.rgb = _rgb(color)
    shp.line.width = Pt(width)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def picture(slide, path, x, y, w=None, h=None, name=None):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    if name:
        pic.name = name
    return pic


def formula(slide, tex, key, x, y, size=20, color="#152C43", scale=1.0, name=None):
    path, w, h = F.math_png(tex, f"math_{key}", size=size, color=color)
    return picture(slide, path, x, y, w=w * scale, name=name)


def notes(slide, body: str):
    slide.notes_slide.notes_text_frame.text = body.strip()


def morph(slide, duration_ms=1400):
    """Add a Morph transition (fade fallback) when entering this slide."""
    p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    xml = (
        '<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
        '<mc:Choice xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main" Requires="p159">'
        f'<p:transition xmlns:p="{p}" xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
        f'spd="slow" p14:dur="{duration_ms}"><p159:morph option="byObject"/></p:transition>'
        "</mc:Choice>"
        f'<mc:Fallback><p:transition xmlns:p="{p}" spd="slow"><p:fade/></p:transition></mc:Fallback>'
        "</mc:AlternateContent>"
    )
    el = etree.fromstring(xml)
    sld = slide._element
    anchor = sld.find(qn("p:clrMapOvr"))
    if anchor is None:
        anchor = sld.find(qn("p:cSld"))
    anchor.addnext(el)


# ------------------------------------------------------------------ slide chrome


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.layout = self.prs.slide_layouts[6]
        self.page = 0

    def slide(self, title, subtitle=None, section=None, new_page=True, transition=False):
        s = self.prs.slides.add_slide(self.layout)
        if new_page:
            self.page += 1
        text(s, 0.67, 0.36, 9.6, 0.7, title, size=28, bold=True, name="!!title")
        if subtitle:
            text(s, 0.67, 1.0, 11.9, 0.45, subtitle, size=14, color=SLATE, name="!!subtitle")
        if section is not None:
            text(s, 9.9, 0.48, 2.77, 0.35, SECTIONS.get(section, section), size=10.5, color=TEAL, bold=True,
                 align=PP_ALIGN.RIGHT, name="!!section")
        text(s, 12.07, 6.98, 0.6, 0.3, f"{self.page:02d}" if section != 0 else "B",
             size=10.5, color=SLATE, align=PP_ALIGN.RIGHT, name="!!page")
        if transition:
            morph(s)
        return s

    def save(self):
        self.prs.save(OUT)


ROADMAP = [
    ("Points", "S:  n × 2"),
    ("Affinity", "A:  n × n"),
    ("Normalize", "L:  n × n"),
    ("Eigenvectors", "X:  n × k"),
    ("Unit rows", "Y:  n × k"),
    ("K-means", "labels"),
]


def roadmap(s, active):
    x0, y, w, h, gap = 0.67, 1.5, 1.86, 0.56, 0.17
    for i, (name, shape) in enumerate(ROADMAP):
        x = x0 + i * (w + gap)
        on = i == active
        done = i < active
        rect(s, x, y, w, h, fill=TEAL if on else CARD, name=f"!!rm{i}")
        text(s, x, y + 0.04, w, h - 0.06,
             [[(name, {"b": True, "s": 11.5, "c": WHITE if on else (NAVY if done else SLATE)})],
              [(shape, {"s": 10, "c": WHITE if on else SLATE})]],
             align=PP_ALIGN.CENTER, space_after=0, spacing=1.0, name=f"!!rmt{i}")
        if i < len(ROADMAP) - 1:
            text(s, x + w - 0.02, y + 0.1, gap + 0.04, 0.35, "›", size=16, color=GREY,
                 align=PP_ALIGN.CENTER, name=f"!!rma{i}")


def card(s, x, y, w, h, head, body, fill=CARD, head_color=TEAL, size=13.5, name=None, body_color=NAVY):
    rect(s, x, y, w, h, fill=fill, name=f"!!{name}_bg" if name else None)
    paras = [[(head, {"b": True, "c": head_color, "s": size + 1})]]
    paras += [[(b, {"c": body_color})] if isinstance(b, str) else b for b in body]
    text(s, x + 0.18, y + 0.12, w - 0.36, h - 0.2, paras, size=size, space_after=5,
         name=f"!!{name}_tx" if name else None)


def takeaway(s, msg, y=6.38, name="!!take"):
    rect(s, 0.67, y, 12.0, 0.5, fill=WARM, name=f"{name}_bg")
    text(s, 0.85, y + 0.04, 11.7, 0.42, msg, size=13.5, color=NAVY, anchor=MSO_ANCHOR.MIDDLE,
         name=f"{name}_tx")


# ------------------------------------------------------------------ scatter panel


class Panel:
    """Square data window mapped onto a slide box; points become named ovals."""

    def __init__(self, x, y, size, lim):
        self.x, self.y, self.size, self.lim = x, y, size, lim

    def xy(self, px, py):
        L = self.lim
        return (self.x + (px + L) / (2 * L) * self.size, self.y + (L - py) / (2 * L) * self.size)

    def axes(self, s, color=LIGHT):
        cx, cy = self.xy(0, 0)
        line(s, self.x, cy, self.x + self.size, cy, color=color, width=0.75, name="!!axx")
        line(s, cx, self.y, cx, self.y + self.size, color=color, width=0.75, name="!!axy")

    def unit_circle(self, s, show=True):
        cx, cy = self.xy(0, 0)
        r = self.size / (2 * self.lim)
        ring(s, cx, cy, r, color=GREY if show else WHITE, name="!!ucirc", width=1.0, dash=True)

    def points(self, s, P, colors, d=0.1, prefix="p"):
        for i, (px, py) in enumerate(P):
            cx, cy = self.xy(px, py)
            dot(s, cx, cy, d, colors[i], name=f"!!{prefix}{i:03d}")


# ------------------------------------------------------------------ slides


def s_title(deck: Deck):
    s = deck.prs.slides.add_slide(deck.layout)
    deck.page += 1
    text(s, 0.67, 0.9, 7.0, 0.8, "On Spectral Clustering", size=36, bold=True)
    text(s, 0.67, 1.65, 7.0, 0.6, "Analysis and an Algorithm", size=24, color=NAVY)
    text(s, 0.67, 2.55, 6.5, 0.9,
         ["Andrew Y. Ng, Michael I. Jordan, Yair Weiss", "NIPS 2001"], size=15, color=SLATE)
    text(s, 0.67, 5.6, 6.5, 0.9,
         [[("COMP3314 Group Project", {"b": True, "c": TEAL})], "Group members:  ________________"],
         size=14, color=SLATE)
    picture(s, ASSETS / "paper_fig1e_njw.png", 7.35, 1.2, w=2.55)
    picture(s, ASSETS / "paper_fig1i_kmeans.png", 10.1, 1.2, w=2.55)
    text(s, 7.35, 4.0, 2.55, 0.35, "NJW", size=13, bold=True, align=PP_ALIGN.CENTER)
    text(s, 10.1, 4.0, 2.55, 0.35, "K-means", size=13, bold=True, align=PP_ALIGN.CENTER)
    text(s, 7.35, 4.4, 5.3, 0.35, "Original paper, Figure 1(e) and 1(i)", size=10.5, color=SLATE,
         align=PP_ALIGN.CENTER)
    notes(s, """
[~15 s] Our paper is "On Spectral Clustering: Analysis and an Algorithm" by Ng, Jordan and Weiss, NIPS 2001.
The picture already shows the problem: two rings. The left panel is their algorithm, the right panel is plain K-means.
The talk has four parts: why K-means fails, the NJW algorithm step by step, our reproduction, and a critical analysis.
""")


def s_motivation(deck: Deck, D):
    s = deck.slide("Why K-means fails on rings", "Same 400 points, same k = 2. Only the method changes.",
                   section=1)
    picture(s, D["img_motivation"], 0.5, 1.6, w=7.9)
    m = D["motivation"]
    card(s, 8.75, 1.75, 3.92, 2.25, "K-means",
         ["Each point goes to the nearest centroid.",
          "Two centroids → one straight boundary (dashed).",
          "A straight line cannot separate two rings."], size=13)
    card(s, 8.75, 4.15, 3.92, 2.0, "Idea of NJW",
         ["Keep K-means.",
          [("Change the coordinates", {"b": True}), (" each point gets, so that the rings become easy "
                                                    "to separate.", {})]],
         size=13, fill=WARM, head_color=ORANGE)
    takeaway(s, f"K-means ARI = {m['ari_km']:.3f} (chance level).   NJW ARI = {m['ari_njw']:.3f}.   "
                "ARI = 1 means identical to the true rings.")
    notes(s, """
[~40 s] These are our 400 points, two rings. K-means assigns each point to its nearest centroid, so with two centroids
the boundary is the perpendicular bisector: a straight line. No straight line separates two concentric rings, so the
result is at chance level, ARI about zero.
NJW keeps K-means as its last step. What it changes is the coordinates: it gives each point new coordinates in which the
two rings are trivially separable. The next slides show how those coordinates are built.
""")


def s_overview(deck: Deck):
    s = deck.slide("NJW in one picture", "Input: n points, the number of clusters k, a scale σ.   "
                   "Output: one label per point.", section=2)
    roadmap(s, -1)
    xs = [0.67 + i * (1.86 + 0.17) for i in range(6)]
    desc = [
        "The raw 2-D coordinates.",
        "How similar is every pair of points?",
        "Rescale by each point's total similarity.",
        "Keep the k largest eigenvectors as columns.",
        "Scale every row to length 1.",
        "Cluster the rows of Y. Row i → point i.",
    ]
    for x, dsc in zip(xs, desc):
        text(s, x + 0.05, 2.2, 1.76, 1.2, dsc, size=12.5, color=NAVY, align=PP_ALIGN.CENTER)
    boxes = [
        ("1  Affinity", r"$A_{ij}=\exp\left(-\frac{\|s_i-s_j\|^2}{2\sigma^2}\right),\ \ A_{ii}=0$", "ov_A"),
        ("2  Normalize", r"$L=D^{-1/2}AD^{-1/2},\ \ D_{ii}=\sum_j A_{ij}$", "ov_L"),
        ("3  Eigenvectors", r"$X=[v_1\ \cdots\ v_k],\ \ Lv_j=\lambda_j v_j$", "ov_X"),
        ("4  Unit rows, then K-means", r"$y_i = x_i\ /\ \|x_i\|$", "ov_Y"),
    ]
    for k, (head, tex, key) in enumerate(boxes):
        x = 0.67 + (k % 2) * 6.07
        yy = 3.45 + (k // 2) * 1.12
        rect(s, x, yy, 5.93, 1.0, fill=CARD)
        text(s, x + 0.18, yy + 0.1, 2.0, 0.8, head, size=13, bold=True, color=TEAL)
        formula(s, tex, key, x + 2.2, yy + 0.2, size=16)
    takeaway(s, [[("Key point: ", {"b": True}),
                  ("point i is now represented by row i of X (then Y). K-means only sees these rows.", {})]],
             y=5.85)
    notes(s, """
[~25 s] Here is the whole algorithm. Five matrices, left to right. A is n by n pairwise similarity. L rescales A by the
degrees. X stacks the top k eigenvectors of L as columns, so it is n by k. Y scales each row of X to unit length.
Finally K-means runs on the n rows of Y, and the label of row i is the label of point i.
The bar at the top will follow us through the next slides, highlighting the current step.
We use a small example: 52 points on two rings, 16 inner and 36 outer, sigma = 0.16.
""")


def toy_panel():
    return Panel(0.67, 2.3, 3.8, 1.18)


HEAT_BOX = (4.85, 2.3, 3.8)


def heat_labels(s, toy, box=HEAT_BOX):
    x, y, size = box
    n, n_in = toy["n"], int((toy["y"] == 0).sum())
    w_in = size * n_in / n
    text(s, x, y - 0.3, w_in, 0.28, "inner", size=10, bold=True, color=SLATE, align=PP_ALIGN.CENTER,
         name="!!hl_in")
    text(s, x + w_in, y - 0.3, size - w_in, 0.28, "outer ring  (columns j)", size=10, bold=True, color=SLATE,
         align=PP_ALIGN.CENTER, name="!!hl_out")
    line(s, x + w_in, y - 0.26, x + w_in, y - 0.04, color=GREY, width=0.75, name="!!hl_sep")


def s_affinity(deck: Deck, D):
    toy = D["toy"]
    P, n = toy["P"], toy["n"]
    hx, hy, hs = HEAT_BOX
    for frame, hi in enumerate(toy["hi"]):
        s = deck.slide("Step 1 · Affinity matrix A", "Every pair of points gets a weight between 0 and 1.",
                       section=2, new_page=frame == 0, transition=True)
        roadmap(s, 1)
        pan = toy_panel()
        pan.axes(s)
        colors = [GREY] * n
        pan.points(s, P, colors)
        cx, cy = pan.xy(*P[hi])
        r = 2 * toy["sigma"] * pan.size / (2 * pan.lim)
        ring(s, cx, cy, r, color=ORANGE, name="!!sig", width=1.5)
        dot(s, cx, cy, 0.16, NAVY, name="!!hi", line=WHITE)
        text(s, cx + r * 0.75, cy - r - 0.28, 0.6, 0.28, "2σ", size=11, color=ORANGE, name="!!sigl")
        text(s, cx - 0.15, cy + 0.08, 0.3, 0.28, "i", size=12, bold=True, color=NAVY, name="!!hil")
        picture(s, toy["img_A"], hx, hy, w=hs, h=hs, name="!!heat")
        rect(s, hx - 0.04, hy + hs * hi / n, hs + 0.08, hs / n, fill=None, line=ORANGE, radius=False,
             name="!!rowbox", line_w=2.0)
        text(s, hx - 0.42, hy + hs * (hi + 0.5) / n - 0.14, 0.4, 0.28, "i", size=12, bold=True,
             color=ORANGE, align=PP_ALIGN.RIGHT, name="!!rowl")
        heat_labels(s, toy)
        formula(s, r"$A_{ij}=\exp\left(-\frac{\|s_i-s_j\|^2}{2\sigma^2}\right)$", "A1", 9.0, 2.3, size=18,
             name="!!fA")
        formula(s, r"$A_{ii}=0$", "A2", 9.0, 3.0, size=16, name="!!fA2")
        body = [
            "Near points: weight close to 1. Far points: close to 0.",
            "σ decides what counts as near.",
        ]
        if frame == 0:
            body.append([("Row i (outer ring): ", {"b": True, "c": ORANGE}),
                         ("its weight goes only to nearby points on the same ring.", {})])
        else:
            body.append([("Row i (inner ring): ", {"b": True, "c": ORANGE}),
                         ("again, all weight stays inside its own ring.", {})])
        text(s, 9.0, 3.55, 3.7, 2.7, body, size=13, space_after=8, name="!!body")
        takeaway(s, "Rows are ordered inner ring first only for display. Then A shows two blocks: "
                    "within-ring weights, almost nothing across.")
        notes(s, """
[~40 s, two clicks] Step one. For every pair of points we compute a Gaussian weight: close to one for near points, close
to zero beyond a few sigma. The orange circle is two sigma around point i. The diagonal is set to zero.
The heat map on the right is A. The highlighted row is point i: all of its weight goes to its neighbours on the same
ring. (click) Pick a point on the inner ring: the same, its row is only non-zero inside the inner block.
Important: we sorted the rows inner ring first only so that you can see the blocks. NJW never uses this order.
""" if frame == 0 else "Second frame of step 1 (Morph): the highlighted point moves to the inner ring.")


def s_normalize(deck: Deck, D):
    toy = D["toy"]
    hx, hy, hs = HEAT_BOX
    s = deck.slide("Step 2 · Normalize by degree: L", "Divide each weight by how connected its two endpoints are.",
                   section=2, transition=True)
    roadmap(s, 2)
    pan = toy_panel()
    pan.axes(s)
    pan.points(s, toy["P"], [GREY] * toy["n"])
    picture(s, toy["img_L"], hx, hy, w=hs, h=hs, name="!!heat")
    heat_labels(s, toy)
    picture(s, toy["img_d"], hx + hs + 0.12, hy, w=0.55, h=hs, name="!!deg")
    text(s, hx + hs + 0.12, hy - 0.3, 0.55, 0.28, "d", size=11, bold=True, color=SLATE,
         align=PP_ALIGN.CENTER, name="!!degl")
    formula(s, r"$d_i=\sum_j A_{ij}$", "L1", 9.75, 2.3, size=17, name="!!fA")
    formula(s, r"$L=D^{-1/2}AD^{-1/2}$", "L2", 9.75, 2.95, size=17, name="!!fA2")
    formula(s, r"$L_{ij}=\frac{A_{ij}}{\sqrt{d_i\,d_j}}$", "L3", 9.75, 3.55, size=17, name="!!fL3")
    text(s, 9.75, 4.45, 2.95, 1.8,
         ["d (grey bars): total weight of each point.",
          "Dense regions no longer dominate; L stays symmetric.",
          "L keeps the two blocks of A."], size=12.5, space_after=7, name="!!body")
    takeaway(s, "Degree normalization changes the matrix. (Row normalization in step 4 changes each point; "
                "they are different steps.)")
    notes(s, """
[~25 s] Step two. The degree d_i is the row sum of A, how strongly point i is connected overall - the grey bars.
L divides each weight by the square root of d_i times d_j. This stops dense regions from dominating, keeps L symmetric,
and keeps the block structure we just saw. L is still n by n.
""")


def s_eigen(deck: Deck, D):
    toy = D["toy"]
    hx, hy, hs = HEAT_BOX
    n = toy["n"]
    Xs = toy["X"] * toy["x_scale"]
    for frame in range(2):
        s = deck.slide("Step 3 · Top-k eigenvectors: X", "Each point gets k = 2 new coordinates: its row of X.",
                       section=2, new_page=frame == 0, transition=True)
        roadmap(s, 3)
        pan = toy_panel()
        pan.axes(s)
        pan.points(s, toy["P"] if frame == 0 else Xs, [GREY] * n)
        if frame == 1:
            rays(s, pan, [toy["Y"][toy["y"] == c].mean(0) for c in (0, 1)])
            text(s, pan.x + pan.size - 1.0, pan.y + pan.size / 2 + 0.02, 1.0, 0.3, "xᵢ₁ →", size=10,
                 color=SLATE, align=PP_ALIGN.RIGHT, name="!!axl1")
            text(s, pan.x + pan.size / 2 + 0.05, pan.y - 0.05, 1.0, 0.3, "xᵢ₂ ↑", size=10, color=SLATE,
                 name="!!axl2")
        picture(s, toy["img_eig"], hx - 0.1, hy - 0.1, w=hs + 0.1, name="!!eig")
        xw = 0.62
        xx = hx + hs + 0.2
        picture(s, toy["img_X"], xx, hy, w=xw, h=hs, name="!!Xheat")
        text(s, xx - 0.05, hy - 0.3, xw + 0.1, 0.28, "v₁   v₂", size=10, bold=True, color=SLATE,
             align=PP_ALIGN.CENTER, name="!!Xl")
        text(s, xx, hy + hs + 0.02, xw, 0.28, "X", size=13, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER, name="!!Xn")
        text(s, hx, 4.55, hs, 1.75,
             [[("Eigenvalues of L are ≤ 1. ", {}), ("Keep the k largest", {"b": True}),
               (" (by value, not absolute value) and put their eigenvectors side by side.", {})],
              [("Teal = positive, orange = negative.", {"c": SLATE, "s": 12})]],
             size=13, space_after=6, name="!!eigtx")
        if frame == 0:
            body = [[("Row i of X", {"b": True}), (" = new coordinates of point i.", {})],
                    "Next: move every point to its row of X."]
        else:
            body = [[("Row i of X", {"b": True}), (" = new coordinates of point i.", {})],
                    [("The two rings became two rays ", {"b": True, "c": TEAL}),
                     ("from the origin. Points on one ray differ only in length.", {})]]
        text(s, 9.8, 2.3, 2.9, 3.9, body, size=13, space_after=10, name="!!body")
        takeaway(s, "Shown scaled up for display. Eigenvector signs are arbitrary; we fixed them so "
                    "the frames line up." if frame == 1 else
                    "With k = 2 we only need the two largest eigenvalues: λ₁ = 1.000, λ₂ = 0.9996, then a gap.")
        notes(s, """
[~35 s, one click] Step three. We compute the eigenvalues of L. They are at most one. We keep the k largest, here two,
and put their eigenvectors side by side as the columns of X - an n by 2 matrix, shown on the right.
Notice the gap after lambda 2: this gap will matter for the theory.
Now the key move: row i of X is the new coordinate of point i. (click) Each point jumps to its row of X.
The two rings become two rays from the origin. Points on the same ray share a direction but have different lengths.
""" if frame == 0 else "Morph: raw points move to their rows of X.")


def s_rows(deck: Deck, D):
    toy = D["toy"]
    n, y, lab = toy["n"], toy["y"], toy["labels"]
    Xs = toy["X"] * toy["x_scale"]
    Y = toy["Y"]
    steps = [
        ("Rows of X", "One ray per ring. Lengths αᵢ differ inside a ring."),
        ("Normalize", "yᵢ = xᵢ / ‖xᵢ‖ slides each point along its ray onto the unit circle."),
        ("K-means on Y", "Two tight groups, 90° apart. K-means cannot miss."),
        ("Back to the points", "Row i's label is point i's label."),
    ]
    for frame in range(4):
        s = deck.slide("Step 4 · Unit rows Y, then K-means",
                       "Row normalization removes the length and keeps only the direction.",
                       section=2, new_page=frame == 0, transition=True)
        roadmap(s, 4 if frame < 2 else 5)
        pan = toy_panel()
        pan.axes(s)
        pan.unit_circle(s, show=frame in (0, 1, 2))
        pts = [Xs, Y, Y, toy["P"]][frame]
        colors = [GREY] * n if frame < 2 else [RING[c] for c in lab]
        pan.points(s, pts, colors)
        if frame in (1, 2):
            angs = [math_deg(Y[y == c].mean(0)) for c in (0, 1)]
            lo, hi = sorted(angs)
            rr = 0.32
            arc = [pan.xy(rr * np.cos(np.radians(t)), rr * np.sin(np.radians(t)))
                   for t in np.linspace(lo, hi, 24)]
            polyline(s, arc, color=NAVY, width=1.5, name="!!arc")
            mid = np.radians((lo + hi) / 2)
            tx, ty = pan.xy(0.55 * np.cos(mid), 0.55 * np.sin(mid))
            text(s, tx - 0.3, ty - 0.17, 0.6, 0.34, "90°", size=14, bold=True, color=NAVY,
                 align=PP_ALIGN.CENTER, name="!!arcl")
        if frame in (0, 1, 2):
            rays(s, pan, [Y[y == c].mean(0) for c in (0, 1)])
        if frame == 2:
            for c in (0, 1):
                v = Y[lab == c].mean(0)
                cx, cy = pan.xy(*(v / np.linalg.norm(v) * 1.12))
                xm = s.shapes.add_shape(MSO_SHAPE.MATH_MULTIPLY, Inches(cx - 0.15), Inches(cy - 0.15),
                                        Inches(0.3), Inches(0.3))
                xm.fill.solid()
                xm.fill.fore_color.rgb = NAVY
                xm.line.fill.background()
                xm.name = f"!!cent{c}"
        for k, (head, body) in enumerate(steps):
            on = k == frame
            yy = 2.3 + k * 0.98
            rect(s, 4.85, yy, 4.6, 0.86, fill=CARD if on else WHITE, line=None if on else LIGHT,
                 name=f"!!st{k}_bg")
            text(s, 5.0, yy + 0.08, 4.35, 0.75,
                 [[(f"{k + 1}  {head}", {"b": True, "c": TEAL if on else GREY, "s": 13.5})],
                  [(body, {"c": NAVY if on else GREY, "s": 12})]],
                 space_after=2, name=f"!!st{k}_tx")
        if frame == 0:
            right = [[("Why is this enough?", {"b": True, "c": TEAL})],
                     "Within a ring, rows of X share a direction but not a length.",
                     "K-means on X would still be pulled by length."]
        elif frame == 1:
            right = [[("Why 90°?", {"b": True, "c": TEAL})],
                     "Different clusters end up in orthogonal directions (next slide).",
                     "Every ring collapses to (almost) one point."]
        elif frame == 2:
            right = [[("K-means on Y", {"b": True, "c": TEAL})],
                     "The same K-means that failed on the raw points.",
                     "In Y its job is trivial: two far-apart dots."]
        else:
            right = [[("Result", {"b": True, "c": TEAL})],
                     f"ARI = {toy['ari']:.2f} on the toy rings.",
                     "Same pipeline, 400 points: ARI = 1.00 (Part 3)."]
        text(s, 9.75, 2.3, 2.95, 3.9, right, size=12.5, space_after=9, name="!!body")
        notes(s, [
            """
[~50 s, three clicks] Step four. Here are the rows of X again, drawn with the unit circle. Inside one ring, the rows share a
direction but have different lengths. (click) We divide every row by its length. Each point slides along its ray onto the
unit circle, and each ring collapses to almost a single point. The two points are 90 degrees apart - the next slide
explains why. (click) Now K-means runs on these rows. It is the same K-means that failed before, but here the task is
trivial: two far-apart dots. (click) Finally the label of row i goes back to point i. The rings are recovered, ARI = 1.
""",
            "Morph: rows of X slide onto the unit circle (row normalization).",
            "K-means on Y: colours appear, crosses mark the centroids.",
            "Morph: points fly back to the original plane with their labels.",
        ][frame])


def rays(s, pan, dirs):
    for c, v in enumerate(dirs):
        u = np.asarray(v) / np.linalg.norm(v) * pan.lim * 0.98
        x0, y0 = pan.xy(0, 0)
        x1, y1 = pan.xy(*u)
        line(s, x0, y0, x1, y1, color=SLATE, width=1.0, dash=True, name=f"!!ray{c}")


def math_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))


def s_ideal(deck: Deck, D):
    s = deck.slide("Why it works: the ideal case", "Suppose there is no weight at all between clusters.",
                   section=2, transition=True)
    picture(s, D["img_ideal"], 0.75, 1.55, w=11.8)
    card(s, 0.67, 5.3, 3.85, 1.55, "Ideal L is block diagonal",
         ["Each block has eigenvalue 1, so λ₁ = … = λₖ = 1.",
          "Each eigenvector lives on one cluster."], size=12.5)
    rect(s, 4.74, 5.3, 3.85, 1.55, fill=CARD)
    text(s, 4.92, 5.42, 3.5, 0.35, "Rows of the ideal X", size=13.5, bold=True, color=TEAL)
    formula(s, r"$x_i=\alpha_i\, r_c\quad(\alpha_i>0)$", "id1", 4.92, 5.85, size=17)
    text(s, 4.92, 6.3, 3.5, 0.5, "direction from the cluster, length αᵢ varies", size=11.5, color=NAVY)
    rect(s, 8.81, 5.3, 3.86, 1.55, fill=WARM)
    text(s, 8.99, 5.42, 3.5, 0.35, "After normalizing", size=13.5, bold=True, color=ORANGE)
    formula(s, r"$y_i=r_c,\quad r_1\perp r_2\perp\cdots\perp r_k$", "id2", 8.99, 5.85, size=17)
    text(s, 8.99, 6.3, 3.5, 0.5, "true for any rotation of the eigenvector basis", size=11.5, color=NAVY)
    notes(s, """
[~40 s] Why does this work? Take the ideal case: remove every weight between the two rings. Then L-hat is block
diagonal. Each block has eigenvalue exactly one, so the top k eigenvalues are all one, and the eigenvectors can be chosen
to live on one cluster each - that is X-hat in the middle.
So row i of X-hat is alpha_i times r_c: the direction depends only on the cluster, the length alpha_i varies.
Dividing by the length removes alpha_i, so y_i = r_c. Different clusters give orthogonal directions - the 90 degrees.
The solver may return any rotation of these eigenvectors; a rotation keeps lengths and angles, so the picture is the same.
""")


def s_general(deck: Deck, D):
    s = deck.slide("Beyond the ideal case: a perturbation", "Real data has some weight across clusters. "
                   "The paper treats it as noise added to the ideal matrix.", section=2, transition=True)
    picture(s, D["img_perturb"], 1.9, 1.55, w=9.5)
    text(s, 1.9, 4.5, 9.5, 0.3, f"Toy rings at σ = {D['perturb_sigma']:.2f}, so that E is visible. "
         "E is scaled up for display.", size=10.5, color=SLATE, align=PP_ALIGN.CENTER)
    card(s, 0.67, 4.95, 5.6, 1.9, "Conditions (paper's A1–A4, in words)",
         ["Each cluster is well connected inside: gap δ = λₖ − λₖ₊₁.",
          "Little weight across clusters: small ε.",
          "No point is only weakly tied to its own cluster."], size=12.5)
    rect(s, 6.45, 4.95, 6.22, 1.9, fill=WARM)
    text(s, 6.63, 5.07, 5.9, 0.35, "Theorem 2", size=13.5, bold=True, color=ORANGE)
    text(s, 6.63, 5.45, 5.9, 0.35, "If δ > (2 + √2) ε, there are k orthogonal unit vectors r₁…rₖ with",
         size=12, color=NAVY)
    formula(s, r"$\frac{1}{n}\sum_i \|y_i-r_{c(i)}\|^2\ \leq\ \frac{4C\,(4+2\sqrt{k})^2\,\epsilon^2}"
            r"{(\delta-\sqrt{2}\,\epsilon)^2}$", "thm2", 6.63, 5.85, size=18)
    notes(s, """
[~25 s] Real data is not ideal. The paper writes A as the ideal within-cluster part plus a perturbation E, the weights
across clusters. Matrix perturbation theory then says: if every cluster is well connected inside - a large eigengap
delta - and the cross-cluster weight epsilon is small, then the rows of Y stay close to k orthogonal points, so K-means
still works. That is Theorem 2. We will test whether its condition actually holds on our data in Part 4.
""")


def s_exp1(deck: Deck, D):
    s = deck.slide("Our reproduction: three datasets", "Our own NJW code (no sklearn SpectralClustering), "
                   "our generated points.", section=3)
    picture(s, D["img_stage1"], 0.45, 1.5, w=8.6)
    st = D["stage1"]
    card(s, 9.3, 1.6, 3.37, 2.65, "Setup",
         ["n = 400, 400, 450.  Data seed 0.",
          "Same points and same k for both methods.",
          "σ = median distance to the 7th neighbour. No labels used.",
          "Labels only for ARI and colours."], size=11.5)
    card(s, 9.3, 4.4, 3.37, 1.85, "Reading",
         [f"Rings: {F.fmt2(st['circles']['km'])} → {F.fmt2(st['circles']['njw'])}",
          f"Moons: {F.fmt2(st['moons']['km'])} → {F.fmt2(st['moons']['njw'])}",
          "Blobs: both 1.00, K-means already works."], size=12, fill=WARM, head_color=ORANGE)
    takeaway(s, "NJW wins where clusters are not convex. On compact blobs it gives the same answer as K-means.")
    notes(s, """
[~40 s] Part three, our reproduction. We implemented NJW ourselves; only the eigensolver and the final K-means come from
libraries. Three datasets we generated, same points and same k for both methods. Sigma comes from a fixed,
label-free rule; labels are only used to score with ARI.
On rings and moons K-means fails - 0.00 and 0.27 - and NJW is perfect. On blobs both are perfect: NJW's advantage
is specifically on non-convex shapes, not everywhere.
""")


def s_exp2(deck: Deck, D):
    pl = D["paper_like"]
    s = deck.slide("Rebuilding the paper's Figure 1 settings", "The paper released no points, n, noise or σ grid, "
                   "so this is a qualitative reproduction.", section=3)
    picture(s, D["img_paper_like"], 0.45, 1.5, w=8.7)
    card(s, 9.35, 1.6, 3.32, 2.3, "σ chosen as in the paper",
         ["Try a grid of σ.",
          "Keep the σ whose rows of Y give the smallest K-means distortion.",
          "No labels involved."], size=11.5)
    n_ok = sum(v["njw"] > 0.99 for v in pl.values())
    card(s, 9.35, 4.05, 3.32, 2.2, "Result",
         [f"NJW = 1.00 on {n_ok} of {len(pl)} settings.",
          "Joined rings included: three rings, recovered.",
          "K-means still fails on the non-convex shapes."],
         size=11.5)
    takeaway(s, "The main claim reproduces: one algorithm, no per-shape tuning, many non-convex shapes.")
    notes(s, f"""
[~40 s] We also rebuilt the settings of the paper's Figure 1 - two circles, line and balls, four clouds, squiggles, the
NIPS letters, and joined rings. The paper did not release data, so this is qualitative.
We used the paper's own rule for sigma: pick the sigma whose rows of Y cluster most tightly. No labels.
NJW gets 1.00 on these rebuilt settings, including the three joined rings. K-means still fails where the shape is
not convex. The limits of sigma and k are the next slides.
""")


def s_sigma(deck: Deck, D):
    sw = D["sweep"]
    y, keep = sw["y"], sw["keep"]
    curve = sw["curve"]
    texts = {
        1.0: ("Blocks are clean.", "Each ring collapses to one point of Y."),
        1.4: ("Still almost no weight across rings.", "Y is still two tight points."),
        2.0: ("Weight appears across the rings.", "Y smears into an arc. ARI falls to 0."),
        4.0: ("A has no block structure left.", "NJW now cuts the plane like K-means."),
    }
    gx, gy, gw, gh = 8.95, 2.45, 3.6, 2.4

    def g(r, a):
        lx = (math.log(r) - math.log(0.25)) / (math.log(8) - math.log(0.25))
        return gx + lx * gw, gy + (1.0 - (a + 0.05) / 1.1) * gh

    for k, fr in enumerate(sw["frames"]):
        s = deck.slide("Sensitivity to the scale σ", "Circles from Part 3 (400 points). σ = r · σ₀, "
                       "σ₀ = our label-free default.", section=4, new_page=k == 0, transition=True)
        picture(s, fr["img_A"], 0.67, 2.3, w=3.55, h=3.55, name="!!sweepA")
        text(s, 0.67, 1.95, 3.55, 0.3, "A (inner ring first)", size=11, color=SLATE, align=PP_ALIGN.CENTER,
             name="!!sAl")
        pan = Panel(4.55, 2.3, 3.55, 1.2)
        pan.axes(s)
        pan.unit_circle(s)
        Y = fr["Y"][keep]
        pan.points(s, Y, [RING[c] for c in y[keep]], d=0.085, prefix="y")
        text(s, 4.55, 1.95, 3.55, 0.3, "rows of Y (colour = true ring)", size=11, color=SLATE,
             align=PP_ALIGN.CENTER, name="!!sYl")
        for name, col, w, dash in [("moons", GREY, 1.25, True), ("circles", TEAL, 2.25, False)]:
            pts = [g(r, a) for r, a in curve[name]]
            cv = polyline(s, pts, color=col, width=w, name=f"!!cv_{name}")
            if dash:
                cv.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        line(s, gx, gy + gh, gx + gw, gy + gh, color=SLATE, width=0.75, name="!!gx")
        line(s, gx, gy, gx, gy + gh, color=SLATE, width=0.75, name="!!gy")
        for r in (0.25, 1, 4):
            px, _ = g(r, 0)
            text(s, px - 0.3, gy + gh + 0.03, 0.6, 0.25, f"{r:g}", size=9.5, color=SLATE,
                 align=PP_ALIGN.CENTER, name=f"!!gt{r:g}")
        text(s, gx, gy + gh + 0.27, gw, 0.25, "r = σ / σ₀ (log scale)", size=9.5, color=SLATE,
             align=PP_ALIGN.CENTER, name="!!gxl")
        for a in (0, 1):
            _, py = g(1, a)
            text(s, gx - 0.4, py - 0.13, 0.35, 0.25, str(a), size=9.5, color=SLATE, align=PP_ALIGN.RIGHT,
                 name=f"!!gya{a}")
        text(s, gx, gy - 0.5, gw, 0.3, "NJW ARI", size=11, color=SLATE, align=PP_ALIGN.CENTER, name="!!gtl")
        text(s, gx + 2.2, g(8, 0.30)[1] - 0.35, 1.4, 0.25, "moons", size=9.5, color=GREY, name="!!glm")
        text(s, gx + 0.1, g(0.3, 1)[1] - 0.32, 1.4, 0.25, "circles", size=9.5, color=TEAL, bold=True,
             name="!!glc")
        mx, my = g(fr["r"], fr["ari"])
        dot(s, mx, my, 0.2, ORANGE, name="!!mark", line=WHITE)
        a1, a2 = texts[fr["r"]]
        ari = 0.0 if abs(fr["ari"]) < 0.005 else fr["ari"]
        text(s, gx - 0.15, 5.5, gw + 0.2, 0.8,
             [[(f"r = {fr['r']:g}   ", {"b": True, "c": NAVY}),
               (f"ARI = {ari:.2f}", {"b": True, "c": TEAL if ari > 0.99 else ORANGE})],
              [(a1 + " " + a2, {"s": 11.5, "c": NAVY})]],
             size=15, space_after=2, name="!!sv")
        takeaway(s, "One global σ. Circles: fine up to 1.4σ₀, ARI 0 from 2σ₀. The method does not reveal "
                    "the safe window.")
        notes(s, """
[~50 s, three clicks] Part four, critical analysis. First, sensitivity to sigma, on the circles from Part 3. Left: A.
Middle: the rows of Y, coloured by the true ring. Right: ARI as we scale sigma.
At our default the blocks are clean and Y is two points. (click) At 1.4 sigma-zero, still fine. (click) At 2 sigma-zero
weight appears between the rings, Y smears into an arc, and ARI drops to zero. (click) At 4, A has lost its blocks, and NJW
cuts the plane just like K-means.
So the method depends on one global scale. On circles the safe window is narrow; on moons the decline is gradual. The
paper's distortion rule chooses sigma without labels, but nothing guarantees it lands in the window.
""" if k == 0 else f"Morph: sigma ratio r = {fr['r']:g}.")


def s_k(deck: Deck, D):
    wk = D["wrong_k"]
    s = deck.slide("k must be given, and is not enough", "Three rings we generated. σ₀ = "
                   f"{wk['sigma0']:.3f} from the same neighbour rule.", section=4)
    picture(s, D["img_wrong_k"], 0.6, 1.5, w=12.1)
    card(s, 0.67, 5.45, 3.85, 1.4, "k = 2",
         [f"Inner ring vs. the other two, every time (ARI {wk['k2_vs_merge']:.2f} against that split)."],
         size=12.5)
    card(s, 4.74, 5.45, 3.85, 1.4, "k = 3 (the true k)",
         [f"{wk[(3, 1.0)]:.2f} at σ₀, {wk[(3, 0.7)]:.2f} at 0.7σ₀. The two outer rings mix at σ₀."], size=12.5)
    card(s, 8.81, 5.45, 3.86, 1.4, "Reading",
         ["k chooses which question is asked. σ still decides the answer."],
         size=12.5, fill=WARM, head_color=ORANGE)
    notes(s, """
[~35 s] Second, the number of clusters k is an input. The paper sets it by hand in every Figure 1 run.
With k = 2 on three rings, NJW consistently separates the inner ring from the other two - a stable, sensible merge.
With the true k = 3, it is still not right at our default sigma: 0.52, the two outer rings mix. At 0.7 sigma-zero it rises
to 0.90. So knowing k is not enough: k picks the question, sigma still decides the answer.
""")


def s_robust(deck: Deck, D):
    br = D["bridge"]
    asm = D["assumptions"]
    rows = {(r["experiment"], r["setting"]): r for r in D["noise"]}
    s = deck.slide("Robustness, and what the theory explains", "Left: noise between clusters. "
                   "Right: Theorem 2 checked on the Part 3 runs.", section=4)
    picture(s, D["img_bridge"], 0.55, 1.55, w=6.5)
    text(s, 0.67, 3.95, 6.3, 0.3, f"Moons + {br['n_bridge']} bridge points (×, left out of ARI)", size=10.5,
         color=SLATE, align=PP_ALIGN.CENTER)
    tbl = [("Setting", "NJW", "Components")]
    for key, label in [("clean", "Clean moons"), ("thin_bridge", "Thin bridge, 25 pts"),
                       ("strong_bridge", "Strong bridge, 80 pts"), ("gap_noise_cloud", "Noise cloud, 120 pts")]:
        r = rows[("moons_bridge", key)]
        tbl.append((label, f"{float(r['ARI_njw']):.2f}", f"{float(r['ARI_connected_components']):.2f}"))
    gt = s.shapes.add_table(len(tbl), 3, Inches(0.67), Inches(4.35), Inches(6.3), Inches(1.9)).table
    gt.columns[0].width = Inches(3.3)
    gt.columns[1].width = Inches(1.4)
    gt.columns[2].width = Inches(1.6)
    for i, row in enumerate(tbl):
        for j, v in enumerate(row):
            c = gt.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 else WHITE)
            tf = c.text_frame
            tf.text = v
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.name, f.size, f.bold = FONT, Pt(12), i == 0
            f.color.rgb = WHITE if i == 0 else (ORANGE if v == "0.00" else NAVY)
    c, m, b = asm["circles"], asm["moons"], asm["blobs"]

    def need(r):
        return float(r["required_delta_(2+sqrt2)eps"])

    rect(s, 7.3, 1.6, 5.37, 3.15, fill=CARD)
    card(s, 7.3, 4.9, 5.37, 1.35, "Row normalization switched off",
         ["350 vs 50-point blobs, same σ and k: ARI 1.00 → 0.96. It helps, by a small margin here."],
         size=12, fill=WARM, head_color=ORANGE)
    text(s, 7.48, 1.72, 5.0, 0.35, "Theorem 2 needs δ > (2 + √2) ε", size=13.5, bold=True, color=TEAL)
    text(s, 7.48, 2.15, 5.0, 2.6,
         [[("Circles: ", {"b": True}), (f"δ = {float(c['delta']):.4f} < {need(c):.3f}. Fails, yet ARI = 1.", {})],
          [("Moons: ", {"b": True}), (f"δ = {float(m['delta']):.4f} < {need(m):.4f}. Fails, yet ARI = 1.", {})],
          [("Blobs: ", {"b": True}),
           (f"holds. Bound {float(b['theorem2_bound']):.1e}, observed "
            f"{float(b['observed_mean_sq_dist_to_orthogonal_centers']):.1e}.", {})]],
         size=12, space_after=9)
    text(s, 7.48, 3.5, 5.0, 1.2,
         [[("Reading: ", {"b": True, "c": ORANGE}),
           ("sufficient, not necessary. Rings have a tiny gap δ, so the theory misses the examples that motivate "
            "the method. Checking it needs the true clusters.", {})]],
         size=11.5)
    takeaway(s, "Weights, not a hard threshold: bridges barely move NJW but break connected components.")
    notes(s, """
[~40 s] Third, robustness and the theory. We added bridge points between the moons. NJW stays at 1.00, because a thin
bridge carries little total weight. A hard-threshold method, connected components, falls to 0: one chain of points
merges the clusters. This matches the paper's own comparison.
On the right we checked Theorem 2's condition on our Part 3 runs, using the true clusters. On circles and moons it fails
- the gap is far too small - yet NJW is perfect. On blobs it holds, but the bound is about ten orders of magnitude
above what we observe. So the theorem is sufficient, not necessary, and it does not explain the ring examples.
Last, the row normalization step: switching it off drops ARI from 1.00 to 0.96 on unbalanced blobs - it helps, modestly.
""")


def s_summary(deck: Deck):
    s = deck.slide("Summary", None, section=5)
    cols = [
        ("What NJW does", TEAL, CARD,
         ["Builds a similarity graph A.", "Normalizes it to L.", "Uses the top-k eigenvectors as new coordinates.",
          "Normalizes rows, then runs K-means."]),
        ("Why it works", TEAL, CARD,
         ["Ideal case: block-diagonal L.", "Each cluster → one direction, all directions orthogonal.",
          "Small cross-cluster weight + large eigengap keep Y close to that picture."]),
        ("Where it is fragile", ORANGE, WARM,
         ["One global σ with a narrow safe window.", "k is given by hand; the true k is not enough.",
          "Theorem 2 does not cover thin shapes; it needs the true clusters to check.",
          "Dense n × n matrix and eigensolver: O(n²) memory."]),
    ]
    for i, (head, hc, fill, body) in enumerate(cols):
        x = 0.67 + i * 4.07
        rect(s, x, 1.45, 3.85, 4.1, fill=fill)
        text(s, x + 0.2, 1.6, 3.5, 0.45, head, size=17, bold=True, color=hc)
        text(s, x + 0.2, 2.2, 3.5, 3.3, [f"•  {b}" for b in body], size=13, space_after=10)
    takeaway(s, "Our part: NJW from scratch, 3 + 6 datasets, tests of σ, k, noise, row normalization and "
                "Theorem 2.", y=5.85)
    notes(s, """
[~30 s] To summarize. NJW turns each point into a row of the top-k eigenvectors, normalizes the rows, and clusters them.
It works because in the ideal case every cluster becomes one direction and the directions are orthogonal, and small
perturbations keep that picture. It is fragile in sigma and k, and its main theorem does not cover the thin shapes that
motivate it. Thank you - we are happy to take questions.
""")


def s_backup(deck: Deck, D):
    # B1: course connections
    s = deck.slide("Backup · Connections to the course", None, section=0)
    rows = [("Concept", "Connection", "Distinction"),
            ("K-means", "Same final step, run on rows of Y", "The embedding changes the geometry"),
            ("PCA", "Both keep top eigenvectors", "PCA: covariance of features. NJW: graph over samples"),
            ("Kernels / similarity", "Gaussian pairwise weight", "Aᵢᵢ = 0, so A need not be a PSD Gram matrix"),
            ("Normalization", "Degree normalization of A", "Row normalization of X is a different step")]
    _table(s, rows, 0.67, 1.5, [2.6, 4.4, 5.0])
    # B2: Q&A
    s = deck.slide("Backup · Questions we prepared", None, section=0)
    qa = [("Why run K-means again?", "On Y the clusters are tight and orthogonal, so its job is easy."),
          ("Why keep the first eigenvector?", "NJW uses the whole top-k subspace; dropping one loses a direction."),
          ("What if the eigenvectors rotate?", "Rotations keep lengths and angles; Y is rotated, clusters stay."),
          ("Does a large eigengap prove correctness?", "It supports subspace stability under A1–A4, not any "
                                                       "particular labelling."),
          ("How do you choose σ and k?", "σ: label-free rule or the paper's distortion search. k: given; "
                                         "see our three-ring test."),
          ("What exactly did you reproduce?", "Our code, our data, Figure-1-like settings. Not the paper's points.")]
    _table(s, [("Question", "Short answer")] + qa, 0.67, 1.5, [4.3, 7.7], size=12)
    # B3: baselines
    s = deck.slide("Backup · Baselines on the Figure 1 settings", "ARI. Kernel methods share the distortion-chosen "
                   "σ. Components search their own threshold.", section=0)
    get = {(r["dataset"], int(r["k"]), r["method"]): r["ARI"] for r in D["paper_like_rows"]}
    sets = [("twocircles", 2), ("lineandballs", 3), ("fourclouds", 2), ("squiggles", 4), ("nips", 8),
            ("threecircles_joined", 3), ("threecircles_joined", 2)]
    meths = [("njw", "NJW"), ("meila_shi", "Meila–Shi"), ("kannan_style", "Kannan-style*"),
             ("connected_components", "Components"), ("kmeans", "K-means")]
    rows = [("Setting",) + tuple(m for _, m in meths)]
    for d, k in sets:
        rows.append((f"{d}, k={k}",) + tuple(F.fmt2(float(get[(d, k, m)])) for m, _ in meths))
    _table(s, rows, 0.67, 1.6, [3.4, 1.7, 1.7, 1.8, 1.7, 1.7], size=12)
    text(s, 0.67, 5.6, 12.0, 0.6, "* Our reading of one sentence in the paper (largest absolute eigenvector "
         "coordinate), not Kannan et al.'s code. Joined rings: NJW ARI here scores the ring points only.",
         size=11, color=SLATE)
    # B4: sigma curves
    s = deck.slide("Backup · σ sensitivity on all three datasets", "Same data and k as Part 3. Dashed line: "
                   "our default σ₀. Labels did not choose σ₀.", section=0)
    picture(s, D["img_sigma_curves"], 0.8, 1.7, w=11.7)
    text(s, 0.8, 5.2, 11.7, 1.0,
         ["Blobs: at r = 0.25 a point's row of X is numerically zero, so row normalization is undefined.",
          "Moons: 1.00 up to r = 2, then 0.81, 0.47, 0.37, 0.30.",
          "Row normalization off (350 vs 50-point blobs, same σ and k): ARI 1.00 → 0.96."], size=12.5)
    # B5: references
    s = deck.slide("Backup · References and attribution", None, section=0)
    text(s, 0.67, 1.5, 12.0, 4.5,
         [[("Primary paper", {"b": True, "c": TEAL})],
          "A. Y. Ng, M. I. Jordan, Y. Weiss. On Spectral Clustering: Analysis and an Algorithm. "
          "NIPS 14, 2001, pp. 849–856.",
          "",
          [("Figures", {"b": True, "c": TEAL})],
          "Slide 1 crops Figure 1 of the paper. Every other figure is our points, our code and our measurements.",
          "",
          [("Code", {"b": True, "c": TEAL})],
          "NJW implemented by us (src/njw_spectral). Libraries: SciPy eigh, scikit-learn KMeans and ARI, "
          "Matplotlib. sklearn SpectralClustering is not used."],
         size=13.5)


def _table(s, rows, x, y, widths, size=12.5):
    t = s.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)),
                           Inches(0.5 * len(rows))).table
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 else WHITE)
            c.text_frame.text = v
            c.text_frame.word_wrap = True
            p = c.text_frame.paragraphs[0]
            f = p.runs[0].font
            f.name, f.size, f.bold = FONT, Pt(size), i == 0 or j == 0
            f.color.rgb = WHITE if i == 0 else NAVY


def main():
    D = F.build_all()
    deck = Deck()
    s_title(deck)
    s_motivation(deck, D)
    s_overview(deck)
    s_affinity(deck, D)
    s_normalize(deck, D)
    s_eigen(deck, D)
    s_rows(deck, D)
    s_ideal(deck, D)
    s_general(deck, D)
    s_exp1(deck, D)
    s_exp2(deck, D)
    s_sigma(deck, D)
    s_k(deck, D)
    s_robust(deck, D)
    s_summary(deck)
    s_backup(deck, D)
    deck.save()
    print(f"wrote {OUT}  ({len(deck.prs.slides._sldIdLst)} slides, {deck.page} numbered pages)")


if __name__ == "__main__":
    main()
