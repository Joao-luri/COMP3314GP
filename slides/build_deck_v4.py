"""Build COMP3314_Spectral_Slides_v4.pptx.

Same deck as v3 through the reproduction slide. The critical section is two
slides (sigma, then k) and the close is where the method fits and what the
2001 paper fixed.

Usage:  .venv/bin/python slides/build_deck_v4.py
"""

from __future__ import annotations

import math as pymath
from pathlib import Path

import build_deck as B
import build_deck_v3 as V3
import figures as F
from build_deck import CARD, GREY, NAVY, ORANGE, SLATE, TEAL, WARM, Deck, card, formula, line, notes, picture, polyline, rect, text
from build_deck_v3 import Mapper, dot
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "COMP3314_Spectral_Slides_v4.pptx"


def _curve_xy(gx, gy, gw, gh, r, a):
    lx = (pymath.log(r) - pymath.log(0.25)) / (pymath.log(8) - pymath.log(0.25))
    return gx + lx * gw, gy + (1.0 - (a + 0.05) / 1.1) * gh


def s_sigma(deck, D):
    sw = D["moons_sweep"]
    P, keep = sw["P"], sw["keep"]
    by_r = {fr["r"]: fr for fr in sw["frames"]}
    s = deck.slide("One σ, and how the paper chooses it",
                   "Same moons. The dashed curve is the two circles.", section=4)
    boxes = [(0.45, 1.0), (3.55, 2.8)]
    for x, r in boxes:
        fr = by_r[r]
        ari = fr["ari"]
        text(s, x, 1.52, 3.15, 0.32,
             [[(f"σ = {r:g} σ₀   ", {"b": True}),
               (f"ARI = {F.fmt2(ari)}", {"b": True, "c": TEAL if ari > 0.99 else ORANGE})]],
             size=13, align=PP_ALIGN.CENTER)
        mp = Mapper(x, 1.88, 3.15, 2.05, (-1.25, 2.25), (-0.75, 1.25))
        for i in keep:
            dot(s, *mp.xy(*P[i]), 0.07, V3.RING[fr["labels"][i]], line=B.WHITE)
    gx, gy, gw, gh = 0.7, 4.35, 6.15, 1.45
    for name, col, wdt, dash in [("circles", GREY, 1.25, True), ("moons", TEAL, 2.25, False)]:
        cv = polyline(s, [_curve_xy(gx, gy, gw, gh, r, a) for r, a in sw["curve"][name]], color=col, width=wdt)
        if dash:
            cv.line.dash_style = B.MSO_LINE_DASH_STYLE.DASH
    line(s, gx, gy + gh, gx + gw, gy + gh, color=SLATE, width=0.75)
    line(s, gx, gy, gx, gy + gh, color=SLATE, width=0.75)
    for r in (0.25, 1, 2, 4, 8):
        px, _ = _curve_xy(gx, gy, gw, gh, r, 0)
        text(s, px - 0.28, gy + gh + 0.02, 0.56, 0.22, f"{r:g}", size=10, color=SLATE, align=PP_ALIGN.CENTER)
    text(s, gx, gy + gh + 0.24, gw, 0.22, "σ / σ₀", size=11, color=SLATE, align=PP_ALIGN.CENTER)
    text(s, gx - 0.55, gy - 0.02, 0.5, 0.22, "1", size=10, color=SLATE, align=PP_ALIGN.RIGHT)
    text(s, gx + 0.08, gy - 0.28, 3.2, 0.24, "ARI    moons (solid), circles (dashed)", size=11, color=SLATE)
    for r, a in ((1.0, by_r[1.0]["ari"]), (2.8, by_r[2.8]["ari"])):
        dot(s, *_curve_xy(gx, gy, gw, gh, r, a), 0.16, ORANGE, line=B.WHITE)

    path, fw, fh = F.math_png(r"$A_{ij}=\exp\left(-\frac{\|s_i-s_j\|^2}{2\sigma^2}\right),\ \ A_{ii}=0$",
                              "sigma_affinity", size=16, color="#152C43")
    scale = min(1.0, 5.15 / fw)
    picture(s, path, 7.45, 1.52, w=fw * scale)
    card(s, 7.4, 2.35, 5.35, 4.35, "How the paper chooses σ",
         ["Try several values of σ.",
          "For each, run K-means on the rows of Y.",
          "Keep the σ with the smallest distortion.",
          [("Distortion ", {"b": True}),
           ("is the sum of squared distances from each row of Y to its centre. True labels are not used.", {})],
          "On our grid that minimum sits where ARI = 1: distortion is about 0. Where ARI collapses, distortion rises to about 30.",
          [("The rule returns one winner. It does not say that ", {}),
           ("2σ₀ already fails", {"b": True}),
           (" on the circles.", {})]],
         size=13)
    notes(s, """
[75 s] Sigma sets the scale of every link. The formula is on the right: Aij is exp of minus the squared distance over
2 sigma squared. The diagonal is 0. A small sigma only connects nearby points. A large sigma also connects points that
are far apart. The whole graph gets one sigma.

On the left, the moons at our default sigma: ARI 1.00. At 2.8 times that sigma, ARI is 0.81. The dashed curve is the
two circles. At 2 sigma-zero it drops to 0, and the cut looks like ordinary K-means.

The paper chooses sigma by trying several values. For each one it runs K-means on the rows of Y and keeps the sigma
with the smallest distortion. Distortion is the sum of squared distances from those rows to their centres. True labels
are not used.

On this grid, distortion is about 0 while ARI is 1, and it rises to about 30 when ARI collapses. So the rule lands in
the successful range. It returns one winner. It does not tell you that doubling sigma already fails on the circles.
""")


def s_k(deck, D):
    k = D["k_page"]
    s = deck.slide("k is given by hand",
                   "Three rings we generated. Every panel uses the same default σ₀.", section=4)
    picture(s, D["img_k_page"], 0.7, 1.55, w=11.9)
    text(s, 0.7, 5.85, 11.9, 0.85,
         [[("k = 2 is a different question: ", {"b": True}),
           (f"ARI {F.fmt2(k['k2_vs_merge'])} against inner vs rest, "
            f"{F.fmt2(k[(2, 1.0)])} against three rings.", {})],
          [(f"The true k = 3 still scores {F.fmt2(k[(3, 1.0)])} at σ₀. ", {}),
           (f"Same k, at 0.7σ₀, ARI rises to {F.fmt2(k[(3, 0.7)])}.", {"b": True})]],
         size=15, align=PP_ALIGN.CENTER)
    notes(s, f"""
[60 s] The algorithm does not decide how many groups there are. We tell it k.

Left: three rings, the groups we want.
Middle: we ask for k = 2. It does not scatter the points at random. The inner ring is one group, and the middle and
outer rings are merged. Against that merge, ARI is {F.fmt2(k['k2_vs_merge'])}. Against the three rings, ARI is
{F.fmt2(k[(2, 1.0)])}. So k = 2 is the right answer to a different question.
Right: we ask for k = 3, the true number. At our default sigma, ARI is still only {F.fmt2(k[(3, 1.0)])}. The inner
ring comes out, and the outer two are mixed.

One more number, and then stop. Same k = 3: shrink sigma to 0.7 times the default, and ARI rises to
{F.fmt2(k[(3, 0.7)])}. The right count is not enough. The scale has to fit as well.
""")


def s_close(deck, D):
    m = D["moons_final"]
    s = deck.slide("Where this method fits",
                   "One scale, and k chosen by hand: that is what the last two slides marked out.", section=5)
    card(s, 0.55, 1.7, 6.0, 4.35, "Where it fits",
         ["Clusters are connected by local neighbours, but not cut by a straight line.",
          "One length scale, and k is known.",
          "Separated round clouds: K-means is already enough.",
          "Very different scales, or the wrong question for k: the last two slides."],
         size=15)
    card(s, 6.8, 1.7, 6.0, 4.35, "What the 2001 paper fixed",
         ["Spectral clustering already existed. People used the eigenvectors differently.",
          "This paper fixed one recipe: affinity graph, top-k eigenvectors, normalize each row, then K-means.",
          "A whole connected piece becomes one point. That is the step K-means could not do."],
         size=15)
    rect(s, 0.55, 6.2, 12.25, 0.72, fill=WARM)
    text(s, 0.75, 6.28, 11.9, 0.56,
         f"On our moons: K-means {F.fmt2(m['km'])}  →  NJW {F.fmt2(m['njw'])}. "
         "The recipe is what made the paper last.",
         size=15, anchor=B.MSO_ANCHOR.MIDDLE)
    notes(s, f"""
[50 s] Where does this method fit? When groups are connected through local neighbours, but a straight line cannot
separate them: the rings and the moons. The graph needs one length scale, and k has to be known. When the clusters are
already separated round clouds, K-means is enough. When the scales differ, or k is asking a different question, that
is the previous two slides.

The paper matters because of the recipe. In 2001, spectral clustering was already around, but authors turned
eigenvectors into clusters in different ways. Ng, Jordan and Weiss fixed one sequence that people then actually used:
an affinity graph, the top k eigenvectors, each row normalized, then K-means. Normalization is the step that makes a
whole connected piece into one point, which is what K-means can cut. On our moons that takes ARI from
{F.fmt2(m['km'])} to {F.fmt2(m['njw'])}.
""")


def main():
    D = F.build_v3()
    D["img_k_page"], D["k_page"] = F.k_page_png()
    deck = Deck()
    V3.s_title(deck)
    V3.s_q1(deck, D)
    V3.s_q2(deck, D)
    V3.s_q3(deck, D)
    V3.s_q4(deck, D)
    V3.s_q5(deck, D)
    V3.s_q6(deck, D)
    V3.s_q7(deck, D)
    s_sigma(deck, D)
    s_k(deck, D)
    s_close(deck, D)
    V3.s_backup_theory(deck, D)
    B.s_backup(deck, D)
    deck.prs.save(OUT)
    print(f"wrote {OUT}  ({len(deck.prs.slides._sldIdLst)} slides)")
    print("k_page", {str(k): (round(v, 3) if isinstance(v, float) else v) for k, v in D["k_page"].items()})


if __name__ == "__main__":
    main()
