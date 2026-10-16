"""Draw an explanatory uniaxial slab ray path and optical indicatrix.

Only the uniaxial illustration uses the terms ordinary/extraordinary. The
independent-index studies can be biaxial and retain the neutral X/Y mode names.
The internal energy-flow directions are recovered from the real optical solver.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lorem_warped import refraction_distortion_matrices

NO, NE, THICKNESS_MM = 1.5, 2.0, 5.0
DIRECTION = (-1.0, 0.0, 1.0)  # 45 degrees from slab normal, in the X-Z plane
REFERENCE = "https://www2.tulane.edu/~sanelson/eens211/uniaxial_minerals.htm"


def _unit(a):
    length = math.sqrt(sum(v * v for v in a))
    return tuple(v / length for v in a)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _convex_hull(points):
    points = sorted(set(points))
    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
    lower, upper = [], []
    for part, subset in ((lower, points), (upper, reversed(points))):
        for point in subset:
            while len(part) >= 2 and cross(part[-2], part[-1], point) <= 0:
                part.pop()
            part.append(point)
    return lower[:-1] + upper[:-1]


def build_schematic(path: Path) -> dict:
    """Save a print-resolution PNG and its exact model parameters to metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorting in the solver is by X polarization, not o/e. For this particular
    # uniaxial optic-axis-Z case, the ordinary ray has the isotropic n_o slope.
    matrices = refraction_distortion_matrices(
        thickness_mm=THICKNESS_MM, n_x=NO, n_y=NO, n_z=NE,
        direction_xyz=DIRECTION)
    offsets = [tuple(float(v) for v in matrices[i, :2, 2]) for i in range(2)]
    air = (DIRECTION[0] / DIRECTION[2], DIRECTION[1] / DIRECTION[2])
    slopes = [(air[0] + dx / THICKNESS_MM, air[1] + dy / THICKNESS_MM)
              for dx, dy in offsets]
    incident = _unit(DIRECTION)
    kz_o = math.sqrt(NO**2 - incident[0]**2 - incident[1]**2)
    ordinary_slope = (incident[0] / kz_o, incident[1] / kz_o)
    ordinary = min(range(2), key=lambda i: math.dist(slopes[i], ordinary_slope))
    extraordinary = 1 - ordinary
    assert math.dist(slopes[ordinary], ordinary_slope) < 0.00001
    assert math.dist(slopes[ordinary], slopes[extraordinary]) > 0.1

    image = Image.new("RGB", (1900, 950), "#f8fbfd")
    d = ImageDraw.Draw(image)
    title = ImageFont.truetype("DejaVuSans-Bold.ttf", 35)
    sub = ImageFont.truetype("DejaVuSans-Bold.ttf", 29)
    body = ImageFont.truetype("DejaVuSans.ttf", 27)
    small = ImageFont.truetype("DejaVuSans.ttf", 25)
    ink, muted, ordinary_color, extra_color = "#193247", "#526777", "#117dac", "#c5662b"
    d.rounded_rectangle((25, 23, 945, 925), radius=20, fill="white", outline="#c9d9e4", width=2)
    d.rounded_rectangle((965, 23, 1875, 925), radius=20, fill="white", outline="#c9d9e4", width=2)
    d.text((65, 57), "A  |  Refraction through a slab", font=title, fill=ink)
    d.text((1005, 57), "B  |  Optical indicatrix", font=title, fill=ink)
    d.text((67, 95), "Light enters 45° from straight-through Z", font=body, fill=muted)
    d.text((1007, 95), "A slice across the light's travel direction", font=body, fill=muted)

    def left(x, y, z):
        return (round(520 + 165*x - 105*y), round(510 + 80*x + 52*y - 210*z))
    xy = [(-1.2, -1), (1.2, -1), (1.2, 1), (-1.2, 1)]
    bottom = [left(x, y, 0) for x, y in xy]
    top = [left(x, y, 1.1) for x, y in xy]
    d.polygon([bottom[0], bottom[1], top[1], top[0]], fill="#dcebf1")
    d.polygon([bottom[1], bottom[2], top[2], top[1]], fill="#edf6f8")
    d.polygon(top, fill="#e3f2f7")
    d.line(bottom + [bottom[0]], fill="#7b9eaf", width=4)
    d.line(top + [top[0]], fill="#527d93", width=5)
    for b, t in zip(bottom, top):
        d.line((b, t), fill="#88a8b8", width=3)

    def arrow(p, q, color, width=8):
        d.line((p, q), fill=color, width=width, joint="curve")
        angle = math.atan2(q[1]-p[1], q[0]-p[0])
        wing = 18
        a = (round(q[0]-wing*math.cos(angle-0.45)), round(q[1]-wing*math.sin(angle-0.45)))
        b = (round(q[0]-wing*math.cos(angle+0.45)), round(q[1]-wing*math.sin(angle+0.45)))
        d.polygon([q, a, b], fill=color)

    # All three unit axes share an origin on the lower slab face. They use
    # the same isometric projection as the slab and the computed rays.
    axis_origin = left(-1.35, 1.1, 0)
    arrow(axis_origin, left(-1.0, 1.1, 0), ink, 4)
    arrow(axis_origin, left(-1.35, 1.45, 0), ink, 4)
    arrow(axis_origin, left(-1.35, 1.1, .33), ink, 4)
    d.text((axis_origin[0] + 53, axis_origin[1] + 20), "X", font=sub, fill=ink)
    d.text((axis_origin[0] - 58, axis_origin[1] + 12), "Y", font=sub, fill=ink)
    d.text((axis_origin[0] - 15, axis_origin[1] - 98), "Z", font=sub, fill=ink)

    entry = (.4, -.35)
    start = left(entry[0] - air[0]*.58, entry[1] - air[1]*.58, -.58)
    boundary = left(*entry, 0)
    arrow(start, boundary, "#245d70", 8)
    d.text((83, 682), "Incident polarized beam", font=sub, fill="#245d70")
    d.text((83, 722), "air → crystal · 45° from upright Z", font=small, fill=muted)
    d.line((520, 718, start[0]-8, start[1]+8), fill="#7898a5", width=2)
    d.ellipse((boundary[0]-8, boundary[1]-8, boundary[0]+8, boundary[1]+8), fill="#245d70")
    for label, idx, color, location in (("Ordinary beam (o)", ordinary, ordinary_color, (65, 326)),
                                         ("Extraordinary (e)", extraordinary, extra_color, (655, 385))):
        slope = slopes[idx]
        at_top = (entry[0] + 1.1*slope[0], entry[1] + 1.1*slope[1])
        exit_point = left(*at_top, 1.1)
        after = left(at_top[0] + air[0]*.38, at_top[1] + air[1]*.38, 1.48)
        arrow(boundary, exit_point, color, 9)
        arrow(exit_point, after, color, 7)
        d.text(location, label, font=sub, fill=color)
        leader_from = (location[0]+195, location[1]+29)
        d.line((leader_from, exit_point), fill=color, width=2)
        d.ellipse((exit_point[0]-5, exit_point[1]-5, exit_point[0]+5, exit_point[1]+5), fill=color)
    d.text((731, 553), "5 mm", font=body, fill=ink)
    d.text((65, 785), "X and Y run across the slab; Z runs through it.",
           font=small, fill=muted)
    d.text((65, 855), "Exit beams are parallel in air; their images shift.",
           font=small, fill=muted)

    # The index ellipsoid is x²/n_o² + y²/n_o² + z²/n_e² = 1.
    # A plane normal to the ordinary transmitted wave normal intersects it
    # in an ellipse with one ordinary semiaxis and one extraordinary semiaxis.
    wave_normal = _unit((incident[0], incident[1], kz_o))
    ordinary_axis = _unit(_cross((0, 0, 1), wave_normal))
    extraordinary_axis = _unit(_cross(wave_normal, ordinary_axis))
    def radius(direction):
        return 1 / math.sqrt((direction[0]**2 + direction[1]**2)/NO**2
                             + direction[2]**2/NE**2)
    r_o, r_e = radius(ordinary_axis), radius(extraordinary_axis)
    def right(p):
        x, y, z = p
        return (round(1412 + 110*x - 70*y), round(465 + 53*x + 43*y - 110*z))
    def vec(a, m, b=None, n: float=0):
        return tuple(a[i]*m + ((b[i]*n) if b is not None else 0) for i in range(3))
    shell = []
    for ia in range(81):
        theta = math.pi * ia / 80
        for ib in range(161):
            phi = 2*math.pi*ib/160
            shell.append(right((NO*math.sin(theta)*math.cos(phi),
                                NO*math.sin(theta)*math.sin(phi), NE*math.cos(theta))))
    d.polygon(_convex_hull(shell), fill="#eef3f8", outline="#99b2c4", width=3)
    for phi in (0, math.pi/4, math.pi/2, 3*math.pi/4):
        d.line([right((NO*math.sin(t)*math.cos(phi),
                       NO*math.sin(t)*math.sin(phi), NE*math.cos(t)))
                for t in [i*math.pi/80 for i in range(81)]], fill="#c2d1de", width=2)
    plane = [right(vec(ordinary_axis, a, extraordinary_axis, b))
             for a, b in ((-2.05,-2.05),(2.05,-2.05),(2.05,2.05),(-2.05,2.05))]
    wash = Image.new("RGBA", image.size)
    ImageDraw.Draw(wash).polygon(plane, fill=(46, 153, 180, 35), outline=(26, 122, 160, 140), width=3)
    image = Image.alpha_composite(image.convert("RGBA"), wash).convert("RGB")
    d = ImageDraw.Draw(image)
    section = [right(vec(ordinary_axis, r_o*math.cos(t),
                         extraordinary_axis, r_e*math.sin(t)))
               for t in [i*2*math.pi/160 for i in range(161)]]
    d.line(section, fill="#087fa4", width=8, joint="curve")
    origin = right((0, 0, 0))
    arrow(origin, right(vec(ordinary_axis, r_o*.72)), ordinary_color, 6)
    arrow(origin, right(vec(extraordinary_axis, r_e*.72)), extra_color, 6)
    arrow(right(vec(wave_normal, -1.75)), right(vec(wave_normal, -.95)), "#245d70", 7)
    d.line((origin, right(vec(wave_normal, 2.15))), fill="#526777", width=3)
    d.text((1150, 708), "X/Y: along the face · Z: through the slab", font=small, fill=muted)
    d.text((1150, 752), "n_o = 1.5  ·  ordinary polarization", font=body, fill=ordinary_color)
    d.text((1150, 793), "n_e = 2.0  ·  optic-axis index", font=body, fill=extra_color)
    d.text((1150, 836), "Slice at 90° to travel: n_o and n_e(θ)",
           font=small, fill=muted)
    d.text((1660, 239), "optic axis Z", font=small, fill=ink)
    d.line((1650, 260, right((0, 0, NE))[0]+5, right((0, 0, NE))[1]+10), fill="#66798b", width=2)
    d.text((1020, 567), "transmitted k_t", font=small, fill="#245d70")
    d.text((1650, 640), "cross-section", font=small, fill="#087fa4")
    image.save(path)
    incident_angle = math.degrees(math.atan2(math.hypot(*DIRECTION[:2]), DIRECTION[2]))
    return {"path": str(path), "optic_case": "uniaxial", "n_x": NO, "n_y": NO,
            "n_z": NE, "thickness_mm": THICKNESS_MM, "direction_xyz": DIRECTION,
            "incident_angle_deg": incident_angle,
            "axes": {"X": "first flat slab edge", "Y": "second flat slab edge",
                     "Z": "through the slab thickness"},
            "mode_offsets_mm": offsets, "ordinary_mode": ordinary,
            "extraordinary_mode": extraordinary,
            "ordinary_slope_xy_per_z": slopes[ordinary],
            "extraordinary_slope_xy_per_z": slopes[extraordinary],
            "section_semiaxes": (r_o, r_e), "reference": REFERENCE}
