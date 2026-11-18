#!/usr/bin/env python3
"""Gridded LAMP (GLMP) pre-processor for Dragon Splashdown Support.

Runs in the repository's Pages workflow (never in the browser). It downloads the newest complete GLMP cycle from
NOMADS, decodes the GRIB2 with eccodes, crops each field to a box, and writes what the page already reads:

  glmp/latest.json             cycle, per-field hours / scale / grid / threshold / status, box, build time
  glmp/<field>_fPPP.bin.gz     gzip'd little-endian uint16 cells (65535 = missing), rows south -> north
  glmp/inventory.txt           every GRIB message seen (shortName, discipline/category/number, probability limits...)

Fields (units before scaling): cig, vis (m) - wspd, wgst (m/s) - t, td (K) - sky, pop, cigp2, visp2 (%).
Deterministic ceiling "unlimited" (-1 / above 12,000 ft) is written as 30,000 m, which the page shows as Unlimited.

The crop box is lon0,lat0,lon1,lat1 from the GLMP_BOX environment variable (set it as a repository secret so the
exact box is not in the code). The published files still show the box's extent, so keep it generous.

  python glmp_build.py --out site/glmp            build
  python glmp_build.py --probe                     list the newest cycle and print every message of one file
"""
import argparse, concurrent.futures as cf, datetime as dt, gzip, json, math, os, re, sys, threading, time, urllib.request

import numpy as np

BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/glmp/prod/"
UA = "DragonSplashdownSupport-GLMP/1.0 (GitHub Pages weather briefing; hourly, <=90 req/min)"
DEFAULT_BOX = (-128.0, 26.0, -112.0, 40.0)
UNLIMITED_M = 30000.0
MIN_HOURLY = 20

# field -> (scale, label). Values are stored as round(value / scale).
FIELDS = {
    "cig":   (1.0,  "ceiling height (m)"),
    "vis":   (1.0,  "visibility (m)"),
    "wspd":  (0.01, "10 m wind speed (m/s)"),
    "wgst":  (0.01, "10 m wind gust (m/s)"),
    "t":     (0.01, "2 m temperature (K)"),
    "td":    (0.01, "2 m dew point (K)"),
    "sky":   (0.1,  "sky cover (%)"),
    "pop":   (0.1,  "1-h PoP (%)"),
    "cigp2": (0.1,  "P(ceiling < 500 ft) (%)"),
    "visp2": (0.1,  "P(visibility < 1 mile) (%)"),
}
PROB_TARGET = {"cigp2": 152.4, "visp2": 1609.344}       # metres
FCSTS_FILE = {"wspd": "wspd", "wgst": "wgst", "t": "t", "td": "td", "sky": "sky", "cig": "cig", "vis": "vis",
              "cigp2": "cigp2", "visp2": "visp2"}       # whole-run files used only for fields the hourly files lack


# ----------------------------------------------------------------------------------------------- polite HTTP
class Throttle:
    def __init__(self, per_min):
        self.gap, self.lock, self.next = 60.0 / per_min, threading.Lock(), 0.0
    def wait(self):
        with self.lock:
            now = time.monotonic(); t = max(now, self.next); self.next = t + self.gap
        time.sleep(max(0.0, t - now))

THROTTLE = Throttle(90)

def get(url, tries=3):
    for k in range(tries):
        THROTTLE.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1: raise
            print(f"  retry {url}: {e}", file=sys.stderr); time.sleep(5 * (k + 1))

def listing(url):
    return re.findall(r'href="([^"?/][^"]*)"', get(url).decode("utf-8", "replace"))


# ----------------------------------------------------------------------------------------------- cycle choice
def choose_cycle():
    """newest cycle with at least MIN_HOURLY hourly CONUS files, looking at the two newest date folders"""
    days = sorted(x.rstrip("/") for x in listing(BASE) if re.match(r"glmp\.\d{8}/?$", x))[-2:]
    hourly, fcsts = {}, {}
    for d in days:
        for f in listing(BASE + d + "/"):
            m = re.match(r"glmp\.t(\d{2})30z\.hourly\.f(\d{3})\.co\.grib2$", f)
            if m and int(m.group(2)) >= 1: hourly.setdefault((d, m.group(1)), set()).add(int(m.group(2)))
            m = re.match(r"glmp\.t(\d{2})30z\.fcsts_(\w+)\.g\.co\.grib2$", f)
            if m: fcsts.setdefault((d, m.group(1)), set()).add(m.group(2))
    ok = sorted((k for k, v in hourly.items() if len(v) >= MIN_HOURLY), reverse=True)
    if not ok: raise SystemExit(f"no GLMP cycle with at least {MIN_HOURLY} hourly files posted")
    d, hh = ok[0]
    return d, hh, sorted(hourly[(d, hh)]), fcsts.get((d, hh), set())


# ----------------------------------------------------------------------------------------------- projection
def earth_radius(h):
    shape = h.get("shapeOfTheEarth", 6)
    if shape == 1: return h["scaledValueOfRadiusOfSphericalEarth"] / 10 ** h["scaleFactorOfRadiusOfSphericalEarth"]
    return {0: 6367470.0, 6: 6371229.0}.get(shape, 6371200.0)

class LCC:
    """Same Lambert conformal maths and index convention as the page's lccIndex()/gIdx()."""
    def __init__(self, g):
        self.g = g; r = math.pi / 180; q = math.pi / 4
        p1, p2 = g["latin1"] * r, g["latin2"] * r
        self.n = math.sin(p1) if abs(p1 - p2) < 1e-10 else \
            math.log(math.cos(p1) / math.cos(p2)) / math.log(math.tan(q + p2 / 2) / math.tan(q + p1 / 2))
        self.F = math.cos(p1) * math.tan(q + p1 / 2) ** self.n / self.n
        self.x0, self.y0 = self.xy(g["la1"], g["lo1"])
        self.si = -1 if g["scan"] & 0x80 else 1
        self.sj = 1 if g["scan"] & 0x40 else -1
    def xy(self, la, lo):
        r = np.pi / 180; q = np.pi / 4
        rho = self.g["R"] * self.F / np.tan(q + np.asarray(la) * r / 2) ** self.n
        dl = (((np.asarray(lo) - self.g["lov"]) % 360 + 540) % 360 - 180) * r
        return rho * np.sin(self.n * dl), -rho * np.cos(self.n * dl)
    def latlon(self, x, y):
        n = self.n; rho = np.sign(n) * np.hypot(x, y)
        th = np.arctan2(np.sign(n) * x, -np.sign(n) * y)
        lon = self.g["lov"] + np.degrees(th / n)
        lat = np.degrees(2 * np.arctan((self.g["R"] * self.F / rho) ** (1 / n)) - np.pi / 2)
        return lat, ((lon + 180) % 360) - 180
    def ij(self, la, lo):                         # fractional grid column / row
        x, y = self.xy(la, lo)
        return (x - self.x0) / (self.si * self.g["dx"]), (y - self.y0) / (self.sj * self.g["dy"])
    def ij_xy(self, i, j):
        return self.x0 + self.si * i * self.g["dx"], self.y0 + self.sj * j * self.g["dy"]

def rows_reversed(vals, g):
    """True if alternate rows are stored reversed (scanning-mode bit 0x10 honoured in storage). Picks the ordering
    with the smaller row-to-row jump, which is unmistakable on any real field (wrong order gives stripes)."""
    if not g["scan"] & 0x10: return False
    def jump(rev):
        A = to_2d(np.where(np.abs(vals) < 1e10, vals, np.nan), g, rev)
        return np.nanmean(np.abs(np.diff(A, axis=0))) if not g["scan"] & 0x20 else np.nanmean(np.abs(np.diff(A, axis=1)))
    with np.errstate(all="ignore"): a, b = jump(True), jump(False)
    return not (b < a)

def to_2d(vals, g, reverse_alt=True):
    """storage order -> A[j, i] (row j, column i) using the scanning-mode bits."""
    nx, ny, s = g["nx"], g["ny"], g["scan"]
    if s & 0x20:
        B = vals.reshape(nx, ny).copy()
        if (s & 0x10) and reverse_alt: B[1::2] = B[1::2, ::-1]
        return B.T
    A = vals.reshape(ny, nx).copy()
    if (s & 0x10) and reverse_alt: A[1::2] = A[1::2, ::-1]
    return A

class Cropper:
    """Crops a native grid to the box and returns (sub-grid definition for the page, function A -> sub-array)."""
    def __init__(self, g, box, reverse_alt):
        self.p = LCC(g); self.reverse_alt = reverse_alt
        lo0, la0, lo1, la1 = box
        edge_lo = np.r_[np.linspace(lo0, lo1, 40), np.full(40, lo1), np.linspace(lo1, lo0, 40), np.full(40, lo0)]
        edge_la = np.r_[np.full(40, la0), np.linspace(la0, la1, 40), np.full(40, la1), np.linspace(la1, la0, 40)]
        fi, fj = self.p.ij(edge_la, edge_lo)
        self.i0 = max(0, int(math.floor(fi.min())) - 1); self.i1 = min(g["nx"] - 1, int(math.ceil(fi.max())) + 1)
        self.j0 = max(0, int(math.floor(fj.min())) - 1); self.j1 = min(g["ny"] - 1, int(math.ceil(fj.max())) + 1)
        if self.i0 > self.i1 or self.j0 > self.j1: raise ValueError("box is outside this grid")
        x, y = self.p.ij_xy(self.i0, self.j0); la, lo = self.p.latlon(x, y)
        # keep the native orientation bits (0x80 / 0x40); the cropped block is stored plainly, row by row
        self.sub = dict(R=g["R"], nx=self.i1 - self.i0 + 1, ny=self.j1 - self.j0 + 1, la1=round(float(la), 6),
                        lo1=round(float(lo) % 360, 6), lad=g["lad"], lov=g["lov"], dx=g["dx"], dy=g["dy"],
                        scan=g["scan"] & 0xC0, latin1=g["latin1"], latin2=g["latin2"])
    def __call__(self, vals, g):
        return to_2d(vals, g, self.reverse_alt)[self.j0:self.j1 + 1, self.i0:self.i1 + 1]


# ----------------------------------------------------------------------------------------------- GRIB decode
KEYS_I = ["discipline", "parameterCategory", "parameterNumber", "productDefinitionTemplateNumber",
          "gridDefinitionTemplateNumber", "Nx", "Ny", "scanningMode", "shapeOfTheEarth",
          "scaledValueOfRadiusOfSphericalEarth", "scaleFactorOfRadiusOfSphericalEarth", "validityDate",
          "validityTime", "probabilityType", "scaleFactorOfLowerLimit", "scaledValueOfLowerLimit",
          "scaleFactorOfUpperLimit", "scaledValueOfUpperLimit"]
KEYS_F = ["latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees", "LaDInDegrees",
          "LoVInDegrees", "DxInMetres", "DyInMetres", "Latin1InDegrees", "Latin2InDegrees", "missingValue",
          "scaledValueOfFirstFixedSurface"]

def header(eccodes, gid):
    """metadata + decoded values (raw storage order, under "_vals") of one eccodes handle"""
    h = {}
    for k in KEYS_I:
        try: h[k] = eccodes.codes_get_long(gid, k)
        except Exception: pass  # noqa: BLE001
    for k in KEYS_F:
        try: h[k] = eccodes.codes_get_double(gid, k)
        except Exception: pass  # noqa: BLE001
    for k in ("shortName", "units", "typeOfLevel", "stepRange"):
        try: h[k] = eccodes.codes_get_string(gid, k)
        except Exception: pass  # noqa: BLE001
    h["_vals"] = eccodes.codes_get_values(gid)
    return h

def grib_messages(path):
    """yield a header dict for every message in a GRIB2 file"""
    import eccodes
    eccodes.codes_grib_multi_support_on()
    with open(path, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None: break
            try: h = header(eccodes, gid)
            finally: eccodes.codes_release(gid)
            yield h

def decode_bytes(buf):
    """header dict for a single GRIB2 message held in memory (an HTTP byte range)"""
    import eccodes
    gid = eccodes.codes_new_from_message(bytes(buf))
    try: return header(eccodes, gid)
    finally: eccodes.codes_release(gid)

def grid_of(h):
    return dict(R=earth_radius(h), nx=h["Nx"], ny=h["Ny"], la1=h["latitudeOfFirstGridPointInDegrees"],
                lo1=h["longitudeOfFirstGridPointInDegrees"] % 360, lad=h["LaDInDegrees"], lov=h["LoVInDegrees"] % 360,
                dx=h["DxInMetres"], dy=h["DyInMetres"], scan=h.get("scanningMode", 0x40),
                latin1=h["Latin1InDegrees"], latin2=h["Latin2InDegrees"])

def limit(h, which):
    s, v = h.get(f"scaleFactorOf{which}Limit"), h.get(f"scaledValueOf{which}Limit")
    return None if s is None or v is None else v * 10.0 ** (-s)

def is_prob(h): return h.get("productDefinitionTemplateNumber") in (5, 9) and "probabilityType" in h

def prob_below(h):
    """threshold for 'probability of value below X' records (type 0: lower limit, type 4: upper limit), else None."""
    t = h.get("probabilityType")
    return limit(h, "Lower") if t == 0 else limit(h, "Upper") if t == 4 else None

def classify(h):
    """-> (field, threshold or None) from GRIB metadata, or None if not a field we keep."""
    d, c, n = h.get("discipline"), h.get("parameterCategory"), h.get("parameterNumber")
    if d != 0: return None
    p = is_prob(h)
    if (c, n) == (6, 13): return ("cigp2", prob_below(h)) if p else ("cig", None)
    if (c, n) == (19, 0): return ("visp2", prob_below(h)) if p else ("vis", None)
    if p:
        if c == 1:
            t = h.get("probabilityType")
            return ("pop", limit(h, "Lower") if t == 3 else limit(h, "Upper") if t == 1 else (limit(h, "Lower") or limit(h, "Upper")))
        return None
    return {(2, 1): ("wspd", None), (2, 22): ("wgst", None), (0, 0): ("t", None), (0, 6): ("td", None),
            (6, 1): ("sky", None)}.get((c, n))

def valid_time(h):
    d, t = str(h["validityDate"]), int(h["validityTime"])
    return dt.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), t // 100, t % 100, tzinfo=dt.timezone.utc)

def clean(vals, h, field):
    v = np.asarray(vals, dtype=np.float64).copy()
    miss = h.get("missingValue")
    bad = ~np.isfinite(v) | (np.abs(v) > 1e10)
    if miss is not None: bad |= v == miss
    if field == "cig":
        unl = (~bad) & (v < 0)
        v[unl] = UNLIMITED_M
    if field in ("sky", "pop", "cigp2", "visp2"):
        bad |= (v < -0.5) | (v > 100.5)
    v[bad] = np.nan
    return v

def encode(a, scale):
    out = np.full(a.shape, 65535, dtype="<u2"); ok = np.isfinite(a)
    out[ok] = np.clip(np.rint(a[ok] / scale), 0, 65533).astype("<u2")
    return gzip.compress(out.tobytes(), compresslevel=6)


# ----------------------------------------------------------------------------------------------- build
def probe():
    d, hh, hours, fc = choose_cycle()
    print(f"newest date folders checked; using {d} cycle {hh}30Z: {len(hours)} hourly files f{hours[0]:03d}-f{hours[-1]:03d}")
    print("fcsts files:", ", ".join(sorted(fc)) or "none")
    names = listing(BASE + d + "/")
    print(".idx files present:", any(x.endswith(".idx") for x in names))
    url = f"{BASE}{d}/glmp.t{hh}30z.hourly.f001.co.grib2"
    data = get(url); print(f"{url.rsplit('/', 1)[1]}: {len(data) / 1e6:.1f} MB")
    path = "/tmp/glmp_probe.grib2"; open(path, "wb").write(data)
    for h in grib_messages(path): print(describe(h))

def describe(h, with_valid=True):
    v = np.asarray(h["_vals"], dtype=float); miss = h.get("missingValue")
    ok = np.isfinite(v) & (np.abs(v) < 1e10) & ((v != miss) if miss is not None else True)
    rng = f"{v[ok].min():.4g}..{v[ok].max():.4g}" if ok.any() else "all missing"
    return (f"{h.get('shortName','?'):>8} d{h.get('discipline')} c{h.get('parameterCategory')} n{h.get('parameterNumber')}"
            f" pdt{h.get('productDefinitionTemplateNumber')} ptype={h.get('probabilityType','-')}"
            f" lo={limit(h,'Lower')} hi={limit(h,'Upper')} level={h.get('typeOfLevel','?')}"
            f"/{h.get('scaledValueOfFirstFixedSurface','')} step={h.get('stepRange','?')} units={h.get('units','?')}"
            + (f" valid={h.get('validityDate')}/{h.get('validityTime', 0):04d}" if with_valid else "") + f" range={rng}")

def build(out, box):
    t_start = time.time()
    os.makedirs(out, exist_ok=True)
    d, hh, hours, fcsts = choose_cycle()
    cycle = dt.datetime.strptime(d[5:] + hh, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)   # HH:00, not HH:30
    print(f"cycle {d} {hh}30Z (valid times counted from {cycle:%Y-%m-%d %H}Z), hourly files: {len(hours)}")
    meta = dict(cycle=cycle.isoformat().replace("+00:00", "Z"), run=f"{hh}30Z",
                built=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                box=list(box), enc="gzip", grids={}, fields={}, source=f"{BASE}{d}/", notes=[])
    inv, croppers, written, thr_used = set(), {}, {}, {}

    def cropper_for(h):
        g = grid_of(h); key = f"{g['nx']}x{g['ny']}_{g['la1']:.3f}_{g['lo1']:.3f}_{g['scan']}"
        if key not in croppers:
            rev = rows_reversed(np.asarray(h["_vals"], dtype=float), g)
            if g["scan"] & 0x10: meta["notes"].append(f"grid {key}: alternate rows {'reversed' if rev else 'in order'} in decoded values")
            croppers[key] = Cropper(g, box, rev); meta["grids"][key] = croppers[key].sub
        return key, g, croppers[key]

    def take(path, label, want):
        """decode one file (main thread only) and write every wanted field/hour it holds"""
        got = 0
        for h in grib_messages(path):
            inv.add(f"[{label.split(' ')[0]}] " + describe(h, with_valid=False))
            cl = classify(h)
            if not cl or cl[0] not in want: continue
            field, thr = cl
            if field in PROB_TARGET or field == "pop":
                target = PROB_TARGET.get(field, 0.254)
                if thr is None or abs(thr - target) > max(0.15 * target, 0.05): continue
            try: vt = valid_time(h)
            except Exception: continue  # noqa: BLE001
            f = round((vt - cycle).total_seconds() / 3600)
            if f < 0 or f > 48 or (field, f) in written: continue
            if label.startswith("hourly f") and f != int(label[8:]):
                meta["notes"].append(f"{label}: GRIB valid time gives f{f:03d} for {field}")
            key, g, crop = cropper_for(h)
            a = crop(clean(h["_vals"], h, field), g)
            with open(os.path.join(out, f"{field}_f{f:03d}.bin.gz"), "wb") as fh: fh.write(encode(a, FIELDS[field][0]))
            written[(field, f)] = key
            if thr is not None: thr_used[field] = thr
            got += 1
        return got

    def run(jobs):
        """download in parallel (4 workers, throttled), decode one file at a time; a failed file is noted and skipped"""
        def dl(job):
            url, label, _ = job; path = "/tmp/glmp_" + re.sub(r"\W", "_", label) + ".grib2"
            data = get(url); open(path, "wb").write(data); return path, len(data)
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(dl, j): j for j in jobs}
            for fu in cf.as_completed(futs):
                url, label, want = futs[fu]
                try:
                    path, size = fu.result(); n = take(path, label, want); os.remove(path)
                    print(f"  {label}: {size / 1e6:.1f} MB, {n} records kept")
                except Exception as e:  # noqa: BLE001
                    meta["notes"].append(f"{label}: failed ({e})"); print(f"  {label}: FAILED {e}", file=sys.stderr)

    run([(f"{BASE}{d}/glmp.t{hh}30z.hourly.f{f:03d}.co.grib2", f"hourly f{f:03d}", set(FIELDS)) for f in hours])
    # whole-run fcsts_* files only for fields the hourly files did not carry (or carried for too few hours)
    need = [fl for fl in FIELDS if sum(1 for (x, _) in written if x == fl) < MIN_HOURLY]
    run([(f"{BASE}{d}/glmp.t{hh}30z.fcsts_{FCSTS_FILE[fl]}.g.co.grib2", f"fcsts_{FCSTS_FILE[fl]}", {fl})
         for fl in need if FCSTS_FILE.get(fl) in fcsts])

    for fl, (scale, label) in FIELDS.items():
        hrs = sorted(f for (x, f) in written if x == fl)
        if not hrs:
            meta["fields"][fl] = dict(status="missing", label=label, hours=[]); continue
        keys = sorted({written[(fl, f)] for f in hrs})
        if len(keys) > 1:      # keep only the hours on the most common grid
            best = max(keys, key=lambda k: sum(written[(fl, f)] == k for f in hrs)); hrs = [f for f in hrs if written[(fl, f)] == best]
            meta["notes"].append(f"{fl}: hours came on {len(keys)} grids; kept {best}"); keys = [best]
        meta["fields"][fl] = dict(status="ok" if len(hrs) >= 36 else "partial", label=label, hours=hrs, scale=scale, grid=keys[0])
        if fl in thr_used: meta["fields"][fl]["threshold_m"] = round(thr_used[fl], 2)
    st = [v["status"] for v in meta["fields"].values()]
    meta["status"] = "ok" if all(x == "ok" for x in st) else "missing" if all(x == "missing" for x in st) else "partial"
    meta["seconds"] = round(time.time() - t_start)
    with open(os.path.join(out, "latest.json"), "w") as fh: json.dump(meta, fh, indent=1)
    with open(os.path.join(out, "inventory.txt"), "w") as fh:
        fh.write(f"GLMP {d} {hh}30Z, built {meta['built']}. One line per distinct record type.\n" + "\n".join(sorted(inv)) + "\n")
    print(json.dumps({k: (v["status"], len(v["hours"])) for k, v in meta["fields"].items()}), meta["status"])
    if meta["status"] == "missing": raise SystemExit("no GLMP fields decoded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="glmp")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe: return probe()
    box = DEFAULT_BOX
    if os.environ.get("GLMP_BOX"):
        box = tuple(float(x) for x in os.environ["GLMP_BOX"].split(","))
        if len(box) != 4: raise SystemExit("GLMP_BOX must be lon0,lat0,lon1,lat1")
    build(a.out, box)

if __name__ == "__main__":
    main()
