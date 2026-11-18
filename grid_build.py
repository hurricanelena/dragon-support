#!/usr/bin/env python3
"""Pre-crops the NBM and RRFS-ensemble (REFS) GRIB2 records that Dragon Splashdown Support draws, so the page can
download small files instead of whole CONUS records and decoding them in the browser.

The page keeps all of its own logic (which run, which hour, 1-h vs 6-h PoP, thresholds, colours, smoothing). This
script only mirrors the records that logic asks for, keyed by "<file URL>#<byte offset>" exactly as the page's
cachedRange() keys them, plus the .idx text of each file so the page can pick the run without a network round trip.

Output per model (nbm/, refs/):
  latest.json      cycle(s), box, grids, records { "<url>#<start>": [file, grid, offset, scale] }, status
  idx.json.gz      { "<url>": "<.idx text>" } for every file mirrored
  r/<hash>.bin.gz  gzip'd uint16 cells; value = offset + u * scale, 65535 = missing

  python grid_build.py --out site            (box from GLMP_BOX, same as Gridded LAMP)
"""
import argparse, concurrent.futures as cf, datetime as dt, gzip, hashlib, json, os, re, sys, time, urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmp_build as G  # noqa: E402  (projection, crop, decode helpers)

UA = "DragonSplashdownSupport-grids/1.0 (GitHub Pages weather briefing)"
HOURS = range(1, 61)

NBM = dict(
    name="nbm", base="https://noaa-nbm-grib2-pds.s3.amazonaws.com", step=1, back=9,   # qmd files are not posted every hour
    url=lambda base, ymd, hh, kind, f: f"{base}/blend.{ymd}/{hh}/{kind}/blend.t{hh}z.{kind}.f{f:03d}.co.grib2",
    kinds={
        "core": [r":WIND:10 m above ground:\d+ hour fcst:$", r":CEIL:cloud ceiling:\d+ hour fcst:$",
                 r":VIS:surface:\d+ hour fcst:$", r":APCP:surface:\d+-\d+ hour acc fcst:prob >0\.254:"],
        "qmd":  [r":WIND:10 m above ground:\d+ hour fcst:ens mean", r":WIND:10 m above ground:\d+ hour fcst:ens std dev"],
    },
    complete=36)
REFS = dict(
    name="refs", base="https://noaa-rrfs-ops-pds.s3.amazonaws.com", step=6, back=4,
    url=lambda base, ymd, hh, kind, f: f"{base}/refs.{ymd}/{hh}/ensprod/refs.t{hh}z.{kind}.f{f:02d}.conus.grib2",
    kinds={
        "eas":  [r":APCP:surface:\d+-\d+ hour acc fcst:prob >0\.254:"],
        "mean": [r":WIND:10 m above ground:\d+ hour fcst:wt ens mean", r":HGT:cloud ceiling:\d+ hour fcst:wt ens mean"],
        "sprd": [r"1::WIND:10 m above ground:\d+ hour fcst:"],   # "1:" = first match only, as the page's indexOf picks
    },
    complete=48)


def get(url, rng=None, tries=3):
    hdr = {"User-Agent": UA}
    if rng: hdr["Range"] = f"bytes={rng[0]}-{'' if rng[1] is None else rng[1]}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404): return None
            if k == tries - 1: raise
        except Exception:  # noqa: BLE001
            if k == tries - 1: raise
        time.sleep(3 * (k + 1))

def records(idx, pats):
    """[(start, end or None, line)] for idx lines matching any pattern, end = next record's start - 1"""
    lines = idx.strip().splitlines(); out = []; done = set()
    for i, l in enumerate(lines):
        hit = [p for p in pats if re.search(p[2:] if p.startswith("1:") else p, l.strip()) and p not in done]
        if hit:
            done.update(p for p in hit if p.startswith("1:"))
            start = int(l.split(":")[1]); end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            out.append((start, end, l))
    return out

def choose_cycle(M, kind, now):
    """newest run (same rule as the page: newest first) whose file at hour M['complete'] is posted"""
    top = now.replace(minute=0, second=0, microsecond=0)
    top -= dt.timedelta(hours=top.hour % M["step"])
    for k in range(M["back"] + 1):
        cyc = top - dt.timedelta(hours=k * M["step"])
        u = M["url"](M["base"], cyc.strftime("%Y%m%d"), cyc.strftime("%H"), kind, M["complete"])
        if get(u + ".idx"): return cyc
    return None

def encode_lin(a):
    ok = np.isfinite(a)
    if not ok.any(): return gzip.compress(np.full(a.shape, 65535, "<u2").tobytes()), 0.0, 1.0
    lo, hi = float(a[ok].min()), float(a[ok].max()); sc = (hi - lo) / 65533 if hi > lo else 1.0
    u = np.full(a.shape, 65535, "<u2"); u[ok] = np.clip(np.rint((a[ok] - lo) / sc), 0, 65533)
    return gzip.compress(u.tobytes(), compresslevel=6), lo, sc

def build_model(M, out, box):
    t0 = time.time(); now = dt.datetime.now(dt.timezone.utc)
    d = os.path.join(out, M["name"]); os.makedirs(os.path.join(d, "r"), exist_ok=True)
    meta = dict(model=M["name"], built=now.isoformat(timespec="seconds").replace("+00:00", "Z"), box=list(box),
                cycles={}, grids={}, records={}, notes=[])
    idx_all, jobs = {}, []
    # 1) pick the run per file type and read every .idx (small) in parallel
    for kind, pats in M["kinds"].items():
        cyc = choose_cycle(M, kind, now)
        if not cyc: meta["notes"].append(f"{kind}: no complete run found"); continue
        meta["cycles"][kind] = cyc.isoformat().replace("+00:00", "Z")
        ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
        urls = [M["url"](M["base"], ymd, hh, kind, f) for f in HOURS]
        with cf.ThreadPoolExecutor(8) as ex:
            for u, txt in zip(urls, ex.map(lambda u: get(u + ".idx"), urls)):
                if not txt: continue
                txt = txt.decode("utf-8", "replace"); idx_all[u] = txt
                jobs += [(u, s, e) for s, e, _ in records(txt, pats)]
    print(f"{M['name']}: runs {meta['cycles']}, {len(idx_all)} files, {len(jobs)} records")
    # 2) fetch the byte ranges in parallel, decode + crop one at a time
    croppers = {}
    def dl(job):
        u, s, e = job; return job, get(u, (s, e))
    with cf.ThreadPoolExecutor(8) as ex:
        for fu in cf.as_completed([ex.submit(dl, j) for j in jobs]):
            try:
                (u, s, e), buf = fu.result()
                if not buf: raise RuntimeError("empty range")
                h = G.decode_bytes(buf); g = G.grid_of(h)
                key = f"{g['nx']}x{g['ny']}_{g['la1']:.3f}_{g['lo1']:.3f}_{g['scan']}"
                if key not in croppers:
                    rev = G.rows_reversed(np.asarray(h["_vals"], float), g)
                    croppers[key] = G.Cropper(g, box, rev); meta["grids"][key] = croppers[key].sub
                v = np.asarray(h["_vals"], float); miss = h.get("missingValue")
                bad = ~np.isfinite(v) | (np.abs(v) > 1e10) | ((v == miss) if miss is not None else False)
                v[bad] = np.nan
                blob, off, sc = encode_lin(croppers[key](v, g))
                rk = f"{u}#{s}"; name = hashlib.sha1(rk.encode()).hexdigest()[:20] + ".bin.gz"
                with open(os.path.join(d, "r", name), "wb") as fh: fh.write(blob)
                meta["records"][rk] = [name, key, off, sc]
            except Exception as ex_:  # noqa: BLE001  one bad record never stops the build
                meta["notes"].append(f"{fu}: {ex_}"[:300])
    with open(os.path.join(d, "idx.json.gz"), "wb") as fh:
        fh.write(gzip.compress(json.dumps({u: t for u, t in idx_all.items()}).encode()))
    n, want = len(meta["records"]), len(jobs)
    meta["status"] = "ok" if want and n >= 0.95 * want else "partial" if n else "missing"
    meta["seconds"] = round(time.time() - t0)
    with open(os.path.join(d, "latest.json"), "w") as fh: json.dump(meta, fh, indent=0)
    print(f"{M['name']}: {n}/{want} records, {meta['status']}, {meta['seconds']} s")
    return meta["status"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="site")
    ap.add_argument("--models", default="nbm,refs"); a = ap.parse_args()
    box = G.DEFAULT_BOX
    if os.environ.get("GLMP_BOX"): box = tuple(float(x) for x in os.environ["GLMP_BOX"].split(","))
    res = [build_model({"nbm": NBM, "refs": REFS}[m], a.out, box) for m in a.models.split(",")]
    if all(r == "missing" for r in res): raise SystemExit("nothing built")

if __name__ == "__main__":
    main()
