#!/usr/bin/env python3
"""GFS-Wave (WAVEWATCH III) seas for Dragon Splashdown Support.

The GFS-Wave GRIB2 files are packed with JPEG 2000, which the page cannot decode in the browser, so this script (run
in the Pages workflow) decodes the significant wave height (HTSGW) and primary wave period (PERPW) records of the
West Coast 0.16 deg grid with eccodes and writes them in the same "fast copy" format as grid_build.py:

  wave/latest.json   cycle, grids (regular lat/lon), records { "<url>#<start>": [file, grid, offset, scale] }, status
  wave/idx.json.gz   { "<url>": "<.idx text>" }
  wave/r/<hash>.bin.gz  gzip'd uint16 cells in the file's own order (value = offset + u * scale, 65535 = missing/land)

Hours: every hour to 120 h, then every 3 h to 240 h (the model's own output steps).

  python wave_build.py --out _site
"""
import argparse, concurrent.futures as cf, datetime as dt, gzip, hashlib, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_build import get, records, encode_lin  # noqa: E402

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
DOMAIN = "wcoast.0p16"
HOURS = list(range(0, 121)) + list(range(123, 241, 3))
PATS = [r":HTSGW:surface:", r":PERPW:surface:"]


def url(cyc, f):
    ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
    return f"{BASE}/gfs.{ymd}/{hh}/wave/gridded/gfswave.t{hh}z.{DOMAIN}.f{f:03d}.grib2"


def choose_cycle(now):
    top = now.replace(minute=0, second=0, microsecond=0); top -= dt.timedelta(hours=top.hour % 6)
    for k in range(5):
        cyc = top - dt.timedelta(hours=6 * k)
        if get(url(cyc, 120) + ".idx"): return cyc
    return None


def decode(buf):
    import eccodes
    gid = eccodes.codes_new_from_message(bytes(buf))
    try:
        g = lambda k: eccodes.codes_get_double(gid, k)
        grid = dict(type="ll", nx=int(eccodes.codes_get_long(gid, "Ni")), ny=int(eccodes.codes_get_long(gid, "Nj")),
                    la1=round(g("latitudeOfFirstGridPointInDegrees"), 6), lo1=round(g("longitudeOfFirstGridPointInDegrees") % 360, 6),
                    di=round(g("iDirectionIncrementInDegrees"), 6), dj=round(g("jDirectionIncrementInDegrees"), 6),
                    scan=int(eccodes.codes_get_long(gid, "scanningMode")))
        grid["dx"] = round(grid["dj"] * 111195.0, 1); grid["dy"] = grid["dx"]   # metres, for the page's 10 NM sampling
        try: eccodes.codes_set(gid, "missingValue", 1e20)
        except Exception: pass  # noqa: BLE001
        v = np.asarray(eccodes.codes_get_values(gid), float)
        v[(~np.isfinite(v)) | (np.abs(v) > 1e10)] = np.nan
        return grid, v
    finally:
        eccodes.codes_release(gid)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="_site"); a = ap.parse_args()
    t0 = time.time(); now = dt.datetime.now(dt.timezone.utc)
    d = os.path.join(a.out, "wave"); os.makedirs(os.path.join(d, "r"), exist_ok=True)
    meta = dict(model="wave", source=f"GFS-Wave {DOMAIN}", built=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                full=True, cycles={}, grids={}, records={}, notes=[])
    cyc = choose_cycle(now)
    if not cyc: raise SystemExit("no GFS-Wave run found")
    meta["cycles"]["wave"] = cyc.isoformat().replace("+00:00", "Z")
    urls = [url(cyc, f) for f in HOURS]; idx_all, jobs = {}, []
    with cf.ThreadPoolExecutor(8) as ex:
        for u, txt in zip(urls, ex.map(lambda u: get(u + ".idx"), urls)):
            if not txt: continue
            txt = txt.decode("utf-8", "replace"); idx_all[u] = txt
            jobs += [(u, s, e) for s, e, _ in records(txt, PATS)]
    print(f"wave: run {meta['cycles']['wave']}, {len(idx_all)} files, {len(jobs)} records")

    def dl(job):
        u, s, e = job; return job, get(u, (s, e))
    with cf.ThreadPoolExecutor(8) as ex:
        for fu in cf.as_completed([ex.submit(dl, j) for j in jobs]):
            try:
                (u, s, e), buf = fu.result()
                if not buf: raise RuntimeError("empty range")
                grid, v = decode(buf)
                key = f"ll{grid['nx']}x{grid['ny']}_{grid['la1']:.3f}_{grid['lo1']:.3f}_{grid['scan']}"
                meta["grids"][key] = grid
                blob, off, sc = encode_lin(v)
                rk = f"{u}#{s}"; name = hashlib.sha1(rk.encode()).hexdigest()[:20] + ".bin.gz"
                with open(os.path.join(d, "r", name), "wb") as fh: fh.write(blob)
                meta["records"][rk] = [name, key, off, sc]
            except Exception as ex_:  # noqa: BLE001
                meta["notes"].append(f"{fu}: {ex_}"[:300])
    if meta["grids"]:      # box = the grid's own extent (the page uses every cell, so no margin test)
        g = next(iter(meta["grids"].values())); la2 = g["la1"] + (g["ny"] - 1) * g["dj"] * (1 if g["scan"] & 0x40 else -1)
        lo2 = g["lo1"] + (g["nx"] - 1) * g["di"]
        meta["box"] = [round(((g["lo1"] + 180) % 360) - 180, 3), round(min(g["la1"], la2), 3), round(((lo2 + 180) % 360) - 180, 3), round(max(g["la1"], la2), 3)]
    with open(os.path.join(d, "idx.json.gz"), "wb") as fh: fh.write(gzip.compress(json.dumps(idx_all).encode()))
    n, want = len(meta["records"]), len(jobs)
    meta["status"] = "ok" if want and n >= 0.95 * want else "partial" if n else "missing"
    meta["seconds"] = round(time.time() - t0)
    with open(os.path.join(d, "latest.json"), "w") as fh: json.dump(meta, fh, indent=0)
    print(f"wave: {n}/{want} records, {meta['status']}, {meta['seconds']} s")
    if meta["status"] == "missing": raise SystemExit("no wave records decoded")


if __name__ == "__main__":
    main()
