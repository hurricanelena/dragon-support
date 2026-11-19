#!/usr/bin/env python3
"""Keeps the previous Gridded LAMP run next to the new one, for Dragon Splashdown Support.

Runs in the Pages workflow right after glmp_build.py. Each deploy replaces the whole site, so this copies the run that
is live right now (from the published site) into glmp/prev/. The page uses prev/ when the newest run has no data for
the selected hour, before falling back to a nearby hour.

  - New run built and different from the live one  -> live run goes to glmp/prev/
  - New run is the same cycle as the live one        -> the live glmp/prev/ is kept as it is
  - New build failed (no glmp/latest.json)           -> the live run and its prev/ are republished unchanged

  python glmp_keep.py --site https://<user>.github.io/<repo>/ --out _site/glmp

Only public NOAA output is copied (the files the site already serves). Failures never stop the workflow.
"""
import argparse, concurrent.futures as cf, datetime as dt, json, os, sys, urllib.request

MAX_AGE_H = 6   # a previous run older than this (relative to the new run) is not worth keeping


def get(url, binary=False):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}), timeout=60) as r:
            data = r.read()
            return data if binary else json.loads(data)
    except Exception:  # noqa: BLE001
        return None


def copy_run(src, meta, dst):
    """Copy every .bin file listed in meta from src (URL ending in /) into dst; latest.json is written last."""
    os.makedirs(dst, exist_ok=True)
    ext = ".bin.gz" if meta.get("enc") == "gzip" else ".bin"
    jobs = [(fl, h) for fl, fd in meta.get("fields", {}).items() for h in fd.get("hours", [])]

    def one(job):
        fl, h = job; name = f"{fl}_f{h:03d}{ext}"; data = get(src + name, binary=True)
        if data is None: return job, False
        with open(os.path.join(dst, name), "wb") as fh: fh.write(data)
        return job, True

    with cf.ThreadPoolExecutor(16) as ex: res = dict(ex.map(one, jobs))
    for fl, fd in meta.get("fields", {}).items():      # drop hours that did not come through
        fd["hours"] = [h for h in fd.get("hours", []) if res.get((fl, h))]
    meta.pop("_prev", None)
    with open(os.path.join(dst, "latest.json"), "w") as fh: json.dump(meta, fh, indent=1)
    got = sum(res.values()); print(f"  copied {got}/{len(jobs)} files of run {meta.get('cycle')} to {dst}")
    return got


def hours_apart(a, b):
    p = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return abs((p(a) - p(b)).total_seconds()) / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="published site URL, ending in /")
    ap.add_argument("--out", default="_site/glmp")
    a = ap.parse_args()
    site = a.site if a.site.endswith("/") else a.site + "/"
    live, livep = get(site + "glmp/latest.json"), get(site + "glmp/prev/latest.json")
    newp = os.path.join(a.out, "latest.json")
    new = json.load(open(newp)) if os.path.exists(newp) else None
    print("new run:", new and new.get("cycle"), "| live run:", live and live.get("cycle"), "| live prev:", livep and livep.get("cycle"))

    if new is None:                                   # build failed: keep serving what is live
        if live: copy_run(site + "glmp/", live, a.out)
        if livep: copy_run(site + "glmp/prev/", livep, os.path.join(a.out, "prev"))
        return
    cand = None
    if live and live.get("cycle") and live["cycle"] != new.get("cycle"): cand = (site + "glmp/", live)
    elif livep and livep.get("cycle") and livep["cycle"] != new.get("cycle"): cand = (site + "glmp/prev/", livep)
    if not cand: print("  no earlier run to keep"); return
    if new.get("cycle") and hours_apart(cand[1]["cycle"], new["cycle"]) > MAX_AGE_H:
        print(f"  earlier run {cand[1]['cycle']} is more than {MAX_AGE_H} h older: not kept"); return
    copy_run(cand[0], cand[1], os.path.join(a.out, "prev"))


if __name__ == "__main__":
    try: main()
    except Exception as e:  # noqa: BLE001
        print("glmp_keep: skipped:", e, file=sys.stderr)
