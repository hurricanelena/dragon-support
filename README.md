# Dragon Splashdown Support

Weather briefing page for Dragon splashdowns, published at `https://<user>.github.io/<repo>/`.

- `dragon_support_index.html`: the page. Sample points and wind criteria are NOT in this repository; each user loads
  their private `dragon_support_settings.json` in the page (kept only in their browser).
- `.github/workflows/site.yml`: publishes the page and, every hour at :40, pre-processed public NOAA data next to it:
  `nbm/`, `refs/` (scripts/grid_build.py) and `glmp/` (scripts/glmp_build.py).
- Optional repository secret `GLMP_BOX` = `lon0,lat0,lon1,lat1`: area the model data is cropped to
  (default -128,26,-112,40). The page uses the fast copies only when every sample point is at least 4° inside it.
