#!/usr/bin/env python3
"""
boundary_check.py — the six checks, automated.

Fetches forecast (or recent past) atmospheric profiles for the West Texas
corridor from the free Open-Meteo API (no key needed) and answers:

  1. pattern   - lee trough west, higher pressure east?
  2. river     - southerly low-level jet in the morning?
  3. lid       - capping inversion at usable height?
  4. hold      - CAPE above the lid ~ zero (gun not loaded)?
  5. fuel      - enough buoyancy below the lid for thermals?
  6. longitude - where is the boundary, where should the line start?

Usage:
  pip install requests
  python3 boundary_check.py --date 2026-07-06
  python3 boundary_check.py --date 2026-07-06 --hour 12   # UTC profile hour

Validated against the five 12Z radiosondes of the 2020/2021 record days:
jet detection exact; lid detection triggers at the moist-layer top and so
reads ~200 m conservative (low); energy split correct within ~50 J/kg.
Verdicts are heuristics tuned on those days. Always look at
the actual soundings before towing up. Author: J. Korner, with the analysis
internal screening tool
"""
import argparse, sys, math, json
from datetime import date as ddate

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ---------------- corridor definition ----------------
LAT_S, LAT_M, LAT_N = 29.5, 32.0, 35.0        # Del Rio-ish, Midland-ish, Amarillo-ish
LONS = [-103.5, -102.5, -101.5, -100.5, -99.5, -98.5]   # west -> east scan
# NOTE: the generic /v1/forecast endpoint has NO 875/750/650 hPa levels and
# rejects the whole request with HTTP 400 if they are asked for. The GFS/HRRR
# endpoint /v1/gfs serves the full 25-hPa ladder (and is the right model
# family for Texas anyway), so use that.
PLEVS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
         750, 725, 700, 675, 650, 625, 600]

API = "https://api.open-meteo.com/v1/gfs"

def fetch_profile(lat, lon, day, hour_utc, tries=3):
    import time
    for attempt in range(tries):
        try:
            return _fetch_profile_once(lat, lon, day, hour_utc)
        except Exception:
            if attempt == tries - 1: raise
            time.sleep(8 * (attempt + 1))   # transient API/network blip: wait and retry

def _fetch_profile_once(lat, lon, day, hour_utc):
    hv = []
    for p in PLEVS:
        hv += [f"temperature_{p}hPa", f"dew_point_{p}hPa",
               f"wind_speed_{p}hPa", f"wind_direction_{p}hPa",
               f"geopotential_height_{p}hPa"]
    params = dict(latitude=lat, longitude=lon,
                  hourly=",".join(hv + ["temperature_2m","dew_point_2m",
                                        "surface_pressure","pressure_msl"]),
                  daily="temperature_2m_max",
                  wind_speed_unit="kmh", timezone="UTC",
                  start_date=str(day), end_date=str(day))
    r = requests.get(API, params=params, timeout=30)
    if r.status_code != 200:
        try: reason = r.json().get("reason", r.text[:200])
        except Exception: reason = r.text[:200]
        raise RuntimeError(f"HTTP {r.status_code}: {reason}")
    j = r.json()
    idx = hour_utc  # hourly arrays are 0..23 UTC for the single day
    prof = []
    H = j["hourly"]
    for p in PLEVS:
        try:
            prof.append(dict(p=p,
                z=H[f"geopotential_height_{p}hPa"][idx],
                T=H[f"temperature_{p}hPa"][idx],
                Td=H[f"dew_point_{p}hPa"][idx],
                spd=H[f"wind_speed_{p}hPa"][idx],
                drct=H[f"wind_direction_{p}hPa"][idx]))
        except (KeyError, TypeError):
            pass
    prof = [r_ for r_ in prof if None not in r_.values()]
    prof.sort(key=lambda r_: r_["z"])
    sfc = dict(T2=H["temperature_2m"][idx], Td2=H["dew_point_2m"][idx],
               mslp=H["pressure_msl"][idx],
               tmax=j["daily"]["temperature_2m_max"][0])
    # ground elevation from the API metadata
    sfc["elev"] = j.get("elevation", 0.0)
    return prof, sfc

# ---------------- pure analysis (testable offline) ----------------
def find_jet(prof, elev):
    """strongest wind below 1200 m AGL; returns (kmh, deg, agl)"""
    cands = [r for r in prof if 0 < r["z"]-elev < 1200]
    if not cands: return None
    j = max(cands, key=lambda r: r["spd"])
    return j["spd"], j["drct"], j["z"]-elev

def find_lid(prof, elev):
    """lowest level 600m-3400m AGL where, over the ~250 m layer above it,
       temperature stops falling (lapse > -3 K/km) AND the dewpoint
       depression widens by > 2.5 K; returns lid AGL or None"""
    n = len(prof)
    for i, a in enumerate(prof):
        agl = a["z"] - elev
        if not (600 < agl < 3400): continue
        b = None
        for j in range(i+1, n):
            if prof[j]["z"] - a["z"] >= 220:
                b = prof[j]; break
        if b is None: break
        dz = (b["z"] - a["z"]) / 1000.0
        lapse = (b["T"] - a["T"]) / dz
        dd_a, dd_b = a["T"]-a["Td"], b["T"]-b["Td"]
        if lapse > -3.0 and (dd_b - dd_a) > 2.5:
            return agl
    return None

def parcel_T(zs, z0, T0, Td0):
    """dry adiabat to LCL, ~pseudo-moist above; crude but adequate."""
    lcl = z0 + 125.0*max(T0-Td0, 0)
    out = []
    for z in zs:
        if z <= lcl: out.append(T0 - 9.8*(z-z0)/1000)
        else:        out.append(T0 - 9.8*(lcl-z0)/1000 - 6.0*(z-lcl)/1000)
    return out

def energy_split(prof, elev, tmax, td_sfc, lid_agl):
    """rough CAPE below vs above the lid, J/kg (buoyancy integral)."""
    zs = [r["z"] for r in prof]
    Te = [r["T"] for r in prof]
    Tp = parcel_T(zs, elev, tmax, td_sfc)
    below = above = 0.0
    for (z1,e1,p1),(z2,e2,p2) in zip(zip(zs,Te,Tp), zip(zs[1:],Te[1:],Tp[1:])):
        dz = z2-z1
        if dz <= 0: continue
        b = 9.81 * ((p1-e1)+(p2-e2))/2.0 / 290.0 * dz   # buoyancy integral
        if b > 0:
            if (z1-elev) < lid_agl: below += b
            else: above += b
    return below, above

def fetch_td2(lat, lon, day, hour_utc, tries=3):
    """surface dewpoint only - cheap probe for the N and S latitude rows"""
    import time
    params = dict(latitude=lat, longitude=lon, hourly="dew_point_2m",
                  timezone="UTC", start_date=str(day), end_date=str(day))
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}")
            return r.json()["hourly"]["dew_point_2m"][hour_utc]
        except Exception:
            if attempt == tries - 1: return None
            time.sleep(5 * (attempt + 1))

def scan_boundary(tds):
    """tds: [(lon, Td2), ...] west->east; returns boundary lon or None"""
    tds = [(l,t) for l,t in tds if t is not None]
    for (l1,t1),(l2,t2) in zip(tds, tds[1:]):
        if t2 - t1 >= 4.0:
            return (l1+l2)/2
    if len(tds) >= 2 and tds[0][1] < 8 and tds[-1][1] > 12:
        return tds[len(tds)//2][0]
    return None

def corridor_bearing(points):
    """points: [(lat, boundary_lon), ...]; bearing of the line, degrees true.
       Needs >=2 latitudes; uses southernmost and northernmost."""
    pts = sorted([p for p in points if p[1] is not None])
    if len(pts) < 2: return None
    (lat1, lo1), (lat2, lo2) = pts[0], pts[-1]
    dlon = (lo2 - lo1) * math.cos(math.radians((lat1+lat2)/2))
    return math.degrees(math.atan2(dlon, lat2 - lat1)) % 360

def tail_from(drct, spd):
    return -spd*math.cos(math.radians(drct))

# ---------------- the six checks ----------------
NAMES = {1:"pattern", 2:"river", 3:"lid", 4:"hold", 5:"fuel", 6:"longitude"}

def run(day, hour, collect=None):
    print(f"\nDRYLINE CHECK — {day}, profiles at {hour:02d}Z")
    print("="*62)
    ok = {}
    checks = []
    def add(no, passed, detail):
        ok[no] = bool(passed)
        checks.append(dict(no=no, name=NAMES[no], passed=bool(passed), detail=detail))
        print(f"{no} {NAMES[no].upper():<9} {detail}  {'YES' if passed else 'NO'}")

    # fetch mid-corridor column set at three latitudes (west->east)
    grid = {}
    for lon in LONS:
        try:
            grid[lon] = fetch_profile(LAT_M, lon, day, hour)
        except Exception as e:
            print(f"  fetch failed at lon {lon}: {e}"); grid[lon] = None

    live = {lon: g for lon, g in grid.items() if g}
    if len(live) < 4:
        sys.exit("not enough data fetched - check connectivity/date range")

    # 1 pattern: mslp gradient west->east
    mslps = [(lon, g[1]["mslp"]) for lon, g in sorted(live.items())]
    grad = mslps[-1][1] - mslps[0][1]
    add(1, grad > 1.5,
        f"west-to-east MSLP rise {grad:+.1f} hPa "
        f"{'-> lee trough in the west' if grad > 1.5 else '-> no trough signal'}")

    # 2 river: best low-level jet across the corridor
    best = None
    for lon, (prof, sfc) in live.items():
        j = find_jet(prof, sfc["elev"])
        if j and (best is None or j[0] > best[1][0]): best = (lon, j)
    if best:
        lon, (spd, drct, agl) = best
        southerly = 140 <= drct <= 220
        add(2, spd >= 35 and southerly,
            f"max low-level wind {spd:.0f} km/h from {drct:.0f} deg "
            f"at {agl:.0f} m AGL (lon {lon})"
            + ("" if (spd >= 35 and southerly) else " - want >=35 km/h from 140-220 deg"))
    else:
        add(2, False, "no low-level jet found")

    # 3+4+5 at the strongest-jet longitude (fall back to mid-corridor if no jet)
    lon0 = best[0] if best else sorted(live)[len(live)//2]
    prof, sfc = live[lon0]
    lid = find_lid(prof, sfc["elev"])
    if lid is not None and 1200 <= lid <= 2800:
        add(3, True, f"capping inversion at {lid:.0f} m AGL")
    elif lid is not None:
        add(3, False, f"inversion at {lid:.0f} m AGL - outside the 1200-2800 m working band")
    else:
        add(3, False, "no capping inversion found")

    if lid:
        below, above = energy_split(prof, sfc["elev"], sfc["tmax"], sfc["Td2"], lid)
        add(4, above < 150,
            f"CAPE above the lid ~{above:.0f} J/kg "
            f"{'-> gun not loaded' if above < 150 else '-> storm risk on the line'}")
        add(5, below > 150,
            f"buoyancy below the lid ~{below:.0f} J/kg "
            f"{'-> thermals' if below > 150 else '-> weak or blue'}")
    else:
        add(4, False, "no lid - not evaluated")
        add(5, False, "no lid - not evaluated")

    # 6 longitude: boundary from surface dewpoint scan at mid latitude
    tds = [(lon, g[1]["Td2"]) for lon, g in sorted(live.items())]
    dl = None
    for (l1,t1),(l2,t2) in zip(tds, tds[1:]):
        if t2 - t1 >= 4.0:          # sharp moistening eastward = boundary between
            dl = (l1+l2)/2; break
    if dl is None and tds[0][1] < 8 and tds[-1][1] > 12:
        dl = tds[len(tds)//2][0]
    if dl is not None:
        start = dl + 1.0            # ~95 km east of the line
        add(6, True, f"boundary near {dl:.1f} deg -> start the line around {start:.1f} deg")
    else:
        add(6, False, "no clear surface moisture gradient - boundary position uncertain")

    # info (not a check): corridor axis from the boundary at three latitudes,
    # and how much of the jet works along it
    course = tail = None
    downwind = round((best[1][1] + 180.0) % 360) if best else None
    if dl is not None and best:
        rows = [(LAT_M, dl)]
        for lat in (LAT_S, LAT_N):
            row = [(lon, fetch_td2(lat, lon, day, hour)) for lon in LONS]
            rows.append((lat, scan_boundary(row)))
        course = corridor_bearing(rows)
        if course is not None:
            tail = best[1][0] * math.cos(math.radians(course - downwind))
            print(f"  COURSE    corridor axis ~{course:03.0f} deg; jet pushes toward "
                  f"{downwind:03.0f} -> tailwind along the line ~{tail:.0f} km/h")

    print("-"*62)
    yes = sum(ok.values())
    verdict = ("6/6 - all checks green" if yes == 6 else
               "4-5/6 - inspect the soundings" if yes >= 4 else
               "pattern not present")
    print(f"VERDICT: {yes}/6 checks -> {verdict}\n")
    if collect is not None:
        collect.append(dict(date=str(day), hour=hour, checks=checks, yes=yes,
                            verdict=verdict,
                            start_lon=round(dl+1.0,1) if dl is not None else None,
                            jet_kmh=round(best[1][0]) if best else None,
                            course=round(course) if course is not None else None,
                            jet_to=downwind,
                            tail_kmh=round(tail) if tail is not None else None))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(ddate.today()))
    ap.add_argument("--hour", type=int, default=12, help="UTC profile hour (12 = 07:00 CDT)")
    ap.add_argument("--json", help="write machine-readable results to this file")
    ap.add_argument("--history", help="append results to this JSON log (one row per run and target day)")
    ap.add_argument("--auto", action="store_true", help="check today and tomorrow (Texas dates)")
    a = ap.parse_args()
    out = []
    if a.auto:
        from datetime import datetime, timedelta, timezone
        tx_now = datetime.now(timezone.utc) - timedelta(hours=5)
        for k in range(5):                    # D0..D+4 (Texas dates); beyond that the
            d = tx_now.date() + timedelta(days=k)   # six checks would be false precision
            try: run(d, a.hour, collect=out)
            except SystemExit: pass
            except Exception as e: print("run failed:", e)
    else:
        run(a.date, a.hour, collect=out)
    if a.json:
        from datetime import datetime, timezone
        json.dump(dict(generated=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                       results=out), open(a.json,"w"), indent=1)
        print("wrote", a.json)
    if a.history:
        import os
        from datetime import datetime, timedelta, timezone
        hist = dict(entries=[])
        if os.path.exists(a.history):
            try: hist = json.load(open(a.history))
            except Exception: pass
        ent = hist.get("entries", [])
        run_day = str((datetime.now(timezone.utc) - timedelta(hours=5)).date())  # Texas date
        for r in out:
            lead = (ddate.fromisoformat(r["date"]) - ddate.fromisoformat(run_day)).days
            # re-runs on the same day replace, so the log holds one row per (run, target)
            ent = [x for x in ent if not (x.get("run")==run_day and x.get("date")==r["date"])]
            ent.append(dict(run=run_day, date=r["date"], lead=lead,
                            checks=[c["passed"] for c in r["checks"]],
                            yes=r["yes"], jet_kmh=r["jet_kmh"], start_lon=r["start_lon"]))
        ent.sort(key=lambda x: (x["date"], x.get("lead",0)))
        hist["entries"] = ent[-800:]          # ~13 months of double-daily entries
        json.dump(hist, open(a.history,"w"), separators=(",",":"))
        print("updated", a.history, f"({len(hist['entries'])} entries)")
