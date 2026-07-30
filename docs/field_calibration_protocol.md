# Field Calibration Protocol — Scripps Pier

Goal: collect **Ground Control Points (GCPs)** that over-determine the camera
geometry and directly measure the **vertical scale** (pixels per meter of
elevation) in the surf zone — where waves actually break. One GCP + priors
(what we have now) only *interpolates* the mid-field; poles at several distances
*measure* it and let the priors go slack. This is what turns "self-built
validator" into "geometrically calibrated coastal imaging station."

Recording runs on its own — anything in the cam's view is captured. Your job is
to put known objects at known places and log when/where.

## Gear
- **A vertical ruler of known length.** Best: a painter's/survey pole (2–3 m)
  with **high-contrast tape bands every 0.5 m**. Fallbacks: a surfboard of
  known length held upright; or **a person of known standing height** — a
  standing person is itself a vertical ruler.
- **Phone** for (a) clock and (b) GPS pin at each spot.
- Optional: a second person to hold the pole while you check the cam.

## Timing
- **Daylight, ideally mid-day** (best light; the dusk session was marginal).
- **Note your phone clock time to the second** at every pose — that's how I pull
  the exact frames. Also lets me fetch the tide from **NOAA La Jolla gauge
  9410230** to tide-correct.

## What makes a GCP valid (learned the hard way)
- **Vertical & plumb.** Use the flat horizon as your level. A tilt foreshortens
  the length and corrupts the scale.
- **Base at the water surface** (swash/shallow), top toward the sky. The exact
  instantaneous waterline doesn't matter — tide is slow and we detrend it, waves
  are the signal. Just get the base *near* the surface.
- **Both ends clearly visible**, held **dead still ~15 s**.
- Vertical does **not** need to be broadside — a plumb pole reads its true height
  from any facing.

## The poses (this is the important part)
Walk **seaward at 3–4 distances**, holding the pole vertical at each. Each
distance lands at a **different image row**, and that's how we build the scale
curve at the rows where waves break (roughly image rows 480–620).

| # | Where | Why |
|---|-------|-----|
| 1 | Right at the swash line | anchors the near field (rows ~680–710) |
| 2 | Knee-deep, a few steps out | mid-near |
| 3 | Waist-deep, as far as safe | reaches into the breaking-zone rows |
| 4 | (optional) hold pole **overhead** at spot 3 | taller ruler = better precision |

At each spot:
1. Stand still, pole vertical, **15 s**.
2. **Raise both arms / wave once** as a sync marker.
3. Note **phone time** + **drop a GPS pin** (or read lat/lon).
4. Stand a bit **left of center** in the frame — the far bottom-right, near the
   railing, is cluttered; center-lower is cleaner.

## Also capture (free, high-value)
- **The fixed railing post** bottom-right is a **PTZ sentinel** — if it ever
  moves in frame, the cam was repositioned and calibration is void. Nothing to
  do; I track it automatically.
- If it's a **good surf day**, footage with surfers gives us the *surfer-as-scale*
  cross-check (pose height as a moving ruler right where the waves are).

## What to record for post-processing
For each pose: **phone time**, **GPS lat/lon**, **pole length** (and where the
bands are). Post-processing then:
1. Pull the exact frames, mark each pole top/base with `mark_scale.py` (vertical),
2. Convert to GCPs, refit the geometry (now **over-determined**),
3. **Validate** the mid-field scale (poles vs the horizon-only interpolation),
4. Convert the motion-energy wave signal to **meters from first principles**, and
5. Compare to CDIP 201 as an **independent** check.

## Safety
Public beach access is fine; you don't need pier deck access. Waist-deep max,
never turn your back on the surf, skip it if it's big.
