# latlas

Estimate where this machine is from ICMP round-trip times to servers whose
locations are known.

No IP geolocation, no client IP address, no Wi-Fi, GPS, timezone, locale, ISP or
device metadata. The position comes from network timing alone.

## Install

```
pip install latlas
```

Python 3.10 or later, with `numpy` and `scipy`. No root or administrator rights
are needed on any supported platform.

## Usage

```
latlas              # scan and print the position
latlas web          # live map on http://0.0.0.0:8080
latlas --json       # machine-readable output
```

Example output:

```
latlas - locating this machine from round-trip times only

  POSITION      52.5200N, 13.4050E
  NEAREST PLACE at Berlin, Germany
  ACCURACY      +/-480 km (speed-of-light bound)

  nearest places to the estimate
        0 km west         Berlin, Germany        pop. 3,426,354
        2 km south        Kreuzberg, Germany     pop. 153,135
        3 km east         Friedrichshain, Germany pop. 117,829

  measured      1598/1600 anchors replied (99.9%), closest 2.10 ms, 61 s
```

`latlas web` opens a map, starts a scan immediately and streams progress as
anchors reply. The finished scan draws the estimate, the uncertainty region and
the nearby towns.

## Accuracy

Measured by replaying the RIPE Atlas anchor mesh (round-trip times between about
1,000 vantage points with published coordinates) through the estimator:

| estimate | median error | 75th percentile |
|---|---|---|
| reported position | 10 km | 130 km |
| model-refined position | 113 km | 252 km |

The reported speed-of-light bound contained the true location in 99.75% of 400
cases. The model-based credible region is over-confident, covering the truth
40.8% of the time at a nominal 90%, and the output labels it as indicative.

## How it works

A round trip covers at least twice the great-circle distance between its
endpoints, and no signal travels faster than light in vacuum, so

```
distance <= (c / 2) * RTT
```

Intersecting that inequality across every anchor gives a region that contains the
machine with no statistical assumptions at all. The reported position is the
centroid of that region.

A fitted delay law, `RTT = alpha * distance / k + beta`, narrows the region
further. The detour factor `alpha` is drawn from a truncated lognormal; the
truncation is required, because without it the model assigns probability to paths
faster than light and contradicts the bound above.

The anchors, a 34,000-city gazetteer and the calibrated delay law are bundled
with the package, so there is no discovery or download step.

## Commands

```
latlas anchors          describe the bundled anchor set
latlas validate mesh    re-measure accuracy against real vantage points
latlas isolation        check that ground truth cannot reach the estimator
latlas --report         full technical report and world map
latlas --help           all options
```

## Data sources

* Anchor coordinates: published by RIPE NCC (Atlas anchors) and Ookla.
* City names: GeoNames, CC BY 4.0.
* Map tiles: OpenStreetMap contributors.

## License

AGPL-3.0-or-later.
