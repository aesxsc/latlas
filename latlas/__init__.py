"""latlas — latency-only geolocation.

A client-localization system that infers the geographic position of the machine
it runs on using nothing but ICMP round-trip times to a set of globally
distributed anchors whose positions are known a priori.

Deliberately excluded as inputs (see `latlas.provenance` for enforcement):
  * IP geolocation of the client or of any observed path
  * the client's public IP address
  * Wi-Fi / BSSID / SSID / signal data
  * GPS, GNSS, cell-tower, or any radio positioning
  * system timezone, locale, NTP conf, or system clock offset
  * ISP / ASN metadata, hostname, or reverse DNS of the client
  * device model, OS hints, browser/user-agent, or language

The only estimator inputs are: (lat, lon) of each anchor, the measured RTT to
each anchor, and the probe schedule/drop statistics.
"""

__version__ = "1.0.0"
