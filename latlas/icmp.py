"""ICMP echo probing with platform-appropriate backends.

Backends, in order of preference per platform:

``icmp-windows``
    ``IcmpSendEcho`` from ``iphlpapi.dll``. Unprivileged, and critically the
    round-trip time is measured by the Windows IP helper service inside the
    network stack rather than by the sleeping caller thread. This matters: a
    userspace timer around a blocking ``connect()`` on Windows is quantised to
    the ~15.6 ms system timer tick, which is useless for latency geolocation.
    ``IcmpSendEcho`` reports whole milliseconds.

``icmp-posix``
    ``SOCK_DGRAM`` ICMP socket, timed by the caller with ``clock_gettime`` so the
    resolution is microseconds. Unprivileged wherever ``net.ipv4.ping_group_range``
    admits the user's group. Linux, the BSDs and macOS all strip the IP header
    from datagram-ICMP replies while raw sockets keep it, so the reply parser
    locates the ICMP header rather than assuming a fixed offset.

``icmp-ping-binary``
    Drives the system ``ping``. Needed because the datagram socket above is *not*
    universally available: Debian and Ubuntu ship
    ``net.ipv4.ping_group_range = 1 0``, which permits no group at all, and the
    tool would otherwise drop to the TCP fallback and lose all precision. The
    ``ping`` binary carries ``cap_net_raw`` on those systems, so it still works
    unprivileged.

``tcp-connect``
    Blocking ``connect()`` timing. Universal but the noisiest backend; a last
    resort so the tool still functions on locked-down networks.

All backends return *samples in milliseconds*. Losses are returned as ``None``
so that drop rate survives into the estimator: packet loss to an anchor is weak
but real evidence about topological distance.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as _w
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

_PROBE_MAGIC = b"LTLS"

#: Default per-echo budget when a backend is asked for one sample at a time.
_DEFAULT_TIMEOUT_MS = 1000


@dataclass
class ProbeSeries:
    """All ICMP samples collected for one anchor in one campaign round."""

    ip: str
    samples_ms: list[float | None] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def sent(self) -> int:
        return len(self.samples_ms)

    @property
    def received(self) -> int:
        return sum(1 for s in self.samples_ms if s is not None)

    @property
    def loss(self) -> float:
        return 1.0 - self.received / self.sent if self.sent else 1.0

    @property
    def rtts(self) -> list[float]:
        return [s for s in self.samples_ms if s is not None]

    def floor_ms(self, q: float = 0.15) -> float | None:
        """Robust estimate of the unqueued RTT floor.

        The minimum of an integer-quantised series is biased upward by ~half a
        quantum and is a high-variance statistic; a low quantile of a modest
        sample is a better floor estimator. Falls back to min for small N.
        """
        r = sorted(self.rtts)
        if not r:
            return None
        if len(r) <= 3:
            return r[0]
        k = max(1, min(len(r) - 1, int(round(q * len(r)))))
        return sum(r[:k]) / k


def _resolve(host: str) -> str | None:
    try:
        socket.inet_aton(host)
        return host
    except OSError:
        pass
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def _int_to_addr(addr_int: int) -> str:
    return socket.inet_ntoa(struct.pack("<I", addr_int))


# --------------------------------------------------------------------------
# Windows backend
# --------------------------------------------------------------------------

_IP_SUCCESS = 0
_IP_STATUS_NAMES = {
    0: "success", 11001: "buffer_too_small", 11002: "dest_net_unreachable",
    11003: "dest_host_unreachable", 11004: "dest_prot_unreachable",
    11005: "dest_port_unreachable", 11006: "no_resources", 11007: "bad_option",
    11008: "hardware_error", 11009: "packet_too_big", 11010: "request_timed_out",
    11011: "bad_request", 11012: "bad_route", 11013: "ttl_expired",
    11014: "bad_dest_addr", 11015: "bad_source_addr", 11016: "pending",
    11050: "general_failure",
}


class _IP_OPTION_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("Ttl", ctypes.c_ubyte), ("Tos", ctypes.c_ubyte),
        ("Flags", ctypes.c_ubyte), ("OptionsSize", ctypes.c_ubyte),
        ("OptionsData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class _ICMP_ECHO_REPLY(ctypes.Structure):
    _fields_ = [
        ("Address", _w.ULONG), ("Status", _w.ULONG), ("RoundTripTime", _w.ULONG),
        ("DataSize", _w.USHORT), ("Reserved", _w.USHORT),
        ("Data", ctypes.POINTER(ctypes.c_ubyte)),
        ("Options", _IP_OPTION_INFORMATION),
    ]


class _WindowsIcmp:
    name = "icmp-windows"
    resolution_ms = 1.0
    kernel_timed = True
    degraded = False

    def __init__(self) -> None:
        # use_last_error=True is required: without it ctypes does not snapshot
        # the thread's last-error value at call time and get_last_error() would
        # report 0 for every failed echo, mislabelling timeouts as successes.
        self._iphlpapi = ctypes.WinDLL("iphlpapi.dll", use_last_error=True)
        self._iphlpapi.IcmpCreateFile.restype = _w.HANDLE
        self._iphlpapi.IcmpSendEcho.restype = _w.ULONG
        self._iphlpapi.IcmpSendEcho.argtypes = [
            _w.HANDLE, _w.ULONG, ctypes.c_char_p, _w.USHORT,
            ctypes.POINTER(_IP_OPTION_INFORMATION), ctypes.c_void_p,
            _w.ULONG, _w.DWORD,
        ]
        self._iphlpapi.IcmpCloseHandle.argtypes = [_w.HANDLE]
        self._iphlpapi.IcmpCloseHandle.restype = _w.BOOL

    def open(self) -> int:
        h = self._iphlpapi.IcmpCreateFile()
        if h == _w.HANDLE(-1).value or not h:
            raise OSError("IcmpCreateFile failed")
        return h

    def close(self, handle: int) -> None:
        try:
            self._iphlpapi.IcmpCloseHandle(handle)
        except Exception:
            pass

    def ping(self, handle: int, addr_int: int, timeout_ms: int,
             seq: int) -> tuple[float | None, str]:
        payload = _PROBE_MAGIC + struct.pack("<I", seq)
        req = _IP_OPTION_INFORMATION(Ttl=128)
        bufsz = ctypes.sizeof(_ICMP_ECHO_REPLY) + len(payload) + 16
        buf = ctypes.create_string_buffer(bufsz)
        n = self._iphlpapi.IcmpSendEcho(handle, addr_int, payload, len(payload),
                                        ctypes.byref(req), buf, bufsz, timeout_ms)
        if n == 0:
            err = ctypes.get_last_error()
            return None, _IP_STATUS_NAMES.get(err, f"winerr{err}")
        rep = ctypes.cast(buf, ctypes.POINTER(_ICMP_ECHO_REPLY)).contents
        if rep.Status != _IP_SUCCESS:
            return None, _IP_STATUS_NAMES.get(rep.Status, f"status{rep.Status}")
        if rep.DataSize < len(payload) or rep.Data is None:
            return None, "short_reply"
        got = bytes(bytearray(rep.Data[i] for i in range(min(rep.DataSize, len(payload)))))
        if got != payload:
            return None, "payload_mismatch"
        if rep.Address != addr_int:
            return None, "address_mismatch"
        return float(rep.RoundTripTime), "success"

    def series(self, handle, addr_int, addr_str, n, interval_s,
               timeout_ms) -> list[float | None]:
        out: list[float | None] = []
        for i in range(n):
            rtt, _ = self.ping(handle, addr_int, timeout_ms, i)
            out.append(rtt)
            if i != n - 1 and interval_s:
                time.sleep(interval_s)
        return out


# --------------------------------------------------------------------------
# POSIX datagram-ICMP backend
# --------------------------------------------------------------------------


def _icmp_checksum(data: bytes) -> int:
    total = 0
    for i in range(0, len(data) - 1, 2):
        total += (data[i] << 8) | data[i + 1]
    if len(data) % 2:
        total += data[-1] << 8
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _icmp_header_offsets(data: bytes):
    """Candidate offsets of the ICMP header within a received datagram.

    A datagram (``SOCK_DGRAM``) ICMP socket on Linux, macOS and the BSDs returns
    the ICMP message *without* the IP header, while a raw socket returns it with
    one. Assuming either layout silently rejects every reply on the other, so
    both are offered and the caller checks which one actually parses.
    """
    if data and (data[0] >> 4) == 4:
        ihl = (data[0] & 0x0F) * 4
        if ihl >= 20:
            yield ihl
    yield 0


def _icmp_payload_matches(data: bytes, payload: bytes) -> bool:
    """True if ``data`` is an echo reply carrying exactly ``payload``."""
    for off in _icmp_header_offsets(data):
        if len(data) < off + 8 + len(payload):
            continue
        if data[off] != 0 or data[off + 1] != 0:   # type 0 = echo reply, code 0
            continue
        if data[off + 8:off + 8 + len(payload)] == payload:
            return True
    return False


class _PosixIcmp:
    name = "icmp-posix"
    kernel_timed = False
    resolution_ms = 1e-3          # perf_counter_ns gives microseconds
    degraded = False

    def open(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP)
        s.settimeout(2.0)
        return s

    def close(self, handle: socket.socket) -> None:
        try:
            handle.close()
        except Exception:
            pass

    def ping(self, handle: socket.socket, addr_int: int, timeout_ms: int,
             seq: int) -> tuple[float | None, str]:
        addr = _int_to_addr(addr_int)
        payload = _PROBE_MAGIC + struct.pack("<I", seq)
        # The kernel overwrites the identifier for datagram ICMP sockets, so it
        # is left at zero; correlation is by payload contents instead.
        hdr = struct.pack("!BBHHH", 8, 0, 0, 0, seq & 0xFFFF)
        pkt = struct.pack("!BBHHH", 8, 0, _icmp_checksum(hdr + payload), 0,
                          seq & 0xFFFF) + payload
        t0 = time.perf_counter_ns()
        try:
            handle.sendto(pkt, (addr, 0))
            deadline = t0 + timeout_ms * 1_000_000
            while True:
                remaining = (deadline - time.perf_counter_ns()) / 1e9
                if remaining <= 0:
                    return None, "request_timed_out"
                handle.settimeout(remaining)
                data, peer = handle.recvfrom(2048)
                if peer[0] != addr:
                    continue
                if _icmp_payload_matches(data, payload):
                    return (time.perf_counter_ns() - t0) / 1e6, "success"
        except socket.timeout:
            return None, "request_timed_out"
        except OSError as e:
            return None, f"oserror{e.errno}"

    def series(self, handle, addr_int, addr_str, n, interval_s,
               timeout_ms) -> list[float | None]:
        out: list[float | None] = []
        for i in range(n):
            rtt, _ = self.ping(handle, addr_int, timeout_ms, i)
            out.append(rtt)
            if i != n - 1 and interval_s:
                time.sleep(interval_s)
        return out


# --------------------------------------------------------------------------
# System ping binary backend
# --------------------------------------------------------------------------

_PING_TIME_RE = re.compile(r"time[=<]\s*([0-9.]+)\s*ms")


class _PingBinary:
    """Drives the platform ``ping`` command.

    One subprocess per anchor rather than per echo: process spawn dominates the
    cost, and ``-c N`` gets all the samples in a single call. Non-root Linux
    refuses intervals below 0.2 s, so that is the floor used regardless of what
    the caller asked for.
    """

    name = "icmp-ping-binary"
    kernel_timed = True           # the ping tool times the exchange itself
    resolution_ms = 1e-3
    degraded = False

    #: `-W` means seconds on iputils (Linux) and milliseconds on macOS/BSD.
    _W_IS_MS = sys.platform == "darwin"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or shutil.which("ping") or "ping"

    def open(self):
        return None

    def close(self, handle) -> None:
        pass

    def series(self, handle, addr_int, addr_str, n, interval_s,
               timeout_ms) -> list[float | None]:
        interval = max(0.2, float(interval_s or 0.0))
        cmd = [self.binary, "-n", "-c", str(n), "-i", f"{interval:g}"]
        if self._W_IS_MS:
            cmd += ["-W", str(int(timeout_ms))]
        else:
            cmd += ["-W", f"{max(0.5, timeout_ms / 1000.0):g}"]
        cmd.append(addr_str)
        budget = n * interval + timeout_ms / 1000.0 + 5.0
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=budget, check=False)
        except (subprocess.TimeoutExpired, OSError):
            return [None] * n
        times = [float(m) for m in _PING_TIME_RE.findall(proc.stdout or "")]
        if not times:
            return [None] * n
        # Losses are not attributable to specific positions, so successes are
        # placed first and the remainder marked as missing.
        return list(times[:n]) + [None] * max(0, n - len(times))

    def ping(self, handle, addr_int: int, timeout_ms: int,
             seq: int) -> tuple[float | None, str]:
        got = self.series(handle, addr_int, _int_to_addr(addr_int), 1, 0.0,
                          timeout_ms)
        v = got[0] if got else None
        return (v, "success") if v is not None else (None, "request_timed_out")


# --------------------------------------------------------------------------
# TCP fallback backend
# --------------------------------------------------------------------------


class _TcpConnect:
    name = "tcp-connect"
    kernel_timed = False
    resolution_ms = 1.0
    #: On Windows a blocking connect() wakeup is quantised to the scheduler tick,
    #: so this backend can only distinguish reachable from unreachable there.
    degraded = True

    def open(self):
        return None

    def close(self, handle) -> None:
        pass

    def ping(self, handle, addr_int: int, timeout_ms: int,
             seq: int) -> tuple[float | None, str]:
        addr = _int_to_addr(addr_int)
        for port in (443, 80, 8080):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout_ms / 1000.0)
            t0 = time.perf_counter_ns()
            try:
                s.connect((addr, port))
                return (time.perf_counter_ns() - t0) / 1e6, "success"
            except socket.timeout:
                continue
            except OSError as e:
                if e.errno in (10061, 111, 61):      # refused => host is up
                    return (time.perf_counter_ns() - t0) / 1e6, "refused"
                continue
            finally:
                s.close()
        return None, "request_timed_out"

    def series(self, handle, addr_int, addr_str, n, interval_s,
               timeout_ms) -> list[float | None]:
        out: list[float | None] = []
        for i in range(n):
            rtt, _ = self.ping(handle, addr_int, timeout_ms, i)
            out.append(rtt)
            if i != n - 1 and interval_s:
                time.sleep(interval_s)
        return out


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


#: Hosts used to confirm a backend can actually reach the internet. A datagram
#: ICMP socket can open and answer on loopback while the network silently drops
#: ICMP to everywhere else -- Azure and many corporate networks do exactly that --
#: so opening a socket is not evidence that probing works.
_REACHABILITY_PROBES = ("1.1.1.1", "8.8.8.8", "9.9.9.9")
_REACHABILITY_TIMEOUT_MS = 900


def _can_reach_internet(be, timeout_ms: int = _REACHABILITY_TIMEOUT_MS) -> bool:
    handle = None
    try:
        handle = be.open()
        for ip in _REACHABILITY_PROBES:
            addr = struct.unpack("<I", socket.inet_aton(ip))[0]
            rtt, _ = be.ping(handle, addr, timeout_ms, 1)
            if rtt is not None:
                return True
        return False
    except Exception:  # noqa: BLE001
        return False
    finally:
        if handle is not None:
            try:
                be.close(handle)
            except Exception:  # noqa: BLE001
                pass


def _make(name: str):
    if name == "windows":
        be = _WindowsIcmp()
        h = be.open()
        be.close(h)
        return be if _can_reach_internet(be) else None
    if name == "posix":
        be = _PosixIcmp()
        h = be.open()
        rtt, _ = be.ping(h, struct.unpack("<I", socket.inet_aton("127.0.0.1"))[0],
                         500, 1)
        be.close(h)
        if rtt is None:
            return None
        return be if _can_reach_internet(be) else None
    if name == "ping-binary":
        if not shutil.which("ping"):
            return None
        be = _PingBinary()
        return be if _can_reach_internet(be) else None
    if name == "tcp":
        be = _TcpConnect()
        return be if _can_reach_internet(be) else None
    return None


def _chain(force: str | None = None) -> list[str]:
    if force:
        return [force]
    if sys.platform == "win32":
        return ["windows", "ping-binary", "tcp"]
    if sys.platform == "darwin":
        return ["posix", "ping-binary", "tcp"]
    return ["posix", "ping-binary", "tcp"]


def select_backend(force: str | None = None):
    """Pick the best working backend, verifying each candidate actually probes."""
    tried = []
    for name in _chain(force):
        try:
            be = _make(name)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{name}({type(e).__name__})")
            continue
        if be is not None:
            return be
        tried.append(f"{name}(unavailable)")
    raise RuntimeError(
        "no probe backend could reach the internet; tried " + ", ".join(tried)
        + ". Outbound ICMP and TCP are both blocked on this network, so "
          "latency cannot be measured from here.")


class Prober:
    """Thread-safe prober; each worker thread lazily gets its own handle."""

    def __init__(self, backend=None, backend_name: str | None = None) -> None:
        self.backend = backend or select_backend(backend_name)
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._handles: list = []

    @property
    def name(self) -> str:
        return self.backend.name

    def _handle(self):
        h = getattr(self._tls, "handle", None)
        if h is None:
            h = self.backend.open()
            self._tls.handle = h
            with self._lock:
                self._handles.append(h)
        return h

    def series(self, ip: str, n: int = 12, interval: float = 0.02,
               timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> ProbeSeries:
        """Send ``n`` echo requests to ``ip``, spaced by ``interval`` seconds."""
        out = ProbeSeries(ip=ip)
        resolved = _resolve(ip)
        if resolved is None:
            out.samples_ms = [None] * n
            out.errors = ["dns_failure"]
            return out
        addr_int = struct.unpack("<I", socket.inet_aton(resolved))[0]
        handle = self._handle()
        out.samples_ms = self.backend.series(handle, addr_int, resolved, n,
                                             interval, timeout_ms)
        out.errors = ["success" if s is not None else "no_reply"
                      for s in out.samples_ms]
        return out

    def close(self) -> None:
        with self._lock:
            for h in self._handles:
                self.backend.close(h)
            self._handles.clear()
        self._tls = threading.local()
