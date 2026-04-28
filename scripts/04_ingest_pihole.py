#!/usr/bin/env python3
"""
04_ingest_pihole.py — Pi-hole DNS Log Feature Extractor
========================================================
Reads Pi-hole's dnsmasq query log and extracts per-device behavioral
features over a rolling time window. Outputs a CSV that can be fed
directly into ids.py scan for anomaly scoring.

Features extracted per device per window:
    - query_count         : total DNS queries in window
    - unique_domains      : number of distinct domains queried
    - blocked_count       : queries blocked by Pi-hole
    - blocked_ratio       : blocked_count / query_count
    - avg_domain_length   : average length of queried domain names
    - max_domain_length   : longest domain queried
    - subdomain_ratio     : ratio of queries that have 3+ labels (e.g. a.b.com)
    - entropy_avg         : average Shannon entropy of queried domains
                            (high entropy = random-looking = C2 red flag)
    - query_rate          : queries per minute in this window
    - https_query_ratio   : ratio of HTTPS/AAAA vs A record queries

Usage:
    # Score the last 10 minutes of DNS activity:
    python 04_ingest_pihole.py

    # Score a specific window and pass straight to ids.py:
    python 04_ingest_pihole.py --window 15 --out /tmp/dns_features.csv
    python ids.py scan --input /tmp/dns_features.csv

    # Tail the log continuously (runs every N minutes):
    python 04_ingest_pihole.py --watch --interval 5
"""

import argparse
import csv
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── Default paths ─────────────────────────────────────────────────────────────

DEFAULT_LOG   = "/var/log/pihole/pihole.log"
DEFAULT_OUT   = "reports/dns_alerts.csv"
DEFAULT_WINDOW = 10   # minutes of log history to analyze

# ── Log line regex ─────────────────────────────────────────────────────────────
# Matches lines like:
# Apr 27 00:02:35 dnsmasq[1791]: query[A] dns.google from 192.168.0.45
# Apr 27 00:02:35 dnsmasq[1791]: /etc/pihole/gravity.list dns.google is 0.0.0.0

QUERY_RE   = re.compile(
    r"(\w{3}\s+\d+\s[\d:]+)\s+dnsmasq\[\d+\]:\s+query\[(\w+)\]\s+(\S+)\s+from\s+([\d.]+)"
)
BLOCKED_RE = re.compile(
    r"(\w{3}\s+\d+\s[\d:]+)\s+dnsmasq\[\d+\]:\s+(?:gravity|black|adlist).*?(\S+)\s+is\s+0\.0\.0\.0"
)


# ── Feature helpers ────────────────────────────────────────────────────────────

def shannon_entropy(s: str) -> float:
    """
    Compute Shannon entropy of a string.
    Random-looking domain names (e.g. C2 DGA domains) have high entropy.
    Normal domains like 'google.com' have low entropy.
    """
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((f / length) * math.log2(f / length) for f in freq.values())


def subdomain_depth(domain: str) -> int:
    """Return number of labels in a domain. google.com = 2, a.b.google.com = 4."""
    return len(domain.strip(".").split("."))


def parse_timestamp(ts_str: str, year: int) -> datetime:
    """Parse dnsmasq timestamp (no year) into a datetime, assuming current year."""
    try:
        return datetime.strptime(f"{year} {ts_str}", "%Y %b %d %H:%M:%S")
    except ValueError:
        return None


# ── Log parsing ────────────────────────────────────────────────────────────────

def parse_log(log_path: str, window_minutes: int) -> tuple[list[dict], set[str]]:
    """
    Parse pihole.log and return:
      - list of query dicts within the time window
      - set of domains that were blocked in the window
    """
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    year   = datetime.now().year

    queries  = []
    blocked  = set()

    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                # Check for blocked domains first
                m_block = BLOCKED_RE.search(line)
                if m_block:
                    ts = parse_timestamp(m_block.group(1), year)
                    if ts and ts >= cutoff:
                        blocked.add(m_block.group(2).lower())
                    continue

                # Check for query lines
                m_query = QUERY_RE.search(line)
                if m_query:
                    ts = parse_timestamp(m_query.group(1), year)
                    if ts and ts >= cutoff:
                        queries.append({
                            "timestamp": ts,
                            "qtype":     m_query.group(2),   # A, AAAA, HTTPS, etc.
                            "domain":    m_query.group(3).lower().rstrip("."),
                            "src_ip":    m_query.group(4),
                        })

    except PermissionError:
        print("[!] Permission denied reading log. Try: sudo python 04_ingest_pihole.py")
        raise
    except FileNotFoundError:
        print(f"[!] Log file not found: {log_path}")
        print("    Check that Pi-hole is running and the path is correct.")
        raise

    return queries, blocked


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(queries: list[dict], blocked: set[str], window_minutes: int) -> list[dict]:
    """
    Group queries by source IP and compute behavioral features per device.
    Returns a list of feature dicts, one per device seen in the window.
    """
    # Group by device IP
    by_device = defaultdict(list)
    for q in queries:
        by_device[q["src_ip"]].append(q)

    rows = []
    for ip, device_queries in by_device.items():
        domains      = [q["domain"] for q in device_queries]
        unique_doms  = set(domains)
        n            = len(domains)

        blocked_hits = sum(1 for d in domains if d in blocked)
        https_count  = sum(1 for q in device_queries if q["qtype"] in ("HTTPS", "AAAA"))
        entropies    = [shannon_entropy(d.split(".")[0]) for d in domains]  # entropy of leftmost label
        lengths      = [len(d) for d in domains]
        depths       = [subdomain_depth(d) for d in domains]

        rows.append({
            # Identity (kept for SOC context, not used as model features)
            "Src IP":             ip,
            "window_minutes":     window_minutes,

            # ── Features ──
            "query_count":        n,
            "unique_domains":     len(unique_doms),
            "blocked_count":      blocked_hits,
            "blocked_ratio":      round(blocked_hits / n, 4) if n else 0,
            "avg_domain_length":  round(sum(lengths) / n, 2) if n else 0,
            "max_domain_length":  max(lengths) if lengths else 0,
            "subdomain_ratio":    round(sum(1 for d in depths if d >= 3) / n, 4) if n else 0,
            "entropy_avg":        round(sum(entropies) / n, 4) if entropies else 0,
            "entropy_max":        round(max(entropies), 4) if entropies else 0,
            "query_rate":         round(n / window_minutes, 4),
            "https_query_ratio":  round(https_count / n, 4) if n else 0,
        })

    return rows


# ── Output ─────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], out_path: str):
    if not rows:
        print("[!] No queries found in window. Nothing to write.")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Wrote {len(rows)} device rows to {out_path}")


def print_summary(rows: list[dict]):
    """Print a quick human-readable summary to terminal."""
    if not rows:
        return

    print("\n" + "═" * 50)
    print("  Pi-hole DNS Feature Summary")
    print("═" * 50)
    print(f"  Devices seen      : {len(rows)}")
    print(f"  Total queries     : {sum(r['query_count'] for r in rows):,}")
    print(f"  Total blocked     : {sum(r['blocked_count'] for r in rows):,}")
    print()
    print(f"  {'Device IP':<18} {'Queries':>8} {'Blocked':>8} {'Entropy':>8} {'Rate/min':>9}")
    print(f"  {'─'*17:<18} {'─'*7:>8} {'─'*7:>8} {'─'*7:>8} {'─'*8:>9}")

    # Sort by entropy_avg descending (most suspicious first)
    for r in sorted(rows, key=lambda x: x["entropy_avg"], reverse=True):
        flag = " ⚠" if r["entropy_avg"] > 3.5 or r["blocked_ratio"] > 0.3 else ""
        print(
            f"  {r['Src IP']:<18} {r['query_count']:>8,} {r['blocked_count']:>8,} "
            f"{r['entropy_avg']:>8.2f} {r['query_rate']:>9.2f}{flag}"
        )

    print()
    print("  ⚠ = high entropy (possible DGA/C2) or high block ratio")
    print("═" * 50)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract DNS behavioral features from Pi-hole logs for ML-IDS scoring."
    )
    parser.add_argument("--log",      default=DEFAULT_LOG,    help="Path to pihole.log")
    parser.add_argument("--out",      default=DEFAULT_OUT,    help="Output features CSV path")
    parser.add_argument("--window",   type=int, default=DEFAULT_WINDOW,
                        help="Minutes of log history to analyze (default: 10)")
    parser.add_argument("--watch",    action="store_true",
                        help="Run continuously, re-analyzing every --interval minutes")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between runs in --watch mode (default: 5)")
    parser.add_argument("--quiet",    action="store_true", help="Suppress terminal summary")
    args = parser.parse_args()

    def run_once():
        print(f"[*] Parsing last {args.window} minutes of {args.log} ...")
        queries, blocked = parse_log(args.log, args.window)
        print(f"[*] Found {len(queries):,} queries from {len(set(q['src_ip'] for q in queries))} devices")

        rows = extract_features(queries, blocked, args.window)

        if not args.quiet:
            print_summary(rows)

        write_csv(rows, args.out)
        print(f"\n[→] To score with ML-IDS:")
        print(f"    python ids.py scan --input {args.out} --sensitivity high\n")

    if args.watch:
        print(f"[*] Watch mode: running every {args.interval} minutes. Ctrl+C to stop.")
        while True:
            run_once()
            print(f"[*] Next run in {args.interval} minutes...")
            time.sleep(args.interval * 60)
    else:
        run_once()


if __name__ == "__main__":
    main()
