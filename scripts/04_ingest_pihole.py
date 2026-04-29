import argparse
import csv
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
 
# set the default paths for pihole logs and output CSV
 
DEFAULT_LOG   = "/var/log/pihole/pihole.log"
DEFAULT_OUT   = "reports/dns_alerts.csv"
DEFAULT_WINDOW = 10   # this is the default value to see how far back in the logs we want to analyze in minutes
 
# log line regexes to extract queries and blocked domains
 
QUERY_RE   = re.compile(
    r"(\w{3}\s+\d+\s[\d:]+)\s+dnsmasq\[\d+\]:\s+query\[(\w+)\]\s+(\S+)\s+from\s+([\d.]+)"
)
BLOCKED_RE = re.compile(
    r"(\w{3}\s+\d+\s[\d:]+)\s+dnsmasq\[\d+\]:\s+(?:gravity|black|adlist).*?(\S+)\s+is\s+0\.0\.0\.0"
)
 
 
# feature extraction helpers
 
def shannon_entropy(s: str) -> float:
    # Calculate Shannon entropy of a string (e.g. domain label) to detect randomness.
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((f / length) * math.log2(f / length) for f in freq.values())
 


def subdomain_depth(domain: str) -> int:
    # Count the number of labels in the domain to measure subdomain depth
    return len(domain.strip(".").split("."))
 
 
def parse_timestamp(ts_str: str, year: int) -> datetime:
    # parse the timestamp from the log line and return a datetime object
    try:
        return datetime.strptime(f"{year} {ts_str}", "%Y %b %d %H:%M:%S")
    except ValueError:
        return None
 
 
# parsing the pihole.log file to extract queries and blocked domains within the specified time window
 
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

    # we read the log file line by line checking for matches to our regexes, for each line we check if it matches the blocked domain pattern first
    # if it doesn't match we check if it matches a query pattern, if it matches there we extract the timestamp and check if it's within our cutoff time
    # if it's within cutoff time, we add the query to our list of queries and if it's a blocked domain we add it to our set of blocked domains
    # we also handle some common exceptions that might occur when trying to read the log file such as permission errors or file not found errors
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
 
 
# feature extraction: for each device (source IP) 
# we compute various features based on their DNS query behavior in the time window
# such as total queries, unique domains, blocked ratio, average domain length, entropy, etc
 
def extract_features(queries: list[dict], blocked: set[str], window_minutes: int) -> list[dict]:
    """
    Group queries by source IP and compute behavioral features per device.
    Returns a list of feature dicts, one per device seen in the window.
    """
    # group by device IP
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
            # identity (kept for SOC context, not used for model training)
            "Src IP":             ip,
            "window_minutes":     window_minutes,
 
            # features
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
 
 
# Output
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
 
# a terminal summary to print out the key features for each device in a human-readable format, sorted by suspiciousness
# useful without having to dig into the CSV 
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
 
    # sort by entropy_avg descending (most suspicious first)
    for r in sorted(rows, key=lambda x: x["entropy_avg"], reverse=True):
        flag = " ⚠" if r["entropy_avg"] > 3.5 or r["blocked_ratio"] > 0.3 else ""
        print(
            f"  {r['Src IP']:<18} {r['query_count']:>8,} {r['blocked_count']:>8,} "
            f"{r['entropy_avg']:>8.2f} {r['query_rate']:>9.2f}{flag}"
        )
 
    print()
    print("  ⚠ = high entropy (possible DGA/C2) or high block ratio")
    print("═" * 50)
 
 
# main function to parse arguments and run the log parsing and feature extraction
 
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
