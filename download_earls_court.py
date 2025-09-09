#!/usr/bin/env python3
# download_earls_court.py
import argparse, hashlib, json, os, re, sys, time
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urljoin, quote_plus

import requests
from bs4 import BeautifulSoup

DEFAULT_SEEDS = [
    "https://www.earlscourt.com/planning/",
    "https://www.lbhf.gov.uk/planning/regeneration-transforming-our-borough/earls-court-development",
]

PDF_EXT = (".pdf",)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EarlsCourtDownloader/1.0; +https://example.local)"
}

ALLOWED_DOMAINS = {
    "earlscourt.com",
    "www.earlscourt.com",
    "rbkc.gov.uk",
    "www.rbkc.gov.uk",
    "lbhf.gov.uk",
    "www.lbhf.gov.uk",
    "public-access.lbhf.gov.uk",
    "drive.google.com",
    "docs.google.com",
    "issuu.com",
    "www.issuu.com",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.max_redirects = 10
TIMEOUT = 30

def safe_filename(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path) or "file"
    # Keep query hint if helpful
    qs = urlparse(url).query
    if qs and not name.lower().endswith(".pdf"):
        name = (name + "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", qs))[:140]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def sha256_file(fp) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_pdf_url(url: str) -> bool:
    # Quick check first
    if url.lower().endswith(PDF_EXT):
        return True
    # Some portals serve PDFs via no-extension routes; we’ll HEAD it
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=TIMEOUT)
        ctype = r.headers.get("Content-Type", "")
        return "application/pdf" in ctype.lower()
    except Exception:
        return False

def goog_drive_direct(u: str) -> str | None:
    # Handle both docs.google.com and drive.google.com viewer links
    parsed = urlparse(u)
    if "google.com" not in parsed.netloc:
        return None
    # patterns: /file/d/<id>/view, /uc?id=<id>, open?id=<id>
    m = re.search(r"/file/d/([^/]+)/", parsed.path)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    qs = parse_qs(parsed.query)
    fid = qs.get("id", [None])[0]
    if fid:
        return f"https://drive.google.com/uc?export=download&id={fid}"
    return None

def try_issuu_pdf(u: str) -> str | None:
    # Best-effort: sometimes Issuu exposes a downloadable PDF in meta tags or JSON.
    # We’ll fetch HTML and look for a direct .pdf in og:video / scripts.
    try:
        html = SESSION.get(u, timeout=TIMEOUT).text
    except Exception:
        return None
    # Look for obvious .pdf URLs
    for m in re.finditer(r"https?://[^\s\"']+\.pdf", html, flags=re.I):
        cand = m.group(0)
        # Sanity check the URL actually returns a PDF
        if is_pdf_url(cand):
            return cand
    return None  # Fall back to manual later if not found

def within_allowed(u: str) -> bool:
    try:
        netloc = urlparse(u).netloc.lower()
        return netloc in ALLOWED_DOMAINS
    except Exception:
        return False

def absolute(seed: str, href: str) -> str:
    return urljoin(seed, href)

def extract_links(seed_url: str) -> set[str]:
    links = set()
    try:
        r = SESSION.get(seed_url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"[warn] failed to fetch {seed_url}: {e}", file=sys.stderr)
        return links

    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Skip anchors/mailto
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        u = absolute(seed_url, href)
        if within_allowed(u):
            links.add(u)

    # Some pages (like LBHF) list many PDFs across sections – also follow obvious pagination in-page
    return links

def download_file(url: str, out_dir: str) -> dict:
    info = {"url": url, "status": "skipped", "path": None, "sha256": None, "note": None}
    try:
        # Convert Google Drive viewers to direct
        gd = goog_drive_direct(url)
        if gd:
            url = gd

        # Try Issuu direct PDF resolution
        if "issuu.com" in urlparse(url).netloc.lower():
            direct = try_issuu_pdf(url)
            if direct:
                url = direct
            else:
                info["status"] = "needs_manual"  # captured in manifest
                info["note"] = "Issuu page saved; no direct PDF found"
                return info

        # If it isn't clearly a PDF, test content-type
        if not is_pdf_url(url):
            info["status"] = "not_pdf"
            return info

        # Stream download
        resp = SESSION.get(url, stream=True, timeout=TIMEOUT)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "application/pdf" not in ctype.lower():
            info["status"] = f"not_pdf ({ctype})"
            return info

        name = safe_filename(url)
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        path = os.path.join(out_dir, name)

        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        info["path"] = path
        info["sha256"] = sha256_file(path)
        info["status"] = "downloaded"
        return info
    except requests.HTTPError as e:
        info["status"] = f"http_error {e.response.status_code if e.response else ''}"
        return info
    except Exception as e:
        info["status"] = f"error: {e}"
        return info

def main():
    ap = argparse.ArgumentParser(description="Earl's Court planning pack downloader")
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS,
                    help="Seed pages to crawl for document links")
    ap.add_argument("--out", default=None, help="Output directory (default: downloads/earls-court-YYYYmmdd-HHMMSS)")
    ap.add_argument("--max", type=int, default=0, help="Max files to download (0 = no limit)")
    ap.add_argument("--include-issuu", action="store_true", help="Attempt best-effort Issuu PDF resolution")
    ap.add_argument("--domain", nargs="*", default=[], help="Extra allowed domains (whitelist)")
    args = ap.parse_args()

    if args.domain:
        for d in args.domain:
            ALLOWED_DOMAINS.add(d.lower())

    out_dir = args.out or os.path.join("downloads", f"earls-court-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")

    all_links = set()
    for seed in args.seeds:
        print(f"[seed] {seed}")
        links = extract_links(seed)
        # Keep direct PDF links and also keep known portals/Issuu/Drive for follow-up
        all_links |= links

    # Expand one level for earlscourt.com/planning pages (often sectioned)
    to_check = set(all_links)
    for u in list(all_links):
        if urlparse(u).netloc.endswith("earlscourt.com") and u.rstrip("/").startswith("https://www.earlscourt.com/planning"):
            to_check |= extract_links(u)

    # Filter to plausible doc links (pdfs, drive viewers, public-access endpoints, issuu pages)
    candidates = set()
    for u in to_check:
        netloc = urlparse(u).netloc.lower()
        if u.lower().endswith(".pdf"):
            candidates.add(u)
        elif "drive.google.com" in netloc or "docs.google.com" in netloc:
            candidates.add(u)
        elif "public-access.lbhf.gov.uk" in netloc:
            candidates.add(u)  # many are direct file links
        elif "rbkc.gov.uk" in netloc and (".pdf" in u.lower() or "/downloads/" in u.lower()):
            candidates.add(u)
        elif "issuu.com" in netloc:
            candidates.add(u)

    results = []
    pdf_count = 0
    for u in sorted(candidates):
        if (args.max and pdf_count >= args.max):
            break
        if "issuu.com" in u and not args.include_issuu:
            results.append({"url": u, "status": "issuu_skipped", "note": "run with --include-issuu to attempt"})
            continue
        info = download_file(u, out_dir)
        results.append(info)
        if info["status"] == "downloaded":
            pdf_count += 1
        print(f"[{info['status']}] {u}")

        # Be nice to hosts
        time.sleep(0.5)

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "results": results}, f, indent=2)

    # Summary
    dl = sum(1 for r in results if r["status"] == "downloaded")
    pending = [r for r in results if r["status"] in ("issuu_skipped", "needs_manual")]
    print(f"\nDone. Downloaded {dl} file(s) to: {out_dir}")
    if pending:
        print(f"{len(pending)} item(s) need manual follow-up (see manifest.json).")

if __name__ == "__main__":
    main()
