#!/usr/bin/env python3
# download_earls_court.py (Drive-fixed)
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
    "drive.usercontent.google.com",  # NEW: actual file host used by Drive
    "issuu.com",
    "www.issuu.com",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.max_redirects = 10
TIMEOUT = 30

def safe_filename(url_or_name: str) -> str:
    # Accept either a URL or a plain filename, normalise to safe
    if "/" in url_or_name or "\\" in url_or_name:
        path = urlparse(url_or_name).path
        name = os.path.basename(path) or "file"
        qs = urlparse(url_or_name).query
        if qs and not name.lower().endswith(".pdf"):
            name = (name + "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", qs))[:140]
    else:
        name = url_or_name or "file"
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
    # Drive behaves badly with HEAD; skip and let Drive handler decide
    netloc = urlparse(url).netloc.lower()
    if "google.com" in netloc or "usercontent.google.com" in netloc:
        return False
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=TIMEOUT)
        ctype = r.headers.get("Content-Type", "")
        return "application/pdf" in ctype.lower()
    except Exception:
        return False

def parse_content_disposition_filename(disposition: str | None) -> str | None:
    if not disposition:
        return None
    # Try RFC 5987 filename*=
    m = re.search(r'filename\*\s*=\s*[^\'"]+\'[^\'"]*\'([^;]+)', disposition, flags=re.I)
    if m:
        return m.group(1)
    # Fallback to filename=
    m = re.search(r'filename\s*=\s*"([^"]+)"', disposition, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r'filename\s*=\s*([^;]+)', disposition, flags=re.I)
    if m:
        return m.group(1).strip()
    return None

def extract_gdrive_id(u: str) -> str | None:
    parsed = urlparse(u)
    if "google.com" not in parsed.netloc and "googleusercontent.com" not in parsed.netloc:
        return None
    # /file/d/<id>/...
    m = re.search(r"/file/d/([^/]+)/?", parsed.path)
    if m:
        return m.group(1)
    # id=...
    qs = parse_qs(parsed.query)
    fid = qs.get("id", [None])[0]
    if fid:
        return fid
    return None

def drive_build_initial_url(file_id: str) -> str:
    # The legacy uc endpoint is still the entry point that sets cookies
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def drive_try_usercontent_url(file_id: str, confirm: str | None = None) -> str:
    # Newer flow often lands here; confirm is optional on small files
    base = "https://drive.usercontent.google.com/download"
    if confirm:
        return f"{base}?id={file_id}&export=download&confirm={confirm}"
    return f"{base}?id={file_id}&export=download"

def drive_extract_confirm_token(html: str) -> str | None:
    # Look for confirm token in links or hidden inputs
    # href="...confirm=<token>..."
    m = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if m:
        return m.group(1)
    # hidden input: name="confirm" value="token"
    m = re.search(r'name=["\']confirm["\']\s+value=["\']([0-9A-Za-z_]+)["\']', html)
    if m:
        return m.group(1)
    return None

def download_gdrive(url: str, out_dir: str) -> dict:
    info = {"url": url, "status": "skipped", "path": None, "sha256": None, "note": None}
    try:
        file_id = extract_gdrive_id(url)
        if not file_id:
            info["status"] = "not_gdrive"
            return info

        # Step 1: hit uc endpoint to set cookies or (if small) get file directly
        init_url = drive_build_initial_url(file_id)
        r = SESSION.get(init_url, timeout=TIMEOUT, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "").lower()

        if "application/pdf" in ctype or "application/octet-stream" in ctype:
            # Direct file response
            disp = r.headers.get("Content-Disposition")
            fname = parse_content_disposition_filename(disp) or f"{file_id}.pdf"
            if not fname.lower().endswith(".pdf"):
                fname += ".pdf"
            path = os.path.join(out_dir, safe_filename(fname))
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            info["path"] = path
            info["sha256"] = sha256_file(path)
            info["status"] = "downloaded"
            return info

        # Step 2: it's an HTML interstitial; parse confirm token
        html = r.text
        if "Quota exceeded" in html or "download quota for this file has been exceeded" in html:
            info["status"] = "quota_exceeded"
            info["note"] = "Google Drive download quota exceeded"
            return info

        confirm = drive_extract_confirm_token(html)

        # Step 3: request usercontent with (maybe) confirm
        u2 = drive_try_usercontent_url(file_id, confirm)
        r2 = SESSION.get(u2, timeout=TIMEOUT, stream=True)
        ctype2 = r2.headers.get("Content-Type", "").lower()

        # Occasionally another HTML page; try once more to harvest token from it
        if "text/html" in ctype2 and confirm is None:
            html2 = r2.text
            confirm = drive_extract_confirm_token(html2)
            if confirm:
                u3 = drive_try_usercontent_url(file_id, confirm)
                r2 = SESSION.get(u3, timeout=TIMEOUT, stream=True)
                ctype2 = r2.headers.get("Content-Type", "").lower()

        if "application/pdf" not in ctype2 and "application/octet-stream" not in ctype2:
            info["status"] = f"not_pdf ({ctype2 or 'unknown'})"
            info["note"] = "Drive interstitial not resolved"
            return info

        disp2 = r2.headers.get("Content-Disposition")
        fname2 = parse_content_disposition_filename(disp2) or f"{file_id}.pdf"
        if not fname2.lower().endswith(".pdf"):
            fname2 += ".pdf"
        path = os.path.join(out_dir, safe_filename(fname2))

        with open(path, "wb") as f:
            for chunk in r2.iter_content(chunk_size=1024 * 256):
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

def try_issuu_pdf(u: str) -> str | None:
    try:
        html = SESSION.get(u, timeout=TIMEOUT).text
    except Exception:
        return None
    for m in re.finditer(r"https?://[^\s\"']+\.pdf", html, flags=re.I):
        cand = m.group(0)
        if is_pdf_url(cand):
            return cand
    return None

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
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        u = absolute(seed_url, href)
        if within_allowed(u):
            links.add(u)
    return links

def download_file(url: str, out_dir: str) -> dict:
    info = {"url": url, "status": "skipped", "path": None, "sha256": None, "note": None}
    try:
        netloc = urlparse(url).netloc.lower()

        # Google Drive (new flow)
        if "google.com" in netloc or "googleusercontent.com" in netloc:
            gd_info = download_gdrive(url, out_dir)
            if gd_info["status"] == "not_gdrive":
                # Not a recognised Drive link; fall through to generic logic
                pass
            else:
                return gd_info

        # Issuu: attempt to resolve direct PDF
        if "issuu.com" in netloc:
            direct = try_issuu_pdf(url)
            if direct:
                url = direct
            else:
                info["status"] = "needs_manual"
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

        # Prefer server-provided filename if any
        disp = resp.headers.get("Content-Disposition")
        fname = parse_content_disposition_filename(disp)
        if fname:
            name = safe_filename(fname)
        else:
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
        all_links |= links

    to_check = set(all_links)
    for u in list(all_links):
        if urlparse(u).netloc.endswith("earlscourt.com") and u.rstrip("/").startswith("https://www.earlscourt.com/planning"):
            to_check |= extract_links(u)

    candidates = set()
    for u in to_check:
        netloc = urlparse(u).netloc.lower()
        if u.lower().endswith(".pdf"):
            candidates.add(u)
        elif "drive.google.com" in netloc or "docs.google.com" in netloc or "drive.usercontent.google.com" in netloc:
            candidates.add(u)
        elif "public-access.lbhf.gov.uk" in netloc:
            candidates.add(u)
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
            print(f"[issuu_skipped] {u}")
            continue

        info = download_file(u, out_dir)
        results.append(info)
        if info["status"] == "downloaded":
            pdf_count += 1
        print(f"[{info['status']}] {u}")

        time.sleep(0.5)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "results": results}, f, indent=2)

    dl = sum(1 for r in results if r["status"] == "downloaded")
    pending = [r for r in results if r["status"] in ("issuu_skipped", "needs_manual")]
    print(f"\nDone. Downloaded {dl} file(s) to: {out_dir}")
    if pending:
        print(f"{len(pending)} item(s) need manual follow-up (see manifest.json).")

if __name__ == "__main__":
    main()
