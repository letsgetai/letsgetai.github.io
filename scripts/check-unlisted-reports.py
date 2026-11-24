#!/usr/bin/env python3
"""Content equivalence for the unlisted reports against main.

Compares the restyled pages with the copies on main. Tables may be rendered
as labeled records; each header and cell must survive in order. Chrome,
spacing, and the index link are ignored.
"""

from __future__ import annotations

import hashlib
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMB = "static/d51eedba8127d40f796861c0/index.html"
LEG = "static/t5x9b3f7/index.html"
MEDIA = [
    "01-gpt-researcher.gif",
    "02-openhands-automate.png",
    "03-mini-swe-agent.gif",
    "04-browser-use-dmv.gif",
    "05-unitree-g1.gif",
    "06-autoresearch-progress.png",
    "07-ai-scientist.gif",
    "08-darwin-godel-machine.gif",
]


def git_show(path: str) -> str:
    return subprocess.check_output(["git", "show", f"main:{path}"], cwd=ROOT, text=True)


def git_show_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"main:{path}"], cwd=ROOT)


def textify(fragment: str) -> str:
    fragment = re.sub(
        r"<sup>\s*(?:<a\b[^>]*>)?(\d+)(?:</a>)?\s*</sup>",
        r"〔\1〕",
        fragment,
        flags=re.S,
    )
    fragment = re.sub(
        r'<span\b[^>]*vertical-align:\s*super[^>]*>(\d+)</span>',
        r"〔\1〕",
        fragment,
        flags=re.S,
    )
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment).replace("\xa0", " ")
    return re.sub(r"\s+", " ", fragment).strip()


def article_of(page: str) -> str:
    match = re.search(r'<article id="article">(.*)</article>', page, re.S)
    if match:
        return match.group(1)
    body = re.search(r"<body[^>]*>(.*)</body>", page, re.S)
    if not body:
        raise SystemExit("no article or body")
    return body.group(1)


def strip_chrome(fragment: str) -> str:
    fragment = re.sub(r'<p class="(?:index-link|backtop)">.*?</p>', "", fragment, flags=re.S)
    fragment = re.sub(r'<section class="records">.*?</section>', "\n", fragment, flags=re.S)
    fragment = re.sub(r'<div class="(?:table-scroll|wrap)"[^>]*>.*?</div>', "\n", fragment, flags=re.S)
    return fragment


def paragraphs(fragment: str) -> list[str]:
    body = strip_chrome(fragment)
    return [textify(p) for p in re.findall(r"<p\b[^>]*>(.*?)</p>", body, re.S) if textify(p)]


def headings(fragment: str) -> list[str]:
    body = strip_chrome(fragment)
    return [textify(h) for h in re.findall(r"<h[1-3]\b[^>]*>(.*?)</h[1-3]>", body, re.S)]


def tables(fragment: str) -> list[list[list[tuple[str, str]]]]:
    found = []
    for table in re.findall(r"<table\b[^>]*>(.*?)</table>", fragment, re.S):
        rows = []
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S):
            rows.append([textify(cell) for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.S)])
        headers = rows[0]
        found.append([list(zip(headers, row)) for row in rows[1:]])
    return found


def records(fragment: str) -> list[list[list[tuple[str, str]]]]:
    found = []
    for section in re.findall(r'<section class="records">(.*?)</section>', fragment, re.S):
        group = []
        for record in re.findall(r'<article class="record">(.*?)</article>', section, re.S):
            pairs = [
                (textify(dt), textify(dd))
                for dt, dd in re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", record, re.S)
            ]
            group.append(pairs)
        found.append(group)
    return found


def resource_groups(fragment: str) -> list[list[list[tuple[str, str]]]]:
    grouped = records(fragment)
    return grouped if grouped else tables(fragment)


def refs(fragment: str) -> list[tuple[str, tuple[str, ...]]]:
    match = re.search(r'<ol class="refs">(.*?)</ol>', fragment, re.S)
    if not match:
        return []
    items = []
    for li in re.findall(r"<li\b[^>]*>(.*?)</li>", match.group(1), re.S):
        hrefs = tuple(
            html.unescape(href)
            for href in re.findall(r'href="([^"]*)"', li)
            if not href.startswith("#")
        )
        items.append((textify(li), hrefs))
    return items


def sups(fragment: str) -> list[str]:
    return re.findall(r"<sup>\s*(?:<a\b[^>]*>)?(\d+)", fragment)


def content_hrefs(fragment: str) -> list[str]:
    cleaned = re.sub(r'<p class="(?:index-link|backtop)">.*?</p>', "", fragment, flags=re.S)
    cleaned = re.sub(r'<nav\b[^>]*>.*?</nav>', "", cleaned, flags=re.S)
    return [html.unescape(href) for href in re.findall(r'href="([^"]*)"', cleaned) if not href.startswith("#")]


def media(fragment: str) -> list[tuple[str, str]]:
    imgs = re.findall(r'<img\b[^>]*alt="([^"]*)"[^>]*src="([^"]*)"', fragment)
    imgs += [(alt, src) for src, alt in re.findall(r'<img\b[^>]*src="([^"]*)"[^>]*alt="([^"]*)"', fragment)]
    # unique preserving order
    seen = []
    for alt, src in imgs:
        item = (html.unescape(alt), html.unescape(src))
        if item not in seen:
            seen.append(item)
    videos = re.findall(r'<video\b[^>]*src="([^"]*)"', fragment)
    return seen, videos


def expect(name: str, left, right) -> None:
    if left != right:
        print(f"[FAIL] {name}")
        if isinstance(left, list) and isinstance(right, list):
            print(f"  lens {len(left)} vs {len(right)}")
            for i, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    print("  first", i)
                    print("  base", str(a)[:400])
                    print("  now ", str(b)[:400])
                    break
            if len(left) != len(right):
                extra = left[len(right):] if len(left) > len(right) else right[len(left):]
                print("  extra", str(extra[:2])[:400])
        else:
            print(" ", left)
            print(" ", right)
        raise SystemExit(1)
    print(f"[OK]   {name}")


def check_embodied() -> None:
    base = article_of(git_show(EMB))
    now = article_of((ROOT / EMB).read_text(encoding="utf-8"))
    expect("embodied paragraphs", paragraphs(base), paragraphs(now))
    expect("embodied headings", headings(base), headings(now))
    expect("embodied records", resource_groups(base), resource_groups(now))
    expect("embodied bibliography", refs(base), refs(now))
    expect("embodied superscripts", sups(base), sups(now))
    if len(sups(now)) != 233:
        raise SystemExit(f"sup count {len(sups(now))}")
    groups = resource_groups(now)
    rows = sum(len(g) for g in groups)
    cells = sum(len(pair) for g in groups for pair in g)
    if (len(groups), rows, cells) != (5, 35, 175):
        # 35 data rows * 5 columns = 175 data cells; plus 5 header rows were 200 cells
        raise SystemExit(f"record shape {(len(groups), rows, cells)}")
    header_cells = 5 * 5
    if cells + header_cells != 200:
        raise SystemExit("cell accounting")
    print("[OK]   embodied 5 groups, 35 records, 200 header+cell slots")
    glued = ["rosa52", "OM168", "eva-021", "github.io/5", "RoboBrain2.559", "08448v35"]
    visible = "\n".join(dd for g in groups for rec in g for _, dd in rec)
    separated = visible.replace("〔", " [").replace("〕", "] ")
    for bad in glued:
        if bad in separated:
            raise SystemExit(f"glued citation {bad}")
    for needle in ["rosa〔52〕", "OM1〔68〕", "harnessvla.github.io/〔5〕", "eva-0〔21〕", "RoboBrain2.5〔59〕", "2607.08448v3〔5〕"]:
        if needle not in visible:
            raise SystemExit(f"missing boundary {needle}")
    print("[OK]   embodied citation boundaries")


def check_legacy() -> None:
    base_page = git_show(LEG)
    now_page = (ROOT / LEG).read_text(encoding="utf-8")
    base = article_of(base_page)
    now = article_of(now_page)
    expect("legacy headings", headings(base), headings(now))
    expect("legacy paragraphs", paragraphs(base), paragraphs(now))
    expect("legacy records", resource_groups(base), resource_groups(now))
    expect("legacy content hrefs", content_hrefs(base), content_hrefs(now))
    expect("legacy images and video", media(base), media(now))
    imgs, videos = media(now)
    if len(imgs) != 8 or len(videos) != 2:
        raise SystemExit(f"media counts {len(imgs)} {len(videos)}")
    for name in MEDIA:
        old = git_show_bytes(f"static/t5x9b3f7/{name}")
        new = (ROOT / "static/t5x9b3f7" / name).read_bytes()
        if old != new:
            raise SystemExit(f"media bytes changed {name}")
        if hashlib.sha256(new).hexdigest() != hashlib.sha256(old).hexdigest():
            raise SystemExit(name)
    print("[OK]   legacy 8 media files unchanged")
    if 'content="noindex, nofollow, noarchive"' not in now_page:
        raise SystemExit("legacy robots")
    print("[OK]   legacy robots")


def check_index_and_nav() -> None:
    pages = sorted((ROOT / "static").glob("*/index.html"))
    index_pages = [p for p in pages if "AI调研索引" in p.read_text(encoding="utf-8")]
    if len(index_pages) != 1:
        raise SystemExit(f"index pages {len(index_pages)}")
    index = index_pages[0]
    token = index.parent.name
    if not re.fullmatch(r"[0-9a-f]{24}", token):
        raise SystemExit(f"index token {token}")
    text = index.read_text(encoding="utf-8")
    if 'content="noindex, nofollow, noarchive"' not in text:
        raise SystemExit("index robots")
    if "<script" in text.lower():
        raise SystemExit("index script")
    for href in ("/d51eedba8127d40f796861c0/", "/t5x9b3f7/"):
        if f'href="{href}"' not in text:
            raise SystemExit(f"index missing {href}")
    for page in (ROOT / EMB, ROOT / LEG):
        body = page.read_text(encoding="utf-8")
        if f'href="/{token}/"' not in body:
            raise SystemExit(f"{page.parent.name} missing index link")
        if 'href="/report-reading.css"' not in body and 'href="/report-reading.css"' not in body:
            raise SystemExit("css")
    # Site navigation surfaces must not name the unlisted paths.
    banned_roots = [
        ROOT / "README.md",
        ROOT / "hugo.toml",
        ROOT / "content",
        ROOT / "layouts",
        ROOT / "i18n",
    ]
    needles = [token, "d51eedba8127d40f796861c0", "t5x9b3f7"]
    for root in banned_roots:
        files = [root] if root.is_file() else list(root.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in raw:
                    raise SystemExit(f"{needle} leaked into {path.relative_to(ROOT)}")
    print(f"[OK]   index /{token}/ stays out of nav, posts, and config")


def main() -> None:
    check_embodied()
    check_legacy()
    check_index_and_nav()
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc)
        raise
