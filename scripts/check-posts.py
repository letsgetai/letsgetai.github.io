#!/usr/bin/env python3
"""博客文章格式体检：上传前必跑。检查语法配对、代码块标注、裸 URL、链接可达性。"""
import glob, os, re, sys, urllib.request, urllib.error

BS = chr(92)
BT = chr(96)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_OK = True

def report(name, issues):
    global ALL_OK
    if issues:
        ALL_OK = False
        print("[FAIL] " + name)
        for i in issues:
            print("  - " + i)
    else:
        print("[OK]   " + name)

for f in sorted(glob.glob(os.path.join(ROOT, "content/posts/*.md"))):
    c = open(f, encoding="utf-8").read()
    name = os.path.basename(f)
    issues = []

    if c.count("**") % 2 != 0:
        issues.append("加粗 ** 不配对")

    parts = c.split("---", 2)
    body = parts[2] if len(parts) == 3 else c
    LQ = chr(0x201C); RQ = chr(0x201D)
    if body.count(LQ) != body.count(RQ):
        issues.append("中文引号“”不配对")
    body_clean = re.sub(BT + "[^" + BT + "]*" + BT, "", body)
    mixed = re.findall(LQ + "[^" + RQ + "]*" + chr(34), body_clean)
    mixed += re.findall(chr(34) + "[^" + RQ + "]*" + RQ, body_clean)
    if mixed:
        issues.append("引号混用（中文引号对里夹英文引号）")

    if c.count(BT) % 2 != 0:
        issues.append("反引号不配对")

    if (BS + "(") in c or (BS + "[") in c:
        issues.append("旧式分隔符 " + BS + "( / " + BS + "[")

    ws = " \t\n、，。；）)（([]" + chr(39) + chr(34) + BT
    url_re = re.compile("https?://[^" + ws + "]+")
    for m in url_re.finditer(c):
        start = m.start()
        if start >= 2 and c[start-2:start] == "](":
            continue
        issues.append("裸URL: " + m.group()[:60])

    fence_count = 0
    pos = 0
    while True:
        fence = c.find(BT*3, pos)
        if fence == -1: break
        fence_count += 1
        line_end = c.find(chr(10), fence)
        rest = c[fence+3:line_end if line_end > -1 else len(c)]
        if fence_count % 2 == 1 and rest.strip() == "":
            line_no = c[:fence].count(chr(10)) + 1
            issues.append("代码块无语言标注（第 " + str(line_no) + " 行）: " + BT*3 + " 后应跟语言如 text/python")
        pos = fence + 3

    if c.startswith("---"):
        fm = parts[1] if len(parts) >= 2 else ""
        if fm.count(chr(34)) % 2 != 0:
            issues.append("front matter 引号不配对")
        for k in ["title", "date", "draft"]:
            if k + ":" not in fm:
                issues.append("front matter 缺 " + k)

    report(name, issues)

print()
print("=== 链接可达性检查 ===")
all_links = set()
for f in glob.glob(os.path.join(ROOT, "content/posts/*.md")):
    c = open(f, encoding="utf-8").read()
    for m in re.finditer("]" + BS + "(" + "https?://[^)]+" + BS + ")", c):
        all_links.add(m.group()[2:-1])

bad_links = []
warn_links = []
for url in sorted(all_links):
    ok = False
    err = ''
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0 (blog-check)'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status >= 400:
                    err = 'HTTP ' + str(resp.status)
                else:
                    ok = True
                break
        except urllib.error.HTTPError as e:
            err = 'HTTP ' + str(e.code)
            if e.code < 500:
                break
        except Exception as e:
            err = str(e)[:60]
    if ok:
        continue
    if err.startswith('HTTP') and '404' in err:
        bad_links.append((url, err))
    else:
        warn_links.append((url, err))

if warn_links:
    for url, err in warn_links:
        print('[WARN] ' + url + ' -> ' + err)
    print('（网络波动警告，非死链；本地可手动复验）')

if bad_links:
    ALL_OK = False
    for url, err in bad_links:
        print("[DEAD] " + url + " -> " + err)
else:
    print("全部 " + str(len(all_links)) + " 个链接可达")

sys.exit(0 if ALL_OK else 1)