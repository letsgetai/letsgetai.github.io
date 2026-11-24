# 未列入导航的调研页

这些页面可以通过网址打开，但不进入首页、归档、菜单、订阅源或 sitemap。页面使用 `noindex, nofollow, noarchive`。不要把它们写进 `content/`、`hugo.toml`、`README.md` 或布局菜单。

## 现有页面

| 路径 | 页面 |
| --- | --- |
| `/b7b38a5beeaa0614c5f52120/` | AI调研索引 |
| `/d51eedba8127d40f796861c0/` | 具身智能体（Embodied Agents）与控制运行时深度调研 |
| `/t5x9b3f7/` | 工作型数字 Agent：环境分类、两根趋势与可展示案例 |

样式在 `static/report-reading.css`。宽表改成带字段名的记录，字段按原表头顺序保留。

## 增加一篇报告

1. 在 `static/` 下新建至少 16 位随机十六进制目录，放入独立的 `index.html`。沿用现有报告的页头：`noindex, nofollow, noarchive`，并链接 `/report-reading.css`。
2. 在索引页加上这篇报告的原标题和链回。报告页放一个指向索引的“调研索引”链接。
3. 不要把路径加进公开导航。不要引入外部字体、CDN 或脚本。
4. 运行 `python3 scripts/check-unlisted-reports.py`。它对照 `main` 上的两篇既有报告，检查正文、记录字段、引用、媒体文件，以及索引路径没有漏进导航和文章目录。
