"""Fill reports/phase1/report_template.md from audit_summary.json and build HTML (+ optional PDF).

    python tools/build_report.py            # -> reports/phase1/build/report.html
    python tools/build_report.py --pdf       # also prints a PDF with headless Chrome/Chromium if found
    python tools/build_report.py --strict    # fail while any [[TODO: ...]] remains (use before the final commit)

Placeholders:  {{key.sub}}  {{key|pct}} (fraction -> %)  {{key|dict}} ("a: 1, b: 2")
"""
import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
PH = re.compile(r"\{\{\s*([^\}\|\s]+)\s*(?:\|\s*(\w+)\s*)?\}\}")
TODO = re.compile(r"\[\[TODO:(.*?)\]\]", re.S)

CSS = """
@page { size: A4; margin: 18mm 15mm; }
body { direction: rtl; text-align: right; font-family: 'Vazirmatn','Vazirmatn NL','Vazir','Noto Naskh Arabic','Tahoma',sans-serif;
       font-size: 11pt; line-height: 1.85; color:#111; }
h1 { font-size: 16pt; border-bottom: 2px solid #4a6fa5; padding-bottom: 4px; margin-top: 26px; }
h1.title { font-size: 19pt; border: none; text-align: center; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 7px; vertical-align: top; }
th { background: #eef3fb; }
code, pre { direction: ltr; text-align: left; unicode-bidi: embed; font-family: Consolas, monospace; }
pre { background: #f5f5f5; padding: 8px; }
img { max-width: 100%; display: block; margin: 8px auto; }
blockquote { border-right: 4px solid #4a6fa5; margin: 8px 0; padding: 2px 12px; background: #f4f7fb; }
mark.todo { background: #fff3a0; }
h1 { break-after: avoid; }
p:has(+ table), p:has(+ img) { break-after: avoid; }
th, td { unicode-bidi: plaintext; overflow-wrap: anywhere; }
body { overflow-wrap: anywhere; }
code { overflow-wrap: anywhere; }
pre { white-space: pre-wrap; }
table { table-layout: auto; max-width: 100%; }
tr { break-inside: avoid; }
"""


def lookup(d, path):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
            continue
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def fmt(v, mode):
    if mode == "pct":
        return f"{float(v) * 100:.1f}%"
    if mode == "dict":
        return "، ".join(f"{k}: {fmt(x, None)}" for k, x in v.items())
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}".replace(",", "")
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


CSV_PH = re.compile(r"\{\{@([\w\./]+)(?:\s*\|\s*([\w,\+ ]+))?\}\}")


def csv_table(m):
    """{{@file.csv|col1,col2}} -> markdown table of that CSV (reports/phase2/tables/), 3 significant decimals."""
    import pandas as pd
    df = pd.read_csv(ROOT / "reports/phase2/tables" / m.group(1))
    if m.group(2):
        df = df[[c.strip() for c in m.group(2).split(",")]]
    def cell(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))


def fill(text, summary):
    missing = []
    text = CSV_PH.sub(csv_table, text)

    def sub(m):
        try:
            return fmt(lookup(summary, m.group(1)), m.group(2))
        except KeyError:
            missing.append(m.group(1))
            return f"⟦MISSING:{m.group(1)}⟧"
    return PH.sub(sub, text), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--listening", default=None)
    ap.add_argument("--phase", type=int, default=1, help="1 or 2: selects default template/summary/output paths")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    ph = f"phase{a.phase}"
    a.template = a.template or str(ROOT / f"reports/{ph}/report_template.md")
    a.summary = a.summary or str(ROOT / (f"reports/{ph}/audit_summary.json" if a.phase == 1 else f"reports/{ph}/phase2_summary.json"))
    a.listening = a.listening or str(ROOT / f"reports/{ph}/listening_summary.json")

    summary = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    lp = Path(a.listening)
    if lp.exists():                                   # produced by: python -m src.energy_review score
        summary["listening"] = json.loads(lp.read_text(encoding="utf-8"))
    text = Path(a.template).read_text(encoding="utf-8")
    text, missing = fill(text, summary)

    title = re.search(r'^title:\s*"(.*)"\s*$', text, re.M)
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    todos = TODO.findall(text)

    out_dir = ROOT / f"reports/{ph}/build"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.filled.md").write_text(text, encoding="utf-8")

    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = TODO.sub(lambda m: f'<mark class="todo">TODO:{html.escape(m.group(1))}</mark>', body)
    page = (f'<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>Phase {a.phase}</title>'
            f"<style>{CSS}</style></head><body><h1 class='title'>{title.group(1) if title else ''}</h1>{body}</body></html>")
    # figures are referenced relative to reports/phase1/ -> place the HTML there so ../../figures resolves
    html_path = out_dir / "report.html"
    page = page.replace('src="../../figures', 'src="../../../figures')
    html_path.write_text(page, encoding="utf-8")

    print(f"HTML: {html_path}")
    if missing:
        print(f"!! {len(missing)} placeholder(s) not found in summary JSON: {sorted(set(missing))}")
    if todos:
        print(f"!! {len(todos)} TODO marker(s) still open:")
        for t in todos:
            print("   -", " ".join(t.split())[:110])
    if (missing or todos) and a.strict:
        sys.exit(1)

    if a.pdf:
        chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser", "chrome",
                                                  "msedge")), None)
        pdf = ROOT / f"reports/{ph}.pdf"
        if not chrome:
            print("No Chrome/Chromium on PATH: open the HTML in a browser and use Print -> Save as PDF "
                  f"-> save as {pdf}")
            return
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", html_path.as_uri()], check=True)
        print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
