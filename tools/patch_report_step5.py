"""One-off patch: updates reports/phase1/report_template.md for step 5 (blinded listening test,
exclusion bookkeeping). Safe: every anchor must match exactly once, otherwise NOTHING is changed.

    python tools/patch_report_step5.py
"""
import re
import shutil
import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1] / "reports/phase1/report_template.md"

NEW_ROW_LOW = ("| انرژی کم (RMS < {{low_energy_threshold_dbfs}} dBFS؛ آستانه‌ی توصیفی) | {{n_low_energy}} فایل؛ جاروب آستانه: "
               "{{low_energy_sweep|dict}}؛ میانه‌ی Frame_Dyn_dB: کم‌انرژی‌ها {{frame_dyn_db_median_low_energy}} در برابر بقیه "
               "{{frame_dyn_db_median_other}} | پرچم `Low_Energy`؛ بدون تغییر برچسب؛ حذف فقط طبق قاعده‌ی پیش‌ثبت‌شده‌ی آزمون "
               "شنیدن کور (پایین) |")

NEW_ROW_EXCL = ("| جمع‌بندی حذف‌ها | {{n_excluded}} فایل؛ دلایل: {{exclude_reason_counts|dict}} | اعمال یکسان روی همه‌ی splitها پس از "
                "split؛ تخصیص گویندگان تغییر نمی‌کند |")

NEW_PARAGRAPH = """**فایل‌های کم‌انرژی.** نسخه‌ی اولیه‌ی تحلیل آستانه را از صدک ۱٪ کمترین دامنه می‌گرفت؛ چنین آستانه‌ای *بدون توجه به داده* همیشه دقیقاً ۱٪ فایل‌ها را پرچم می‌زند و دلیل علمی ندارد. در این نسخه، RMS بر حسب dBFS فقط یک پرچم **توصیفی** (`Low_Energy`) است و تصمیم حذف با یک **آزمون شنیدن کور** با قاعده‌ی پیش‌ثبت‌شده گرفته می‌شود. این فایل‌ها در پوشه‌ی کلمه‌ی خود هستند و **خطای برچسب یا «کلاس سکوت» تلقی نمی‌شوند**: ممکن است گوینده با بهره‌ی بسیار کم یا میکروفون ضعیف ضبط کرده باشد. ساختن کلاس «سکوت» از آن‌ها نادرست است، چون همچنان برچسب کلمه دارند و نمی‌توانند نماینده‌ی ورودی «بدون گفتار» باشند.

**شاخص عینی مکمل.** برای تفکیک «گفتار ضعیف» از «فقط نویز/هیس»، پویایی فریمی (`Frame_Dyn_dB` = صدک ۹۵ منهای صدک ۱۰ سطح RMS فریم‌های ۲۵ms، با کف ‎−۱۰۰ dBFS) محاسبه شد: گفتار بین هجاها و مکث‌ها پویایی زیاد دارد و نویز ایستا ندارد. میانه‌ی آن برای فایل‌های کم‌انرژی {{frame_dyn_db_median_low_energy}} dB در برابر {{frame_dyn_db_median_other}} dB برای بقیه است (شکل ۱۱). این شاخص فقط توصیفی است و در قاعده‌ی تصمیم نقشی ندارد.

![انرژی در برابر پویایی](../../figures/phase1/fig11_energy_vs_dynamics.png)

**پروتکل شنیدن کور (پیش از شنیدن در `configs/config.yaml` ثبت شد).** (۱) کلیپ‌ها بر اساس RMS در شش بازه‌ی <−۷۰، −۷۰..−۶۰، −۶۰..−۵۰، −۵۰..−۴۰، −۴۰..−۳۰ و ≥ −۳۰ dBFS دسته‌بندی شدند؛ بازه‌ی آخر (سطح عادی) **کنترل** است. (۲) از هر بازه حداکثر ۶ کلیپ با seed ثابت ۴۲ انتخاب شد. (۳) همه‌ی کلیپ‌ها peak-normalize (۰٫۷)، با نام بی‌معنا (`clip_001` و ...) و به ترتیب تصادفی ارائه شدند؛ شنونده نه برچسب را می‌دید و نه سطح انرژی را. (۴) برای هر کلیپ یکی از ۸ کلمه یا «؟» (قابل‌تشخیص نیست) ثبت شد. (۵) کلیپ *غیرقابل‌استفاده* است اگر کلمه‌ی شنیده‌شده با برچسب یکی نباشد («؟» هم شامل است). (۶) **قاعده:** اگر در بازه‌ای زیر کنترل نرخ غیرقابل‌استفاده ≥ ۵۰٪ باشد، لبه‌ی بالایی بالاترین چنین بازه‌ای آستانه‌ی حذف می‌شود و همه‌ی فایل‌های با RMS کمتر از آن از train/val/test حذف می‌شوند؛ در غیر این صورت هیچ فایلی به دلیل انرژی حذف نمی‌شود. (۷) اگر نرخ غیرقابل‌استفاده‌ی کنترل از ۲۰٪ بیشتر باشد، آزمون نامعتبر است و تکرار می‌شود.

**نتیجه‌ی آزمون** ({{listening.n_listened}} کلیپ؛ نرخ نامطابقت کنترل: {{listening.control_unusable_rate|pct}}):

{{listening.table_md}}

{{listening.decision_text}}

**محدودیت‌ها.** یک شنونده (خودم)؛ حدود ۶ کلیپ در هر بازه، یعنی توان آماری کم و فاصله‌های اطمینان پهن؛ و قابل‌فهم‌بودن برای انسان دقیقاً با قابل‌استفاده‌بودن برای مدل یکی نیست. به همین دلیل در فاز ۴ تحلیل حساسیت (با/بدون این فایل‌ها) انجام می‌شود. حذف‌ها پس از split و فقط با حذف ردیف اعمال می‌شوند، پس تخصیص گویندگان به splitها تغییر نمی‌کند.
"""

NEW_SPLIT_TABLE = """| split | گوینده | فایل (پیش از حذف) | سهم فایل‌ها | فایل (پس از حذف) |
|---|---|---|---|---|
| train | {{split.speakers_per_split.train}} | {{split.files_per_split.train}} | {{split.file_share_per_split.train|pct}} | {{split.files_per_split_after_exclusion.train}} |
| val | {{split.speakers_per_split.val}} | {{split.files_per_split.val}} | {{split.file_share_per_split.val|pct}} | {{split.files_per_split_after_exclusion.val}} |
| test | {{split.speakers_per_split.test}} | {{split.files_per_split.test}} | {{split.file_share_per_split.test|pct}} | {{split.files_per_split_after_exclusion.test}} |
"""


def main():
    text = T.read_text(encoding="utf-8")
    if "آزمون شنیدن کور" in text:
        print("Already patched; nothing to do.")
        return
    ops = [
        ("low-energy table row", re.compile(r"^\| انرژی کم \(RMS < \{\{low_energy_threshold_dbfs\}\} dBFS\).*$", re.M),
         lambda m: NEW_ROW_LOW),
        ("outlier table row", re.compile(r"^\| نمونه‌های پرت \(Tukey.*$", re.M),
         lambda m: m.group(0) + "\n" + NEW_ROW_EXCL),
        ("low-energy paragraph", re.compile(r"\*\*فایل‌های کم‌انرژی\.\*\*.*?(?=\n# ۵\.)", re.S),
         lambda m: NEW_PARAGRAPH),
        ("split table", re.compile(r"\| split \| گوینده \| فایل \| سهم فایل‌ها \|\n.*?\n\| test \|[^\n]*\n", re.S),
         lambda m: NEW_SPLIT_TABLE),
        ("baseline sentence", re.compile(re.escape("نتایج روی test ({{split.baselines_on_test.n_test_files}} کلیپ):")),
         lambda m: "نتایج روی test پس از حذف‌ها ({{split.baselines_on_test.n_test_files}} کلیپ):"),
    ]
    problems = []
    for name, rx, _ in ops:
        n = len(rx.findall(text))
        if n != 1:
            problems.append(f"{name}: found {n} match(es), expected exactly 1")
    if problems:
        print("NOT patched (template was edited in those places):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    for name, rx, fn in ops:
        text = rx.sub(fn, text, count=1)
    shutil.copy(T, T.with_suffix(".md.bak"))
    T.write_text(text, encoding="utf-8")
    print("Patched. Backup of the old template: report_template.md.bak (delete it before committing).")


if __name__ == "__main__":
    main()
