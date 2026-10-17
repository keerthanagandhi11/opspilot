"""Generate the two sample incident PDFs shipped with OpsPilot.

Run:  python scripts/generate_samples.py
Needs: pip install reportlab  (dev dependency only)"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "backend" / "app" / "sample_data"

REPORT = [
    (
        "Incident Report INC-1042 - Summary",
        [
            "Incident INC-1042: nightly_batch_07 failed at 02:10 and the morning sales report was "
            "delayed by about 50 minutes. Severity was medium.",
            "Affected service: nightly sales reporting. Business impact: regional managers received "
            "the morning sales report at 07:50 instead of 07:00.",
            "Detected by: automated job monitor alert at 02:11.",
        ],
    ),
    (
        "Root Cause Analysis",
        [
            "Root cause: the upstream feed was delivered at 02:55, after nightly_batch_07 had already "
            "started at 02:10. The job failed with return code 12 because its input file was missing.",
            "The scheduler has no file-arrival dependency for this job, so it started on the clock "
            "even though the feed had not landed.",
            "The documents received do not explain why the upstream feed was late.",
        ],
    ),
    (
        "Resolution and Follow-up Actions",
        [
            "The rerun was approved at 03:05 by the shift lead. Follow-ups: ask the data team why the "
            "feed was late, and review the SLA alert threshold.",
            "Proposed improvement: add a file-watcher dependency so nightly_batch_07 waits for the "
            "feed instead of starting at a fixed time.",
            "Status: resolved. Incident closed after review.",
        ],
    ),
]

SHIFT_LOG = [
    (
        "Night Shift Log - Handover",
        [
            "21:00 Shift handover completed. All evening batch jobs finished with return code 0.",
            "21:30 Backup window opened for the reporting database.",
        ],
    ),
    (
        "Night Shift Log - Routine Checks",
        [
            "23:45 Backup window closed successfully. Disk usage on the batch server at 61 percent.",
            "01:30 Pre-batch checks passed. Scheduler queue healthy.",
        ],
    ),
    (
        "Night Shift Log - Failure",
        [
            "02:10 nightly_batch_07 started and aborted with return code 12. 02:55 the upstream feed "
            "file was delivered to the landing folder.",
            "02:12 Operator acknowledged the monitor alert and checked the landing folder; it was empty.",
        ],
    ),
    (
        "Night Shift Log - Recovery",
        [
            "03:05 the rerun started. 03:40 the job completed with return code 0 and the morning report was released.",
        ],
    ),
    (
        "Night Shift Log - Operator Notes",
        [
            "Operator note: no data was lost. report_gen_02 started after the rerun and finished normally.",
            "06:00 End of shift. No other incidents recorded.",
        ],
    ),
]


def build(path: Path, title: str, pages: list[tuple[str, list[str]]]) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title, author="OpsPilot sample data")
    story = []
    for i, (heading, paras) in enumerate(pages):
        story.append(Paragraph(heading, styles["Heading1"]))
        story.append(Spacer(1, 8))
        for p in paras:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 6))
        if i < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build(OUT / "incident_report_INC-1042.pdf", "Incident Report INC-1042", REPORT)
    build(OUT / "shift_log_night.pdf", "Night Shift Log", SHIFT_LOG)
    print("Wrote sample PDFs to", OUT)
