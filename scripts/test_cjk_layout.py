"""Regression tests: run with python -m unittest discover -s scripts -p 'test_*.py'."""
from io import BytesIO
from pathlib import Path
import os
import unittest

import pdfplumber
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

from cjk_layout import audit_layout, make_cjk_styles


TEXT = (
    "MedMCQA、MedQA、PubMedQA 和 HealthBench 等基准，通过测试模型回答医学选择题与长篇问答的能力，"
    "为评估 LLM 在医疗语境中的表现迈出了良好第一步。<super>3–6</super> "
    "但它们并未评估 LLM 如何在真实医疗环境与临床工作流中工作。在这样的环境里，模型必须从 EHR "
    "中搜索患者信息，从多模态数据源收集相关上下文，然后在 EHR 内执行任务。"
    "原作者用 12 个模型进行评测，Claude 3.5 Sonnet v2 的整体成功率最高，为 69.67%。"
)


def render(flowables):
    stream = BytesIO()
    SimpleDocTemplate(stream, pagesize=(595, 842), leftMargin=60, rightMargin=60,
                      topMargin=60, bottomMargin=60).build(flowables)
    stream.seek(0)
    return stream


class CJKLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [os.environ.get("CJK_TEST_FONT", ""), "C:/Windows/Fonts/msyh.ttc",
                      "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
                      "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
        font = next((p for p in candidates if p and Path(p).is_file()), None)
        if not font:
            raise RuntimeError("Set CJK_TEST_FONT to a static CJK TrueType font to run layout regressions")
        pdfmetrics.registerFont(TTFont("TestCJK", font))
        cls.styles = make_cjk_styles("TestCJK", "TestCJK")

    def test_old_word_justification_is_detected(self):
        bad = ParagraphStyle("Legacy", fontName="TestCJK", fontSize=9.5,
                             leading=16.5, alignment=TA_JUSTIFY)
        with pdfplumber.open(render([Paragraph(TEXT, bad)])) as document:
            report = audit_layout(document, len(document.pages))
        self.assertTrue(report["findings"])
        self.assertTrue(any(f["kind"] == "sparse-mixed-line" for f in report["findings"]))

    def test_fixed_prose_preserves_text_and_spacing(self):
        with pdfplumber.open(render([Paragraph(TEXT, self.styles["body"])])) as document:
            report = audit_layout(document, len(document.pages))
            text = "".join(p.extract_text() or "" for p in document.pages)
            self.assertFalse(report["findings"])
            for expected in ("MedMCQA", "HealthBench", "69.67%", "Sonnet", "PubMedQA"):
                self.assertIn(expected, text)
            self.assertNotIn("\x00", text)

    def test_explicit_local_exclusion_is_recorded(self):
        bad = ParagraphStyle("ReviewedRegion", fontName="TestCJK", fontSize=9.5,
                             leading=16.5, alignment=TA_JUSTIFY)
        with pdfplumber.open(render([Paragraph(TEXT, bad)])) as document:
            original = audit_layout(document, 1)
            region = original["findings"][0]["bbox"]
            excluded = {"page": 1, "bbox": region, "reason": "Synthetic exclusion test only"}
            reviewed = audit_layout(document, 1, [excluded])
            self.assertLess(len(reviewed["findings"]), len(original["findings"]))
            self.assertEqual(reviewed["exclusions"], [excluded])
            self.assertTrue(reviewed["findings"])  # Other bad lines remain visible.

    def test_thin_font_name_is_not_hidden_by_registration_alias(self):
        with pdfplumber.open(render([Paragraph(TEXT, self.styles["body"])])) as document:
            self.assertFalse(audit_layout(document, 1)["thin_body_fonts"])
            # Simulate the embedded font metadata found in the reported PDF.
            for char in document.pages[0].chars:
                char["fontname"] = "AAAAAA+NotoSansSC-Thin"
            self.assertEqual(audit_layout(document, 1)["thin_body_fonts"],
                             ["AAAAAA+NotoSansSC-Thin"])

    def test_every_prose_role_wraps_without_stretch(self):
        for name, style in self.styles.items():
            with self.subTest(role=name):
                with pdfplumber.open(render([Paragraph(TEXT, style)])) as document:
                    self.assertFalse(audit_layout(document, len(document.pages))["findings"])

    def test_ruled_table_gutter_is_not_a_prose_gap(self):
        rows = [[Paragraph("模型名称", self.styles["cell"]),
                 Paragraph("这是一段连续中文说明文字用来检验表格栏间空白", self.styles["cell"])],
                [Paragraph("MedAgentBench", self.styles["cell"]),
                 Paragraph("正常表格不能因为两列之间存在空白被误判为正文拉伸", self.styles["cell"])]]
        table = Table(rows, colWidths=[210, 220])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.black)]))
        with pdfplumber.open(render([table])) as document:
            self.assertTrue(document.pages[0].find_tables())
            self.assertFalse(audit_layout(document, 1)["findings"])

    def test_bad_justification_inside_table_is_not_suppressed(self):
        bad = ParagraphStyle("BadCell", fontName="TestCJK", fontSize=9.5,
                             leading=16.5, alignment=TA_JUSTIFY)
        table = Table([[Paragraph("表格内的中英混排正文", self.styles["cell"])],
                       [Paragraph(TEXT, bad)]], colWidths=[430])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.black)]))
        with pdfplumber.open(render([table])) as document:
            findings = audit_layout(document, 1)["findings"]
        self.assertTrue(any(f.get("region") == "table-cell" for f in findings))

    def test_lists_and_long_identifiers(self):
        items = [Paragraph("改进工具接口：fhir_medication_request_create 与中文说明连续排版。" * 3,
                           self.styles["bullet"], bulletText="•"),
                 Paragraph("记忆组件保留任务结果。" * 5, self.styles["bullet"], bulletText="•")]
        with pdfplumber.open(render(items)) as document:
            self.assertFalse(audit_layout(document, 1)["findings"])

    def test_reportlab_font_mapping_preserves_symbols(self):
        # No NFKD or global underscore conversion: names, code and notation are data.
        text = "中文公式说明：x<sub>0</sub>、q<sub>i</sub>(x3)、lambda<sub>i</sub>(x1)，接口 fhir_patient_search。"
        with pdfplumber.open(render([Paragraph(text, self.styles["body"])])) as document:
            extracted = document.pages[0].extract_text()
            self.assertIn("fhir_patient_search", extracted)
            self.assertNotIn("<sub>", extracted)
            self.assertNotIn("\x00", extracted)


if __name__ == "__main__":
    unittest.main()
