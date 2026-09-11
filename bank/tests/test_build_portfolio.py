import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_portfolio import add_home_link, homepage_html, publish


class PortfolioBuildTests(unittest.TestCase):
    def test_homepage_links_all_three_competitions(self):
        page = homepage_html()
        self.assertIn('href="bank/"', page)
        self.assertIn('href="playground-series-s5e4/"', page)
        self.assertIn('href="playground-series-s5e6/"', page)
        self.assertIn('href="https://yunseo326.github.io/"', page)
        self.assertIn("@media(max-width:620px)", page)

    def test_home_link_is_added_once(self):
        source = "<html><body><main>report</main></body></html>"
        result = add_home_link(source)
        self.assertEqual(result.count('aria-label="Data Study 전체 대회"'), 1)
        self.assertEqual(add_home_link(result).count('aria-label="Data Study 전체 대회"'), 1)

    def test_publish_creates_home_and_three_subpages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for slug in ["bank", "playground-series-s5e4", "playground-series-s5e6"]:
                source = root / "sources" / slug / "index.html"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"<html><body>{slug}</body></html>", encoding="utf-8")
                sources[slug] = source
            destination = root / "public"
            publish(destination, sources)
            self.assertIn("Kaggle 대회 학습 기록", (destination / "index.html").read_text(encoding="utf-8"))
            for slug in sources:
                page = (destination / slug / "index.html").read_text(encoding="utf-8")
                self.assertIn(slug, page)
                self.assertIn("전체 대회", page)

    def test_publish_reuses_existing_page_when_source_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "public"
            published = destination / "bank" / "index.html"
            published.parent.mkdir(parents=True, exist_ok=True)
            published.write_text("<html><body>existing bank</body></html>", encoding="utf-8")
            publish(destination, {"bank": root / "missing" / "bank.html"})
            self.assertIn("existing bank", published.read_text(encoding="utf-8"))
            self.assertIn("전체 대회", published.read_text(encoding="utf-8"))

    def test_published_bank_report_explains_models_without_source_check_framing(self):
        report = (ROOT.parent / "bank" / "docs" / "index.html").read_text(encoding="utf-8")
        self.assertIn("모델 개선 과정", report)
        self.assertIn("검증 설계와 신뢰도", report)
        self.assertIn("최고 모델 구성", report)
        self.assertIn("최고 모델 진단", report)
        self.assertNotIn("원문으로 보완된", report)
        self.assertNotIn("공식 변수 사전", report)
        self.assertNotIn("UCI 공식 정의에 따르면", report)


if __name__ == "__main__":
    unittest.main()
