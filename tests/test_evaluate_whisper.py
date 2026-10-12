import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_whisper import normalize_persian


class PersianNormalizationTests(unittest.TestCase):
    def test_arabic_characters_and_zero_width_non_joiner(self):
        self.assertEqual(normalize_persian("مي‌كند"), "می کند")

    def test_diacritics_punctuation_and_spacing(self):
        self.assertEqual(normalize_persian("  سَلَام،   دنیا! "), "سلام دنیا")


if __name__ == "__main__":
    unittest.main()
