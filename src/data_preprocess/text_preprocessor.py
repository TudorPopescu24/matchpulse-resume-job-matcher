import re
import unicodedata
import contractions
from bs4 import BeautifulSoup

class TextPreprocessor:
    def clean_html(self, text):
        """Remove HTML tags and decode HTML entities."""
        return BeautifulSoup(text, "html.parser").get_text()

    def expand_contractions(self, text):
        """Expand common English contractions (e.g., don't → do not)."""
        return contractions.fix(text)

    def normalize_unicode(self, text):
        """Normalize Unicode characters (e.g., accented → base forms)."""
        return unicodedata.normalize("NFKC", text)

    def preprocess(self, text):
        text = str(text)
        text = self.clean_html(text)
        text = self.expand_contractions(text)
        text = self.normalize_unicode(text)

        text = re.sub(r"[ \t]+", " ", text)  # normalize spaces
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multiple empty lines to max 2

        text = text.lower()

        text = re.sub(r"[^\x00-\x7F]+", "", text)

        return text.strip()