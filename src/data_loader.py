"""Download, verify and validate the mini_speech_commands subset."""
import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class SpeechDataLoader:
    """Reproducible acquisition of the data. Fails loudly instead of returning silently."""

    def __init__(self, cfg: dict):
        d = cfg["data"]
        self.cfg = cfg
        self.base_dir = Path(cfg["_root"]) / d["base_dir"]
        self.zip_path = self.base_dir / "mini_speech_commands.zip"
        self.extracted_dir = self.base_dir / "mini_speech_commands"
        self.url = d["url"]
        self.expected_sha = d.get("zip_sha256")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _zip_ok(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None
        except zipfile.BadZipFile:
            return False

    def _safe_extract(self) -> None:
        """extractall with a path-traversal guard."""
        base = self.base_dir.resolve()
        with zipfile.ZipFile(self.zip_path) as z:
            for member in z.namelist():
                target = (base / member).resolve()
                if base not in target.parents and target != base:
                    raise RuntimeError(f"Unsafe path in zip: {member}")
            z.extractall(base)

    # ------------------------------------------------------------------- main
    def download_and_extract(self) -> Path:
        if self.validate_extraction(raise_on_error=False):
            logger.info("Data already extracted and structurally valid.")
            return self.extracted_dir

        if self.zip_path.exists() and not self._zip_ok(self.zip_path):
            logger.warning("Existing zip is corrupt; deleting it.")
            self.zip_path.unlink()

        if not self.zip_path.exists():
            logger.info("Downloading %s", self.url)
            urllib.request.urlretrieve(self.url, self.zip_path)   # raises on failure
            if not self._zip_ok(self.zip_path):
                raise RuntimeError("Downloaded zip is corrupt.")

        digest = _sha256(self.zip_path)
        logger.info("zip SHA-256 = %s", digest)
        if self.expected_sha and digest != self.expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: expected {self.expected_sha}, got {digest}")
        if not self.expected_sha:
            logger.warning("configs/config.yaml: data.zip_sha256 is null. Paste the hash above "
                           "into the config so later runs verify the archive.")

        self._safe_extract()
        self.validate_extraction(raise_on_error=True)
        return self.extracted_dir

    def validate_extraction(self, raise_on_error: bool = True) -> bool:
        """Check the folder structure against the config (classes + files per class)."""
        d = self.cfg["data"]
        problems = []
        if not self.extracted_dir.is_dir():
            problems.append(f"missing folder {self.extracted_dir}")
        else:
            found = sorted(p.name for p in self.extracted_dir.iterdir() if p.is_dir())
            if found != sorted(d["expected_classes"]):
                problems.append(f"classes {found} != expected {sorted(d['expected_classes'])}")
            for c in found:
                n = len(list((self.extracted_dir / c).glob("*.wav")))
                if n != d["expected_files_per_class"]:
                    problems.append(f"class '{c}': {n} wav files, expected {d['expected_files_per_class']}")
        if problems:
            msg = "Dataset validation problems: " + "; ".join(problems)
            if raise_on_error:
                raise RuntimeError(msg)
            logger.info(msg)
            return False
        return True
