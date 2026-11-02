"""Deterministic fakes and fixture builders shared by tests."""

import io
import tarfile
from pathlib import Path

import numpy as np

OBS_HEADER = (
    "observation_uuid\tobserver_id\tlatitude\tlongitude\tpositional_accuracy\t"
    "taxon_id\tquality_grade\tobserved_on\tanomaly_score"
)
PHOTO_HEADER = (
    "photo_uuid\tphoto_id\tobservation_uuid\tobserver_id\textension\t"
    "license\twidth\theight\tposition"
)
TAXA_HEADER = "taxon_id\tancestry\trank_level\trank\tname\tactive"
OBSERVER_HEADER = "observer_id\tlogin\tname"

# Observation keys: "0..." and "1..." are in a 0.25 sample, "f..." is not.
OBS_PLANT = "0a000000-0000-4000-8000-000000000001"
OBS_NO_ID = "0b000000-0000-4000-8000-000000000002"
OBS_BIRD = "0c000000-0000-4000-8000-000000000003"
OBS_PLANT_OUT = "f0000000-0000-4000-8000-000000000004"
OBS_LIFE = "1a000000-0000-4000-8000-000000000005"

OBSERVATIONS = [
    f"{OBS_PLANT}\t10\t45.5\t3.25\t20\t52775\tresearch\t2021-05-01\t0.5",
    f"{OBS_NO_ID}\t11\t-12.0\t130.0\t\t\tneeds_id\t2019-01-02\t",
    f"{OBS_BIRD}\t10\t1.0\t2.0\t5\t3\tresearch\t2020-01-01\t",
    f"{OBS_PLANT_OUT}\t12\t1.0\t2.0\t5\t52775\tresearch\t2020-01-01\t",
    f"{OBS_LIFE}\t12\t10.0\t20.0\t3000\t48460\tcasual\t2018-07-07\t",
]
PHOTOS = [
    f"p-1\t101\t{OBS_PLANT}\t10\tjpg\tCC-BY\t240\t180\t0",
    f"p-2\t102\t{OBS_PLANT}\t10\tjpeg\tCC0\t240\t180\t1",
    f"p-3\t103\t{OBS_NO_ID}\t11\tjpg\tCC-BY-NC\t180\t240\t0",
    f"p-4\t104\t{OBS_NO_ID}\t11\tjpg\tCC-BY-SA\t180\t240\t1",  # licence not kept
    f"p-5\t105\t{OBS_BIRD}\t10\tjpg\tCC-BY\t240\t180\t0",  # bird
    f"p-6\t106\t{OBS_PLANT_OUT}\t12\tjpg\tCC-BY\t240\t180\t0",  # not sampled
    f"p-7\t107\t{OBS_LIFE}\t12\tpng\tcc-by\t240\t240\t0",
]
TAXA = [
    "3\t48460/1/2\t70\tclass\tAves\ttrue",
    "48460\t\t100\tstateofmatter\tLife\ttrue",
    "52775\t48460/47126/211194/47125\t10\tspecies\tBellis perennis\ttrue",
    "47126\t48460\t70\tkingdom\tPlantae\ttrue",
]
OBSERVERS = ["10\tana\tAna B", "11\tbob\t", '12\tcyd\tCyd "C" D']

DEFAULT_MEMBERS = {
    "observations.csv": [OBS_HEADER, *OBSERVATIONS],
    "observers.csv": [OBSERVER_HEADER, *OBSERVERS],
    "photos.csv": [PHOTO_HEADER, *PHOTOS],
    "taxa.csv": [TAXA_HEADER, *TAXA],
}


def make_dump(members: dict[str, list[str]] | None = None) -> bytes:
    """Build a gzipped tar like the iNaturalist Open Data metadata dump."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, lines in (members or DEFAULT_MEMBERS).items():
            data = ("\n".join(lines) + "\n").encode()
            info = tarfile.TarInfo(f"inaturalist-open-data-20990101/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeHub:
    """A hub backed by a local directory, standing in for a remote repository."""

    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.uploads: list[str] = []
        self.folder_uploads: list[tuple[list[str], list[str]]] = []

    def list_files(self, prefix: str) -> list[str]:
        return sorted(
            str(p.relative_to(self.remote))
            for p in self.remote.rglob("*")
            if p.is_file() and str(p.relative_to(self.remote)).startswith(prefix)
        )

    def download(self, path: str, root: Path) -> Path:
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((self.remote / path).read_bytes())
        return dest

    def upload_file(self, local: Path, path: str, message: str) -> None:
        dest = self.remote / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(local.read_bytes())
        self.uploads.append(path)

    def upload_folder(self, root, message, allow, delete) -> None:
        self.folder_uploads.append((list(allow), list(delete)))
        for p in root.rglob("*"):
            if p.is_file() and any(p.relative_to(root).match(a) for a in allow):
                self.upload_file(p, str(p.relative_to(root)), message)


class FakeEncoder:
    """Deterministic encoder: an image maps to its mean colour, a text to its length."""

    model_id = "fake/fake-encoder"
    scale = 10.0
    bias = -5.0

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_images(self, images):
        self.image_calls += len(images)
        return np.array(
            [[*np.asarray(im, np.float32).mean(axis=(0, 1)) / 255, 0.1] for im in images],
            np.float32,
        )

    def encode_texts(self, texts):
        return np.array([[len(t) % 3 + 0.5, 1.0, 0.5, 0.2] for t in texts], np.float32)
