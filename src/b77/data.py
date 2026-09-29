"""Banking77 loading, pinned to one Hugging Face revision and verified by sha256.

The raw parquet files are downloaded once into `data/raw/` (not committed) and
checked against the hashes below on every load. Item ids come from row order in
the upstream files (`train-00042`, `test-00017`), so they are stable across
machines without a mapping file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATASET_ID = "legacy-datasets/banking77"
REVISION = "f54121560de48f2852f90be299010d1d6dc612ec"
LICENSE = "CC-BY-4.0"
N_CLASSES = 77


@dataclass(frozen=True)
class RawFile:
    path_in_repo: str
    sha256: str
    rows: int


RAW_FILES: dict[str, RawFile] = {
    "train": RawFile(
        "data/train-00000-of-00001.parquet",
        "3c648a31689f4ab3acbd4f4f4d120944bb521cf6ba57da77aa87c57e8979be81",
        10_003,
    ),
    "test": RawFile(
        "data/test-00000-of-00001.parquet",
        "318da70fb77a0e01bcfaecc97ef6e3645313aab98c429f0f0630c9d48e703ecc",
        3_080,
    ),
}


class DataIntegrityError(ValueError):
    """A downloaded file or a loaded split does not match what the code was built for."""


@dataclass(frozen=True)
class Split:
    """One split in upstream row order."""

    name: str
    ids: list[str]
    texts: list[str]
    labels: np.ndarray  # int64, values in [0, 77)

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, indices: np.ndarray | list[int], name: str | None = None) -> Split:
        idx = np.asarray(indices, dtype=np.int64)
        return Split(
            name=name or self.name,
            ids=[self.ids[i] for i in idx],
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
        )


def download_url(split: str) -> str:
    raw = RAW_FILES[split]
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{raw.path_in_repo}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(raw_dir: Path, split: str) -> Path:
    return raw_dir / f"{split}.parquet"


def fetch(raw_dir: Path = Path("data/raw")) -> dict[str, Path]:
    """Download any missing split, then verify every file's sha256."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, raw in RAW_FILES.items():
        path = raw_path(raw_dir, split)
        if not path.exists():
            _download(download_url(split), path)
        actual = sha256_file(path)
        if actual != raw.sha256:
            raise DataIntegrityError(
                f"{path} has sha256 {actual}, expected {raw.sha256} for {DATASET_ID}@{REVISION}. "
                "Delete the file and run `make data` again."
            )
        paths[split] = path
    return paths


def _download(url: str, dest: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=".parquet")
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, Path(tmp).open("wb") as out:
            shutil.copyfileobj(response, out)
        Path(tmp).replace(dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_split(split: str, raw_dir: Path = Path("data/raw")) -> Split:
    """Load one split, downloading it if needed, and check its shape and label names."""
    path = fetch(raw_dir)[split]
    table = pq.read_table(path)
    names = _label_names_from_metadata(table.schema.metadata)
    if names != label_names():
        raise DataIntegrityError(f"{path}: label names differ from the pinned list")
    texts = table.column("text").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    expected = RAW_FILES[split].rows
    if len(texts) != expected:
        raise DataIntegrityError(f"{path}: {len(texts)} rows, expected {expected}")
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        raise DataIntegrityError(f"{path}: label outside [0, {N_CLASSES})")
    ids = [f"{split}-{i:05d}" for i in range(len(texts))]
    return Split(name=split, ids=ids, texts=texts, labels=labels)


def _label_names_from_metadata(metadata: dict[bytes, bytes] | None) -> list[str]:
    if not metadata or b"huggingface" not in metadata:
        raise DataIntegrityError("parquet file has no Hugging Face feature metadata")
    info = json.loads(metadata[b"huggingface"])
    return list(info["info"]["features"]["label"]["names"])


def label_names() -> list[str]:
    """The 77 intent names in label-id order, exactly as the dataset spells them."""
    return list(LABEL_NAMES)


LABEL_NAMES: tuple[str, ...] = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)
