# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=2.0.0"]
# ///
"""Start, follow and fetch the Qwen3-8B QLoRA Hugging Face Job.

Needs HF_TOKEN in the environment and a positive Hugging Face credit balance.
The job timeout is the spending cap: flavor price x timeout.

    uv run scripts/launch_hf_job.py start --lr 1e-4 --smoke   # a few cents
    uv run scripts/launch_hf_job.py start --lr 1e-4           # the real run
    uv run scripts/launch_hf_job.py logs <job_id>
    uv run scripts/launch_hf_job.py fetch                     # results into artifacts/hf-job/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import fetch_job_logs, run_uv_job, sync_bucket, sync_job_volume

SCRIPT = Path(__file__).with_name("hf_job_qwen3_8b_qlora.py")
OUT = Path("artifacts/hf-job")
FLAVOR = "a10g-large"
USD_PER_HOUR = 1.50  # a10g-large list price, huggingface.co/docs/hub/jobs-pricing, 2026-09-28
TIMEOUT = {"smoke": "30m", "full": "3h"}


def start(lr: float, smoke: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    volume = sync_job_volume(OUT, "/output", read_only=False)
    args = ["--lr", str(lr), "--out", "/output", "--flavor", FLAVOR]
    args += ["--usd-per-hour", str(USD_PER_HOUR)]
    if smoke:
        args.append("--smoke")
    timeout = TIMEOUT["smoke" if smoke else "full"]
    job = run_uv_job(
        str(SCRIPT),
        script_args=args,
        flavor=FLAVOR,
        timeout=timeout,
        name="b77-qwen3-8b-qlora-smoke" if smoke else "b77-qwen3-8b-qlora",
        volumes=[volume],
    )
    print(f"job {job.id} on {FLAVOR}, timeout {timeout}: {job.url}")
    print(f"output volume {volume.source}/{volume.path}")


def fetch() -> None:
    volume = sync_job_volume(OUT, "/output", read_only=False)
    sync_bucket(f"hf://buckets/{volume.source}/{volume.path}", str(OUT))
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--lr", type=float, required=True)
    s.add_argument("--smoke", action="store_true")
    lg = sub.add_parser("logs")
    lg.add_argument("job_id")
    sub.add_parser("fetch")
    args = ap.parse_args()
    if args.cmd == "start":
        start(args.lr, args.smoke)
    elif args.cmd == "logs":
        for line in fetch_job_logs(job_id=args.job_id, tail=40):
            print(line)
    else:
        fetch()


if __name__ == "__main__":
    main()
