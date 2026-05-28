import argparse
import re
from pathlib import Path


ITER_LINE_RE = re.compile(
    r"iter:\s*(\d+).*?l_pix:\s*([0-9.eE+-]+)(?:\s+l_g_mae:\s*([0-9.eE+-]+))?"
)
VAL_HEADER_RE = re.compile(r"Validation\s+(.+)$")
METRIC_RE = re.compile(r"#\s*(psnr|ssim):\s*([0-9.]+)", re.IGNORECASE)


def parse_log(log_path: Path):
    train_at_iter = {}
    val_at_iter = {}
    current_iter = None
    current_dataset = None

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        iter_match = ITER_LINE_RE.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            l_pix = float(iter_match.group(2))
            l_g_mae = float(iter_match.group(3)) if iter_match.group(3) is not None else None
            train_at_iter[current_iter] = {"l_pix": l_pix, "l_g_mae": l_g_mae}
            current_dataset = None
            continue

        header_match = VAL_HEADER_RE.search(line)
        if header_match and current_iter is not None:
            current_dataset = header_match.group(1).strip()
            val_at_iter.setdefault(current_iter, {})
            val_at_iter[current_iter].setdefault(current_dataset, {})
            continue

        if current_dataset is not None and current_iter is not None:
            metric_match = METRIC_RE.search(line)
            if metric_match:
                metric_name = metric_match.group(1).lower()
                metric_value = float(metric_match.group(2))
                val_at_iter[current_iter][current_dataset][metric_name] = metric_value

    return train_at_iter, val_at_iter


def fmt(v):
    return "-" if v is None else f"{v:.4f}"


def build_markdown(target_iters, train_at_iter, val_at_iter):
    header = [
        "| iter | l_pix | l_g_mae | canon_psnr | canon_ssim | nikon_psnr | nikon_ssim |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []

    for it in target_iters:
        train = train_at_iter.get(it, {})
        vals = val_at_iter.get(it, {})

        canon = vals.get("RealSR_Canon", {})
        nikon = vals.get("RealSR_Nikon", {})

        rows.append(
            "| {it} | {l_pix} | {l_g_mae} | {c_psnr} | {c_ssim} | {n_psnr} | {n_ssim} |".format(
                it=it,
                l_pix=fmt(train.get("l_pix")),
                l_g_mae=fmt(train.get("l_g_mae")),
                c_psnr=fmt(canon.get("psnr")),
                c_ssim=fmt(canon.get("ssim")),
                n_psnr=fmt(nikon.get("psnr")),
                n_ssim=fmt(nikon.get("ssim")),
            )
        )

    return "\n".join(header + rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize DRCT training log for reviewer milestones.")
    parser.add_argument("--log", required=True, help="Path to train log file.")
    parser.add_argument(
        "--iters",
        default="50000,100000,200000",
        help="Comma-separated milestone iterations (default: 50000,100000,200000).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output markdown file path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    target_iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]

    train_at_iter, val_at_iter = parse_log(log_path)
    md = build_markdown(target_iters, train_at_iter, val_at_iter)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(md + "\n", encoding="utf-8")
        print(f"Saved summary to: {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
