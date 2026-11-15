"""跨 run 对比：读多个 run 的 results.json，画 DL vs baseline 柱状图。

用法：
    python compare.py                          # 扫所有含 results.json 的 run
    python compare.py --runs run1 run2 run3    # 指定 run 目录名
"""
import argparse

import configs
from Analysis import compare_runs


def main():
    ap = argparse.ArgumentParser(description="跨 run 对比（DL vs baseline 柱状图）")
    ap.add_argument("--runs", nargs="+", default=None, help="run 目录名（不填则扫所有含 results.json 的）")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [configs.PARAMETER_DIR / r for r in args.runs]
    else:
        run_dirs = sorted(d for d in configs.PARAMETER_DIR.glob("*")
                          if d.is_dir() and (d / "val" / "results.json").exists())
    if not run_dirs:
        print("没有含 results.json 的 run，请先跑 evaluate 或 --final")
        return

    out = configs.PARAMETER_DIR / "compare"
    compare_runs(run_dirs, out)
    print(f"已存 {out}")


if __name__ == "__main__":
    main()
