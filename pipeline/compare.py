"""跨 run 对比：读多个 run 的 results.json，画 DL vs baseline 柱状图。

用法：
    python compare.py --runs run1 run2 run3                 # 指定 run 目录名（默认扫 parameter/）
    python compare.py --previous                            # 从 previous-run/{all,baseline,warp} 按相同 scale 分组比较
    python compare.py --previous --mode seg --fold 1        # previous 模式可指定阶段/折

--previous 会对缺失 results.json 的 run 自动评估（加载 best.pt 生成 results.json）。
"""
import argparse
import json
import re

import torch

import configs
from Analysis import compare_runs
from model import build_model
from evaluate import evaluate_model
from utils.data import load_volume
from utils.dataset.split import split_inside


def extract_scale(run_dir):
    """从 config.json 提取 scale，回退到 run 名 -s{N}- 解析。"""
    cfg = run_dir / "config.json"
    if cfg.exists():
        s = json.load(open(cfg))["config"].get("scale")
        if s:
            return s
    m = re.search(r"-s(\d+)-", run_dir.name)
    return int(m.group(1)) if m else None


def auto_evaluate(run_dir, mode="develop", fold=0):
    """对 run 自动评估（缺失 results.json 时），加载 best.pt 生成 results.json。返回 results dict。"""
    p = run_dir / "val" / "results.json"
    if p.exists():
        return json.load(open(p))

    cfg = json.load(open(run_dir / "config.json"))["config"]
    # 恢复训练时的 scale + 输入开关（否则口径错乱）
    restore = {"scale": cfg["scale"]} if "scale" in cfg else {}
    restore.update(cfg.get("input_switches", {}))
    if restore:
        configs.apply_cli([f"{k}={v}" for k, v in restore.items()])

    train_ids, val_ids, test_ids = split_inside(mode, fold)
    ids = test_ids or val_ids
    if not ids:
        raise RuntimeError(f"{run_dir.name}: 阶段 {mode} fold {fold} 无评估集")

    hr, lr = load_volume(ids[0])
    model = build_model(cfg["model"], hr.shape[0], lr.shape[0]).to(configs.DEVICE)
    best = run_dir / "best.pt"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=configs.DEVICE))
    else:
        print(f"⚠️ {run_dir.name} 无 best.pt，用随机权重")

    (run_dir / "guess").mkdir(parents=True, exist_ok=True)
    per_volume, mean_dl, mean_base, mean_reg = evaluate_model(
        model, ids, configs.DEVICE, sample_steps=configs.SAMPLE_STEPS,
        guess_dir=run_dir / "guess")
    results = {"run": run_dir.name, "per_volume": per_volume,
               "mean_dl": mean_dl, "mean_base": mean_base, "mean_reg": mean_reg}
    json.dump(results, open(p, "w"), indent=2)
    print(f"[auto-eval] {run_dir.name}: dl l1={mean_dl['l1']:.6f} base l1={mean_base['l1']:.6f}")
    return results


def compare_previous(previous_dir, out_dir, mode="develop", fold=0):
    """从 previous-run/{all,baseline,warp} 按相同 scale 分组比较。"""
    groups = {}
    for cat in ["all", "baseline", "warp"]:
        cat_dir = previous_dir / cat
        if not cat_dir.exists():
            continue
        for d in sorted(cat_dir.iterdir()):
            if not d.is_dir() or not (d / "config.json").exists():
                continue
            scale = extract_scale(d)
            groups.setdefault(scale, []).append(d)

    if not groups:
        return {}

    for scale in sorted(groups, key=lambda s: (s is None, s or 0)):
        run_dirs = groups[scale]
        for d in run_dirs:
            auto_evaluate(d, mode, fold)
        label = f"s{scale}" if scale is not None else "unknown"
        out = out_dir / label
        compare_runs(run_dirs, out)
        print(f"[scale {label}] {len(run_dirs)} 个 run -> {out}")
    return groups


def main():
    ap = argparse.ArgumentParser(description="跨 run 对比（DL vs baseline 柱状图）")
    ap.add_argument("--runs", nargs="+", default=None, help="run 目录名（不填则扫所有含 results.json 的）")
    ap.add_argument("--previous", action="store_true",
                    help="从 previous-run/{all,baseline,warp} 按相同 scale 分组比较")
    ap.add_argument("--mode", default="develop", choices=["develop", "seg", "test"])
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    if args.previous:
        prev = configs.ROOT_DIR / "previous-run"
        groups = compare_previous(prev, configs.PARAMETER_DIR / "compare", args.mode, args.fold)
        if not groups:
            print("previous-run 里没有可比较的 run（缺 config.json）")
        return

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
