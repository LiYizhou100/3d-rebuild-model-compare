"""训练入口：configs 指定模型名，自动装配 model/config/train/infer/dataset。

用法：
    python main.py --mode develop --fold 0             # 单折训练（炼丹调参）
    python main.py --mode seg --fold 1 epoch=5         # key=value 覆盖配置（优先级最高）
    python main.py --mode test --final                 # 最终测试：多折选 epoch → 训除测试集外 → 测试评估
"""
import argparse
import json

import configs
from model import build_model
from train import train, evaluate
from infer import infer
from utils.data import load_volume
from utils.dataset.split import split_inside
from Log import new_run, add_step, add_epoch, combile
from Analysis import analyse_epoch_loss, plot_step_loss, plot_all_fold


def _fold_runs(prefix, cfg):
    """按 run 名（fold 写在名字里）找该配置的所有折。"""
    return sorted(configs.PARAMETER_DIR.glob(
        f"{prefix}-model{configs.MODEL}-cfg{cfg}-s{configs.SCALE}-fold*"))


def _pick_epoch(prefix, cfg, default):
    """多折 val/loss.json 汇总，选平均 val_loss 最小的 epoch。"""
    runs = _fold_runs(prefix, cfg)
    if not runs:
        return default
    combile(runs, "val", "loss.json")
    p = runs[0].parent / "all_fold" / "val" / "loss.json"
    if not p.exists():
        return default
    merged = json.load(open(p)).get("epoch", {})
    avg = {int(ep): v["loss"] for ep, v in merged.items() if "loss" in v}
    return min(avg, key=avg.get) if avg else default


def main():
    ap = argparse.ArgumentParser(description="OCT 帧插值：阶段/折/最终测试 + key=value 配置覆盖")
    ap.add_argument("--mode", default="develop", choices=["develop", "seg", "test"],
                    help="阶段：develop(20炼丹) / seg(30分割) / test(150测试)")
    ap.add_argument("--fold", type=int, default=0, help="交叉验证折号")
    ap.add_argument("--final", action="store_true", help="最终测试：训除测试集外数据 + 测试集评估")
    ap.add_argument("--cfg", default="base", help="配置标签，写进 run 名")
    ap.add_argument("--prefix", default="exp", help="run 名前缀")
    # ★ 剩余 key=value（如 scale=4 epoch=5 lr=0.001）作为配置覆盖，优先级最高
    args, overrides = ap.parse_known_args()
    configs.apply_cli(overrides)

    epochs = configs.EPOCH
    patience = configs.PATIENCE
    train_ids, val_ids, test_ids = split_inside(args.mode, args.fold)

    # 探测尺寸
    hr, lr = load_volume(train_ids[0])
    y_hr, y_lr = hr.shape[0], lr.shape[0]

    if args.final:
        if not test_ids:
            raise SystemExit(f"[final] 阶段 {args.mode} 没有最终测试集，请用 seg/test")
        best_epoch = _pick_epoch(args.prefix, args.cfg, epochs)
        plot_all_fold(_fold_runs(args.prefix, args.cfg))
        all_train = train_ids + val_ids
        run_dir = new_run(args.prefix, configs.MODEL, args.cfg, configs.SCALE, "final",
                          train_ids=all_train, test_ids=test_ids)
        print(f"[final] {run_dir.name}  best_epoch={best_epoch}  train={len(all_train)}  test={len(test_ids)}")
        model = build_model(configs.MODEL, y_hr, y_lr).to(configs.DEVICE)
        train(model, all_train, [], best_epoch, configs.LR,
              configs.SCHEDULER_STEP, configs.SCHEDULER_TIMES, patience,
              log=lambda s, r: add_step(run_dir, "train", "loss.json", s, r))
        test_loss = evaluate(model, test_ids)
        add_epoch(run_dir, "val", "loss.json", 1, {"loss": test_loss})
        print(f"[final] test_loss={test_loss:.6f}（{len(test_ids)} 卷）")
        # 推理演示（测试集第 1 卷前 3 帧，避免 T=1000 全卷太慢）
        _, test_lr = load_volume(test_ids[0])
        sr = infer(model, test_lr[:3], configs.DEVICE)
        print(f"[infer] {test_lr[:3].shape} -> {sr.shape}")
    else:
        run_dir = new_run(args.prefix, configs.MODEL, args.cfg, configs.SCALE, args.fold,
                          train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        print(f"[train] {run_dir.name}  train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
        model = build_model(configs.MODEL, y_hr, y_lr).to(configs.DEVICE)
        history = train(model, train_ids, val_ids, epochs, configs.LR,
                        configs.SCHEDULER_STEP, configs.SCHEDULER_TIMES, patience,
                        log=lambda s, r: add_step(run_dir, "train", "loss.json", s, r))
        for ep, v in history.items():
            add_epoch(run_dir, "train", "loss.json", ep, {"loss": v["train_loss"]})
            add_epoch(run_dir, "val", "loss.json", ep, {"loss": v["val_loss"]})
        plot_step_loss(run_dir)
        analyse_epoch_loss(run_dir)
        print(f"完成 {run_dir.name}")


if __name__ == "__main__":
    main()
