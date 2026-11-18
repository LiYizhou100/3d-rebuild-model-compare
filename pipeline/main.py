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
from Analysis import analyse_epoch_loss, plot_step_loss, plot_all_fold, plot_eval_results, plot_guess_slice, plot_verify_slice
from evaluate import evaluate_model, linear_interp


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


def val_plot(model, val_ids, epoch, save_dir):
    """对验证集第一个卷推理，画三组逐像素红绿对比图，存 analysis/{epoch}/{sid}.png。"""
    sid = val_ids[0]
    hr, lr = load_volume(sid)
    sr = infer(model, lr, device=configs.DEVICE, sample_steps=configs.SAMPLE_STEPS)
    base = linear_interp(lr, model.scale)
    reg = model.infer_reg(lr, device=configs.DEVICE) if hasattr(model, "infer_reg") else None
    out = save_dir / "analysis" / f"{epoch:02d}"
    out.mkdir(parents=True, exist_ok=True)
    plot_verify_slice(hr, sr, base, out / f"{sid}.png", reg=reg)


def _load_reg_net(model):
    """加载第 1 步训练好的配准权重。Diffusion 冻结 reg_net；ResidualInterp 已在 build 里加载且需端到端微调。"""
    import torch
    if hasattr(model, "input_builder"):
        return   # ResidualInterp：build 里已加载 reg_net.pt（在 input_builder 里），保持端到端微调
    reg_path = configs.PARAMETER_DIR / "reg_net.pt"
    if reg_path.exists():
        model.reg_net.load_state_dict(torch.load(reg_path, map_location=configs.DEVICE))
        for p in model.reg_net.parameters():
            p.requires_grad_(False)
        print(f"加载配准权重并冻结 {reg_path}")
    else:
        print("⚠️ 无 reg_net.pt，配准随机初始化（请先 python reg_train.py）")


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
        _load_reg_net(model)
        train(model, all_train, [], best_epoch, configs.LR,
              configs.SCHEDULER_STEP, configs.SCHEDULER_TIMES, patience,
              log=lambda s, r: add_step(run_dir, "train", "loss.json", s, r), save_dir=run_dir)
        per_volume, mean_dl, mean_base, mean_reg = evaluate_model(
            model, test_ids, configs.DEVICE,
            sample_steps=20, guess_dir=run_dir / "guess")
        json.dump({"run": run_dir.name, "per_volume": per_volume,
                   "mean_dl": mean_dl, "mean_base": mean_base, "mean_reg": mean_reg},
                  open(run_dir / "val" / "results.json", "w"), indent=2)
        add_epoch(run_dir, "val", "loss.json", 1, {"loss": mean_dl["l1"]})
        print(f"[final] dl l1={mean_dl['l1']:.6f} psnr={mean_dl['psnr']:.3f} | base l1={mean_base['l1']:.6f}")
        plot_eval_results(run_dir)
        plot_guess_slice(run_dir, test_ids[0])
    else:
        run_dir = new_run(args.prefix, configs.MODEL, args.cfg, configs.SCALE, args.fold,
                          train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        print(f"[train] {run_dir.name}  train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
        model = build_model(configs.MODEL, y_hr, y_lr).to(configs.DEVICE)
        _load_reg_net(model)
        history = train(model, train_ids, val_ids, epochs, configs.LR,
                        configs.SCHEDULER_STEP, configs.SCHEDULER_TIMES, patience,
                        log=lambda s, r: add_step(run_dir, "train", "loss.json", s, r),
                        save_dir=run_dir, val_plot=val_plot)
        for ep, v in history.items():
            add_epoch(run_dir, "train", "loss.json", ep, {"loss": v["train_loss"]})
            add_epoch(run_dir, "val", "loss.json", ep, {"loss": v["val_loss"]})
        # ★ 训练结束自动评估验证集（加载 best.pt），生成 results.json，供 compare 直接读
        best = run_dir / "best.pt"
        if best.exists() and val_ids:
            model.load_state_dict(torch.load(best, map_location=configs.DEVICE))
            (run_dir / "guess").mkdir(parents=True, exist_ok=True)
            per_volume, mean_dl, mean_base, mean_reg = evaluate_model(
                model, val_ids, configs.DEVICE, sample_steps=configs.SAMPLE_STEPS,
                guess_dir=run_dir / "guess")
            json.dump({"run": run_dir.name, "per_volume": per_volume,
                       "mean_dl": mean_dl, "mean_base": mean_base, "mean_reg": mean_reg},
                      open(run_dir / "val" / "results.json", "w"), indent=2)
            print(f"[auto-eval] dl l1={mean_dl['l1']:.6f} base l1={mean_base['l1']:.6f}")
        plot_step_loss(run_dir)
        analyse_epoch_loss(run_dir)
        print(f"完成 {run_dir.name}")


if __name__ == "__main__":
    main()
