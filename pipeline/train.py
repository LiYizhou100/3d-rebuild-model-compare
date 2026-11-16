"""统一训练接口：多 epoch + 早停。所有模型共用，不 import 具体模型。"""
import torch

from utils.data import load_volume


def train(model, train_ids, val_ids, epochs, lr, step_size, gamma, patience, log=None, save_dir=None, val_plot=None):
    """通用训练循环 + 早停。

    model 需实现：make_batches(hr,lr) / train_step(batch,opt,sch) / eval_loss(hr,lr)。
    patience：连续多少次「既没创新低、也没比上一 epoch 下降」就早停（超参）。
              ★ 下降但未创新低 → 不累计（仍在进步，不停）。
    save_dir：给定时，best 时存 best.pt，结束时存 last.pt。
    val_plot：可选回调 val_plot(model, val_ids, epoch, save_dir)，每 epoch 验证后画图。
    返回 history = {epoch: {"train_loss":.., "val_loss":..}}。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma)
    best_val = float("inf")
    prev_val = float("inf")
    no_improve = 0
    history = {}
    step = 0
    for ep in range(epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for sid in train_ids:
            hr, lr = load_volume(sid)
            for batch in model.make_batches(hr, lr):
                record = model.train_step(batch, opt, sch)
                step += 1
                ep_loss += record["loss"]
                n += 1
                if log:
                    log(step, record)
        train_loss = ep_loss / n if n else float("nan")
        val_loss = evaluate(model, val_ids) if val_ids else float("nan")
        history[ep + 1] = {"train_loss": train_loss, "val_loss": val_loss}
        print(f"epoch {ep + 1}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")
        if val_ids and val_plot and save_dir:
            val_plot(model, val_ids, ep + 1, save_dir)
        if val_ids:
            if val_loss < best_val:
                # 创新低 → 存 best，重置
                best_val = val_loss
                no_improve = 0
                if save_dir:
                    torch.save(model.state_dict(), save_dir / "best.pt")
            elif val_loss < prev_val:
                # 没创新低，但比上一 epoch 下降 → 仍在进步，重置（不停）
                no_improve = 0
            else:
                # 既没创新低也没下降（停滞/上升）→ 累计
                no_improve += 1
                if no_improve >= patience:
                    print(f"早停：val loss 连续 {patience} 个 epoch 未下降（best={best_val:.6f}）")
                    break
            prev_val = val_loss
    if save_dir:
        torch.save(model.state_dict(), save_dir / "last.pt")
    return history


def evaluate(model, ids):
    """在 ids 集上平均 eval_loss（用于验证/测试）。"""
    if not ids:
        return float("nan")
    return sum(model.eval_loss(*load_volume(sid)) for sid in ids) / len(ids)
