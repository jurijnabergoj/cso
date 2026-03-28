import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from cso.data.count_dataset import CountDataset, collate_fn
from cso.models.count_predictor import build_model
from pathlib import Path
from tqdm import tqdm
from cfg.utils import load_config
from cfg.configs import ExperimentConfig
import pprint
import json
import os
import sys


def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    assert local_rank < torch.cuda.device_count(), (
        f"LOCAL_RANK={local_rank}, "
        f"but only {torch.cuda.device_count()} GPUs visible"
    )

    return local_rank


def ddp_sum(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def initialize_dataloader(cfg: ExperimentConfig):
    feat_file = cfg.model.image_feat_file
    cont_file = cfg.model.container_image_feat_file
    obj_file = cfg.model.object_image_feat_file
    train_dataset = CountDataset(
        cfg.data.train_dirs,
        image_feat_file=feat_file,
        container_image_feat_file=cont_file,
        object_image_feat_file=obj_file,
    )
    val_dataset = CountDataset(
        cfg.data.val_dir,
        image_feat_file=feat_file,
        container_image_feat_file=cont_file,
        object_image_feat_file=obj_file,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if dist.get_rank() == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        val_loader = None
    return train_loader, val_loader


def loss_fn(pred_count, true_count, cfg: ExperimentConfig):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    if cfg.loss.log_scale:
        pred_count = torch.clamp(pred_count, min=0)
        true_count = torch.clamp(true_count, min=0)

        pred_count = torch.log1p(pred_count)
        true_count = torch.log1p(true_count)

    if cfg.loss.use_l1 and cfg.loss.use_mse:
        return 0.5 * mse_loss(pred_count, true_count) + 0.5 * l1_loss(
            pred_count, true_count
        )
    elif cfg.loss.use_mse:
        return mse_loss(pred_count, true_count)
    elif cfg.loss.use_l1:
        return l1_loss(pred_count, true_count)


def train(cfg: ExperimentConfig, train_loader, val_loader, model, device):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda")

    num_epochs = cfg.train.epochs
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            break
        model.train()
        train_loader.sampler.set_epoch(epoch)

        train_loss = torch.zeros(1, device=device)
        train_mae = torch.zeros(1, device=device)
        num_train_samples = torch.zeros(1, device=device)

        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Training...")

        iterator = train_loader
        if dist.get_rank() == 0:
            iterator = tqdm(train_loader)

        for sample in iterator:
            container_out = sample["container_outputs"]
            object_out = sample["object_outputs"]
            true_count = sample["true_count"].to(device, non_blocking=True)

            shape_latent_container = container_out["shape_latent"].to(
                device, non_blocking=True
            )
            shape_latent_object = object_out["shape_latent"].to(
                device, non_blocking=True
            )

            slat_features_container = container_out["slat_features"].to(
                device, non_blocking=True
            )
            slat_features_object = object_out["slat_features"].to(
                device, non_blocking=True
            )

            geom_feat = sample["geom_features"].to(device, non_blocking=True)

            # Geometric estimate
            cont_scale = (
                container_out["scale"].to(device, non_blocking=True).mean(dim=-1)
            )  # (B,)
            obj_scale = (
                object_out["scale"].to(device, non_blocking=True).mean(dim=-1)
            )  # (B,)
            geometric_estimate = 0.64 * (cont_scale / obj_scale.clamp(min=1e-4)) ** 3

            image_feats = sample.get("image_feats")
            if image_feats is not None:
                image_feats = image_feats.to(device, non_blocking=True)

            container_image_feats = sample.get("container_image_feats")
            if container_image_feats is not None:
                container_image_feats = container_image_feats.to(
                    device, non_blocking=True
                )

            object_image_feats = sample.get("object_image_feats")
            if object_image_feats is not None:
                object_image_feats = object_image_feats.to(device, non_blocking=True)

            with torch.autocast("cuda"):
                result = model(
                    shape_latent_container,
                    shape_latent_object,
                    slat_features_container,
                    slat_features_object,
                    geom_feat,
                    geometric_estimate,
                    image_feats=image_feats,
                    container_image_feats=container_image_feats,
                    object_image_feats=object_image_feats,
                )
                pred_count = result[0] if isinstance(result, tuple) else result
                loss = loss_fn(pred_count.float(), true_count, cfg)

            # Handle NaN/Inf losses safely when training with DDP. In DDP every process must run backward() so that gradient all-reduce
            # synchronization can complete. If one process skips backward(), the others
            # will hang waiting for the all-reduce.
            #
            # Therefore we do not skip backward() even when the loss is NaN/Inf.
            # Instead we:
            #   1. Detect the invalid loss
            #   2. Still run backward() so all processes participate in gradient sync
            #   3. Let AMP GradScaler detect the invalid gradients and skip optimizer.step()
            #   4. Skip logging/metrics accumulation for this batch

            loss_valid = torch.isfinite(loss)
            if not loss_valid and dist.get_rank() == 0:
                print(
                    f"WARNING: Non-finite loss ({loss.item():.4f}), skipping accumulation."
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if loss_valid:
                train_loss += loss.detach()
                train_mae += torch.abs(pred_count.float() - true_count).sum()
                num_train_samples += true_count.numel()

        scheduler.step()

        train_loss = ddp_sum(train_loss)
        train_mae = ddp_sum(train_mae)
        num_train_samples = ddp_sum(num_train_samples)

        avg_train_loss = (train_loss / max(len(train_loader), 1)).item()
        avg_train_mae = (train_mae / num_train_samples.clamp(min=1)).item()

        if dist.get_rank() == 0:
            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Train MAE: {avg_train_mae:.4f}")

        # Validation
        if dist.get_rank() == 0:
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print("Validating")
                model.eval()

                val_loss = 0
                val_mae = 0
                num_val_samples = 0

                all_preds = []
                all_gts = []

                with torch.no_grad():
                    val_iterator = tqdm(val_loader)

                    for sample in val_iterator:
                        container_out = sample["container_outputs"]
                        object_out = sample["object_outputs"]
                        true_count = sample["true_count"].to(device, non_blocking=True)

                        shape_latent_container = container_out["shape_latent"].to(
                            device, non_blocking=True
                        )
                        shape_latent_object = object_out["shape_latent"].to(
                            device, non_blocking=True
                        )

                        slat_features_container = container_out["slat_features"].to(
                            device, non_blocking=True
                        )
                        slat_features_object = object_out["slat_features"].to(
                            device, non_blocking=True
                        )

                        geom_feat = sample["geom_features"].to(
                            device, non_blocking=True
                        )

                        cont_scale = (
                            container_out["scale"]
                            .to(device, non_blocking=True)
                            .mean(dim=-1)
                        )
                        obj_scale = (
                            object_out["scale"]
                            .to(device, non_blocking=True)
                            .mean(dim=-1)
                        )
                        geometric_estimate = (
                            0.64 * (cont_scale / obj_scale.clamp(min=1e-4)) ** 3
                        )

                        image_feats = sample.get("image_feats")
                        if image_feats is not None:
                            image_feats = image_feats.to(device, non_blocking=True)

                        container_image_feats = sample.get("container_image_feats")
                        if container_image_feats is not None:
                            container_image_feats = container_image_feats.to(
                                device, non_blocking=True
                            )

                        object_image_feats = sample.get("object_image_feats")
                        if object_image_feats is not None:
                            object_image_feats = object_image_feats.to(
                                device, non_blocking=True
                            )

                        with torch.autocast("cuda"):
                            result = model.module(
                                shape_latent_container,
                                shape_latent_object,
                                slat_features_container,
                                slat_features_object,
                                geom_feat,
                                geometric_estimate,
                                image_feats=image_feats,
                                container_image_feats=container_image_feats,
                                object_image_feats=object_image_feats,
                            )
                            pred_count = (
                                result[0] if isinstance(result, tuple) else result
                            )
                            loss = loss_fn(pred_count, true_count, cfg)

                        all_preds.append(pred_count.float().cpu())
                        all_gts.append(true_count.cpu())

                        val_loss += loss.item()
                        val_batch_sum = (
                            torch.abs(pred_count.float() - true_count).sum().item()
                        )
                        val_mae += val_batch_sum
                        num_val_samples += true_count.size(0)

                    avg_val_loss = val_loss / len(val_loader)
                    avg_val_mae = val_mae / num_val_samples

                    print(f"Epoch {epoch+1}/{num_epochs}")
                    print(
                        f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.2f}"
                    )
                    print(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.2f}")

                    all_preds = torch.cat(all_preds)
                    all_gts = torch.cat(all_gts)
                    metrics = compute_counting_metrics(all_gts, all_preds)

                    for k, v in metrics.items():
                        print(f"{k}: {v:.4f}")

                    if avg_val_mae < best_val_mae:
                        best_val_mae = avg_val_mae
                        epochs_without_improvement = 0

                        if cfg.train.save_checkpoints:
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "best_val_mae": best_val_mae,
                                },
                                cfg.train.output_dir
                                / Path(f"{cfg.exp_name}_best_model.pth"),
                            )
                        print(f"  Saved new best model with VAL MAE: {avg_val_mae:.2f}")
                    else:
                        epochs_without_improvement += 1
                        if (
                            cfg.train.patience > 0
                            and epochs_without_improvement >= cfg.train.patience
                        ):
                            print(
                                f"  Early stopping: no improvement for {epochs_without_improvement} val checks."
                            )
                            early_stop = True

                    if cfg.train.save_checkpoints and (epoch + 1) % 10 == 0:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                            },
                            cfg.train.output_dir
                            / Path(f"{cfg.exp_name}_checkpoint_epoch_{epoch + 1}.pth"),
                        )

    if dist.get_rank() == 0:
        return {
            "avg_train_loss": avg_train_loss,
            "avg_train_mae": avg_train_mae,
            "avg_val_loss": avg_val_loss,
            "avg_val_mae": avg_val_mae,
            "best_val_mae": best_val_mae,
            "metrics": metrics,
        }
    return {}


def compute_counting_metrics(y_true, y_pred, eps=1e-8):
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    y_true = y_true.float()
    y_pred = y_pred.float()

    abs_err = torch.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2

    mae = abs_err.mean()
    nae = abs_err.sum() / (y_true.sum() + eps)
    sre = sq_err.sum() / (y_true.pow(2).sum() + eps)
    smape = 100 * (abs_err / ((y_true + y_pred).clamp(min=eps) / 2.0)).mean()
    r2 = 1.0 - (sq_err.sum()) / ((((y_true - (y_true.mean())) ** 2).sum()) + eps)

    return {
        "MAE": mae.item(),
        "NAE": nae.item(),
        "SRE": sre.item(),
        "sMAPE": smape.item(),
        "R2": r2.item(),
    }


def display_results(dict):
    if dist.get_rank() == 0:
        print("\nTraining completed")
        pprint.pprint(dict)

        print("\nFinal validation metrics")
        pprint.pprint(dict["metrics"])


def main(config_path):
    torch.backends.cudnn.benchmark = True
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    cfg = load_config(config_path)

    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    train_loader, val_loader = initialize_dataloader(cfg)
    results_dict = train(cfg, train_loader, val_loader, model, device)

    display_results(results_dict)

    if dist.get_rank() == 0 and results_dict:
        results_path = cfg.train.output_dir / f"{cfg.exp_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    k: (
                        float(v)
                        if not isinstance(v, dict)
                        else {ik: float(iv) for ik, iv in v.items()}
                    )
                    for k, v in results_dict.items()
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("/d/hpc/home/jn16867/cso/cfg/baseline.yaml")
    )
    main(config_path)
