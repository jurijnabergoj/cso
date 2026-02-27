import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cso.data.count_dataset import CountDataset, collate_fn
from cso.models.sam3d_count_predictor import SAM3DCountPredictor
from pathlib import Path
from tqdm import tqdm
import argparse


def initialize_dataloader(train_dirs, val_dir):
    train_dataset = CountDataset(train_dirs)
    val_dataset = CountDataset(val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
    
    
def pool_slat(batch_slat):
    pooled = []
    for x in batch_slat:
        pooled.append(
            torch.cat([x.max(dim=0)[0], x.mean(dim=0)], dim=-1)
        )
    return torch.stack(pooled, dim=0)


def train(train_loader, val_loader, model, device, num_epochs, lr, use_hybrid, output_dir, load_best_model=False, exp_name="default_exp"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    checkpoint = torch.load(Path("/d/hpc/home/jn16867/cso/outputs/best_model.pth"), map_location=device)

    if load_best_model:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    best_val_mae = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        num_train_samples = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Training...")
        
        for sample in tqdm(train_loader):
            container_out = sample['container_outputs']
            object_out = sample['object_outputs']
            true_count = sample["true_count"].to(device)
            
            shape_latent_container = container_out['shape_latent'].to(device)
            shape_latent_object = object_out['shape_latent'].to(device)
                
            slat_features_container = pool_slat(container_out['slat_features']).to(device)
            slat_features_object = pool_slat(object_out['slat_features']).to(device)
            
            geom_feat = sample["geom_features"].to(device)
            geometric_estimate = sample["geom_estimate"].to(device)
            
            if use_hybrid:
                pred_count, correction = model(
                    shape_latent_container,
                    shape_latent_object,
                    slat_features_container,
                    slat_features_object,
                    geom_feat,
                    geometric_estimate
                )
            else:
                pred_count, _ = model(
                    shape_latent_container,
                    shape_latent_object,
                    slat_features_container,
                    slat_features_object,
                    geom_feat,
                    None
                )
                
            loss = 0.5 * mse_loss(pred_count, true_count) + 0.5 * l1_loss(pred_count, true_count)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_mae_sum = torch.abs(pred_count - true_count).sum().item()
            train_mae += batch_mae_sum
            num_train_samples += true_count.size(0)
            
        print(f"Train loss at epoch {epoch + 1}: {train_loss}")
        print(f"Train mae at epoch {epoch + 1}: {train_mae}")
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / num_train_samples

        # Validation
        if epoch % 10 == 0 or epoch == num_epochs - 1:            
            print("Validating")
            model.eval()

            val_loss = 0
            val_mae = 0
            num_val_samples = 0
            
            all_preds = []
            all_gts = []
            
            with torch.no_grad():
                for sample in tqdm(val_loader):
                    container_out = sample['container_outputs']
                    object_out = sample['object_outputs']
                    true_count = sample["true_count"].to(device)
                    
                    shape_latent_container = container_out['shape_latent'].to(device)
                    shape_latent_object = object_out['shape_latent'].to(device)
                        
                    slat_features_container = pool_slat(container_out['slat_features']).to(device)
                    slat_features_object = pool_slat(object_out['slat_features']).to(device)
                    
                    geom_feat = sample["geom_features"].to(device)
                    geometric_estimate = sample["geom_estimate"].to(device)
            
                    pred_count, _ = model(
                        shape_latent_container,
                        shape_latent_object,
                        slat_features_container,
                        slat_features_object,
                        geom_feat,
                        None
                    )
                    
                    all_preds.append(pred_count.cpu())
                    all_gts.append(true_count.cpu())
                    
                    loss = 0.5 * mse_loss(pred_count, true_count) + 0.5 * l1_loss(pred_count, true_count)
                    
                    val_loss += loss.item()
                    val_batch_sum = torch.abs(pred_count - true_count).sum().item()
                    val_mae += val_batch_sum
                    num_val_samples += true_count.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_mae = val_mae / num_val_samples
                
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.2f}")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.2f}")
                
                all_preds = torch.cat(all_preds)
                all_gts = torch.cat(all_gts)
                metrics = compute_counting_metrics(all_gts, all_preds)
                
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
                    
                if avg_val_mae < best_val_mae:
                    best_val_mae = avg_val_mae
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_mae': best_val_mae 
                    }, output_dir / f"{exp_name}_best_model.pth")
                    print(f"  Saved new best model with VAL MAE: {avg_val_mae:.2f}")
                
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, output_dir / f"{exp_name}_checkpoint_epoch_{epoch + 1}.pth")
 

    return avg_train_loss, avg_train_mae, avg_val_loss, avg_val_mae, best_val_mae, metrics


def compute_counting_metrics(y_true, y_pred, eps=1e-8):
    """
    Args:
        y_true: (N,) torch.Tensor or np.ndarray
        y_pred: (N,) torch.Tensor or np.ndarray
    Returns:
        dict with MAE, NAE, SRE, sMAPE, R2
    """
    
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
        "R2": r2.item()
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=False)
    args = parser.parse_args()
    
    train_dir1 = Path("/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part1")
    train_dir2 = Path("/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2")
    val_dir = Path("/d/hpc/projects/FRI/jn16867/3d-counting/scenes_val")
    output_dir = Path("/d/hpc/home/jn16867/cso/outputs")

    train_dirs = [train_dir1, train_dir2]

    exp_name = args.exp_name
    device = "cuda"
    num_epochs = 300
    lr = 1e-3

    use_hybrid = False
    model = SAM3DCountPredictor(use_hybrid=use_hybrid).to(device)
        
    train_loader, val_loader = initialize_dataloader(train_dirs, val_dir)
    
    avg_train_loss, avg_train_mae, avg_val_loss, avg_val_mae, best_val_mae, final_metrics = train(train_loader=train_loader, val_loader=val_loader, model=model, device=device, num_epochs=num_epochs, lr=lr, use_hybrid=use_hybrid, output_dir=output_dir, exp_name=exp_name)
    
    
    print("\nTraining completed")
    print(f"Average Train Loss: {avg_train_loss:.4f}, Average Train MAE: {avg_train_mae:.4f}")
    print(f"Average Val Loss: {avg_val_loss:.4f}, Average Val MAE: {avg_val_mae:.4f}")
    print(f"Best Validation MAE: {best_val_mae:.4f}")
    
    print("\nFinal validation metrics")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
                
    
    """
    # Load best current model and continue training
    checkpoint = torch.load(Path("../model_checkpoints/stacks-3d/checkpoint_epoch_100.pth"), map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint["epoch"] + 1
    best_val_mae = checkpoint["best_val_mae"]
    """
