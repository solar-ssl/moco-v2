"""
SSL for Solar Panel Segmentation - BYOL-Style (Small Dataset Optimized)
=========================================================================
BYOL (Bootstrap Your Own Latent) doesn't need negative samples, making it 
much more stable for small datasets like yours (2308 images).

Key differences from MoCo:
1. No queue/negatives - uses predictor to prevent collapse
2. Symmetric loss - both views predict each other
3. More stable for small datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageOps
import random
import os
import glob
import pandas as pd
from tqdm import tqdm
import math
import copy

# ==========================================
# 1. CONFIGURATION
# ==========================================
config = {
    "Dataset Path": "/kaggle/working/moco_data",
    
    # BYOL Hyperparameters
    "Projection Dim": 256,
    "Hidden Dim": 4096,
    "Momentum": 0.996,            # EMA momentum (starts here, increases to 1.0)
    
    # Training Setup
    "Training Epochs": 200,           
    "Warmup Epochs": 10,
    "Batch Size": 128,            # BYOL works with larger batches
    "Base Learning Rate": 0.2,    # BYOL uses higher LR with LARS
    "Weight Decay": 1.5e-6,       # BYOL uses lower weight decay
    
    # Accumulation for effective batch 512
    "Accumulation Steps": 4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running BYOL-Style SSL on {device}")

# ==========================================
# 2. AUGMENTATION (Asymmetric - Critical for BYOL)
# ==========================================
class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class Solarize:
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, x):
        return ImageOps.solarize(x, self.threshold)

# View 1: Stronger augmentation
transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # More aggressive crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # Satellite-specific
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),  # 90° rotation
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=1.0),  # Always blur for view 1
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# View 2: Slightly different augmentation (asymmetric is key!)
transform_2 = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=0.1),  # Less blur for view 2
    transforms.RandomApply([Solarize()], p=0.2),      # Solarize only view 2
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. DATASET
# ==========================================
class SolarPanelDataset(Dataset):
    def __init__(self, root_dir, transform1, transform2):
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_paths = glob.glob(os.path.join(root_dir, "*"))
        self.image_paths = [p for p in self.image_paths 
                           if "_label" not in p and not p.endswith('.txt')]
        print(f"✅ Loaded {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            v1 = self.transform1(image)
            v2 = self.transform2(image)
            return v1, v2
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self) - 1))

# ==========================================
# 4. MODEL COMPONENTS
# ==========================================
class MLP(nn.Module):
    """Projector/Predictor MLP"""
    def __init__(self, in_dim, hidden_dim, out_dim, use_bn=True):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent
    - Online network: encoder + projector + predictor
    - Target network: encoder + projector (EMA of online)
    - No negatives needed!
    """
    def __init__(self, base_encoder, projection_dim=256, hidden_dim=4096):
        super().__init__()
        
        # Online encoder
        self.online_encoder = base_encoder(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        encoder_dim = self.online_encoder.fc.in_features
        self.online_encoder.fc = nn.Identity()  # Remove classifier
        
        # Online projector
        self.online_projector = MLP(encoder_dim, hidden_dim, projection_dim)
        
        # Online predictor (key component that prevents collapse!)
        self.predictor = MLP(projection_dim, hidden_dim // 2, projection_dim)
        
        # Target network (EMA of online, no gradients)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update_target(self, momentum):
        """EMA update of target network"""
        for online_params, target_params in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data
    
    def forward(self, x1, x2):
        # Online network forward
        z1_online = self.online_projector(self.online_encoder(x1))
        z2_online = self.online_projector(self.online_encoder(x2))
        
        # Predictions
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)
        
        # Target network forward (no gradients)
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))
            
        return p1, p2, z1_target.detach(), z2_target.detach(), z1_online

# ==========================================
# 5. LOSS FUNCTION
# ==========================================
def byol_loss(p, z):
    """
    Negative cosine similarity loss
    p: predictions from online network
    z: projections from target network
    """
    p = nn.functional.normalize(p, dim=-1)
    z = nn.functional.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

# ==========================================
# 6. LARS OPTIMIZER (Important for BYOL)
# ==========================================
class LARS(optim.Optimizer):
    """
    LARS optimizer for large batch training
    Adapted from: https://github.com/facebookresearch/vissl
    """
    def __init__(self, params, lr, momentum=0.9, weight_decay=0, eta=0.001, exclude_bias_and_bn=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)
        self.exclude_bias_and_bn = exclude_bias_and_bn

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # LARS scaling
                param_norm = p.norm()
                grad_norm = grad.norm()
                
                if param_norm > 0 and grad_norm > 0:
                    trust_ratio = group['eta'] * param_norm / grad_norm
                else:
                    trust_ratio = 1.0
                    
                # Update momentum
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p)
                    
                buf = self.state[p]['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad, alpha=trust_ratio)
                
                p.add_(buf, alpha=-group['lr'])
                
        return loss

# ==========================================
# 7. LEARNING RATE SCHEDULE
# ==========================================
def cosine_schedule(epoch, warmup_epochs, total_epochs, base_value, final_value=0):
    """Cosine schedule with warmup"""
    if epoch < warmup_epochs:
        return base_value * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))

def momentum_schedule(epoch, total_epochs, base_momentum=0.996, final_momentum=1.0):
    """Momentum increases from base to 1.0"""
    return final_momentum - (final_momentum - base_momentum) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2

# ==========================================
# 8. TRAINING LOOP
# ==========================================
def train():
    dataset = SolarPanelDataset(config['Dataset Path'], transform_1, transform_2)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config['Batch Size'], 
        shuffle=True, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    model = BYOL(
        models.resnet50, 
        projection_dim=config['Projection Dim'],
        hidden_dim=config['Hidden Dim']
    ).to(device)
    
    # Use LARS optimizer
    optimizer = LARS(
        model.parameters(),
        lr=config['Base Learning Rate'],
        momentum=0.9,
        weight_decay=config['Weight Decay']
    )
    
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = config['Accumulation Steps']
    
    print("🔥 STARTING BYOL TRAINING...")
    print(f"   Effective Batch Size: {config['Batch Size'] * accumulation_steps}")
    print(f"   Base LR: {config['Base Learning Rate']}")
    print(f"   Warmup Epochs: {config['Warmup Epochs']}")
    
    history = {'epoch': [], 'loss': [], 'f_std': [], 'lr': [], 'momentum': []}
    best_loss = float('inf')

    for epoch in range(config['Training Epochs']):
        model.train()
        running_loss = 0.0
        running_std = 0.0
        num_batches = 0
        
        # Update learning rate
        current_lr = cosine_schedule(
            epoch, 
            config['Warmup Epochs'], 
            config['Training Epochs'], 
            config['Base Learning Rate']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        # Update momentum
        current_momentum = momentum_schedule(
            epoch, 
            config['Training Epochs'], 
            config['Momentum']
        )
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for batch_idx, (v1, v2) in enumerate(pbar):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                p1, p2, z1_target, z2_target, z_online = model(v1, v2)
                
                # Symmetric loss: both views predict each other
                loss = byol_loss(p1, z2_target) + byol_loss(p2, z1_target)
                loss = loss / accumulation_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update target network
                model.update_target(current_momentum)
            
            running_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            # Monitor feature diversity
            with torch.no_grad():
                z_norm = nn.functional.normalize(z_online, dim=-1)
                f_std = z_norm.std(dim=0).mean().item()
                running_std += f_std
                
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'std': f'{f_std:.4f}',
                'lr': f'{current_lr:.5f}',
                'm': f'{current_momentum:.4f}'
            })
        
        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model.update_target(current_momentum)
        
        # Epoch stats
        epoch_loss = running_loss / num_batches
        epoch_std = running_std / num_batches
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['f_std'].append(epoch_std)
        history['lr'].append(current_lr)
        history['momentum'].append(current_momentum)
        
        # Status indicators
        std_status = "✅" if epoch_std > 0.1 else "⚠️" if epoch_std > 0.05 else "❌"
        print(f"📊 Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Std: {epoch_std:.4f} {std_status} | LR: {current_lr:.5f} | Mom: {current_momentum:.4f}")
        
        # Save best
        if epoch_loss < best_loss and epoch >= config['Warmup Epochs']:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': model.online_encoder.state_dict(),
                'projector_state_dict': model.online_projector.state_dict(),
                'loss': epoch_loss,
                'std': epoch_std
            }, "byol_best.pth")
            print(f"💾 Saved best model!")
        
        # Checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': model.online_encoder.state_dict(),
                'full_model_state_dict': model.state_dict(),
                'loss': epoch_loss,
            }, f"byol_epoch_{epoch+1}.pth")
    
    # Final save
    torch.save(model.online_encoder.state_dict(), "byol_encoder_final.pth")
    pd.DataFrame(history).to_csv("byol_training_log.csv", index=False)
    print("🏆 TRAINING COMPLETE!")
    
    return model, history

# ==========================================
# 9. EXTRACT BACKBONE FOR SEGMENTATION
# ==========================================
def load_pretrained_backbone(checkpoint_path):
    """
    Load the pretrained encoder for downstream segmentation.
    Use this with DeepLabV3 or U-Net for fine-tuning.
    """
    # Create ResNet50 backbone
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'encoder_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        backbone.load_state_dict(checkpoint)
    
    print(f"✅ Loaded pretrained backbone from {checkpoint_path}")
    return backbone

# ==========================================
# 10. DIAGNOSTIC
# ==========================================
def diagnose_training(history):
    print("\n" + "="*50)
    print("🔍 BYOL TRAINING DIAGNOSIS")
    print("="*50)
    
    # Feature diversity
    final_std = history['f_std'][-1]
    if final_std < 0.05:
        print(f"❌ COLLAPSE: Feature Std = {final_std:.4f}")
    elif final_std < 0.1:
        print(f"⚠️  LOW DIVERSITY: Feature Std = {final_std:.4f}")
    else:
        print(f"✅ Good Diversity: Feature Std = {final_std:.4f}")
    
    # Loss trend
    early_loss = sum(history['loss'][:5]) / 5
    late_loss = sum(history['loss'][-5:]) / 5
    
    if late_loss >= early_loss * 0.95:
        print(f"⚠️  LOSS PLATEAU: {early_loss:.4f} -> {late_loss:.4f}")
    else:
        reduction = (early_loss - late_loss) / early_loss * 100
        print(f"✅ Loss Reduced: {early_loss:.4f} -> {late_loss:.4f} ({reduction:.1f}%)")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    model, history = train()
    diagnose_training(history)
