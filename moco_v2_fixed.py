import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import random
import os
import glob
import pandas as pd
from tqdm import tqdm
import math

# ==========================================
# 1. CONFIGURATION (Properly Tuned)
# ==========================================
config = {
    "Dataset Path": "/kaggle/working/moco_data",
    
    # MoCo Hyperparameters
    "Queue Size": 4096,           # Larger queue = more negatives
    "Temperature": 0.2,           # FIXED: 0.07 is too aggressive, 0.2 is safer
    "Momentum": 0.999,            # Standard
    
    # Training Setup
    "Training Epochs": 200,           
    "Warmup Epochs": 10,          # WILL ACTUALLY BE USED NOW
    "Batch Size": 64,                 
    "Base Learning Rate": 0.03,   # Will be scaled
    "Weight Decay": 1e-4,
    
    # Gradient Accumulation (effective batch = 64 * 4 = 256)
    "Accumulation Steps": 4,
}

# FIXED: Proper LR scaling
config["Learning Rate"] = config["Base Learning Rate"] * (config["Batch Size"] * config["Accumulation Steps"]) / 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Fixed MoCo v2 on {device}")
print(f"📊 Effective Batch Size: {config['Batch Size'] * config['Accumulation Steps']}")
print(f"📊 Scaled Learning Rate: {config['Learning Rate']}")

# ==========================================
# 2. AUGMENTATION (Satellite Specific - Enhanced)
# ==========================================
class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    """Solarization augmentation - important for SSL"""
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, x):
        from PIL import ImageOps
        return ImageOps.solarize(x, self.threshold)

# MoCo v2+ style augmentation
moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomApply([Solarize()], p=0.2),  # Added solarization
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Satellite specific
    transforms.RandomRotation(90),    # Added: 90° rotations for satellite
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. DATASET
# ==========================================
class SolarPanelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*"))
        # Filter labels if any leaked
        self.image_paths = [p for p in self.image_paths if "_label" not in p and not p.endswith('.txt')]
        print(f"✅ Loaded {len(self.image_paths)} clean images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                q = self.transform(image)
                k = self.transform(image)
            return q, k
        except Exception as e:
            # Fallback for corrupt images
            return self.__getitem__(random.randint(0, len(self.image_paths)-1))

# ==========================================
# 4. MODEL (Fixed Architecture)
# ==========================================
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=4096, m=0.999, T=0.2):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # Encoders
        self.encoder_q = base_encoder(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder_k = base_encoder(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Get feature dimension
        dim_mlp = self.encoder_q.fc.in_features
        
        # FIXED: 3-layer MLP projector with LayerNorm (more stable than BN for small batches)
        # This is closer to BYOL/SimSiam architecture which is more stable
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, 2048),
            nn.LayerNorm(2048),      # LayerNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),   # Extra layer
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, dim)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, dim)
        )

        # Initialize Key Encoder (copy from query)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        # Queue - will be filled properly during training
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Track if queue is warmed up
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Handle wrap around
            rem = self.K - ptr
            self.queue[:, ptr:self.K] = keys[:rem].T
            self.queue[:, :batch_size-rem] = keys[rem:].T
            
        ptr = (ptr + batch_size) % self.K 
        self.queue_ptr[0] = ptr
        
        # Mark queue as filled after one full cycle
        if ptr < batch_size and not self.queue_filled:
            self.queue_filled[0] = True
            print("📦 Queue fully initialized!")

    def forward(self, im_q, im_k):
        # Query features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # Key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # Positive logits: (N, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: (N, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Combine logits: (N, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=im_q.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels, q  # Return q for monitoring

# ==========================================
# 5. LEARNING RATE SCHEDULER WITH WARMUP
# ==========================================
def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    """Cosine annealing with linear warmup"""
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ==========================================
# 6. TRAINING LOOP (With Gradient Accumulation)
# ==========================================
def train():
    dataset = SolarPanelDataset(config['Dataset Path'], transform=moco_transform)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config['Batch Size'], 
        shuffle=True, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    model = MoCo(
        models.resnet50, 
        K=config['Queue Size'], 
        T=config['Temperature']
    ).to(device)
    
    # Use LARS optimizer for better stability (optional, but recommended)
    # Or use standard SGD with momentum
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['Learning Rate'], 
        momentum=0.9, 
        weight_decay=config['Weight Decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision for speed

    print("🔥 STARTING TRAINING (Fixed Architecture with Warmup)...")
    print(f"   Queue Size: {config['Queue Size']}")
    print(f"   Temperature: {config['Temperature']}")
    print(f"   Warmup Epochs: {config['Warmup Epochs']}")
    print(f"   Accumulation Steps: {config['Accumulation Steps']}")

    history = {'epoch': [], 'loss': [], 'f_std': [], 'lr': [], 'accuracy': []}
    best_loss = float('inf')
    
    accumulation_steps = config['Accumulation Steps']

    for epoch in range(config['Training Epochs']):
        model.train()
        running_loss = 0.0
        running_std = 0.0
        running_acc = 0.0
        num_batches = 0
        
        # FIXED: Proper warmup implementation
        current_lr = get_lr(epoch, config['Warmup Epochs'], config['Training Epochs'], config['Learning Rate'])
        set_lr(optimizer, current_lr)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for batch_idx, (images_q, images_k) in enumerate(pbar):
            images_q = images_q.to(device, non_blocking=True)
            images_k = images_k.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                logits, labels, q_features = model(images_q, images_k)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps  # Normalize loss
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            # Monitor metrics
            with torch.no_grad():
                # Feature diversity (std should be 0.1-0.3, NOT 0.02!)
                q_std = q_features.std(dim=0).mean().item()
                running_std += q_std
                
                # Top-1 accuracy (should be increasing)
                acc = (logits.argmax(dim=1) == labels).float().mean().item()
                running_acc += acc
                
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.3f}', 
                'std': f'{q_std:.4f}',
                'acc': f'{acc:.2%}',
                'lr': f'{current_lr:.5f}'
            })

        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Epoch Stats
        epoch_loss = running_loss / num_batches
        epoch_std = running_std / num_batches
        epoch_acc = running_acc / num_batches
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['f_std'].append(epoch_std)
        history['lr'].append(current_lr)
        history['accuracy'].append(epoch_acc)
        
        print(f"📊 Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Std: {epoch_std:.4f} | Acc: {epoch_acc:.2%} | LR: {current_lr:.5f}")
        
        # COLLAPSE WARNING
        if epoch_std < 0.05:
            print(f"⚠️  WARNING: Feature Std ({epoch_std:.4f}) is LOW! Model may be collapsing!")
        
        # Save best model
        if epoch_loss < best_loss and epoch > config['Warmup Epochs']:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.encoder_q.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'std': epoch_std
            }, "moco_v2_best.pth")
            print(f"💾 Saved best model at epoch {epoch+1}")
        
        # Save checkpoints
        if (epoch+1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.encoder_q.state_dict(),
                'full_model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"moco_v2_epoch_{epoch+1}.pth")

    # Save final
    torch.save(model.encoder_q.state_dict(), "moco_v2_final.pth")
    pd.DataFrame(history).to_csv("training_log.csv", index=False)
    print("🏆 DONE.")
    
    return model, history

# ==========================================
# 7. DIAGNOSTIC FUNCTION
# ==========================================
def diagnose_training(history):
    """Check for common issues"""
    print("\n" + "="*50)
    print("🔍 TRAINING DIAGNOSIS")
    print("="*50)
    
    # Check for collapse
    final_std = history['f_std'][-1]
    if final_std < 0.05:
        print(f"❌ COLLAPSE DETECTED: Feature Std = {final_std:.4f} (should be > 0.1)")
    elif final_std < 0.1:
        print(f"⚠️  LOW DIVERSITY: Feature Std = {final_std:.4f} (ideally > 0.1)")
    else:
        print(f"✅ Feature Diversity OK: Std = {final_std:.4f}")
    
    # Check loss trend
    early_loss = sum(history['loss'][:5]) / 5
    late_loss = sum(history['loss'][-5:]) / 5
    
    if late_loss >= early_loss:
        print(f"❌ LOSS NOT DECREASING: {early_loss:.3f} -> {late_loss:.3f}")
    else:
        reduction = (early_loss - late_loss) / early_loss * 100
        print(f"✅ Loss Reduced: {early_loss:.3f} -> {late_loss:.3f} ({reduction:.1f}% reduction)")
    
    # Check accuracy
    final_acc = history['accuracy'][-1]
    if final_acc < 0.1:
        print(f"❌ VERY LOW ACCURACY: {final_acc:.2%} (model not learning)")
    elif final_acc < 0.5:
        print(f"⚠️  LOW ACCURACY: {final_acc:.2%} (may need more training)")
    else:
        print(f"✅ Good Accuracy: {final_acc:.2%}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    model, history = train()
    diagnose_training(history)
