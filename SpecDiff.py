import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import os
import cv2
import numpy as np
from torchvision import transforms
from torchmetrics.classification import MulticlassJaccardIndex
from diffusers import DPMSolverMultistepScheduler
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_size=(256, 256)):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir) if lbl.endswith(('.png', '.jpg', '.jpeg'))])
        self.class_colors = {
            (255, 255, 255): 0, (160, 160, 160): 1,
            (80, 80, 80): 2, (0, 0, 0): 3
        }
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        label_mask = np.zeros(self.target_size, dtype=np.uint8)
        label_resized = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)

        for rgb, class_idx in self.class_colors.items():
            label_mask[np.all(label_resized == rgb, axis=-1)] = class_idx

        if self.transform:
            image = self.transform(image_resized)
            
        label_mask = torch.from_numpy(label_mask).long()
        return image, label_mask


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )
    def forward(self, x):
        return x * self.excitation(self.pool(x))

class FSBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_c)
    def forward(self, x):
        h = self.relu(self.norm1(self.conv1(x)))
        h = self.relu(self.norm2(self.conv2(h)))
        h = self.se(h)
        return h

class TCS(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels) 
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels) 
        self.act2 = nn.ReLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels*2)
        self.se = SEBlock(out_channels)

    def forward(self, x, t_emb):  
        h = self.act1(self.norm1(self.conv1(x)))

        modulation = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = modulation.chunk(2, dim=1)
        h = h * scale
        h = self.act2(self.norm2(self.conv2(h)))
        h = h + shift
        h = self.se(h)
        return h

class CustomGatedAttention(nn.Module):
    def __init__(self, channels, time_emb_dim, num_iterations=4, num_bottleneck_blocks=2):
        super().__init__()
        self.n_attention_iterations = num_iterations
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        self.q_learnable_bases = nn.ParameterList([nn.Parameter(torch.randn(1, channels * 2, 1, 1)) for _ in range(num_iterations)])
        self.k_learnable_bases = nn.ParameterList([nn.Parameter(torch.randn(1, channels * 2, 1, 1)) for _ in range(num_iterations)])
        self.qk_base_feature_conv3x3s = nn.ModuleList([nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1) for _ in range(num_iterations)])
        self.qk_final_projection_1x1s = nn.ModuleList([nn.Conv2d(channels * 2, channels, kernel_size=1) for _ in range(num_iterations)])

        self.v_learnable_bases = nn.ParameterList([nn.Parameter(torch.randn(1, channels, 1, 1)) for _ in range(num_iterations)])
        self.v_base_feature_conv3x3s = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(num_iterations)])
        self.v_final_projection_1x1s = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_iterations)])

        self.qk_norms = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(num_iterations)])
        self.v_norms = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(num_iterations)])
        
        self.bottleneck_blocks = nn.ModuleList([
            TCS(channels, channels, time_emb_dim) for _ in range(num_bottleneck_blocks)
        ])
        self.bottleneck_gate_convs = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_iterations)])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, h_bot, c_bot, t_emb):
        current_q = h_bot
        low_freq_k_initial = self.avg_pool(c_bot)
        high_freq_k_initial = c_bot - low_freq_k_initial
        k_combined_initial = torch.cat([high_freq_k_initial, low_freq_k_initial], dim=1)
        v_initial = c_bot
        
        attention_output = None

        for i in range(self.n_attention_iterations):
            q_base_i, k_base_i, v_base_i = self.q_learnable_bases[i], self.k_learnable_bases[i], self.v_learnable_bases[i]
            qk_conv3x3_i, v_conv3x3_i = self.qk_base_feature_conv3x3s[i], self.v_base_feature_conv3x3s[i]
            qk_proj1x1_i, v_proj1x1_i = self.qk_final_projection_1x1s[i], self.v_final_projection_1x1s[i]
            
            q_norm_i, k_norm_i, v_norm_i = self.qk_norms[i], self.qk_norms[i], self.v_norms[i]
            
            low_freq_q = self.avg_pool(current_q)
            high_freq_q = current_q - low_freq_q
            q_combined = torch.cat([high_freq_q, low_freq_q], dim=1)
            
            q_modulator = qk_conv3x3_i(q_base_i)
            q_modulated = q_combined * q_modulator
            q_processed = qk_proj1x1_i(q_modulated)
            q_processed = q_norm_i(q_processed)

            k_modulator = qk_conv3x3_i(k_base_i)
            k_modulated = k_combined_initial * k_modulator
            k_processed = qk_proj1x1_i(k_modulated)
            k_processed = k_norm_i(k_processed)

            v_modulator = v_conv3x3_i(v_base_i)
            v_modulated = v_initial * v_modulator
            v_processed = v_proj1x1_i(v_modulated)
            v_processed = v_norm_i(v_processed)
            
            zqk = q_processed * k_processed

            h = zqk
            for TCS in self.bottleneck_blocks:
                h = TCS(h, t_emb)
            
            gate = self.sigmoid(h)
            
            attention_output = gate * v_processed
            
            current_q = attention_output

        return attention_output



class ControlEncoder(nn.Module):
    def __init__(self, initial_latent_channels, time_emb_dim=128):
        super().__init__()
        self.down1 = TCS(initial_latent_channels, initial_latent_channels, time_emb_dim)
        self.down2 = TCS(initial_latent_channels, initial_latent_channels, time_emb_dim)
        self.down3 = TCS(initial_latent_channels, initial_latent_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bot1 = TCS(initial_latent_channels, initial_latent_channels, time_emb_dim)

        self.zt_parser_conv1 = nn.Conv2d(initial_latent_channels, initial_latent_channels, kernel_size=1)
        self.zt_parser_norm1 = nn.BatchNorm2d(initial_latent_channels)
        self.zt_parser_act1 = nn.ReLU()

        self.zt_parser_conv2 = nn.Conv2d(initial_latent_channels, initial_latent_channels, kernel_size=1)
        self.zt_parser_norm2 = nn.BatchNorm2d(initial_latent_channels)
        self.zt_parser_act2 = nn.ReLU()

        self.zt_parser_conv3 = nn.Conv2d(initial_latent_channels, initial_latent_channels, kernel_size=1)
        self.zt_parser_norm3 = nn.BatchNorm2d(initial_latent_channels)
        self.zt_parser_act3 = nn.ReLU()

        self.zt_parser_conv_bot = nn.Conv2d(initial_latent_channels, initial_latent_channels, kernel_size=1)
        self.zt_parser_norm_bot = nn.BatchNorm2d(initial_latent_channels)
        self.zt_parser_act_bot = nn.ReLU()

    def forward(self, z_condition, t_emb, zt_features):
        h_zt1, h_zt2, h_zt3, h_zt_bot = zt_features

        cond_features = []
        
        c1 = self.down1(z_condition, t_emb)
        parsed_h_zt1 = self.zt_parser_act1(self.zt_parser_norm1(self.zt_parser_conv1(h_zt1)))
        cond_features.append(c1 + parsed_h_zt1)

        c2 = self.down2(self.pool(c1), t_emb)
        parsed_h_zt2 = self.zt_parser_act2(self.zt_parser_norm2(self.zt_parser_conv2(h_zt2)))
        cond_features.append(c2 + parsed_h_zt2)
        
        c3 = self.down3(self.pool(c2), t_emb)
        parsed_h_zt3 = self.zt_parser_act3(self.zt_parser_norm3(self.zt_parser_conv3(h_zt3)))
        cond_features.append(c3 + parsed_h_zt3)

        c_bot = self.bot1(self.pool(c3), t_emb)
        parsed_h_zt_bot = self.zt_parser_act_bot(self.zt_parser_norm_bot(self.zt_parser_conv_bot(h_zt_bot)))
        cond_features.append(c_bot + parsed_h_zt_bot)

        return cond_features

class SimpleUNet(nn.Module):
    def __init__(self, initial_latent_channels, time_emb_dim=128):
        super().__init__()
        self.down1 = TCS(initial_latent_channels, initial_latent_channels, time_emb_dim)
        self.down2 = TCS(initial_latent_channels * 2, initial_latent_channels, time_emb_dim)
        self.down3 = TCS(initial_latent_channels * 2, initial_latent_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bot1 = TCS(initial_latent_channels * 2, initial_latent_channels, time_emb_dim)
        
        self.bot_attention = CustomGatedAttention(channels=initial_latent_channels, time_emb_dim=time_emb_dim)
        self.bot2 = TCS(initial_latent_channels, initial_latent_channels, time_emb_dim)

        self.up_trans1 = nn.ConvTranspose2d(initial_latent_channels, initial_latent_channels, 4, 2, 1)
        self.up1 = TCS(initial_latent_channels * 2, initial_latent_channels, time_emb_dim)
        
        self.up_trans2 = nn.ConvTranspose2d(initial_latent_channels, initial_latent_channels, 6, 2, 2)
        self.up2 = TCS(initial_latent_channels * 2, initial_latent_channels, time_emb_dim)
        
        self.up_trans3 = nn.ConvTranspose2d(initial_latent_channels, initial_latent_channels, 8, 2, 3)
        self.up3 = TCS(initial_latent_channels * 2, initial_latent_channels, time_emb_dim)
        
        self.out = nn.Conv2d(initial_latent_channels, initial_latent_channels, 1)

    def forward(self, x_t, t_emb, cond_features):
        c1, c2, c3, c_bot = cond_features

        h1 = self.down1(x_t, t_emb)
        h1_cond = torch.cat([h1, c1], dim=1)
        
        h2 = self.down2(self.pool(h1_cond), t_emb)
        h2_cond = torch.cat([h2, c2], dim=1)
        
        h3 = self.down3(self.pool(h2_cond), t_emb)
        h3_cond = torch.cat([h3, c3], dim=1)
        
        h_bot = self.bot1(self.pool(h3_cond), t_emb)

        attn_out = self.bot_attention(h_bot, c_bot, t_emb)
        h_bot = h_bot + attn_out
        h_bot = self.bot2(h_bot, t_emb)

        h_up = self.up_trans1(h_bot)
        h_up = torch.cat([h_up, h3], dim=1)
        h_up = self.up1(h_up, t_emb)
        
        h_up = self.up_trans2(h_up)
        h_up = torch.cat([h_up, h2], dim=1)
        h_up = self.up2(h_up, t_emb)

        h_up = self.up_trans3(h_up)
        h_up = torch.cat([h_up, h1], dim=1)
        h_up = self.up3(h_up, t_emb)
        
        return self.out(h_up)

class SpecDiff(nn.Module):
    def __init__(self, in_channels=3, latent_dim_base=32, num_classes=4, time_emb_dim=128):
        super().__init__()

        c0 = latent_dim_base      
        c1 = int(c0 * 1.5)        
        c2 = int(c1 * 2)          
        c3 = int(c2 * 2)          
        c4 = int(c3 * 1.5)          

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 5), nn.ReLU(),
            nn.Linear(time_emb_dim * 5, time_emb_dim),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_in = FSBlock(in_channels, c0)
        self.enc_block1 = FSBlock(c0, c1)
        self.enc_block2 = FSBlock(c1, c2)
        self.enc_block3 = FSBlock(c2, c3)
        self.enc_block4 = FSBlock(c3, c4)
        self.encoder_to_latent_proj = nn.Conv2d(c4, c0, kernel_size=1)
        self.norm_encoder_to_latent = nn.BatchNorm2d(c0)
        self.act_encoder_to_latent = nn.ReLU()

        self.control_encoder = ControlEncoder(c0, time_emb_dim)
        self.denoiser = SimpleUNet(c0, time_emb_dim)

        self.zt_encoder_down1 = TCS(c0, c0, time_emb_dim)
        self.zt_encoder_down2 = TCS(c0, c0, time_emb_dim)
        self.zt_encoder_down3 = TCS(c0, c0, time_emb_dim)
        self.zt_encoder_pool = nn.MaxPool2d(2)
        self.zt_encoder_bot1 = TCS(c0, c0, time_emb_dim)
        
        self.final_noise_predictor = nn.Conv2d(c0, c0, kernel_size=1)

        self.latent_to_decoder_proj = nn.Conv2d(c0, c4, kernel_size=1)
        self.norm_latent_to_decoder = nn.BatchNorm2d(c4)
        self.act_latent_to_decoder = nn.ReLU()

        self.dec_proc = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
            SEBlock(c4)
        )
        self.dec_up1 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            SEBlock(c3)
        )
        self.dec_up2 = nn.ConvTranspose2d(c3, c2, kernel_size=8, stride=2, padding=3)
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            SEBlock(c2)
        )
        self.dec_up3 = nn.ConvTranspose2d(c2, c1, kernel_size=8, stride=2, padding=3)
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            SEBlock(c1)
        )
        self.dec_up4 = nn.ConvTranspose2d(c1, c0, kernel_size=8, stride=2, padding=3)
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(c0 + c0, c0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
            SEBlock(c0)
        )
        
        self.dec_out = nn.Conv2d(c0, num_classes, kernel_size=3, padding=1)


    def _forward_encoder(self, x):
        skip_features = []
        x = self.enc_in(x)
        skip_features.append(x)

        x = self.enc_block1(self.pool(x))
        skip_features.append(x)

        x = self.enc_block2(self.pool(x))
        skip_features.append(x)

        x = self.enc_block3(self.pool(x))
        skip_features.append(x)

        z0 = self.enc_block4(self.pool(x))
        
        z0_compressed = self.act_encoder_to_latent(self.norm_encoder_to_latent(self.encoder_to_latent_proj(z0)))
        
        return z0_compressed, skip_features

    def _get_zt_features(self, x_t, t_emb):
        h_zt1 = self.zt_encoder_down1(x_t, t_emb)
        h_zt2 = self.zt_encoder_down2(self.zt_encoder_pool(h_zt1), t_emb)
        h_zt3 = self.zt_encoder_down3(self.zt_encoder_pool(h_zt2), t_emb)
        h_zt_bot = self.zt_encoder_bot1(self.zt_encoder_pool(h_zt3), t_emb)
        return [h_zt1, h_zt2, h_zt3, h_zt_bot]
        
    def _forward_decoder(self, z, skip_features):
        z_expanded = self.act_latent_to_decoder(self.norm_latent_to_decoder(self.latent_to_decoder_proj(z)))

        skips_r = skip_features.pop(0) 
        skips_h = skip_features.pop(0) 
        skips_q = skip_features.pop(0) 
        skips_e = skip_features.pop(0) 
        
        x = self.dec_proc(z_expanded)

        up1 = self.dec_up1(x)
        x = torch.cat([up1, skips_e], dim=1)
        x = self.dec_block1(x)

        up2 = self.dec_up2(x)
        x = torch.cat([up2, skips_q], dim=1)
        x = self.dec_block2(x)

        up3 = self.dec_up3(x)
        x = torch.cat([up3, skips_h], dim=1)
        x = self.dec_block3(x)
        
        up4 = self.dec_up4(x)
        x = torch.cat([up4, skips_r], dim=1)
        x = self.dec_block4(x)
        
        return self.dec_out(x)
        
    def forward(self, x_t, t, condition_z0):
        t_emb = self.time_mlp(t)
        
        zt_features = self._get_zt_features(x_t, t_emb)

        cond_features = self.control_encoder(condition_z0, t_emb, zt_features)
        
        predicted_noise = self.denoiser(x_t, t_emb, cond_features)
        
        predicted_noise = self.final_noise_predictor(predicted_noise)
        return predicted_noise


def evaluate(model, dataloader, device, iou_metric):
    model.eval()
    total_loss = 0.0
    iou_metric.reset()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            z0_latent_compressed, skip_features = model._forward_encoder(imgs)
            mask_logits = model._forward_decoder(z0_latent_compressed, skip_features)
            
            loss = F.cross_entropy(mask_logits, lbls)
            preds = torch.argmax(mask_logits, dim=1)
            iou_metric.update(preds, lbls)
            total_loss += loss.item()
            pbar.set_postfix(eval_loss=loss.item(), mIoU=iou_metric.compute().item())
            
    avg_loss = total_loss / len(dataloader)
    avg_iou = iou_metric.compute().item()
    model.train()
    return avg_loss, avg_iou

def generate_mask(model, condition_image, num_inference_steps, device):
    model.eval()
    scheduler = DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    with torch.no_grad():
        condition_z0_compressed, skip_features_orig = model._forward_encoder(condition_image.to(device))
        
        z = torch.randn_like(condition_z0_compressed)

        for t in timesteps:
            t_tensor = t.unsqueeze(0) if t.dim() == 0 else t
            
            predicted_noise = model(z, t_tensor, condition_z0=condition_z0_compressed)
            z = scheduler.step(predicted_noise, t, z).prev_sample
            
        final_mask_logits = model._forward_decoder(z, [sf.clone() for sf in skip_features_orig])
        final_mask = torch.argmax(final_mask_logits, dim=1)
    return final_mask


def main():
    # --- Hyperparameters ---
    IMG_SIZE = 256
    LATENT_DIM_BASE = 32
    IN_CHANNELS = 3
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 60 
    TIME_EMB_DIM = 128
    MODEL_SAVE_PATH = "SpecDiff.pth"

    TRAIN_TIMESTEPS = 1000
    SAMPLING_STEPS = 20
    
    RECON_WEIGHT = 0.5
    DIFF_WEIGHT = 1.0 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_image_dir = "/kaggle/input/radarcomm/train/input"
    train_label_dir = "/kaggle/input/radarcomm/train/label"
    val_image_dir = "/kaggle/input/radarcomm/val/input"
    val_label_dir = "/kaggle/input/radarcomm/val/label"
    
    train_dataset = SemanticSegmentationDataset(train_image_dir, train_label_dir, transform=transform, target_size=(IMG_SIZE, IMG_SIZE))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataloader = None
    if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
        val_dataset = SemanticSegmentationDataset(val_image_dir, val_label_dir, transform=transform, target_size=(IMG_SIZE, IMG_SIZE))
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print(f"validation doesn't exist")

    model = SpecDiff(in_channels=IN_CHANNELS, latent_dim_base=LATENT_DIM_BASE, num_classes=NUM_CLASSES, 
                     time_emb_dim=TIME_EMB_DIM).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='macro').to(device)
    
    betas = torch.linspace(0.0001, 0.02, TRAIN_TIMESTEPS, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    best_val_iou = -1.0
    
    for epoch in range(EPOCHS):
        model.train()
        iou_metric.reset()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, lbls in pbar:
            optimizer.zero_grad()
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.shape[0]

            z0_compressed, skip_features = model._forward_encoder(imgs)

            t = torch.randint(0, TRAIN_TIMESTEPS, (B,), device=device).long()
            noise_gt = torch.randn_like(z0_compressed)
            sqrt_alphas_cumprod_t = alphas_cumprod[t].sqrt().view(B, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = (1. - alphas_cumprod[t]).sqrt().view(B, 1, 1, 1)
            zt = sqrt_alphas_cumprod_t * z0_compressed + sqrt_one_minus_alphas_cumprod_t * noise_gt
            
            predicted_noise = model(zt, t, condition_z0=z0_compressed)
            
            loss_diff = F.mse_loss(predicted_noise, noise_gt)

            pred_z0 = (zt - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            mask_logits = model._forward_decoder(pred_z0, [sf.clone() for sf in skip_features])
            
            loss_recon = F.cross_entropy(mask_logits, lbls)
            
            total_loss = (RECON_WEIGHT * loss_recon) + (DIFF_WEIGHT * loss_diff)
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                iou_metric.update(torch.argmax(mask_logits, dim=1), lbls)
            
            pbar.set_postfix(
                TotalLoss=f"{total_loss.item():.4f}",
                ReconLoss=f"{loss_recon.item():.4f}", 
                DiffLoss=f"{loss_diff.item():.4f}",
                mIoU=f"{iou_metric.compute().item():.4f}",
                LR=f"{optimizer.param_groups[0]['lr']:.1e}"
            )
            

        if val_dataloader:
            eval_loss, eval_iou = evaluate(model, val_dataloader, device, iou_metric)
            print(f"===> Epoch {epoch+1} Validation Stats: Eval Loss: {eval_loss:.4f} | Eval mIoU: {eval_iou:.4f}")
            
            scheduler.step(eval_loss)

            if eval_iou > best_val_iou:
                best_val_iou = eval_iou
                print(f" mIoU validation better ({best_val_iou:.4f}). Saving...")
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            if epoch == EPOCHS - 1:
                print(f"Training completed. Saving model to {MODEL_SAVE_PATH}...")
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if val_dataloader:
        print(f"Loading model from {MODEL_SAVE_PATH} for inference.")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        
        test_image, _ = next(iter(val_dataloader))
        condition_image_tensor = test_image[0].unsqueeze(0).to(device)
        
        generated_mask_array = generate_mask(model, condition_image_tensor, SAMPLING_STEPS, device)
        scale_factor = (255 // (NUM_CLASSES - 1)) if NUM_CLASSES > 1 else 255
        output_mask_image = (generated_mask_array.squeeze().cpu().numpy() * scale_factor).astype(np.uint8)
        cv2.imwrite("generated_mask_flat_latent_channels.png", output_mask_image)
        print(f"saving generated_mask_flat_latent_channels.png")

if __name__ == '__main__':
    main()