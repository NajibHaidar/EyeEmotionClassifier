import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNBackbone(nn.Module):
    """CNN Backbone to extract low-level visual features."""
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1, 128)  # Reshape for Transformer (batch_size, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    """Transformer Encoder to relate global relationships in image patches."""
    def __init__(self, embed_dim=128, num_heads=8, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=0.1, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        return x

class EmotionClassifier(nn.Module):
    """Full model: Pretrained CNN Backbone + Transformer + Classification Head"""
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        
        # Use a pre-trained ResNet backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify the ResNet to output 128-dim features instead of 1000 (this is crucial)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)
        
        # Transformer layer (optional, can be removed if ResNet is good enough)
        self.transformer = TransformerEncoder(embed_dim=128)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to reduce overfitting
            nn.Linear(512, num_classes)  
        )
        
    def forward(self, x):
        x = self.backbone(x)  # Extract features with ResNet
        x = x.unsqueeze(1)  # Add a sequence dimension for Transformer (if needed)
        x = self.transformer(x)  # Pass through Transformer
        x = x[:, 0, :]  # Take only the first token (like in BERT/ViT)
        x = self.classifier(x)  # Pass through classification head
        return x