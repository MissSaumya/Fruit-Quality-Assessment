import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ==========================================
# 1. CUSTOM MODEL ARCHITECTURE
# ==========================================
class ViTTwoHead(nn.Module):
    def __init__(self, num_fruits=5, num_qualities=3):
        super().__init__()
        self.base_model = models.vit_b_16(pretrained=False)
        num_ftrs = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Identity() 
        self.fruit_head = nn.Linear(num_ftrs, num_fruits)
        self.qual_head = nn.Linear(num_ftrs, num_qualities)

    def forward(self, x):
        features = self.base_model(x)
        fruit_out = self.fruit_head(features)
        qual_out = self.qual_head(features)
        return fruit_out, qual_out

# ==========================================
# 2. CORRECT CLASS DEFINITIONS (From Notebook)
# ==========================================
# 0: Apple, 1: Banana, 2: Grape, 3: Mango, 4: Orange (Alphabetical)
FRUIT_CLASSES = ["Apple", "Banana", "Grape", "Mango", "Orange"]

# IMPORTANT: Order MUST match notebook: ['fresh', 'rotten', 'formalin-mixed']
QUALITY_CLASSES = ["Fresh", "Rotten", "Formalin-mixed"]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def load_model(path):
    model = ViTTwoHead(num_fruits=5, num_qualities=3)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, img_path):
    # Standard SOTA Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        fruit_logits, qual_logits = model(img_tensor)
        
        # Softmax for probabilities
        fruit_prob = torch.nn.functional.softmax(fruit_logits, dim=1)
        qual_prob = torch.nn.functional.softmax(qual_logits, dim=1)
        
        fruit_conf, fruit_pred = torch.max(fruit_prob, 1)
        qual_conf, qual_pred = torch.max(qual_prob, 1)
        
    fruit_name = FRUIT_CLASSES[fruit_pred.item()]
    quality_name = QUALITY_CLASSES[qual_pred.item()]
    
    return fruit_name, quality_name, fruit_conf.item()*100, qual_conf.item()*100