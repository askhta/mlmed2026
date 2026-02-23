import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress

class HCDataset(Dataset):
    def __init__(self, root_dir, img_size=256, has_annotations=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.has_annotations = has_annotations
        self.images = [f for f in os.listdir(root_dir)
                       if f.endswith(".png") and "Annotation" not in f]
        
        # Load pixel size data
        self.pixel_sizes = {}
        
        # Try to load from training or test CSV
        csv_candidates = [
            os.path.join(root_dir, '..', 'training_set_pixel_size_and_HC.csv'),
            os.path.join(root_dir, '..', 'test_set_pixel_size.csv'),
            os.path.join(root_dir, 'pixel_sizes.csv')
        ]
        
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # Try different column name variations
                    pixel_col = None
                    if 'pixel size(mm)' in df.columns:
                        pixel_col = 'pixel size(mm)'
                    elif 'pixel_size' in df.columns:
                        pixel_col = 'pixel_size'
                    elif 'pixel size' in df.columns:
                        pixel_col = 'pixel size'
                    
                    if pixel_col:
                        for idx, row in df.iterrows():
                            self.pixel_sizes[row['filename']] = row[pixel_col]
                    break
                except Exception as e:
                    print(f"Warning: Could not load pixel sizes from {csv_path}: {e}")
                    continue

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def fit_ellipse_from_mask(self, mask):
        """Fit ellipse to mask and return parameters"""
        ys, xs = np.where(mask > 0)
        
        if len(xs) < 5:  # Need at least 5 points for ellipse fitting
            return [self.img_size/2, self.img_size/2, 10.0, 10.0, 0.0]
        
        # Stack points for ellipse fitting
        points = np.column_stack([xs, ys])
        
        # Fit ellipse
        ellipse = cv2.fitEllipse(points.astype(np.float32))
        
        # Extract ellipse parameters: (center), (axes), angle
        center, axes, angle = ellipse
        cx, cy = center
        a, b = axes[0] / 2, axes[1] / 2  # Convert diameter to semi-axes
        angle_rad = np.deg2rad(angle)
        
        return [cx, cy, a, b, angle_rad]

    def extract_ellipse_params(self, mask_path):
        """Extract ellipse parameters from annotation mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            return torch.tensor([self.img_size/2, self.img_size/2, 10.0, 10.0, 0.0], dtype=torch.float32)
        
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Fill ellipse and extract parameters
        params = self.fit_ellipse_from_mask(mask)
        
        return torch.tensor(params, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        ann_path = os.path.join(
            self.root_dir,
            img_name.replace(".png", "_Annotation.png")
        )

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)  # Keep as (3, H, W) for CNN
        
        if self.has_annotations:
            target = self.extract_ellipse_params(ann_path)
        else:
            target = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        
        # Get pixel size
        pixel_size = self.pixel_sizes.get(img_name, 0.1)  # Default 0.1 mm/pixel

        return img, target, pixel_size, img_name


# Neural network regression model for ellipse parameters - CNN based
class HeadCircumferenceModel(nn.Module):
    def __init__(self):
        super(HeadCircumferenceModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Adaptive pooling to fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)  # Output: cx, cy, semi_a, semi_b, angle_rad
        
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Training function
def train_model(model, train_loader, epochs=10, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for images, targets, pixel_sizes, img_names in train_loader:
            images = images.to(device)
            # Reshape images back to (B, 3, 256, 256) instead of flattened
            images = images.reshape(-1, 3, 256, 256)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return model


def calculate_head_circumference(semi_a, semi_b):
    """
    Calculate head circumference from ellipse semi-axes
    Uses Ramanujan's approximation for ellipse perimeter
    """
    a, b = max(semi_a, semi_b), min(semi_a, semi_b)
    h = ((a - b)**2) / ((a + b)**2)
    circumference = np.pi * (a + b) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))
    return circumference


# Main execution
if __name__ == "__main__":
    # Load dataset
    train_dir = r'd:\DS\ML in medicine\practical_2\training_set'
    test_dir = r'd:\DS\ML in medicine\practical_2\test_set'
    
    dataset = HCDataset(train_dir, img_size=256)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = HeadCircumferenceModel()
    
    # Train model
    print("Training Head Circumference Model...")
    model = train_model(model, train_loader, epochs=10)
    
    # Test on test images and generate output CSV
    test_dataset = HCDataset(test_dir, img_size=256, has_annotations=False)
    
    if len(test_dataset) == 0:
        print("\nNo test images found. Using training images for demonstration...")
        test_dataset = HCDataset(train_dir, img_size=256, has_annotations=True)
    
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Store results
    results = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for images, targets, pixel_sizes, img_names in test_loader:
            images = images.to(device)
            # Reshape images back to (B, 3, 256, 256) instead of flattened
            images = images.reshape(-1, 3, 256, 256)
            predictions = model(images)
            
            # Convert to numpy
            pred_values = predictions.cpu().numpy()[0]
            pixel_size = pixel_sizes[0].item() if isinstance(pixel_sizes[0], torch.Tensor) else pixel_sizes[0]
            img_name = img_names[0]
            
            # Extract ellipse parameters
            center_x_px, center_y_px, semi_a_px, semi_b_px, angle_rad = pred_values
            
            # Convert from pixels to mm
            center_x_mm = center_x_px * pixel_size
            center_y_mm = center_y_px * pixel_size
            semi_a_mm = semi_a_px * pixel_size
            semi_b_mm = semi_b_px * pixel_size
            
            # Calculate head circumference
            hc_mm = calculate_head_circumference(semi_a_mm, semi_b_mm)
            
            # Store result
            results.append({
                'filename': img_name,
                'center_x_mm': center_x_mm,
                'center_y_mm': center_y_mm,
                'semi_axes_a': semi_a_mm,
                'semi_axes_b': semi_b_mm,
                'angle_rad': angle_rad,
                'head_circumference_mm': hc_mm
            })
            
            # Print first result
            if len(results) == 1:
                print(f"\n=== Sample Prediction ===")
                print(f"Image: {img_name}")
                print(f"Center X (mm): {center_x_mm:.2f}")
                print(f"Center Y (mm): {center_y_mm:.2f}")
                print(f"Semi-axis A (mm): {semi_a_mm:.2f}")
                print(f"Semi-axis B (mm): {semi_b_mm:.2f}")
                print(f"Angle (rad): {angle_rad:.4f}")
                print(f"Head Circumference (mm): {hc_mm:.2f}")
    
    # Save results to CSV
    output_csv = r'd:\DS\ML in medicine\practical_2\predictions_output.csv'
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    
    print(f"\nPredictions saved to: {output_csv}")
    print(f"Total predictions: {len(results)}")
          
