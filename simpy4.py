import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm

# Image dimensions
width = 750
height = 618

def create_radar_background(width, height):
    """Generate dark radar background with speckle noise"""
    # Start with very dark background
    background_value = 2
    radar_img = np.full((height, width), background_value, dtype=float)
    
    # Add speckle noise (multiplicative noise - characteristic of radar)
    # Using gamma distribution for realistic speckle
    speckle = np.random.gamma(shape=2.0, scale=1.0, size=(height, width))
    radar_img = radar_img * speckle
    
    # Add some additive Gaussian noise for texture
    gaussian_noise = np.random.normal(0, 2, (height, width))
    radar_img = radar_img + gaussian_noise
    
    # Clip and normalize to keep background very dark (like the example)
    radar_img = np.clip(radar_img, 0, 20)
    radar_img = radar_img.astype(np.uint8)
    
    return radar_img

def generate_targets(width, height, num_targets=15):
    """Generate random target positions using pandas DataFrame"""
    targets_data = {
        'x': np.random.randint(10, width - 10, num_targets),
        'y': np.random.randint(10, height - 10, num_targets)
    }
    targets_df = pd.DataFrame(targets_data)
    return targets_df

def draw_radar_targets(img, targets_df):
    """Draw 3x3 pixel targets with strongest center pixel"""
    for _, target in targets_df.iterrows():
        x, y = int(target['x']), int(target['y'])
        
        # Ensure target is within bounds
        if x < 1 or x > img.shape[1] - 2 or y < 1 or y > img.shape[0] - 2:
            continue
        
        # 3x3 target with graduated intensity
        # Corner pixels - medium intensity
        img[y-1, x-1] = 180
        img[y-1, x+1] = 180
        img[y+1, x-1] = 180
        img[y+1, x+1] = 180
        
        # Edge pixels - bright
        img[y-1, x] = 220
        img[y+1, x] = 220
        img[y, x-1] = 220
        img[y, x+1] = 220
        
        # Center pixel - strongest intensity (white)
        img[y, x] = 255
    
    return img

def apply_jet_colormap(gray_img):
    """Convert grayscale image to jet colormap RGB"""
    # Normalize to 0-1 range
    normalized = gray_img.astype(float) / 255.0
    
    # Apply jet colormap
    jet = cm.get_cmap('jet')
    colored_img = jet(normalized)
    
    # Convert to RGB (remove alpha channel) and scale to 0-255
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    
    return rgb_img

import os
import json

# Create dataset directory
base_dir = r'C:\vscode\cremlin\foldem'
dataset_dir = os.path.join(base_dir, 'radar_dataset')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Dataset parameters
num_images = 100  # Number of images to generate
min_targets = 5
max_targets = 20

print(f"Generating dataset with {num_images} radar images...")
print(f"Target range: {min_targets}-{max_targets} per image")
print(f"Saving to: {dataset_dir}/")
print(f"Color format: Jet colormap (RGB)")

# Store dataset metadata
dataset_metadata = {
    'num_images': num_images,
    'image_width': width,
    'image_height': height,
    'color_format': 'jet_colormap_RGB',
    'target_size': '3x3 pixels',
    'min_targets': min_targets,
    'max_targets': max_targets,
    'images': []
}

# Generate dataset
for idx in range(num_images):
    # Random number of targets for this image
    num_targets = np.random.randint(min_targets, max_targets + 1)
    
    # Create radar image
    radar_img = create_radar_background(width, height)
    
    # Generate targets
    targets_df = generate_targets(width, height, num_targets=num_targets)
    
    # Draw targets on image
    radar_img = draw_radar_targets(radar_img, targets_df)
    
    # Apply jet colormap to convert to RGB
    colored_img = apply_jet_colormap(radar_img)
    
    # Save RGB image with jet colormap
    img_filename = f'radar_image_{idx:04d}.png'
    img_path = os.path.join(images_dir, img_filename)
    pil_img = Image.fromarray(colored_img, mode='RGB')
    pil_img.save(img_path)
    
    # Save labels (target positions) as JSON
    label_filename = f'radar_image_{idx:04d}.json'
    label_path = os.path.join(labels_dir, label_filename)
    labels_data = {
        'image_id': idx,
        'image_filename': img_filename,
        'num_targets': len(targets_df),
        'targets': targets_df.to_dict('records')
    }
    with open(label_path, 'w') as f:
        json.dump(labels_data, f, indent=2)
    
    # Update metadata
    dataset_metadata['images'].append({
        'id': idx,
        'filename': img_filename,
        'num_targets': len(targets_df)
    })
    
    # Progress indicator
    if (idx + 1) % 10 == 0:
        print(f"Generated {idx + 1}/{num_images} images...")

# Save dataset metadata
metadata_path = os.path.join(dataset_dir, 'dataset_info.json')
with open(metadata_path, 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

# Create CSV summary of all targets
all_targets = []
for idx in range(num_images):
    label_path = os.path.join(labels_dir, f'radar_image_{idx:04d}.json')
    with open(label_path, 'r') as f:
        data = json.load(f)
        for target in data['targets']:
            all_targets.append({
                'image_id': idx,
                'image_filename': data['image_filename'],
                'target_x': target['x'],
                'target_y': target['y']
            })

targets_summary_df = pd.DataFrame(all_targets)
csv_path = os.path.join(dataset_dir, 'targets_summary.csv')
targets_summary_df.to_csv(csv_path, index=False)

print(f"\n✓ Dataset generation complete!")
print(f"\nDataset structure:")
print(f"  {dataset_dir}/")
print(f"  ├── images/          ({num_images} RGB PNG images with jet colormap)")
print(f"  ├── labels/          ({num_images} JSON label files)")
print(f"  ├── dataset_info.json (metadata)")
print(f"  └── targets_summary.csv (all targets in CSV format)")
print(f"\nTotal targets: {len(all_targets)}")
print(f"Average targets per image: {len(all_targets)/num_images:.1f}")

# Display one sample image
sample_idx = 0
sample_img_path = os.path.join(images_dir, f'radar_image_{sample_idx:04d}.png')
sample_img = np.array(Image.open(sample_img_path))

fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
im = ax.imshow(sample_img, aspect='auto', interpolation='nearest')
ax.set_xlabel('Velocity', color='white', fontsize=11, fontweight='bold')
ax.set_ylabel('Range', color='white', fontsize=11, fontweight='bold')
ax.set_title(f'Sample Image: {sample_img_path}', color='white', fontsize=13, 
             fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=9)
plt.tight_layout()
plt.show()

print(f"\nSample image displayed: radar_image_{sample_idx:04d}.png")
print("Images are now saved in jet colormap format (RGB)")