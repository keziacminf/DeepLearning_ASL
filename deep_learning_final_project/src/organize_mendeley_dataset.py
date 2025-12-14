#!/usr/bin/env python3
"""
Organize Mendeley ASL dataset after download
"""

import os
import shutil
from pathlib import Path
import random

def organize_mendeley_dataset(source_dir='data/mendeley_downloaded'):
    """Organize Mendeley ASL dataset into train/test structure"""
    print("="*60)
    print("Organizing Mendeley ASL Dataset")
    print("="*60)

    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"\n✗ Error: Source directory not found: {source_dir}")
        print("\nPlease:")
        print("1. Download from: https://data.mendeley.com/datasets/48dg9vhmyk/2")
        print("2. Extract to: data/mendeley_downloaded/")
        print("3. Run this script again")
        return False

    # Find all image directories
    print(f"\nSearching in: {source_path}")

    train_dir = Path('data/train')
    test_dir = Path('data/test')

    # Clear existing data
    print("\nClearing old data...")
    for d in [train_dir, test_dir]:
        if d.exists():
            for item in d.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    shutil.rmtree(item)
        d.mkdir(parents=True, exist_ok=True)

    # Find class directories
    class_dirs = {}

    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)

        # Look for image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(image_files) > 50:  # Must have at least 50 images
            # Try to identify class name from directory structure
            dir_name = root_path.name.upper()

            # Check if it's a valid ASL letter or special class
            if len(dir_name) == 1 and dir_name.isalpha():
                class_name = dir_name
            elif dir_name in ['DELETE', 'SPACE']:
                class_name = dir_name
            elif 'DEL' in dir_name.upper():
                class_name = 'DELETE'
            elif 'SPACE' in dir_name.upper():
                class_name = 'SPACE'
            else:
                # Try parent directory
                parent_name = root_path.parent.name.upper()
                if len(parent_name) == 1 and parent_name.isalpha():
                    class_name = parent_name
                else:
                    continue

            if class_name not in class_dirs:
                class_dirs[class_name] = []

            class_dirs[class_name].extend([root_path / f for f in image_files])

    if not class_dirs:
        print("\n✗ No class directories found!")
        print("\nPlease check the structure of the downloaded dataset.")
        print(f"Expected to find letter directories (A-Z) in: {source_path}")
        return False

    print(f"\n✓ Found {len(class_dirs)} classes")

    # Organize each class
    total_train = 0
    total_test = 0

    for class_name in sorted(class_dirs.keys()):
        images = class_dirs[class_name]
        print(f"\nProcessing '{class_name}': {len(images)} images")

        # Shuffle
        random.seed(42)
        random.shuffle(images)

        # Limit to reasonable number (e.g., 500 per class for faster training)
        max_per_class = 500
        if len(images) > max_per_class:
            print(f"  Limiting to {max_per_class} images")
            images = images[:max_per_class]

        # Create class directories
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)

        # Split 80/20
        split_idx = int(len(images) * 0.8)

        # Copy images
        for i, img_path in enumerate(images):
            if i < split_idx:
                dest = train_class_dir / f"{class_name}_{i}.jpg"
                total_train += 1
            else:
                dest = test_class_dir / f"{class_name}_{i-split_idx}.jpg"
                total_test += 1

            shutil.copy2(img_path, dest)

        print(f"  → Train: {split_idx}, Test: {len(images) - split_idx}")

    print("\n" + "="*60)
    print("Dataset Organization Complete!")
    print("="*60)
    print(f"Total training images: {total_train}")
    print(f"Total test images: {total_test}")
    print(f"Classes: {', '.join(sorted(class_dirs.keys()))}")
    print("\nYou can now train with:")
    print("  python asl_project/train_model.py")
    print("="*60)

    return True

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = 'data/mendeley_downloaded'

    organize_mendeley_dataset(source)
