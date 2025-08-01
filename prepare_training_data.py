import glob
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)

# Configuration
PROCESSED_DATA_DIR = Path("./Processed_data")
TRAINING_DATA_DIR = Path("./training_data")
NEW_TRAINING_DATA_DIR = Path("./new_training_data")

# Files that should use existing test splits
FILES_WITH_TEST_SPLITS = {
    "flores.jsonl",
    "mintaka.jsonl",
    "mldr.jsonl", 
    "mlqa.jsonl",
    "amazon_review.jsonl",
    "squad.jsonl",
    "stackoverflow.jsonl"
}

# Files to exclude from final training data
EXCLUDE_FILES = {'xnli', 'Wikireranking', 'samanantar_language_classification'}

# Files containing English data (use only half)
ENGLISH_DATA_FILES = {'amazon_review', 'crosssum_english_english', 'eli5', 'squad'}

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def create_train_val_split(file_path: Path, output_dir: Path) -> None:
    """Create train/validation split for a dataset."""
    data = load_jsonl(file_path)
    if not data:
        return
    
    random.shuffle(data)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    dataset_name = file_path.stem
    dataset_dir = output_dir / dataset_name
    
    save_jsonl(train_data, dataset_dir / "train.jsonl")
    save_jsonl(val_data, dataset_dir / "val.jsonl")
    print(f"Created train/val split for {dataset_name}: {len(train_data)} train, {len(val_data)} val")

def use_existing_test_split(file_path: Path, output_dir: Path) -> None:
    """Use existing test file as validation set."""
    # Load training data
    train_data = load_jsonl(file_path)
    if not train_data:
        return
    
    # Find corresponding test file
    test_file_path = file_path.with_name(f"{file_path.stem}_test{file_path.suffix}")
    val_data = load_jsonl(test_file_path)
    
    dataset_name = file_path.stem
    dataset_dir = output_dir / dataset_name
    
    save_jsonl(train_data, dataset_dir / "train.jsonl")
    save_jsonl(val_data, dataset_dir / "val.jsonl")
    print(f"Used existing test split for {dataset_name}: {len(train_data)} train, {len(val_data)} val")

def process_individual_datasets() -> None:
    """Process individual datasets into train/val splits."""
    print("Processing individual datasets...")
    
    # Get all JSONL files in processed data directory
    data_files = list(PROCESSED_DATA_DIR.glob("*.jsonl"))
    
    # Filter out test files
    data_files = [f for f in data_files if "_test" not in f.name]
    
    for file_path in data_files:
        if file_path.name in FILES_WITH_TEST_SPLITS or "crosssum" in file_path.name:
            use_existing_test_split(file_path, TRAINING_DATA_DIR)
        else:
            create_train_val_split(file_path, TRAINING_DATA_DIR)

def load_dataset_data(file_path: Path, dataset_name: str) -> List[Dict[str, Any]]:
    """Load data for a specific dataset with special handling for English datasets."""
    data = load_jsonl(file_path)
    
    if dataset_name in ENGLISH_DATA_FILES:
        # Use only half of English data
        half_length = len(data) // 2
        return data[:half_length]
    
    return data

def combine_datasets() -> None:
    """Combine all individual datasets into final training data."""
    print("Combining datasets...")
    
    train_files = list(TRAINING_DATA_DIR.glob("*/train.jsonl"))
    val_files = list(TRAINING_DATA_DIR.glob("*/val.jsonl"))
    
    all_train_data = []
    all_val_data = []
    
    # Process training files
    for file_path in train_files:
        dataset_name = file_path.parent.name
        
        if dataset_name in EXCLUDE_FILES:
            continue
        
        print(f"Loading training data from {dataset_name}")
        data = load_dataset_data(file_path, dataset_name)
        all_train_data.extend(data)
    
    # Process validation files
    for file_path in val_files:
        dataset_name = file_path.parent.name
        
        if dataset_name in EXCLUDE_FILES:
            continue
        
        data = load_jsonl(file_path)
        all_val_data.extend(data)
    
    # Shuffle combined data
    random.shuffle(all_train_data)
    random.shuffle(all_val_data)
    
    # Save final combined datasets
    save_jsonl(all_train_data, NEW_TRAINING_DATA_DIR / "train_data.jsonl")
    save_jsonl(all_val_data, NEW_TRAINING_DATA_DIR / "val_data.jsonl")
    
    print(f"Final dataset created: {len(all_train_data)} train samples, {len(all_val_data)} val samples")

def main() -> None:
    """Main function to orchestrate the data processing pipeline."""
    print("Starting data processing pipeline...")
    
    # Create output directories
    TRAINING_DATA_DIR.mkdir(exist_ok=True)
    NEW_TRAINING_DATA_DIR.mkdir(exist_ok=True)
    
    # Process individual datasets
    process_individual_datasets()
    
    # Combine all datasets
    combine_datasets()
    
    print("Data processing pipeline completed successfully!")

if __name__ == "__main__":
    main()