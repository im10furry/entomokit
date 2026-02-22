"""Dataset splitting utilities."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DatasetSplitter:
    """Split datasets into train/test/unknown classes."""
    
    def __init__(
        self,
        raw_image_csv: str,
        out_dir: str = "datasets",
        seed: int = 42
    ):
        self.raw_image_csv = raw_image_csv
        self.out_dir = Path(out_dir)
        self.seed = seed
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.class_count_dir = self.out_dir / "class_count"
        self.class_count_dir.mkdir(exist_ok=True)
        
        self.all_data: Optional[pd.DataFrame] = None
        self.class_counts: Optional[pd.DataFrame] = None
        self.total_samples = 0
    
    def load_data(self) -> None:
        """Load and validate input CSV."""
        if not os.path.exists(self.raw_image_csv):
            raise FileNotFoundError(f"Input file not found: {self.raw_image_csv}")
        
        self.all_data = pd.read_csv(self.raw_image_csv)
        self.total_samples = len(self.all_data)
        
        class_counts = self.all_data['label'].value_counts().reset_index()
        class_counts.columns = ['label', 'count']
        self.class_counts = class_counts
        
        class_counts.to_csv(self.class_count_dir / 'class.count', index=False)
        
        print(f"Loaded {self.total_samples} samples, {len(class_counts)} classes.")
    
    def split_ratio_mode(
        self,
        unknown_test_classes_ratio: float = 0.0,
        known_test_classes_ratio: float = 0.1
    ) -> dict:
        """Split dataset using ratio-based mode."""
        if self.all_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        known_data = self.all_data.copy()
        test_unknown_data = pd.DataFrame()
        test_known_data = pd.DataFrame()
        train_data = pd.DataFrame()
        
        if unknown_test_classes_ratio > 0:
            target_unknown = self.total_samples * unknown_test_classes_ratio
            shuffled_classes = self.class_counts.sample(frac=1, random_state=self.seed)
            accumulated = 0
            unknown_labels = []
            
            for _, row in shuffled_classes.iterrows():
                unknown_labels.append(row['label'])
                accumulated += row['count']
                if accumulated >= target_unknown:
                    break
            
            test_unknown_data = self.all_data[self.all_data.label.isin(unknown_labels)].reset_index(drop=True)
            known_data = self.all_data[~self.all_data.label.isin(unknown_labels)].reset_index(drop=True)
            
            test_unknown_data.to_csv(self.out_dir / 'test.unknown.csv', index=False)
            test_unknown_data.label.value_counts().to_csv(
                self.class_count_dir / 'class.test.unknown.count', index=False
            )
        else:
            print("Skip unknown ratio split, ratio=0")
        
        if len(known_data) > 0:
            train_idx = known_data.groupby('label', group_keys=False).sample(
                frac=1 - known_test_classes_ratio, random_state=self.seed
            ).index
            train_data = known_data.loc[train_idx].reset_index(drop=True)
            test_known_data = known_data.drop(train_idx).reset_index(drop=True)
            
            train_data.to_csv(self.out_dir / 'train.csv', index=False)
            test_known_data.to_csv(self.out_dir / 'test.known.csv', index=False)
            
            train_data.label.value_counts().to_csv(
                self.class_count_dir / 'class.train.count', index=False
            )
            test_known_data.label.value_counts().to_csv(
                self.class_count_dir / 'class.test.known.count', index=False
            )
        else:
            print("No known data after unknown split.")
        
        return {
            'test_unknown': len(test_unknown_data),
            'test_known': len(test_known_data),
            'train': len(train_data)
        }
    
    def split_count_mode(
        self,
        unknown_test_classes_count: int = 0,
        known_test_classes_count: int = 0,
        min_count_per_class: int = 0,
        max_count_per_class: Optional[int] = None
    ) -> dict:
        """Split dataset using count-based mode."""
        if self.all_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.all_data.copy()
        np.random.seed(self.seed)
        
        test_unknown_data = pd.DataFrame()
        test_known_data = pd.DataFrame()
        train_data = pd.DataFrame()
        
        if unknown_test_classes_count > 0:
            target_unknown = unknown_test_classes_count
            shuffled_classes = self.class_counts.sample(frac=1, random_state=self.seed)
            accumulated = 0
            unknown_labels = []
            
            for _, row in shuffled_classes.iterrows():
                unknown_labels.append(row['label'])
                accumulated += row['count']
                if accumulated >= target_unknown:
                    break
            
            test_unknown_data = df[df.label.isin(unknown_labels)].reset_index(drop=True)
            df = df[~df.label.isin(unknown_labels)].reset_index(drop=True)
            
            test_unknown_data.to_csv(self.out_dir / 'test.unknown.csv', index=False)
            test_unknown_data.label.value_counts().to_csv(
                self.class_count_dir / 'class.test.unknown.count', index=False
            )
        
        if known_test_classes_count > 0:
            target_known = known_test_classes_count
            known_test_samples = []
            accumulated = 0
            
            for lbl, group in df.groupby('label'):
                group = group.sample(frac=1, random_state=self.seed)
                for idx, row in group.iterrows():
                    known_test_samples.append(row)
                    accumulated += 1
                    if accumulated >= target_known:
                        break
                if accumulated >= target_known:
                    break
            
            test_known_data = pd.DataFrame(known_test_samples)
            remain_df = df.drop(test_known_data.index, errors='ignore').reset_index(drop=True)
        else:
            test_known_data = pd.DataFrame()
            remain_df = df.copy()
        
        if len(test_known_data) > 0:
            test_known_data.to_csv(self.out_dir / 'test.known.csv', index=False)
            test_known_data.label.value_counts().to_csv(
                self.class_count_dir / 'class.test.known.count', index=False
            )
        
        train_rows = []
        for lbl, group in remain_df.groupby('label'):
            n = len(group)
            if n < min_count_per_class:
                continue
            if max_count_per_class is not None:
                take = min(n, max_count_per_class)
            else:
                take = n
            sampled = group.sample(n=take, random_state=self.seed)
            train_rows.append(sampled)
        
        if len(train_rows) > 0:
            train_data = pd.concat(train_rows).reset_index(drop=True)
            train_data.to_csv(self.out_dir / 'train.csv', index=False)
            train_data.label.value_counts().to_csv(
                self.class_count_dir / 'class.train.count', index=False
            )
        
        return {
            'test_unknown': len(test_unknown_data),
            'test_known': len(test_known_data),
            'train': len(train_data)
        }
    
    def split(
        self,
        mode: str = 'ratio',
        unknown_test_ratio: float = 0.0,
        known_test_ratio: float = 0.1,
        unknown_test_count: int = 0,
        known_test_count: int = 0,
        min_count_per_class: int = 0,
        max_count_per_class: Optional[int] = None
    ) -> dict:
        """Split dataset into train/test sets.
        
        Args:
            mode: Split mode - 'ratio' or 'count'
            unknown_test_ratio: Ratio of samples for unknown test classes
            known_test_ratio: Ratio of known class samples for test
            unknown_test_count: Number of samples for unknown test classes
            known_test_count: Number of samples for known test classes
            min_count_per_class: Minimum samples per class for train
            max_count_per_class: Maximum samples per class for train
        
        Returns:
            Dictionary with split statistics
        """
        self.load_data()
        
        if mode == 'ratio':
            return self.split_ratio_mode(
                unknown_test_classes_ratio=unknown_test_ratio,
                known_test_classes_ratio=known_test_ratio
            )
        elif mode == 'count':
            return self.split_count_mode(
                unknown_test_classes_count=unknown_test_count,
                known_test_classes_count=known_test_count,
                min_count_per_class=min_count_per_class,
                max_count_per_class=max_count_per_class
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'ratio' or 'count'.")
