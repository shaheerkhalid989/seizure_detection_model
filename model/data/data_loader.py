import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyedflib import highlevel
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple
from config.settings import Config
from .preprocessor import EEGPreprocessor

class SeizureDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = EEGPreprocessor(config)
        self.class_mapping = config.data.event_mapping
        self.inverse_mapping = config.data.inverse_mapping
        
    def find_file_pairs(self, data_dir: Path) -> List[Tuple[Path, Path]]:
        """Find matching EDF/TSV file pairs"""
        edf_files = list(data_dir.glob("*_eeg.edf"))
        return [
            (edf, edf.with_name(edf.name.replace("_eeg.edf", "_events.tsv")))  # Fixed extra )
            for edf in edf_files
            if edf.with_name(edf.name.replace("_eeg.edf", "_events.tsv")).exists()
        ]
    
    def load_edf(self, edf_path: Path) -> Tuple[np.ndarray, datetime]:
        """Load EDF file and preprocess data"""
        signals, _, header = highlevel.read_edf(str(edf_path))
        signals = np.array(signals)[:self.config.data.n_channels]
        
        # Preprocess each channel
        processed = np.zeros_like(signals)
        for i in range(signals.shape[0]):
            processed[i] = self.preprocessor.preprocess(signals[i])
            
        return processed, header['startdate']
    
    # def process_file_pair(self, edf_path: Path, tsv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    #     """Process a single EDF/TSV file pair"""
    #     print(f"\nProcessing: {edf_path.name}")
    #     try:
    #         # Load data
    #         signals, start_time = self.load_edf(edf_path)
    #         print(f"EDF loaded - Signals shape: {signals.shape}")
            
    #         # Load labels
    #         labels_df = self.load_labels(tsv_path)
    #         print(f"TSV loaded - {len(labels_df)} annotations found")
            
    #         # Create epochs
    #         epochs, labels = self.create_epochs(signals, labels_df, start_time)
    #         print(f"Created {len(epochs)} epochs")
            
    #         return epochs, labels
            
    #     except Exception as e:
    #         print(f"Error processing {edf_path.name}: {str(e)}")
    #         return np.array([]), np.array([])

    def process_file_pair(self, edf_path: Path, tsv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Process EDF/TSV file pair into epochs and labels"""
        try:
            # 1. Verify file existence
            if not edf_path.exists() or not tsv_path.exists():
                return np.empty((0, 0, 0), dtype=np.float32), np.empty(0, dtype=np.int32)
            
            # 2. Load EDF data
            signals, signal_headers, header = highlevel.read_edf(str(edf_path))
            signals = np.array(signals[:self.config.data.n_channels], dtype=np.float32)
            
            # 3. Preprocess signals
            processed_signals = np.zeros_like(signals)
            for ch_idx in range(signals.shape[0]):
                processed_signals[ch_idx] = self.preprocessor.preprocess(signals[ch_idx])
            
            # 4. Load and process events
            event_df = pd.read_csv(tsv_path, sep='\t')
            event_df['eventType'] = event_df['eventType'].str.split(':').str[0]
            event_df['label'] = event_df['eventType'].map(self.config.data.event_mapping).fillna(0)
            
            # 5. Create epochs
            epoch_samples = self.config.data.fs * self.config.data.epoch_seconds
            n_epochs = processed_signals.shape[1] // epoch_samples
            
            epochs = np.zeros((n_epochs, epoch_samples, self.config.data.n_channels), dtype=np.float32)
            labels = np.zeros(n_epochs, dtype=np.int32)
            
            for epoch_idx in range(n_epochs):
                start = epoch_idx * epoch_samples
                end = start + epoch_samples
                epochs[epoch_idx] = processed_signals[:, start:end].T
                
                # Calculate label
                epoch_start = epoch_idx * self.config.data.epoch_seconds
                epoch_end = (epoch_idx + 1) * self.config.data.epoch_seconds
                labels[epoch_idx] = self._calculate_epoch_label(event_df, epoch_start, epoch_end)
            
            return epochs, labels
            
        except Exception as e:
            print(f"Error processing {edf_path.name}: {str(e)}")
            return np.empty((0, 0, 0), dtype=np.float32), np.empty(0, dtype=np.int32)
    
    def _calculate_epoch_label(self, event_df: pd.DataFrame, start: float, end: float) -> int:
        """Determine label for an epoch using majority voting"""
        overlaps = event_df[
            (event_df['onset'] < end) & 
            (event_df['onset'] + event_df['duration'] > start)
        ]
        if len(overlaps) == 0:
            return 0
        return int(overlaps['label'].mode()[0]) if len(overlaps['label'].mode()) > 0 else 0
    
    def load_labels(self, tsv_path: Path) -> pd.DataFrame:
        """Load and process event annotations"""
        df = pd.read_csv(tsv_path, sep='\t')
        df['eventType'] = df['eventType'].str.split(':').str[0]
        df['label'] = df['eventType'].map(self.config.data.event_mapping).fillna(0)
        return df
    
    def create_epochs(self, signals: np.ndarray, labels: pd.DataFrame, 
                     start_time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        # Convert to float32 immediately
        signals = signals.astype(np.float32)

        """Create labeled epochs with validation"""
        if signals.shape[1] < self.config.data.fs * self.config.data.epoch_seconds:
            print(f"⚠️ File too short: {signals.shape[1]} samples")
            return np.array([]), np.array([])
        
        """Create labeled epochs from continuous data"""
        epoch_samples = self.config.data.fs * self.config.data.epoch_seconds
        n_epochs = signals.shape[1] // epoch_samples
        
        epoch_data = np.zeros((n_epochs, epoch_samples, self.config.data.n_channels))
        epoch_labels = np.zeros(n_epochs, dtype=np.int32)
        timestamps = []
        
        for i in range(n_epochs):
            start = i * epoch_samples
            end = start + epoch_samples
            epoch_data[i] = signals[:, start:end].T
            epoch_time = start_time + timedelta(seconds=i*self.config.data.epoch_seconds)
            
            # Labeling logic
            epoch_start = i * self.config.data.epoch_seconds
            epoch_end = (i+1) * self.config.data.epoch_seconds
            epoch_labels[i] = any(
                (row['onset'] < epoch_end) and (row['onset'] + row['duration'] > epoch_start)
                for _, row in labels.iterrows()
            )
            
        return epoch_data.astype(np.float32), epoch_labels.astype(np.int32)