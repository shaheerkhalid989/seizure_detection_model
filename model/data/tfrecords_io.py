import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .data_loader import SeizureDataLoader  # Import your data loader

class TFRecordHandler:
    def __init__(self, config):
        self.config = config
        self.data_loader = SeizureDataLoader(config)
        
    def create_tfrecords(self, pairs, output_dir: Path, examples_per_file: int = 5000):
        """Convert EDF/TSV pairs to TFRecord format"""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_count = 0
        writer = None
        
        with tqdm(total=len(pairs), desc="Creating TFRecords") as pbar:
            for edf_path, tsv_path in pairs:
                try:
                    epochs, labels = self.data_loader.process_file_pair(edf_path, tsv_path)
                    if epochs.size == 0:
                        continue
                        
                    for idx in range(0, len(epochs), examples_per_file):
                        if writer: writer.close()
                        file_path = output_dir / f"data_{file_count:04d}.tfrecord"
                        writer = tf.io.TFRecordWriter(str(file_path))
                        file_count += 1
                        
                        batch = zip(
                            epochs[idx:idx+examples_per_file],
                            labels[idx:idx+examples_per_file]
                        )
                        
                        for epoch, label in batch:
                            example = self._create_example(epoch, label)
                            writer.write(example.SerializeToString())
                            
                except Exception as e:
                    print(f"\nError processing {edf_path.name}: {str(e)}")
                finally:
                    pbar.update(1)
                    
            if writer: writer.close()
    
    def _create_example(self, epoch: np.ndarray, label: int):
        """Ensure consistent shape storage"""
        return tf.train.Example(features=tf.train.Features(feature={
            'eeg': tf.train.Feature(float_list=tf.train.FloatList(value=epoch.flatten())),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=epoch.shape))
        }))
    
    @staticmethod
    def parse_tfrecord(example: tf.train.Example):
        """Parse TFRecord example with explicit shape handling"""
        feature_description = {
            'eeg': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'shape': tf.io.FixedLenFeature([2], tf.int64)  # Fixed shape definition
        }
        parsed = tf.io.parse_single_example(example, feature_description)
        return (
            tf.reshape(parsed['eeg'], parsed['shape']),  # Use known shape
            parsed['label']
        )

    @staticmethod
    def read_tfrecord(example):
        """Parse TFRecord example"""
        feature_description = {
            'eeg': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example, feature_description)
        return tf.reshape(parsed['eeg'], (1024, 19)), parsed['label']