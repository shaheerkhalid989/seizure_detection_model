from config.settings import Config
from data.data_loader import SeizureDataLoader
from models.model_builder import ModelBuilder
from models.callbacks import SeizureCallbacks
from data.tfrecords_io import TFRecordHandler
from evaluation.reporter import ReportGenerator
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

def create_dataset(tfrecord_files: list, config: Config, is_training: bool = True) -> tf.data.Dataset:
    """Create optimized TFRecord dataset pipeline"""
    dataset = (
        tf.data.Dataset.from_tensor_slices(tfrecord_files)
        .interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .map(
            TFRecordHandler.parse_tfrecord,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(config.model.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    if is_training:
        dataset = dataset.shuffle(config.model.shuffle_buffer)
    return dataset

def main():
    config = Config()
    data_loader = SeizureDataLoader(config)
    model_builder = ModelBuilder(config)
    handler = TFRecordHandler(config)

    # Prepare Data Directories
    train_tfrecord_dir = config.paths.processed_train / "tfrecords"
    test_tfrecord_dir = config.paths.processed_test / "tfrecords"
    
    for directory in [train_tfrecord_dir, test_tfrecord_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Create Training TFRecords
    if not list(train_tfrecord_dir.glob("*.tfrecord")):
        print("Creating Training TFRecords...")
        train_pairs = data_loader.find_file_pairs(config.paths.train_data)
        handler.create_tfrecords(train_pairs, train_tfrecord_dir)

    # Create Test TFRecords
    if not list(test_tfrecord_dir.glob("*.tfrecord")):
        print("Creating Test TFRecords...")
        test_pairs = data_loader.find_file_pairs(config.paths.test_data)
        handler.create_tfrecords(test_pairs, test_tfrecord_dir)

    # Create Datasets
    train_files = [str(p) for p in train_tfrecord_dir.glob("*.tfrecord")]
    test_files = [str(p) for p in test_tfrecord_dir.glob("*.tfrecord")]
    
    train_dataset = create_dataset(train_files, config, is_training=True)
    test_dataset = create_dataset(test_files, config, is_training=False)

    train_labels = np.array([
        y for x, y in train_dataset.unbatch().as_numpy_iterator()
    ])

    if len(train_labels) == 0:
        raise ValueError("No training labels found. Check your dataset.")

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Build and Train Model
    model = model_builder.build_hybrid_model()
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config.model.epochs,
        callbacks=SeizureCallbacks(config).callbacks,
        class_weight = class_weight_dict
    )

    # Evaluate Model
    print("\nEvaluating on test set...")
    test_preds = model.predict(test_dataset)
    predicted_classes = test_preds.argmax(axis=1)  # Convert to class indices

    # Get True Labels
    # Get True Labels
    test_labels = np.concatenate(
        [np.array(y)[np.newaxis] for x, y in test_dataset.unbatch().as_numpy_iterator()],
        axis=0
    )

    # Generate Reports
    reporter = ReportGenerator(config)
    print("\nGenerating classification report...")
    reporter.save_report(test_labels, predicted_classes)
    
    # Save Model
    model.save(config.paths.model_save)
    print(f"\nModel saved to {config.paths.model_save}")

if __name__ == "__main__":
    main()