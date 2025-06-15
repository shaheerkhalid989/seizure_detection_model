from tensorflow.keras import callbacks
from config.settings import Config

class SeizureCallbacks:
    def __init__(self, config: Config):
        self.config = config.model
        self.callbacks = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.patience,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=str(config.paths.model_save),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]