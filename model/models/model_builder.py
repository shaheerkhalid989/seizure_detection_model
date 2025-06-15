import tensorflow as tf
from tensorflow.keras import layers, regularizers
from config.settings import Config

class ModelBuilder:
    def __init__(self, config: Config):
        self.config = config.model  # Direct access to ModelConfig
    
    def build_hybrid_model(self) -> tf.keras.Model:
        """Build CNN-LSTM hybrid model with correct config access"""
        inputs = tf.keras.Input(
            shape=self.config.input_shape,  # Direct access without .model
            batch_size=self.config.batch_size,
            name='eeg_input'
        )
        
        # CNN Block
        x = layers.Conv1D(8, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(4)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # LSTM Block
        x = layers.LSTM(16, return_sequences=False, 
                       kernel_regularizer=regularizers.l2(self.config.l2_reg))(x)
        
        # Classification Head
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        outputs = layers.Dense(self.config.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model