import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.proj = layers.Dense(d_model)
        self.pos_encoding = positional_encoding(seq_len, d_model)

    def call(self, x):
        x = self.proj(x)
        x = x * tf.math.sqrt(tf.cast(self.d_model, x.dtype))
        pos = tf.cast(self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :], x.dtype)
        return x + pos

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                             dropout=dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.drop = layers.Dropout(dropout)
        self._last_scores = None

    def call(self, x, training=False, return_attention=False):
        attn_out, scores = self.att(x, x, training=training, return_attention_scores=True)
        self._last_scores = scores
        x = self.norm1(x + self.drop(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + ffn_out)
        if return_attention:
            return x, scores
        return x

def build_transformer(seq_len, n_features, d_model=64, num_heads=4, ff_dim=128,
                      num_layers=2, dropout=0.1, learning_rate=1e-3):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = PositionalEmbedding(seq_len, d_model)(inputs)
    x = layers.Dropout(dropout)(x)
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        a = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(-a * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss

def build_transformer_focal(*args, gamma=2.0, alpha=0.25, learning_rate=1e-3, **kwargs):
    kwargs.pop("compile_loss", None)
    model = build_transformer(*args, learning_rate=learning_rate, **kwargs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=focal_loss(gamma, alpha),
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model
