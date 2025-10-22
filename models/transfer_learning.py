"""Transfer learning helpers (placeholder implementations)."""
from tensorflow.keras import applications, layers, models


def build_mobilenetv2(input_shape=(128, 128, 3), num_classes=43, freeze_base=True):
    base = applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    if freeze_base:
        base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model
