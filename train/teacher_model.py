"""EfficientNetB4 teacher model: paper-reported params (Adam, dropout 0.2)
plus reasonable defaults for choices the paper did not report (learning
rate, batch size, which blocks to unfreeze, pseudo-label confidence
threshold). See data/const.py for the actual values.

Uses tf.keras.applications.EfficientNetB4 (TensorFlow's own maintained
implementation) rather than the vendored efficientnet/ package in this repo:
that vendored copy predates Keras 3 and calls a legacy
tensorflow.python.keras.backend API that no longer recognizes Keras 3
tensors, so it crashes on any TF version that defaults to Keras 3
(TF >= 2.16 without the tf_keras legacy shim). tf.keras.applications uses the
same block1..block7 / top_conv / top_bn layer naming convention, so
UNFREEZE_BLOCK_PREFIXES still applies unchanged. The one tradeoff: it only
supports weights='imagenet' or None, not the vendored package's
'noisy-student' option.
"""
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import Adam

from data.const import (
    IMG_SIZE, TOP_DROPOUT_RATE, LEARNING_RATE, WEIGHTS, UNFREEZE_BLOCK_PREFIXES,
)


def build_teacher_model(model_name, num_classes, learning_rate=LEARNING_RATE,
                         top_dropout_rate=TOP_DROPOUT_RATE, weights=WEIGHTS) -> Model:
    """Build an EfficientNetB4 classifier with the entire backbone frozen
    (linear probe: only the new GAP/Dropout/Dense head trains). This is
    stage 1 of training -- see unfreeze_for_finetuning() for stage 2.
    """
    base_model = EfficientNetB4(
        include_top=False, weights=weights,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)

    model = Model(base_model.input, outputs, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def unfreeze_for_finetuning(model, learning_rate,
                             unfreeze_block_prefixes=UNFREEZE_BLOCK_PREFIXES) -> Model:
    """Stage 2: unfreeze `unfreeze_block_prefixes` (e.g. the last couple of
    EfficientNetB4 blocks) on an already linear-probed model, keeping the
    backbone's BatchNorm layers frozen (protects running stats at this batch
    size; the head's own 'head_bn' is excluded -- it keeps training), and
    recompile at a lower learning rate for fine-tuning.
    """
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization) and layer.name != 'head_bn':
            layer.trainable = False
            continue
        if layer.name.startswith(unfreeze_block_prefixes):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_finetune_model(model_name, num_classes, learning_rate, top_dropout_rate=TOP_DROPOUT_RATE,
                         weights=WEIGHTS) -> Model:
    """Original SLAN-tool paper recipe: one fine-tuning stage, no linear
    probe. ImageNet-pretrained EfficientNetB4 with the whole backbone
    trainable from the start, GAP + BatchNorm + Dropout(0.2) + softmax head,
    Adam. The backbone's BatchNorm layers stay frozen (inference mode), as
    Keras recommends when fine-tuning EfficientNet: their ImageNet running
    statistics would otherwise be overwritten by noisy per-batch statistics.
    """
    base_model = EfficientNetB4(include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)

    model = Model(base_model.input, outputs, name=model_name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
