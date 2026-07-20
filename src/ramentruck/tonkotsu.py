"""Deep learning interface for RamenTruck.

Requires TensorFlow, an optional dependency. Install with
``pip install ramentruck[deep]``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
except ImportError as exc:
    raise ImportError(
        "tonkotsu requires TensorFlow. Install it with: pip install ramentruck[deep]"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "tonkotsu requires matplotlib. Install it with: pip install ramentruck[deep]"
    ) from exc


DEFAULT_METRICS = ["accuracy"]


@dataclass(frozen=True)
class SipResult:
    """Result returned by :func:`simmer` after training a Keras model."""

    model: keras.Model
    history: keras.callbacks.History
    history_df: pd.DataFrame
    best_epoch: int | None
    stopped_early: bool
    train_time_s: float


def build_dense(
    input_dim: int,
    hidden_layers: Iterable[int],
    output_dim: int,
    *,
    activation: str = "relu",
    output_activation: str = "sigmoid",
    dropout_rate: float = 0.0,
    l2_lambda: float = 0.0,
    batch_norm: bool = False,
) -> keras.Model:
    """
    Build a dense feedforward network using the Keras functional API.

    Parameters
    ----------
    input_dim
        Number of input features.
    hidden_layers
        Number of units in each hidden layer, in order.
    output_dim
        Number of output units.
    activation
        Activation function used by hidden layers.
    output_activation
        Activation function used by the output layer. Use ``"sigmoid"`` for
        binary classification, ``"softmax"`` for multi-class classification,
        or ``"linear"`` for regression.
    dropout_rate
        Dropout rate applied after each hidden layer. Disabled when ``0.0``.
    l2_lambda
        L2 regularization strength applied to each hidden layer's kernel.
        Disabled when ``0.0``.
    batch_norm
        Whether to apply batch normalization after each hidden layer.

    Returns
    -------
    keras.Model
        An uncompiled Keras model. Compile it directly or pass it to
        :func:`simmer`, which compiles it before training.
    """

    regularizer = regularizers.l2(l2_lambda) if l2_lambda > 0 else None

    inputs = keras.Input(shape=(input_dim,), name="dense_input")
    x = inputs

    for index, units in enumerate(hidden_layers, start=1):
        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            name=f"hidden_{index}",
        )(x)

        if batch_norm:
            x = layers.BatchNormalization(name=f"batch_norm_{index}")(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{index}")(x)

    outputs = layers.Dense(output_dim, activation=output_activation, name="dense_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="dense_model")


def simmer(
    model: keras.Model,
    X_train: Any,
    y_train: Any,
    X_val: Any | None = None,
    y_val: Any | None = None,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    optimizer: str | keras.optimizers.Optimizer = "adam",
    loss: str = "binary_crossentropy",
    metrics: list[str] | None = None,
    early_stopping: bool = True,
    patience: int = 10,
    checkpoint_path: str | Path | None = None,
    callbacks: list[Callback] | None = None,
    verbose: int = 1,
) -> SipResult:
    """
    Compile and train a Keras model, tracking timing and stopping behavior.

    Parameters
    ----------
    model
        An uncompiled (or already compiled) Keras model. It is (re)compiled
        with ``optimizer``, ``loss``, and ``metrics`` before training.
    X_train, y_train
        Training data.
    X_val, y_val
        Optional validation data. Required when ``early_stopping=True``.
    epochs
        Maximum number of training epochs.
    batch_size
        Number of samples per gradient update.
    optimizer, loss, metrics
        Passed to ``model.compile``.
    early_stopping
        Whether to stop training early on stalled validation loss.
    patience
        Number of epochs with no improvement before early stopping fires.
    checkpoint_path
        Optional path to save the best-performing model weights.
    callbacks
        Additional Keras callbacks to run alongside the built-in ones.
    verbose
        Keras verbosity level passed to ``model.fit``.

    Returns
    -------
    SipResult
        The trained model, its training history, and derived metadata.
    """

    metric_names = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_names)

    has_validation = (X_val is not None) and (y_val is not None)
    validation_data = (X_val, y_val) if has_validation else None

    fit_callbacks = list(callbacks) if callbacks is not None else []
    early_stop_callback: EarlyStopping | None = None

    if early_stopping:
        if not has_validation:
            raise ValueError(
                "early_stopping requires both X_val and y_val to be provided."
            )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
        fit_callbacks.append(early_stop_callback)

    if checkpoint_path is not None:
        fit_callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss" if has_validation else "loss",
                save_best_only=True,
            )
        )

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=fit_callbacks,
        verbose=verbose,
    )
    train_time_s = time.perf_counter() - start

    stopped_early = bool(
        early_stop_callback is not None and early_stop_callback.stopped_epoch > 0
    )

    best_epoch = None
    if "val_loss" in history.history:
        best_epoch = int(np.argmin(history.history["val_loss"]))

    return SipResult(
        model=model,
        history=history,
        history_df=pd.DataFrame(history.history),
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        train_time_s=train_time_s,
    )


def plot_history(
    result: SipResult,
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Plot training vs. validation curves for each tracked metric.

    Parameters
    ----------
    result
        A :class:`SipResult` returned by :func:`simmer`.
    metrics
        Metric keys to plot. Defaults to every non-validation key in the
        training history (e.g. ``["loss", "accuracy"]``).
    figsize
        Overall figure size passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with one subplot per metric, each showing training vs.
        validation curves (when validation data was used).
    """

    history_dict = result.history.history

    if metrics is None:
        metrics = sorted(key for key in history_dict if not key.startswith("val_"))

    if not metrics:
        raise ValueError("No metrics available to plot in this training history.")

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)

    for ax, metric_key in zip(axes[0], metrics):
        _plot_metric(ax, history_dict, metric_key)

    fig.tight_layout()
    return fig


def _plot_metric(ax: Any, history_dict: dict[str, list[float]], metric_key: str) -> None:
    """Plot a single metric's training and validation curves onto an axis."""

    train_key = metric_key
    val_key = f"val_{metric_key}"
    epochs = np.arange(1, len(history_dict[train_key]) + 1)

    ax.plot(epochs, history_dict[train_key], "r-", label=f"Training {metric_key}")

    if val_key in history_dict:
        ax.plot(epochs, history_dict[val_key], "b-", label=f"Validation {metric_key}")

    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric_key)
    ax.grid()
    ax.legend()


class EveryNEpochs(Callback):
    """Keras callback that invokes a user function every N epochs."""

    def __init__(self, n: int, fn: Callable[[int, dict[str, float]], None]) -> None:
        """
        Initialize the callback.

        Parameters
        ----------
        n
            Number of epochs between invocations of ``fn``.
        fn
            Called with ``(epoch, logs)`` at the end of every Nth epoch and
            at the final epoch.
        """

        super().__init__()
        self.n = n
        self.fn = fn

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Invoke ``fn`` when the current epoch is a reporting epoch."""

        logs = logs or {}
        last_epoch = self.params.get("epochs", epoch + 1) - 1

        if epoch % self.n == 0 or epoch == last_epoch:
            self.fn(epoch, logs)


# ----------------------------------------------------------------------
# CNN family: composable residual blocks and the build_resnet preset
# ----------------------------------------------------------------------


DEFAULT_RESNET_STAGE_FILTERS: list[tuple[int, int, int]] = [
    (64, 64, 256),
    (128, 128, 512),
    (256, 256, 1024),
    (512, 512, 2048),
]

DEFAULT_RESNET_BLOCKS_PER_STAGE: list[int] = [3, 4, 6, 3]


def _kernel_initializer() -> keras.initializers.Initializer:
    """Return a seeded Glorot-uniform initializer for convolution kernels."""

    return keras.initializers.GlorotUniform(seed=0)


def residual_identity_block(
    x: Any,
    kernel_size: int,
    filters: tuple[int, int, int],
    stage: int,
    block: str,
) -> Any:
    """
    Build a ResNet identity block.

    Three convolutions plus a shortcut connection straight back to the
    block's input. Output shape matches input shape, so this block does not
    change tensor dimensions and can be stacked freely.

    Parameters
    ----------
    x
        Input tensor.
    kernel_size
        Kernel size used by the middle convolution.
    filters
        Number of filters for each of the three convolutions, as
        ``(filters1, filters2, filters3)``.
    stage
        Network stage number, used to build unique layer names.
    block
        Block identifier within the stage, used to build unique layer names.

    Returns
    -------
    tensor
        Output tensor after the residual connection and final activation.
    """

    filters1, filters2, filters3 = filters
    name_prefix = f"stage{stage}_block{block}"
    shortcut = x

    x = layers.Conv2D(filters1, 1, kernel_initializer=_kernel_initializer(), name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act1")(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer=_kernel_initializer(),
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act2")(x)

    x = layers.Conv2D(filters3, 1, kernel_initializer=_kernel_initializer(), name=f"{name_prefix}_conv3")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)

    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_out")(x)

    return x


def residual_conv_block(
    x: Any,
    kernel_size: int,
    filters: tuple[int, int, int],
    stage: int,
    block: str,
    *,
    strides: int = 2,
) -> Any:
    """
    Build a ResNet convolutional block.

    Three convolutions plus a projected shortcut, used when the block needs
    to change the tensor shape (downsampling between stages).

    Parameters
    ----------
    x
        Input tensor.
    kernel_size
        Kernel size used by the middle convolution.
    filters
        Number of filters for each of the three convolutions, as
        ``(filters1, filters2, filters3)``.
    stage
        Network stage number, used to build unique layer names.
    block
        Block identifier within the stage, used to build unique layer names.
    strides
        Stride applied to the first convolution and the projected shortcut.

    Returns
    -------
    tensor
        Output tensor after the residual connection and final activation.
    """

    filters1, filters2, filters3 = filters
    name_prefix = f"stage{stage}_block{block}"
    shortcut = x

    x = layers.Conv2D(
        filters1,
        1,
        strides=strides,
        kernel_initializer=_kernel_initializer(),
        name=f"{name_prefix}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act1")(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer=_kernel_initializer(),
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act2")(x)

    x = layers.Conv2D(filters3, 1, kernel_initializer=_kernel_initializer(), name=f"{name_prefix}_conv3")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)

    shortcut = layers.Conv2D(
        filters3,
        1,
        strides=strides,
        kernel_initializer=_kernel_initializer(),
        name=f"{name_prefix}_shortcut_conv",
    )(shortcut)
    shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_out")(x)

    return x


def build_resnet(
    input_shape: tuple[int, int, int],
    classes: int,
    *,
    stage_filters: list[tuple[int, int, int]] | None = None,
    blocks_per_stage: list[int] | None = None,
    output_activation: str = "softmax",
    optimizer: str = "rmsprop",
    loss: str = "categorical_crossentropy",
) -> keras.Model:
    """
    Build and compile a ResNet-style convolutional network.

    Calling this with only ``input_shape`` and ``classes`` returns a working,
    compiled ResNet50-equivalent model - understanding residual blocks is not
    required. ``stage_filters`` and ``blocks_per_stage`` are overridable for a
    shallower or deeper variant.

    Parameters
    ----------
    input_shape
        Shape of a single input image, e.g. ``(224, 224, 3)``.
    classes
        Number of output classes for the final softmax layer.
    stage_filters
        Per-stage ``(filters1, filters2, filters3)`` tuples. Defaults to the
        standard ResNet50 filter progression.
    blocks_per_stage
        Number of residual blocks in each stage. The first block of each
        stage is a downsampling :func:`residual_conv_block`; the rest are
        :func:`residual_identity_block`. Defaults to ``[3, 4, 6, 3]``
        (ResNet50). Must be the same length as ``stage_filters``.
    output_activation
        Activation for the final dense layer.
    optimizer, loss
        Passed to ``model.compile``. Metrics default to accuracy.

    Returns
    -------
    keras.Model
        A compiled Keras model named ``"resnet"``.

    Raises
    ------
    ValueError
        If ``stage_filters`` and ``blocks_per_stage`` have different lengths.
    """

    stage_filters = stage_filters if stage_filters is not None else DEFAULT_RESNET_STAGE_FILTERS
    blocks_per_stage = (
        blocks_per_stage if blocks_per_stage is not None else DEFAULT_RESNET_BLOCKS_PER_STAGE
    )

    if len(stage_filters) != len(blocks_per_stage):
        raise ValueError(
            "stage_filters and blocks_per_stage must have the same length; "
            f"got {len(stage_filters)} and {len(blocks_per_stage)}."
        )

    inputs = keras.Input(shape=input_shape, name="resnet_input")

    x = layers.ZeroPadding2D(3, name="stem_pad")(inputs)
    x = layers.Conv2D(64, 7, strides=2, kernel_initializer=_kernel_initializer(), name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_act")(x)
    x = layers.MaxPooling2D(3, strides=2, name="stem_pool")(x)

    for stage_index, (filters, num_blocks) in enumerate(
        zip(stage_filters, blocks_per_stage), start=1
    ):
        stage_strides = 1 if stage_index == 1 else 2
        x = residual_conv_block(
            x, kernel_size=3, filters=filters, stage=stage_index, block="a", strides=stage_strides
        )

        for block_index in range(1, num_blocks):
            block_letter = chr(ord("a") + block_index)
            x = residual_identity_block(
                x, kernel_size=3, filters=filters, stage=stage_index, block=block_letter
            )

    x = layers.GlobalAveragePooling2D(name="pool_out")(x)
    outputs = layers.Dense(
        classes,
        activation=output_activation,
        kernel_initializer=_kernel_initializer(),
        name="resnet_output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet")
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model
