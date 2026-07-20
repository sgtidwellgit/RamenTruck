"""Unit tests for ramentruck.tonkotsu."""

import numpy as np
import pytest
from tensorflow import keras

from ramentruck import tonkotsu


# ----------------------------------------------------------------------
# build_dense
# ----------------------------------------------------------------------


def test_build_dense_shapes():
    """Verify the model has the requested input/output shapes and depth."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[8, 4], output_dim=1)

    assert model.input_shape == (None, 4)
    assert model.output_shape == (None, 1)
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    assert len(dense_layers) == 3


def test_build_dense_applies_regularization_options():
    """Verify dropout, batch norm, and L2 layers are added when requested."""

    model = tonkotsu.build_dense(
        input_dim=4,
        hidden_layers=[8],
        output_dim=1,
        dropout_rate=0.3,
        l2_lambda=0.01,
        batch_norm=True,
    )

    layer_types = [type(layer) for layer in model.layers]
    assert keras.layers.Dropout in layer_types
    assert keras.layers.BatchNormalization in layer_types

    hidden_dense = model.get_layer("hidden_1")
    assert hidden_dense.kernel_regularizer is not None


def test_build_dense_no_regularization_by_default():
    """Verify no dropout or batch norm layers exist without opting in."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[8], output_dim=1)

    layer_types = [type(layer) for layer in model.layers]
    assert keras.layers.Dropout not in layer_types
    assert keras.layers.BatchNormalization not in layer_types
    assert model.get_layer("hidden_1").kernel_regularizer is None


# ----------------------------------------------------------------------
# simmer / SipResult
# ----------------------------------------------------------------------


def _toy_binary_data():
    rng = np.random.default_rng(42)
    X_train = rng.random((40, 4)).astype("float32")
    y_train = rng.integers(0, 2, size=(40,)).astype("float32")
    X_val = rng.random((10, 4)).astype("float32")
    y_val = rng.integers(0, 2, size=(10,)).astype("float32")
    return X_train, y_train, X_val, y_val


def test_simmer_returns_sip_result():
    """Verify simmer trains the model and returns populated result fields."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[4], output_dim=1)
    X_train, y_train, X_val, y_val = _toy_binary_data()

    result = tonkotsu.simmer(
        model, X_train, y_train, X_val, y_val, epochs=3, patience=2, verbose=0
    )

    assert isinstance(result, tonkotsu.SipResult)
    assert result.model is model
    assert result.train_time_s >= 0.0
    assert set(result.history_df.columns) == {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert result.best_epoch is not None
    assert isinstance(result.stopped_early, bool)


def test_simmer_early_stopping_requires_validation_data():
    """Verify early_stopping=True without validation data raises ValueError."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[4], output_dim=1)
    X_train, y_train, _, _ = _toy_binary_data()

    with pytest.raises(ValueError, match="early_stopping"):
        tonkotsu.simmer(model, X_train, y_train, epochs=2, verbose=0)


def test_simmer_without_early_stopping_allows_no_validation_data():
    """Verify training proceeds without validation data when early stopping is off."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[4], output_dim=1)
    X_train, y_train, _, _ = _toy_binary_data()

    result = tonkotsu.simmer(
        model, X_train, y_train, epochs=2, early_stopping=False, verbose=0
    )

    assert set(result.history_df.columns) == {"loss", "accuracy"}
    assert result.best_epoch is None
    assert result.stopped_early is False


def test_simmer_saves_checkpoint(tmp_path):
    """Verify a checkpoint file is written when checkpoint_path is given."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[4], output_dim=1)
    X_train, y_train, X_val, y_val = _toy_binary_data()
    checkpoint_path = tmp_path / "best_model.keras"

    tonkotsu.simmer(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,
        patience=1,
        checkpoint_path=checkpoint_path,
        verbose=0,
    )

    assert checkpoint_path.exists()


# ----------------------------------------------------------------------
# plot_history
# ----------------------------------------------------------------------


def test_plot_history_default_metrics():
    """Verify one subplot is produced per non-validation metric."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[4], output_dim=1)
    X_train, y_train, X_val, y_val = _toy_binary_data()
    result = tonkotsu.simmer(
        model, X_train, y_train, X_val, y_val, epochs=2, patience=1, verbose=0
    )

    fig = tonkotsu.plot_history(result)

    assert len(fig.get_axes()) == 2


def test_plot_history_specific_metrics():
    """Verify only the requested metrics are plotted."""

    model = tonkotsu.build_dense(input_dim=4, hidden_layers=[4], output_dim=1)
    X_train, y_train, X_val, y_val = _toy_binary_data()
    result = tonkotsu.simmer(
        model, X_train, y_train, X_val, y_val, epochs=2, patience=1, verbose=0
    )

    fig = tonkotsu.plot_history(result, metrics=["loss"])

    assert len(fig.get_axes()) == 1
    assert fig.get_axes()[0].get_ylabel() == "loss"


def test_plot_history_raises_without_metrics():
    """Verify a clear error is raised when there is nothing to plot."""

    class _EmptyHistory:
        history = {}

    class _FakeResult:
        history = _EmptyHistory()

    with pytest.raises(ValueError, match="No metrics"):
        tonkotsu.plot_history(_FakeResult())


# ----------------------------------------------------------------------
# EveryNEpochs
# ----------------------------------------------------------------------


def test_every_n_epochs_fires_on_schedule_and_last_epoch():
    """Verify the callback fires every N epochs and on the final epoch."""

    seen_epochs = []
    callback = tonkotsu.EveryNEpochs(n=2, fn=lambda epoch, logs: seen_epochs.append(epoch))
    callback.params = {"epochs": 5}

    for epoch in range(5):
        callback.on_epoch_end(epoch, logs={"loss": 0.1})

    assert seen_epochs == [0, 2, 4]


# ----------------------------------------------------------------------
# CNN family: residual blocks and build_resnet
# ----------------------------------------------------------------------


def test_residual_identity_block_preserves_shape():
    """Verify an identity block's output shape matches its input shape."""

    inputs = keras.Input(shape=(8, 8, 16))
    outputs = tonkotsu.residual_identity_block(
        inputs, kernel_size=3, filters=(4, 4, 16), stage=1, block="a"
    )

    assert outputs.shape[1:] == (8, 8, 16)


def test_residual_conv_block_downsamples_and_changes_channels():
    """Verify a conv block downsamples spatially and projects channel depth."""

    inputs = keras.Input(shape=(8, 8, 16))
    outputs = tonkotsu.residual_conv_block(
        inputs, kernel_size=3, filters=(4, 4, 32), stage=1, block="a", strides=2
    )

    assert outputs.shape[1:] == (4, 4, 32)


def test_build_resnet_default_preset_compiles():
    """Verify build_resnet with defaults returns a compiled model with the right output shape."""

    model = tonkotsu.build_resnet(
        input_shape=(32, 32, 3),
        classes=5,
        stage_filters=[(8, 8, 32), (16, 16, 64)],
        blocks_per_stage=[1, 1],
    )

    assert model.output_shape == (None, 5)
    assert model.optimizer is not None
    assert model.name == "resnet"


def test_build_resnet_mismatched_stage_lengths_raises():
    """Verify a ValueError is raised when stage_filters and blocks_per_stage disagree in length."""

    with pytest.raises(ValueError, match="same length"):
        tonkotsu.build_resnet(
            input_shape=(32, 32, 3),
            classes=5,
            stage_filters=[(8, 8, 32)],
            blocks_per_stage=[1, 1],
        )
