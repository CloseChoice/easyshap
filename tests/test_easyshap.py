import numpy as np
from easyshap import eshap_compare
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pytest
import xarray as xr


def test_eshap_compare_multiple_output():
    tf = pytest.importorskip("tensorflow")
    models = pytest.importorskip("tensorflow.keras.models")
    layers = pytest.importorskip("tensorflow.keras.layers")

    train_images = np.ones(shape=(10, 32, 32, 3))
    train_labels = np.random.randint(0, 2, size=10)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    # this model has multiple outputs, one for each class
    model.add(layers.Dense(2))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        epochs=1,
    )
    train_images = xr.DataArray(
        train_images,
        dims=("observations", "pixel_x", "pixel_y", "rgb"),
        coords={
            "pixel_x": np.arange(32),
            "pixel_y": np.arange(32),
            "rgb": ["r", "g", "b"],
        },
    )
    result = eshap_compare(train_images, model, model, explainer="deep")
    assert result.sum().values == np.array(0.0)


def test_eshap_compare_single_output():
    tf = pytest.importorskip("tensorflow")
    models = pytest.importorskip("tensorflow.keras.models")
    layers = pytest.importorskip("tensorflow.keras.layers")

    train_images = np.ones(shape=(10, 32, 32, 3))
    train_labels = np.random.randint(0, 2, size=10)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    # this model has only a single output
    model.add(layers.Dense(1))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        epochs=1,
    )
    train_images = xr.DataArray(
        train_images,
        dims=("observations", "pixel_x", "pixel_y", "rgb"),
        coords={
            "pixel_x": np.arange(32),
            "pixel_y": np.arange(32),
            "rgb": ["r", "g", "b"],
        },
    )
    result = eshap_compare(train_images, model, model, explainer="deep")
    assert result.dims == ("observations", "pixel_x", "pixel_y", "rgb")
    expected_coords = {
        "pixel_x": np.arange(32),
        "pixel_y": np.arange(32),
        "rgb": ["r", "g", "b"],
    }
    assert result.coords.equals(
        xr.DataArray(
            dims=("observations", "pixel_x", "pixel_y", "rgb"), coords=expected_coords
        ).coords
    )
    assert result.sum().values == np.array(0.0)


def test_eshap_compare_pandas_input():
    X, y = shap.datasets.adult(n_points=100)
    dtc = DecisionTreeClassifier(max_depth=2).fit(X, y)
    rfc = RandomForestClassifier(max_depth=2, n_estimators=2).fit(X, y)

    result = eshap_compare(X, dtc, rfc)
    assert result.dims == ("output", "dim_0", "dim_1")
    expected_coords = {"dim_0": X.index, "dim_1": X.columns, "output": [0, 1]}
    assert result.coords.equals(
        xr.DataArray(dims=("output", "dim_0", "dim_1"), coords=expected_coords).coords
    )
