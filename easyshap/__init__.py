import xarray as xr
import pandas as pd
import numpy as np
import shap
from typing import Any


def _get_explainer(explainer: str | shap.Explainer | None) -> shap.Explainer:
    if explainer == "gradient":
        explainer = shap.explainers.GradientExplainer
    elif explainer == "tree":
        explainer = shap.explainers.TreeExplainer
    elif explainer == "deep":
        explainer = shap.explainers.DeepExplainer
    elif explainer == "kernel":
        explainer = shap.explainers.KernelExplainer
    elif explainer == "linear":
        explainer = shap.explainers.LinearExplainer
    elif explainer == "sampling":
        explainer = shap.explainers.SamplingExplainer
    elif explainer == "partition":
        explainer = shap.explainers.PartitionExplainer
    elif explainer == "permutation":
        explainer = shap.explainers.PermutationExplainer
    elif explainer == "additive":
        explainer = shap.explainers.AdditiveExplainer
    elif explainer == "exact":
        explainer = shap.explainers.ExactExplainer
    elif explainer is None:
        explainer = shap.Explainer
    else:
        raise ValueError(
            "explainer must be one of 'gradient', 'tree', 'deep', 'kernel', 'linear', 'sampling', "
            f"'partition', 'permutation', 'additive', 'exact' or None. Received: {explainer}"
        )
    return explainer


def eshap_compare(
    data: xr.DataArray | pd.DataFrame | np.ndarray,
    model1: Any,
    model2: Any,
    feature_names: list[str] | None = None,
    explainer: str | shap.Explainer | None = None,
    model1_name: str = "model1",
    model2_name: str = "model2",
) -> xr.DataArray:
    """Compare two models using SHAP values. Return the difference of the SHAP values of the two models.

    Args:
        data (xr.DataArray | pd.DataFrame | np.ndarray): Data with features to explain.
        model1 (Any): First model that can be explained by shap.
        model2 (Any): Second model that can be explained by shap.
        output_names (list[str] | None, optional): _description_. Defaults to None.  # todo: not yet implemented
        explainer (str | shap.Explainer | None, optional): Explainer to use. Note that the explainer is applied to both models,
            so if the two models cannot be explained by the same explainer (e.g. a NN and a DecisionTree) leave this empty. Defaults to None.
        model1_name (str): Name of model 1
        model2_name (str): Name of model 2

    Raises:
        ValueError: Raises if explainer an invalid explainer is provided.

    Returns:
        xr.DataArray: _description_
    """
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        data = xr.DataArray(data)
    explainer = _get_explainer(explainer)
    explainer_model1 = explainer(model1, data.values)
    explainer_model2 = explainer(model2, data.values)
    try:
        shap_values1 = explainer_model1.shap_values(data.values)
        shap_values2 = explainer_model2.shap_values(data.values)
    except TypeError:
        raise TypeError(
            f"Selected explainer {explainer.__call__.__name__} automatically. This did not work, please provide an explainer yourself"
        )
    if feature_names is None:
        if isinstance(data, pd.DataFrame):
            feature_names = list(data.columns)
        # todo: make case distinction if we have an xarray here
        else:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
    if isinstance(shap_values1, list) and isinstance(shap_values2, list):
        assert len(shap_values1) == len(
            shap_values2
        ), f"shap_values1 and shap_values2 must have the same length. Received: {len(shap_values1)} and {len(shap_values2)}"
        if len(shap_values1) == 1 and len(shap_values2) == 1:
            shap_values1_xarr = xr.DataArray(
                shap_values1[0], dims=data.dims, coords=data.coords, name=model1_name
            )
            shap_values2_xarr = xr.DataArray(
                shap_values2[0], dims=data.dims, coords=data.coords, name=model2_name
            )
            shap_values_diff = xr.DataArray(
                shap_values1_xarr - shap_values2_xarr,
                dims=data.dims,
                coords=data.coords,
                name="diff"
            )
        else:
            shap_values1_xarr = xr.DataArray(
                shap_values1,
                dims=("output", *data.dims),
                coords=data.coords.assign({"output": list(range(len(shap_values1)))}),
                name=f"{model1_name}"
            )
            shap_values2_xarr = xr.DataArray(
                shap_values2,
                dims=("output", *data.dims),
                coords=data.coords.assign({"output": list(range(len(shap_values2)))}),
                name=f"{model2_name}"
            )
            shap_values_diff = xr.DataArray(
                shap_values1_xarr - shap_values2_xarr,
                dims=("output", *data.dims),
                coords=data.coords.assign({"output": list(range(len(shap_values2)))}),
                name="diff"
            )

    elif isinstance(shap_values1, list) or isinstance(shap_values2, list):
        raise ValueError(
            f"shap_values1 and shap_values2 must both be lists or neither be lists. Received: {type(shap_values1)} and {type(shap_values2)}"
        )
    else:
        shap_values1_xarr = xr.DataArray(
            shap_values1, dims=data.dims, coords=data.coords, name=f"{model1_name}"
        )
        shap_values2_xarr = xr.DataArray(
            shap_values2, dims=data.dims, coords=data.coords, name=f"{model2_name}"
        )
        shap_values_diff = xr.DataArray(shap_values1_xarr - shap_values2_xarr,
                                        dims=data.dims,
                                        coords=data.coords,
                                        name="diff")
    shap_ds = xr.merge((shap_values1_xarr, shap_values2_xarr, shap_values_diff))
    return shap_ds
