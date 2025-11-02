import logging
import os
from typing import Optional
import warnings
import click
from nerfbaselines.evaluation import evaluate, run_inside_eval_container, build_evaluation_protocol
from nerfbaselines._types import DatasetFeature, CameraModel
from nerfbaselines.datasets import load_dataset
from ._common import NerfBaselinesCliCommand
from .._registry import evaluation_protocols_registry


@click.command(
    "evaluate",
    cls=NerfBaselinesCliCommand,
    short_help="Evaluate predictions",
    help=(
        "Evaluate predictions (e.g., obtained by running `nerfbaselines render`) against the ground truth. "
        "The predictions are evaluated using the correct evaluation protocol and the results are saved in the specified JSON output file. "
        "The predictions can be provided as a directory or a tar.gz/zip file."
    ),
)
@click.argument("predictions", type=click.Path(file_okay=True, dir_okay=True, path_type=str), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=str), required=True, help="Path to the output JSON file to save the evaluation results.")
@click.option(
    "--data",
    type=str,
    required=False,
    help=(
        "A path to the dataset to train on. The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. If the dataset is a local path, the dataset will be loaded directly from the specified path."
    ),
)
@click.option(
    "--evaluation-protocol",
    default=None,
    type=click.Choice(list(evaluation_protocols_registry.keys())),
    help="Override the default evaluation protocol. WARNING: This is strongly discouraged.",
    hidden=True,
)
@click.option("--eval-num-patches", type=int, default=3, help="Number of patches on the smaller image axis. (Number of patches on larger image axis determined automatically based on image dimension, keeping patches nearly square.)", show_default=True)
def evaluate_command(predictions: str, output: str, data: Optional[str], evaluation_protocol=None, eval_num_patches: int = 3) -> None:
    evaluation_protocol_obj = None
    if evaluation_protocol is not None:
        warnings.warn(f"Overriding the evaluation protocol to {evaluation_protocol}. This is strongly discouraged.")
        evaluation_protocol_obj = build_evaluation_protocol(evaluation_protocol)

    dataset = None
    if data is not None:
        _data = data
        if "://" not in _data:
            _data = os.path.abspath(_data)

        # Load eval dataset
        logging.info("Loading eval dataset")
        dataset = load_dataset(_data, split="test", features=frozenset(("points3D_xyz", "images_points3D_indices")), supported_camera_models=frozenset(("pinhole",)), load_features=True)

    with run_inside_eval_container():
        evaluate(predictions, output, num_patches=eval_num_patches, evaluation_protocol=evaluation_protocol_obj, dataset=dataset)
