import argparse
import random
import pickle
import logging
import yaml
import time
from pathlib import Path
from typing import List, Union

from sklearn.metrics import accuracy_score
import numpy as np

from wisard.encoders import (
    ThermometerEncoder2,
    DistributiveThermometerEncoder,
    encode_dataset,
)
from wisard.utils import untie
from wisard.wisard import WiSARD
from wisard.optimize import find_best_bleach_bayesian, find_best_bleach_bin_search


def thermometer_encoder(
    x_train,
    x_test,
    resolution: int,
    fit_on_train_test: bool = False,
    use_tqdm: bool = False,
):
    logging.info("Encoding dataset...")
    thermometer = ThermometerEncoder2(resolution=resolution)
    if fit_on_train_test:
        x_merged = np.concat([x_train, x_test])
    else:
        x_merged = x_train
    thermometer.fit(x_merged)
    x_train = encode_dataset(thermometer, x_train, use_tqdm=use_tqdm)
    x_test = encode_dataset(thermometer, x_test, use_tqdm=use_tqdm)
    logging.info("Encoding done")
    return x_train, x_test


def distributive_thermometer_encoder(
    x_train,
    x_test,
    resolution: int,
    fit_on_train_test: bool = False,
    use_tqdm: bool = False,
):
    logging.info("Encoding dataset...")
    distributive_thermometer = DistributiveThermometerEncoder(
        resolution=resolution)
    if fit_on_train_test:
        x_merged = np.concat([x_train, x_test])
    else:
        x_merged = x_train
    distributive_thermometer.fit(x_merged)
    x_train = encode_dataset(distributive_thermometer,
                             x_train,
                             use_tqdm=use_tqdm)
    x_test = encode_dataset(distributive_thermometer, x_test, use_tqdm=use_tqdm)
    logging.info("Encoding done")
    return x_train, x_test


def do_train_and_evaluate(
    x_train,
    y_train,
    x_test,
    y_test,
    tuple_size: int,
    input_indexes: List[int],
    bleach: Union[int, str] = "bin_search",
    use_tqdm: bool = False,
    verbose: bool = False,
) -> dict:
    num_classes = len(np.unique(y_train))

    logging.info(" ----- Training model ----- ")
    logging.info(f"input_indexes: {input_indexes}")
    logging.info(f"tuple size: {tuple_size}")
    logging.info(f"x_train.shape: {x_train.shape}")
    logging.info(f"x_test.shape: {x_test.shape}")
    logging.info(f"num classes: {num_classes}")
    logging.info(f"Using bleach mode: {bleach}")

    model = WiSARD(
        num_inputs=x_train[0].size,
        num_classes=num_classes,
        unit_inputs=tuple_size,
        unit_entries=1,
        unit_hashes=1,
        input_idxs=input_indexes,
        shared_rand_vals=False,
        randomize=False,
        use_dict=True,
    )

    model.fit(x_train, y_train, use_tqdm=use_tqdm)
    max_bleach = model.max_bleach()
    logging.info(f"Max bleach is: {max_bleach}\n")

    logging.info(" ----- Evaluating model ----- ")

    if isinstance(bleach, int):
        y_pred = model.predict(x_test, y_test, bleach=bleach, use_tqdm=use_tqdm)
        y_pred, ties = untie(y_pred, use_tqdm=False)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"[b={bleach}] Accuracy={accuracy:.3f}, ties={ties}")
        history = {"accuracy": {bleach: accuracy}, "ties": {bleach: ties}}

    elif bleach == "bin_search":
        bleach, history = find_best_bleach_bin_search(
            model,
            X=x_test,
            y=y_test,
            min_bleach=1,
            max_bleach=max_bleach // 2,
            use_tqdm=use_tqdm,
            verbose=verbose,
        )
    elif bleach == "bayesian_search":
        bleach, history = find_best_bleach_bayesian(
            model,
            X=x_test,
            y=y_test,
            min_bleach=1,
            max_bleach=max_bleach // 2,
            use_tqdm=use_tqdm,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Invalid value for bleach: '{bleach}'")

    return model, bleach, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train wisard and find bleach (MNIST)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_path",
                        type=str,
                        action="store",
                        help="Input dataset file (pickle)")
    parser.add_argument(
        "--tuple-size",
        type=int,
        required=True,
        help="Size of the tuple for Wisard",
    )

    # Non-required args
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Default seed for pseudo-random number generator",
    )
    parser.add_argument("--output",
                        type=str,
                        action="store",
                        help="Output file")
    parser.add_argument("--linear-index",
                        action="store_true",
                        help="Use linear index for RAMs")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Increases verbosity")
    parser.add_argument(
        "--bleach",
        action="store",
        default="bin_search",
        help="Bleach value or algorithm",
    )
    parser.add_argument("--tqdm", action="store_true", help="Use TQDM bar")

    # Encoder parsers
    subparsers = parser.add_subparsers(help="Encoder parser",
                                       required=True,
                                       dest="encoder")

    # The thermometer
    tparser = subparsers.add_parser("thermometer", help="Thermometer encoding")
    tparser.add_argument("--resolution",
                         type=int,
                         required=True,
                         help="Thermometer resolution")
    # tparser.add_argument(
    #     "--min-val",
    #     type=float,
    #     required=True,
    #     help="Minimum value for encoder (thermometer)",
    # )
    # tparser.add_argument(
    #     "--max-val",
    #     type=float,
    #     required=True,
    #     help="Maximum value for encoder (thermometer)",
    # )

    # The distribuive thermometer
    dtparser = subparsers.add_parser("distributive-thermometer",
                                     help="Distributive thermometer encoding")
    dtparser.add_argument("--resolution",
                          type=int,
                          required=True,
                          help="Thermometer resolution")
    dtparser.add_argument("--fit-on-train-test",
                          action="store_true",
                          help="Concat train and test to fit")
    args = parser.parse_args()

    start_time = time.time()

    ################################
    if args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN

    logging.basicConfig(level=log_level, format='[%(asctime)s] [%(levelname)s]: %(message)s')

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        raise ValueError(f"Invalid dataset at: {dataset_path}")

    with dataset_path.open("rb") as f:
        (x_train, y_train), (x_test, y_test) = pickle.load(f)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.encoder == "thermometer":
        x_train, x_test = thermometer_encoder(
            x_train,
            x_test,
            # args.min_val,
            # args.max_val,
            args.resolution,
            use_tqdm=args.tqdm,
        )

        input_indexes = (np.arange(x_train[0].size).reshape(
            args.resolution, -1).T.ravel())

    elif args.encoder == "distributive-thermometer":
        x_train, x_test = distributive_thermometer_encoder(
            x_train,
            x_test,
            args.resolution,
            # fit_on_train_test=args.fit_on_train_test,
            use_tqdm=args.tqdm,
        )

        input_indexes = (np.arange(x_train[0].size).reshape(
            args.resolution, -1).T.ravel())

    # Should not reach here...
    else:
        raise ValueError(f"Invalid encoder: {args.encoder}")

    # Use linear?
    if not args.linear_index:
        np.random.shuffle(input_indexes)

    # Use defined bleach or bleach search?
    try:
        bleach = int(args.bleach)
    except ValueError:
        if args.bleach not in ["bin_search", "bayesian_search"]:
            raise ValueError(f"Invalid value for bleach: {args.bleach}")
        bleach = args.bleach

    # Finally, train the model
    model, bleach, history = do_train_and_evaluate(
        x_train,
        y_train,
        x_test,
        y_test,
        input_indexes=input_indexes,
        tuple_size=args.tuple_size,
        bleach=bleach,
        use_tqdm=args.tqdm,
        verbose=args.verbose
    )
    logging.info(f"The best bleach was: {bleach}")

    end_time = time.time()

    config = {
        "input args": dict(vars(args)),
        "tuple_size": int(args.tuple_size),
        "best_bleach": int(bleach),
        "x_train": list(x_train.shape),
        "x_test": list(x_test.shape),
        "history": history,
        "start": float(start_time),
        "end": float(end_time),
        "elapsed": float(end_time - start_time),
        "input_indexes": [int(i) for i in input_indexes],
    }

    if args.output:
        logging.info("Saving output...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            yaml.dump(config, f)
        logging.info("Saving output done")

    print(f"Best bleach: {bleach}")
    print(history)
