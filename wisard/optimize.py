from .wisard import WiSARD
import numpy as np
from sklearn.metrics import accuracy_score
from .utils import untie
from bayes_opt import BayesianOptimization


def find_best_bleach_bin_search(model: WiSARD, X: np.ndarray, y: np.ndarray,
                                min_bleach: int, max_bleach: int, use_tqdm: bool = False,
                                verbose: bool = False):
    best_bleach = max_bleach // 2
    step = max(max_bleach // 4, 1)
    bleach_accuracies = {}
    bleach_ties = {}
    
    while True:  # This guy can deadlock. max_val is 3, step will be 0
        values = [best_bleach - step, best_bleach, best_bleach + step]
        accuracies = []
        for b in values:
            if verbose:
                print(f"Testing with bleach={b}")
            if b in bleach_accuracies:
                accuracies.append(bleach_accuracies[b])
                if verbose:
                    print(f"[b={b}] Accuracy={bleach_accuracies[b]:.3f}")
            elif b < 1:
                accuracies.append(0)
                if verbose:
                    print(f"[b={b}] Accuracy=0")
            else:
                y_pred = model.predict(X, y, bleach=b, use_tqdm=use_tqdm)
                y_pred, ties = untie(y_pred, use_tqdm=False)
                accuracy = accuracy_score(y, y_pred)
                bleach_accuracies[b] = float(accuracy)
                bleach_ties[b] = int(ties)
                accuracies.append(accuracy)
                if verbose:
                    print(f"[b={b}] Accuracy={accuracy:.3f}, ties={ties}")
        new_best_bleach = values[accuracies.index(max(accuracies))]
        if (new_best_bleach == best_bleach) and (step == 1):
            break
        best_bleach = new_best_bleach
        if step > 1:
            step //= 2

    if verbose:
        print(f"Best bleach: {best_bleach}....")

    history = {
        "accuracy": bleach_accuracies,
        "ties": bleach_ties
    }

    return best_bleach, history


def find_best_bleach_bayesian(model: WiSARD,
                              X: np.ndarray,
                              y: np.ndarray,
                              min_bleach: int,
                              max_bleach: int,
                              init_points: int = 3,
                              n_iter: int = 10,
                              seed: int = None,
                              use_tqdm: bool = False,
                              verbose: bool = False):

    def inference(bleach):
        y_pred = model.predict(X, y, bleach=bleach, use_tqdm=use_tqdm)
        y_pred, ties = untie(y_pred, use_tqdm=False)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    if verbose:
        bounds = {"bleach": (min_bleach, max_bleach)}

    optimizer = BayesianOptimization(f=inference,
                                     pbounds=bounds,
                                     random_state=seed,
                                     verbose=2)

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    if verbose:
        print(f'Best bleach: {optimizer.max["params"]["bleach"]}....')

    return int(optimizer.max["params"]["bleach"]), optimizer
