import numpy as np
import pandas as pd
import argparse
from functools import partial

from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

# from tqdm import tqdm
from time import perf_counter
import json
import pickle


model_dict = {
    "SVC": {
        "clf": LinearSVC(),
        "param_space": {
            "C": hp.choice("C", [10e-4, 10e-3, 10e-2, 10e-1, 1]),
            "penalty": hp.choice("penalty", ["l1", "l2"]),
            },
        },
    "KNN": {
        "clf": KNeighborsClassifier(metric="correlation", n_jobs=-1),
        "param_space": {
            "n_neighbors": hp.choice("n_neighbors", range(1, 8, 1)),
            "weights": hp.choice("weights", ["uniform", "distance"]),
            "leaf_size": hp.choice("leaf_size", range(3, 15, 2)),
            },
        },
    "RF" : {
        "clf": RandomForestClassifier(n_jobs=-1),
        "param_space": {
            "max_depth": hp.choice("RF_max_depth", range(2, 9)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "n_estimators": hp.choice("n_estimators", range(90, 141, 10)),
            "max_features": hp.choice("max_features", ["sqrt", "log2"]),
            "class_weight": hp.choice("class_weight", ["balanced",
                                                    "balanced_subsample"]),
                                                    },
        },
    "HGB": {
        "clf": HistGradientBoostingClassifier(),
        "param_space": {
            "learning_rate" : hp.uniform("learning_rate", 0., 0.5),
            "max_depth": hp.choice("HGB_max_depth", range(2, 9, 1)),
            "l2_regularization": hp.uniform("l2_regularization", 0.5, 1.),
            },
        },
    "MLP": {
        "clf": MLPClassifier(early_stopping=True),
        "param_space": {
            "hidden_layer_sizes": hp.choice("hidden_layer_sizes",
                                            [(100,), (1000, 100,)]),
            "alpha": hp.choice("alpha", [1e-4, 1e-3, 1e-2, 1e-1]),
            },
    }
}


optim_params = {
                "HGB": { "l2_regularization": 0.6506914295928992,
                         "learning_rate": 0.3144676106516742,
                         "max_depth": 7 },
                "KNN": { "leaf_size": 10,
                         "n_neighbors": 2,
                         "weights": "distance"},
                "MLP": {"alpha": 0.01, "hidden_layer_sizes": [1000, 100]},
                "RF": { "class_weight": "balanced_subsample",
                        "criterion": "entropy",
                        "max_depth": 9,
                        "max_features": "sqrt",
                        "n_estimators": 130 },
                "SVC": {"C": 1, "penalty": "l2"}
            }


def loaddata():
    tic = perf_counter()
    datafile = "../data/gene_FPKM_transposed_UMR75.gzip"
    metafactor = "AboveBelow"
    print("Loading data...")

    arabidf = pd.read_parquet(datafile)
    X_arabi = arabidf[arabidf.columns[14:]].astype("float64")
    X_arabi = X_arabi.apply(lambda x: np.log2(x+1.0))
    Y_arabi = arabidf[metafactor]
    class_names = Y_arabi.unique().tolist()
    print(f"RNAseq data shape: {X_arabi.shape}")
    toc = perf_counter()
    print(f"Data loaded. Time elapsed: {toc - tic}")
    
    return X_arabi, Y_arabi, class_names


def model_eval(params, clf, X_tr, Y_tr):
    clf["classifier"].set_params(**params)
    score = cross_val_score(clf, X_tr, Y_tr, cv=3, scoring="f1_weighted").mean()
    return {"loss": -1.0*score, "status": STATUS_OK}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Specify learning model')
    parser.add_argument('-clf', '--classifier', metavar ='model',
                        required = True, dest ='model', action ='store',
                        help ='Classifier model')
    parser.add_argument('-m', '--mode', required = False, dest ='train_mode',
                        action ='store_true', help ='Enable Training Mode')
    args = parser.parse_args()
    if args.model not in model_dict.keys():
        print("Specified model not implemented.")
        print(f"Available models are: {list(model_dict.keys())}")
        exit(1)

    model_name = args.model
    train_mode = args.train_mode
    print(model_name, train_mode)

    X_arabi, Y_arabi, class_names = loaddata()
    print(X_arabi.shape, Y_arabi.shape)
    print(class_names)

    tic = perf_counter()
    Xtr, Xte, Ytr, Yte = train_test_split(X_arabi, Y_arabi, test_size=0.3,
                                          random_state=42, stratify=Y_arabi)

    # scaler = StandardScaler()
    pca = PCA(n_components=4)
    labelenc = LabelEncoder().fit(class_names)
    class_codes = labelenc.transform(class_names).tolist()
    Ytr = labelenc.transform(Ytr)
    Yte = labelenc.transform(Yte)

    print(f"Training data shapes: {Xtr.shape}, {Ytr.shape}")
    print(f"Test data shapes: {Xte.shape}, {Yte.shape}")
    print(f"Train / test splitting done. Time elapsed: {perf_counter() - tic}")

    if model_name in ["SVC", "MLP", "RF"]:
        clf = Pipeline(steps=[("pca", pca),
                              ("classifier", model_dict[model_name]["clf"])])
    else:
        clf = Pipeline(steps=[("classifier", model_dict[model_name]["clf"])])

    if train_mode:
        print("Beginning Model Training and Selection using Hyperopt")
        space = model_dict[model_name]["param_space"]
        objective = partial(model_eval, clf=clf, X_tr=Xtr, Y_tr=Ytr)
        algo = tpe.suggest
        trials = Trials()

        best_result = fmin( fn=objective,
                            space=space,
                            algo=algo,
                            max_evals=60,
                            trials=trials)
        opt_params = space_eval(space, best_result)
        print(opt_params)
    else:
        opt_params = optim_params[model_name]

    print("Training Model with Tuned Parameters")
    clf["classifier"].set_params(**opt_params)
    clf.fit(Xtr, Ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(Yte, pred)
    confmat = confusion_matrix(Yte, pred, normalize="true")
    p, r, f, _ = precision_recall_fscore_support(Yte, pred,
                                                 average="weighted", zero_division=0)
    model_performance = { "class_names": class_names,
                          "label_encoding": class_codes,
                          "accuracy": acc,
                          "precision": p,
                          "recall": r,
                          "f1_score": f,
                          "confusion_matrix": confmat.tolist()}
    print("Model test performance")
    print(model_performance)

    outf1 = f"../results/arabidopsis/{model_name}_performance.json"
    with open(outf1, "w") as fp:
        json.dump(model_performance, fp)

    outf2 = f"../results/arabidopsis/{model_name}_model.pickle"
    with open(outf2, "wb") as fp:
        pickle.dump(clf, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote model performance metrics to file: {outf1}")
    print(f"Saved model to file: {outf2}")

    print("Done")
