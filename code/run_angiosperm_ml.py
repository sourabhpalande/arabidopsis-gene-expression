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
        "clf": LinearSVC(dual=False),
        "param_space": {
            "C": hp.choice("C", [pow(10, x) for x in range(-5, 1)]),
            "penalty": hp.choice("penalty", ["l1", "l2"]),
            },
        },
    "KNN": {
        "clf": KNeighborsClassifier(metric="correlation", n_jobs=-1),
        "param_space": {
            "n_neighbors": hp.choice("n_neighbors", range(1, 15)),
            "weights": hp.choice("weights", ["uniform", "distance"]),
            "leaf_size": hp.choice("leaf_size", range(3, 13)),
            },
        },
    "RF" : {
        "clf": RandomForestClassifier(n_jobs=-1),
        "param_space": {
            "max_depth": hp.choice("RF_max_depth", range(2, 17)),
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
            "max_depth": hp.choice("HGB_max_depth", range(2, 17)),
            "l2_regularization": hp.uniform("l2_regularization", 0.5, 1.),
            },
        },
    "MLP": {
        "clf": MLPClassifier(early_stopping=True),
        "param_space": {
            "hidden_layer_sizes": hp.choice("hidden_layer_sizes",
                                            [(128,),
                                             (1024, 128,),
                                             (1024, 512, 128)]),
            "alpha": hp.choice("alpha", [pow(10, x) for x in range(-5, 1)]),
            },
    }
}


optim_params = { 
                "HGB": { "l2_regularization": 0.6506914295928992,
                         "learning_rate": 0.3144676106516742,
                         "max_depth": 7 },
                "KNN": { "leaf_size": 10,
                         "n_neighbors": 2,
                         "weights": "distance" },
                "MLP": {"alpha": 0.001, "hidden_layer_sizes": [1000, 100]},
                "RF": { "class_weight": "balanced_subsample",
                        "criterion": "gini",
                        "max_depth": 9,
                        "max_features": "sqrt",
                        "n_estimators": 110 },
                "SVC": {"C": 0.0001, "penalty": "l1"}
}


def loaddata():
    tic = perf_counter()
    arabifile = "../data/gene_FPKM_transposed_UMR75.gzip"
    angiofile = "../data/Angiosperm_data_clean.csv"
    metafactor = "AboveBelow"
    print("Loading data...")

    angiodf = pd.read_csv(angiofile)
    meta_cols = angiodf.columns[:6]
    gene_cols = angiodf.columns[6:]
    meta_angio = angiodf[meta_cols]
    X_angio = angiodf[gene_cols].astype("float64")
    X_angio = X_angio.apply(lambda x: np.log2(x+1.0))
    angiometa = angiodf[metafactor]
    print(f"Angiosperm RNAseq data shape: {X_angio.shape}")

    arabidf = pd.read_parquet(arabifile)
    X_arabi = arabidf[gene_cols].astype("float64")
    X_arabi = X_arabi.apply(lambda x: np.log2(x+1.0))
    arabimeta = arabidf[metafactor]
    print(f"Arabidopsis RNAseq data shape: {X_arabi.shape}")

    all_labels = pd.concat([arabimeta, angiometa], ignore_index=True)
    class_names = all_labels.unique().tolist()
    print(f"Number of all labels: {all_labels.shape}")
    print(f"Unique labels: {class_names}")

    toc = perf_counter()
    print(f"Data loaded. Time elapsed: {toc - tic}")

    return(X_arabi, arabimeta, X_angio, angiometa, class_names, meta_angio)


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

    X_arabi, Y_arabi, X_angio, Y_angio, class_names, meta_angio = loaddata()

    tic = perf_counter()
    Xtr, Xte, Ytr, Yte = train_test_split(X_arabi, Y_arabi, test_size=0.3,
                                          random_state=42, stratify=Y_arabi)
    # scaler = StandardScaler()
    pca = PCA(n_components=4)
    labelenc = LabelEncoder().fit(class_names)
    class_codes = labelenc.transform(class_names).tolist()
    Ytr = labelenc.transform(Ytr)
    Yte = labelenc.transform(Yte)
    Yangio = labelenc.transform(Y_angio)

    print(f"Training data shapes: {Xtr.shape}, {Ytr.shape}")
    print(f"Test data shapes: {Xte.shape}, {Yte.shape}")
    print(f"Angio data shapes: {X_angio.shape}, {Y_angio.shape}")
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

        best_result = fmin( fn=objective, space=space, algo=algo,
                            max_evals=60, trials=trials)
        opt_params = space_eval(space, best_result)
        print(opt_params)
    else:
        opt_params = optim_params[model_name]

    print("Training Model with Tuned Parameters")
    clf["classifier"].set_params(**opt_params)
    clf.fit(Xtr, Ytr)
    pred_ar = clf.predict(Xte)
    pred_an = clf.predict(X_angio)
    a_ar = accuracy_score(Yte, pred_ar)
    c_ar = confusion_matrix(Yte, pred_ar, normalize="true")
    p_ar, r_ar, f_ar, _ = precision_recall_fscore_support(Yte, pred_ar,
                                        average="weighted", zero_division=0)

    a_an = accuracy_score(Yangio, pred_an)
    c_an = confusion_matrix(Yangio, pred_an, normalize="true")
    p_an, r_an, f_an, _ = precision_recall_fscore_support(Yangio, pred_an,
                                        average="weighted", zero_division=0)
    model_performance = dict()
    model_performance["class_names"] = class_names
    model_performance["label_encoding"] = class_codes
    model_performance["arabidopsis"] = {
                                        "accuracy": a_ar,
                                        "precision": p_ar,
                                        "recall": r_ar,
                                        "f1_score": f_ar,
                                        "confusion_matrix": c_ar.tolist()
                                        }

    model_performance["angiosperm"] = {
                                        "accuracy": a_an,
                                        "precision": p_an,
                                        "recall": r_an,
                                        "f1_score": f_an,
                                        "confusion_matrix": c_an.tolist()
    }

    print("Test performance metrics")
    print(model_performance)

    outf1 = f"../results/angiosperm/{model_name}_performance.json"
    with open(outf1, "w") as fp:
        json.dump(model_performance, fp)

    outf2 = f"../results/angiosperm/{model_name}_model.pickle"
    with open(outf2, "wb") as fp:
        pickle.dump(clf, fp, protocol=pickle.HIGHEST_PROTOCOL)

    outf3 = f"../results/angiosperm/{model_name}_metadata.csv"
    meta_angio["true_labels"] = Yangio
    meta_angio["predictions"] = pred_an
    meta_angio.to_csv(outf3, index=False)

    print(f"Wrote model performance metrics to file: {outf1}")
    print(f"Saved model to file: {outf2}")
    print(f"Saved angiosperm metadata to file: {outf3}")

    print("Done")
