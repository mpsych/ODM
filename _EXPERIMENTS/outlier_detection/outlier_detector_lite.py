import glob
import os
import pickle

import sklearn
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from outlier_detector import OutlierDetector
from feature_extractor import Features

warnings.filterwarnings("ignore", category=ConvergenceWarning)
THRESHOLD = 0.0001

OUTPUTDIR = "/raid/mpsych/ODL/ABLATION/"


class OutlierDetectorLite:
    def __init__(self, DATAPATH="/raid/mpsych/"):
        """Initializes the class"""
        self.datapath = DATAPATH
        self.ALGORITHMS = [
            "AE",
            "VAE",
            "SOGAAL",
            "DeepSVDD",
            "AnoGAN",
            "HBOS",
            "LOF",
            "OCSVM",
            "IForest",
            "CBLOF",
            "COPOD",
            "KNN",
            "AvgKNN",
            "MedKNN",
            "SOS",
            "KDE",
            "Sampling",
            "PCA",
            "LMDD",
            "COF",
            "ECOD",
            "SOD",
            "INNE",
            "FB",
            "LODA",
            "SUOD",
        ]

    def load_data(self, DATASET):
        """Loads the dataset
        Parameters
        ----------
        DATASET : str
            The name of the dataset to load
        Returns
        -------
        imgs : np.ndarray
            The dataset
        """
        with open(os.path.join(self.datapath, f"dataset{DATASET}.pkl"), "rb") as f:
            imgs = pickle.load(f)
        return imgs

    def load_configs(self, DATASET):
        """Loads the dataset
        Parameters
        ----------
        DATASET : str
            The name of the dataset to load
        Returns
        -------
        configs : dict
            The dataset
        """
        with open(os.path.join(self.datapath, f"dataset{DATASET}_configs.pkl"), "rb") as f:
            configs = pickle.load(f)
        return configs

    def load_ground_truth(self, DATASET):
        """Loads the ground truth
        Parameters
        ----------
        DATASET : str
            The name of the dataset to load
        Returns
        -------
        ground_truth : np.ndarray
            The ground truth
        """
        with open(os.path.join(self.datapath, f"dataset{DATASET}_labels.pkl"), "rb") as f:
            ground_truth = pickle.load(f)
        return ground_truth

    def load_results(self, resultsfile):
        """Loads the results
        Parameters
        ----------
        resultsfile : str
            The path to the results file
        Returns
        -------
        results : dict
            The results
        """
        with open(resultsfile, "rb") as f:
            results = pickle.load(f)

        return results

    def display_best_results(self, resultsfile):
        """Displays the best results
        Parameters
        ----------
        results : dict
            The results
        Returns
        -------
        None
        """
        all_evals = self.load_results(resultsfile)
        best_tp = 0
        best_tp_alg = ""
        for alg in all_evals.keys():
            tp = all_evals[alg][0]["evaluation"]["tp"]
            print(alg, tp)
            if best_tp < tp:
                best_tp = tp
                best_tp_alg = alg

    def run(
        self,
        DATASET,
        ALGORITHM,
        imgs=None,
        feature_vector=None,
        groundtruth=None,
        default_config=False,
        custom_config=None,
    ):
        """Runs the outlier detection algorithm on the dataset
        Parameters
        ----------
        DATASET : str
          The name of the dataset to run the algorithm on
        ALGORITHM : str
          The name of the algorithm to run
        imgs : np.ndarray, None
            A dataset to use instead of the default one
        feature_vector : np.ndarray, None
            A feature vector to use instead of the default one
        groundtruth : np.ndarray, None
            A ground truth to use instead of the default one
        default_config : bool
          Whether to use the default configuration or not
        custom_config : dict, None
            A custom configuration to use
        Returns
        -------
        results : dict
          The results of the algorithm
        """

        DATASETS = {
            "A": 0.08,
            "B": 0.13,
            "C": 0.24,
            "D": 0.24,
            "ASTAR": 0.063,
            "BSTAR": 0.050,
            "CSTAR": 0.015,
        }

        CONTAMINATION = DATASETS[DATASET]

        # load data
        if imgs is None:
            imgs = self.load_data(DATASET)

        print("Loaded images.")

        if custom_config is None:
            # setup algorithm w/ the best config w/ the best feat w/ best norm
            configs = self.load_configs(DATASET)

            CONFIG = configs[ALGORITHM]["config"]

            if default_config:
                CONFIG = {}  # clear the config and use default
                CONFIG = {"contamination": CONTAMINATION}

            NORM = configs[ALGORITHM]["norm"]
            FEAT = configs[ALGORITHM]["feat"]

            print("Loaded config, norm, and feats.")
        else:
            CONFIG = custom_config
            NORM = CONFIG["norm"]
            FEAT = CONFIG["feat"]

            print("Loaded custom config, norm, and feats.")
        CONFIG["verbose"] = 0
        CONFIG["return_decision_function"] = True
        CONFIG["accuracy_score"] = False
        print("config: ", CONFIG)

        print("FEAT: ", FEAT)
        print("NORM: ", NORM)

        if feature_vector is None:
            feature_vector = Features.get_features(imgs, FEAT, NORM, **CONFIG)
        else:
            print("Using provided feature vector.")

        print("Calculated features!")

        scores, labels, decision_function = OutlierDetector.detect_outliers(
            features=feature_vector,
            imgs=imgs,
            pyod_algorithm=ALGORITHM,
            display=False,
            number_bad=self.__number_bad(DATASET),
            **CONFIG,
        )

        print("Trained!")

        #
        # EVALUATE
        #

        # load groundtruth
        if groundtruth is None:
            groundtruth = self.load_ground_truth(DATASET)
        else:
            print("Using provided ground truth.")

        evaluation = None

        if labels is None:
            evaluation = {
                "groundtruth_indices": np.where(np.array(groundtruth) > 0),
                "pred_indices": np.where(np.array(scores) > 0),
                "roc_auc": 0,
                "f1_score": 0,
                "acc_score": 0,
                "jaccard_score": 0,
                "precision_score": 0,
                "average_precision": 0,
                "recall_score": 0,
                "hamming_loss": 0,
                "log_loss": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }

        elif len(labels) == len(groundtruth):
            evaluation = self.evaluate(groundtruth, labels)
        return {
            "algorithm": ALGORITHM,
            "norm": NORM,
            "feat": FEAT,
            "dataset": DATASET,
            "scores": scores,
            "labels": labels,
            # 'decision_function': decision_function,
            "groundtruth": groundtruth,
            "evaluation": evaluation,
        }

    @staticmethod
    def evaluate(groundtruth, pred):
        """Evaluates the results of the outlier detection algorithm
        Parameters
        ----------
        groundtruth : list
          The groundtruth labels
        pred : list
          The predicted labels
        Returns
        -------
        evaluation : dict
          The evaluation metrics
        """

        cm = sklearn.metrics.confusion_matrix(groundtruth, pred)

        return {
            "groundtruth_indices": np.where(np.array(groundtruth) > 0),
            "pred_indices": np.where(np.array(pred) > 0),
            "roc_auc": sklearn.metrics.roc_auc_score(groundtruth, pred),
            "f1_score": sklearn.metrics.f1_score(groundtruth, pred),
            "acc_score": sklearn.metrics.accuracy_score(groundtruth, pred),
            "jaccard_score": sklearn.metrics.jaccard_score(groundtruth, pred),
            "precision_score": sklearn.metrics.precision_score(groundtruth, pred),
            "average_precision": sklearn.metrics.average_precision_score(
                groundtruth, pred
            ),
            "recall_score": sklearn.metrics.recall_score(groundtruth, pred),
            "hamming_loss": sklearn.metrics.hamming_loss(groundtruth, pred),
            "log_loss": sklearn.metrics.log_loss(groundtruth, pred),
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        }

    def print_results(self, resultsfile):
        """Prints the results of the outlier detection algorithm
        Parameters
        ----------
        resultsfile : str
          The path to the results file
        Returns
        -------
        None
        """
        with open(resultsfile, "rb") as f:
            results = pickle.load(f)

        NO_RUNS = len(results[list(results.keys())[0]])

        for algo in results.keys():
            metrics = {
                method: []
                for method in results[algo][0]["evaluation"].keys()
                if method.find("indices") == -1
            }
            for run in range(0, NO_RUNS):
                for m, value in metrics.items():
                    cur = results[algo][run]["evaluation"][m]

                    value.append(cur)

            print(algo)
            for m, value_ in metrics.items():
                print("   ", m, np.mean(value_), "+/-", np.std(metrics[m]))

    def convert_norm_feature(self, norm, feat):
        """Converts the norm and feature to full names with upper case first letter
        Parameters
        ----------
        norm : str
          The name of the norm
        feat : str
          The name of the feature
        Returns
        -------
        norm, feat : str
          The full names of the norm and feature
        """
        if norm == "max":
            norm = "Max"
        elif norm == "minmax":
            norm = "Min-Max"
        elif norm == "gaussian":
            norm = "Gaussian"

        if feat == "hist":
            feat = "Histogram"
        elif feat == "downsample":
            feat = "Downsample"
        elif feat == "orb":
            feat = "ORB"
        elif feat == "sift":
            feat = "SIFT"

        return norm, feat

    def extract_data(
        self, results_dict, variable="jaccard_score", sort=True, sort_by="mean"
    ):
        """Extracts the data from the results dictionary for use in tables and plots
        Parameters
        ----------
        results_dict : dict
          The results dictionary
        variable : str
          The variable to extract
        sort : bool
            Whether to sort the results
        sort_by : str
            The variable to sort by
        Returns
        -------
        data : dict
          The data dictionary
        """
        values = {}
        NO_RUNS = len(results_dict[list(results_dict.keys())[0]])
        for algo in results_dict.keys():
            norm = results_dict[algo][0]["norm"]
            feat = results_dict[algo][0]["feat"]
            norm, feat = self.convert_norm_feature(norm, feat)
            dataset = results_dict[algo][0]["dataset"]
            scores = []
            for run in range(0, NO_RUNS):
                # if variable is 'f1' and dataset is ASTAR, BSTAR or CSTAR then make it 'f1_score'
                # if variable is 'f1_score' and dataset is A, B or C then make it 'f1'
                if variable == "f1" and dataset in ["ASTAR", "BSTAR", "CSTAR"]:
                    variable = "f1_score"
                elif variable == "f1_score" and dataset in ["A", "B", "C"]:
                    variable = "f1"
                try:
                    scores.append(results_dict[algo][run]["evaluation"][variable])
                except (TypeError, KeyError):
                    continue
            try:
                mean = np.mean(scores)
                std = np.std(scores)
            except ValueError:
                print("error in the following results")
                print("scores", scores)
                print(f"algo: {algo}, norm: {norm}, feat: {feat}, dataset: {dataset}")

            values[algo] = {
                "mean": mean,
                "std": std,
                "norm": norm,
                "feat": feat,
                "dataset": dataset,
            }

        if sort:
            values = sorted(values.items(), key=lambda x: x[1][sort_by], reverse=True)
            values = dict(values)

        return values

    def results_to_latex(
        self, dataset_paths: list, variable="jaccard_score", display=True
    ) -> str:
        """Prints the results from a list of datasets in a latex table format
        similar to the above method, execpt changes based on the number of datasets
        Parameters
        ----------
        dataset_paths : list
            List of paths to the results files
        variable : str
            The variable to print in the table
        display : bool
            If True, the latex table is printed to the console
        Returns
        -------
        latex : str
            The latex table
        """
        results = []
        for dataset in dataset_paths:
            with open(dataset, "rb") as f:
                results.append(pickle.load(f))

        values = [self.extract_data(result, variable) for result in results]
        dataset_names = [value[list(value.keys())[0]]["dataset"] for value in values]
        tabular = "".join("ccl|" for _ in dataset_names)
        tabular = tabular[:-1]

        multicolumn = "".join(
            "\\multicolumn{3}{c}{\\textbf{Dataset %s}} & " % dataset_name
            for dataset_name in dataset_names
        )
        multicolumn = multicolumn[:-2] + "\\\\"

        header = "".join(
            (
                "\\textbf{Algorithm} & \\textbf{Norm. + Feature}  & \\textbf{%s} &"
                % variable.replace("_", " ").title()
            )
            for _ in dataset_names
        )
        header = header[:-1] + "\\\\"

        # create the latex table
        latex = """
        \\begin{table}[!h]
        \\centering
        \\label{tab:results}
        \\caption{ }
        \\resizebox{\\textwidth}{!}{
        \\begin{tabular}{%s}
        \\toprule
        %s
        %s
        \\midrule
        """ % (
            tabular,
            multicolumn,
            header,
        )
        for i in range(0, len(values[0].keys())):
            for value_ in values:
                algo = list(value_.keys())[i]
                latex += (
                    "%s & %s + %s & %.4f &"
                    % (
                        algo,
                        value_[algo]["norm"],
                        value_[algo]["feat"],
                        value_[algo]["mean"],
                    )
                    if value_[algo]["std"] < THRESHOLD
                    else "%s & %s + %s & $%.4f\pm%.4f$ &"
                    % (
                        algo,
                        value_[algo]["norm"],
                        value_[algo]["feat"],
                        value_[algo]["mean"],
                        value_[algo]["std"],
                    )
                )
            latex = latex[:-1] + "\\\\" + "\n"

        latex += """
        \\bottomrule
        \end{tabular}
        }
        \end{table}
        """
        if display:
            print(latex)
        return latex

    def results_to_latex_alt_style(
        self, dataset_paths: list, variable="jaccard_score", display=True, d=4
    ) -> str:
        """Prints the results from a list of datasets in a latex table format
        similar to the above method, execpt format of latex table is different.


        Key to abbreviations:
        Norms: G: Gaussian, M: Max, MM: Min-Max, R: Robust, Z: Z-score
        Features: H: Histogram, S: SIFT, O: ORB,

        Parameters
        ----------
        dataset_paths : list
            List of paths to the results files
        variable : str
            The variable to print in the table
        display : bool
            If True, the latex table is printed to the console
        d : int
            The number of decimal places to print

        Returns
        -------
        latex : str
            The latex table
        """
        results = []
        for dataset in dataset_paths:
            with open(dataset, "rb") as f:
                results.append(pickle.load(f))

        values = [
            self.extract_data(result, variable, sort=False) for result in results
        ]
        dataset_names = [value[list(value.keys())[0]]["dataset"] for value in values]
        # create the latex table
        tabular = "ccl|cl|cl|cl|cl|cl"
        header = """\\textbf{Algorithm}&\multicolumn{2}{c}{\\textbf{A (n=100, x\%)}} & \multicolumn{2}{c}{\\textbf{B (n=100, x\%)}} & \multicolumn{2}{c}{\\textbf{C (n=100, x\%)}}
        &\multicolumn{2}{c}{\textbf{A (x\%)}} & \multicolumn{2}{c}{\\textbf{B (x\%)}} & \multicolumn{2}{c}{\\textbf{C (x\%)}}  \\\\"""

        latex = (
            """
        \\begin{table}[!ht]
        \centering
        \label{tab:final_results}
        \caption{\\textbf{Outlier Detection Results.} AUC ROC scores (the higher, the better) for the evaluated algorithms with best performing normalization and features (G+H: Gaussian and Histogram, M+H: Max and Histogram, MM+S: Min-Max and SIFT\TODO{...}) on our test datasets with varying properties of unwanted images. Our method performed best overall with an average score of \TODO{X$\pm$Y}.}
        \resizebox{\\textwidth}{!}{
        \begin{tabular}{"""
            + tabular
            + """}
        \toprule
        """
            + header
            + """
        \midrule
        """
        )
        for i in range(0, len(values[0].keys())):
            for counter, value_ in enumerate(values):
                if counter == 0:
                    algo = list(value_.keys())[i]
                    norm = self.norm_abbr(value_[algo]["norm"])
                    feat = self.feature_abbr(value_[algo]["feat"])
                    # if std is less than threshold or nan then we don't print std
                    latex += (
                        "%s & %s + %s & %.4f &"
                        % (algo, norm, feat, value_[algo]["mean"])
                        if value_[algo]["std"] < THRESHOLD
                        or value_[algo]["std"] == np.nan
                        else "%s & %s + %s & $%.4f\pm%.4f$ &"
                        % (
                            algo,
                            norm,
                            feat,
                            value_[algo]["mean"],
                            value_[algo]["std"],
                        )
                    )
                else:
                    norm = self.norm_abbr(value_[algo]["norm"])
                    feat = self.feature_abbr(value_[algo]["feat"])
                    latex += (
                        "%s + %s & %.4f &" % (norm, feat, value_[algo]["mean"])
                        if value_[algo]["std"] < THRESHOLD
                        or value_[algo]["std"] == np.nan
                        else "%s + %s & $%.4f\pm%.4f$ &"
                        % (norm, feat, value_[algo]["mean"], value_[algo]["std"])
                    )
            latex = latex[:-1] + "\\\\" + "\n"

        latex += """
        \\bottomrule
        \end{tabular}
        }
        \end{table}
        """
        if display:
            print(latex)
        return latex

    def norm_abbr(self, norm):
        if norm in ["gaussian", "Gaussian"]:
            return "G"
        elif norm in ["max", "Max"]:
            return "M"
        elif norm in ["minmax", "MinMax", "min-max", "Min-Max"]:
            return "MM"
        elif norm in ["robust", "Robust"]:
            return "R"
        elif norm in ["zscore", "ZScore", "z-score", "Z-Score"]:
            return "Z"
        else:
            return norm

    def feature_abbr(self, feature):
        if feature in ["histogram", "Histogram"]:
            return "H"
        elif feature in ["sift", "SIFT"]:
            return "S"
        elif feature in ["orb", "ORB"]:
            return "O"
        else:
            return feature

    def divid_by(self, dataset: str):
        """returns the lenght of the unwanted images in the dataset"""
        if dataset == "A":
            return 8
        elif dataset == "B":
            return 13
        elif dataset == "C":
            return 24
        elif dataset in {"ASTAR", "A*"}:
            return 63
        elif dataset in {"BASTAR", "B*"}:
            return 50
        elif dataset in {"CASTAR", "C*"}:
            return 15
        else:
            return 0

    def get_best_results_for_each_algorithm(
        self,
        dataset_paths: list = None,
        cache_root=OUTPUTDIR,
        variable="roc_auc",
        display=True,
        to_latex=False,
    ) -> dict:
        """For each algorithm prints the best DataSet, Norm + Feature,
        and variable score. So there should be a total of 26 results in final
        dict.
        Parameters
        ----------
        dataset_paths : list
            List of paths to the results files
        cache_root : str
            The root directory to search for results files
        variable : str
            The variable to print in the table
        Returns
        -------
        best_results : dict
            The best results for each dataset based on the variable
        """
        results = []
        if dataset_paths is None:
            dataset_paths = glob.glob(os.path.join(cache_root, "*.pkl"))
        for dataset in dataset_paths:
            with open(dataset, "rb") as f:
                try:
                    results.append(pickle.load(f))
                except:
                    continue
        values = [self.extract_data(result, variable) for result in results]
        dataset_names = [value[list(value.keys())[0]]["dataset"] for value in values]
        # get the best results for each algorithm
        best_results = {}
        for i in range(len(dataset_names)):
            for algo in values[i].keys():
                if algo in best_results:
                    if values[i][algo]["mean"] > best_results[algo]["mean"]:
                        best_results[algo] = values[i][algo]

                else:
                    best_results[algo] = values[i][algo]
        # sort the results by the highest mean
        best_results = sorted(
            best_results.items(), key=lambda x: x[1]["mean"], reverse=True
        )
        best_results = dict(best_results)

        # print the results
        if display:
            print(
                f'Dataset & Algorithm & Norm. + Feature & {variable.replace("_", " ").title()}'
            )
            for algo, value_ in best_results.items():
                print(
                    "%s & %s & %s + %s & %.4f"
                    % (
                        value_["dataset"],
                        algo,
                        best_results[algo]["norm"],
                        best_results[algo]["feat"],
                        best_results[algo]["mean"],
                    )
                )

        if to_latex:
            latex = """
            \\begin{table}[!h]
            \\centering
            \\label{tab:results}
            \\caption{ }
            \\resizebox{\\textwidth}{!}{
            \\begin{tabular}{ccll}
            \\toprule
            \\textbf{Dataset} & \\textbf{Algorithm} & \\textbf{Norm. + Feature}  & \\textbf{%s} \\\\
            \\midrule
            """ % (
                variable.replace("_", " ").title()
            )
            for algo, value__ in best_results.items():
                latex += "%s & %s & %s + %s & %.4f \\\\" % (
                    value__["dataset"],
                    algo,
                    best_results[algo]["norm"],
                    best_results[algo]["feat"],
                    best_results[algo]["mean"],
                )
                latex += "\n"
            latex += """
            \\bottomrule
            \end{tabular}
            }
            \end{table}
            """
            print(latex)

        return best_results

    def get_best_results_per_dataset(
        self,
        dataset_paths: list = None,
        cache_root=OUTPUTDIR,
        variable="roc_auc",
        display=True,
    ) -> dict:
        """Prints the best Algorithm, Norm + Feature and variable score for each dataset
        found in the cache root
        Parameters
        ----------
        dataset_paths : list
            List of paths to the results files
        cache_root : str
            The root directory to search for results files
        variable : str
            The variable to print in the table
        Returns
        -------
        best_results : dict
            The best results for each dataset based on the variable
        """
        results = []
        if dataset_paths is None:
            dataset_paths = glob.glob(os.path.join(cache_root, "*.pkl"))
        for dataset in dataset_paths:
            with open(dataset, "rb") as f:
                try:
                    results.append(pickle.load(f))
                except:
                    continue
        values = [self.extract_data(result, variable) for result in results]
        dataset_names = [value[list(value.keys())[0]]["dataset"] for value in values]
        # get the best results for each dataset
        best_results = {}
        best_alg = ""
        for i in range(len(dataset_names)):
            for algo in values[i].keys():
                if dataset_names[i] in best_results:
                    if values[i][algo]["mean"] > best_results[dataset_names[i]]["mean"]:
                        best_results[dataset_names[i]] = values[i][algo]
                        best_alg = algo

                else:
                    best_results[dataset_names[i]] = values[i][algo]
                    best_alg = algo
        # sort the results by the highest mean
        best_results = sorted(
            best_results.items(), key=lambda x: x[1]["mean"], reverse=True
        )
        best_results = dict(best_results)

        # print the results
        if display:
            print(
                f'Dataset & Algorithm & Norm. + Feature & {variable.replace("_", " ").title()}'
            )
            for dataset, value_ in best_results.items():
                print(
                    "%s & %s & %s + %s & %.4f"
                    % (
                        dataset,
                        best_alg,
                        value_["norm"],
                        best_results[dataset]["feat"],
                        best_results[dataset]["mean"],
                    )
                )

        return best_results

    def print_all_experiments_results(
        self,
        dataset_paths: list = None,
        cache_root=OUTPUTDIR,
        variable="roc_auc",
        to_latex=False,
        display=True,
    ) -> dict:
        """Prints out a report of ALL the results from all the experiments in
         the cache root

        Parameters
        ----------
        dataset_paths : list
            List of paths to the results files
        cache_root : str
            The root directory to search for results files
        variable : str
            The variable to print in the table
        to_latex : bool
            Whether to print the table in latex format
        display : bool
            Whether to print the table to the console

        Returns
        -------
        results : dict
            The results for each dataset

        """
        results = []
        if dataset_paths is None:
            dataset_paths = glob.glob(os.path.join(cache_root, "*.pkl"))
        for dataset in dataset_paths:
            with open(dataset, "rb") as f:
                try:
                    results.append(pickle.load(f))
                except:
                    continue
        values = [self.extract_data(result, variable) for result in results]
        # print out the results
        if display:
            print(
                f'Dataset & Algorithm & Norm. + Feature & {variable.replace("_", " ").title()}'
            )
            for value in values:
                for algo in value.keys():
                    print(
                        "%s & %s & %s + %s & %.4f"
                        % (
                            value[algo]["dataset"],
                            algo,
                            value[algo]["norm"],
                            value[algo]["feat"],
                            value[algo]["mean"],
                        )
                    )

        # print out the results in latex format
        if to_latex:
            latex = """
            \\begin{table}[h]
            \\centering
            \\label{tab:results}
            \\caption{ }
            \\resizebox{\\textwidth}{!}{
            \\begin{tabular}{ccll}
            \\toprule
            \\textbf{Dataset} & \\textbf{Algorithm} & \\textbf{Norm. + Feature}  & \\textbf{%s} \\\\
            \\midrule
            """ % (
                variable.replace("_", " ").title()
            )
            for value in values:
                for algo in value.keys():
                    latex += "%s & %s & %s + %s & %.4f \\\\" % (
                        value[algo]["dataset"],
                        algo,
                        value[algo]["norm"],
                        value[algo]["feat"],
                        value[algo]["mean"],
                    )
                    latex += "\n"
            latex += """
            \\bottomrule
            \end{tabular}
            }
            \end{table}
            """
            print(latex)

        return results

    # def generate_report_tables(self,
    #                            dataset_paths: list = None,
    #                            cache_root=OUTPUTDIR,
    #                            variable='roc_auc',
    #                            to_latex=False,
    #                            display=True) -> dict:
    #     """ Prints out a report of ALL the results from all the experiments in
    #     the cache root
    #
    #     Parameters
    #     ----------
    #     dataset_paths : list
    #         List of paths to the results files
    #     cache_root : str
    #         The root directory to search for results files
    #     variable : str
    #         The variable to print in the table
    #     to_latex : bool
    #         Whether to print the table in latex format
    #     display : bool
    #         Whether to print the table to the console
    #
    #     Returns
    #     -------
    #     results : dict
    #         The results for each dataset
    #
    #     """
    #     results = []
    #     if dataset_paths is None:
    #         dataset_paths = glob.glob(os.path.join(cache_root, '*.pkl'))
    #     for dataset in dataset_paths:
    #         with open(dataset, 'rb') as f:
    #             try:
    #                 results.append(pickle.load(f))
    #             except:
    #                 continue
    #     # extract the data from the results
    #     values = []
    #     for result in results:
    #         values.append(self.extract_data(result, variable))
    #
    #     # print out values if dataset is BStar or CStar
    #     for value in values:
    #         for algo in value.keys():
    #             if value[algo]['dataset'] == 'BStar' or 'CStar':
    #                 print('%s & %s & %s + %s & %.4f' % (value[algo]['dataset'],
    #                                                     algo,
    #                                                     value[algo]['norm'],
    #                                                     value[algo]['feat'],
    #                                                     value[algo]['mean']))
    #
    #     datasetA = []
    #     datasetB = []
    #     datasetC = []
    #     datasetASTAR = []
    #     datasetBSTAR = []
    #     datasetCSTAR = []
    #
    #     # move the results into the correct dataset
    #     for value in values:
    #         for algo in value.keys():
    #             # if mean is nan then skip
    #             if np.isnan(value[algo]['mean']):
    #                 continue
    #             if value[algo]['dataset'] == 'A':
    #                 datasetA.append(
    #                     ({'algorithm': algo, 'results': value[algo]}))
    #             elif value[algo]['dataset'] == 'B':
    #                 datasetB.append(
    #                     ({'algorithm': algo, 'results': value[algo]}))
    #             elif value[algo]['dataset'] == 'C':
    #                 datasetC.append(
    #                     ({'algorithm': algo, 'results': value[algo]}))
    #             elif value[algo]['dataset'] == 'ASTAR' or 'A*':
    #                 datasetASTAR.append(
    #                     ({'algorithm': algo, 'results': value[algo]}))
    #             elif value[algo]['dataset'] == 'BSTAR' or 'B*':
    #                 datasetBSTAR.append(
    #                     ({'algorithm': algo, 'results': value[algo]}))
    #             elif value[algo]['dataset'] == 'CSTAR' or 'C*':
    #                 datasetCSTAR.append(
    #                     ({'algorithm': algo, 'results': value[algo]}))
    #
    #     # sort the results by the mean
    #     datasetA = sorted(datasetA, key=lambda k: k['results']['mean'],
    #                       reverse=True)
    #     datasetB = sorted(datasetB, key=lambda k: k['results']['mean'],
    #                       reverse=True)
    #     datasetC = sorted(datasetC, key=lambda k: k['results']['mean'],
    #                       reverse=True)
    #     datasetASTAR = sorted(datasetASTAR, key=lambda k: k['results']['mean'],
    #                           reverse=True)
    #     datasetBSTAR = sorted(datasetBSTAR, key=lambda k: k['results']['mean'],
    #                           reverse=True)
    #     datasetCSTAR = sorted(datasetCSTAR, key=lambda k: k['results']['mean'],
    #                           reverse=True)
    #
    #     best_each_alg_datasetA = self._best_algorithms(datasetA)
    #     best_each_alg_datasetB = self._best_algorithms(datasetB)
    #     best_each_alg_datasetC = self._best_algorithms(datasetC)
    #     best_each_alg_datasetASTAR = self._best_algorithms(datasetASTAR)
    #     best_each_alg_datasetBSTAR = self._best_algorithms(datasetBSTAR)
    #     best_each_alg_datasetCSTAR = self._best_algorithms(datasetCSTAR)
    #
    #     print('Dataset BSTAR')
    #     print(datasetBSTAR)
    #
    #     print('Dataset best BSTAR')
    #     pprint(best_each_alg_datasetBSTAR)

    # print out the results
    # if display:
    #     print('Dataset A')
    #     self._print_results(best_each_alg_datasetA, variable)
    #     print('Dataset B')
    #     self._print_results(best_each_alg_datasetB, variable)
    #     print('Dataset C')
    #     self._print_results(best_each_alg_datasetC, variable)
    #     print('Dataset A*')
    #     self._print_results(best_each_alg_datasetASTAR, variable)
    #     print('Dataset B*')
    #     self._print_results(best_each_alg_datasetBSTAR, variable)
    #     print('Dataset C*')
    #     self._print_results(best_each_alg_datasetCSTAR, variable)
    #
    # names = ['A', 'B', 'C', 'A*', 'B*', 'C*']
    # best_each_alg = [best_each_alg_datasetA, best_each_alg_datasetB,
    #                  best_each_alg_datasetC, best_each_alg_datasetASTAR,
    #                  best_each_alg_datasetBSTAR, best_each_alg_datasetCSTAR]
    #
    # if to_latex:
    #     for name, value in zip(names, best_each_alg):
    #         latex = '''
    #         \\begin{table}[H]
    #         \\centering
    #         \\caption{Results for dataset %s}
    #         \\label{tab:results%s}
    #         {
    #         \\begin{tabular}{lrrrr}
    #         \\toprule
    #         Algorithm & Norm & Features & Mean & Std \\
    #         \\midrule
    #         ''' % (name, name)
    #         for algo in value.keys():
    #             latex += '%s & %s & %s & %.3f & %.3f \\\\' % (
    #                 algo,
    #                 value[algo]['norm'],
    #                 value[algo]['feat'],
    #                 value[algo]['mean'],
    #                 value[algo]['std'])
    #             latex += '\n'
    #         latex += '''
    #         \\bottomrule
    #         \end{tabular}
    #         }
    #         \end{table}
    #         '''
    #         print(latex)

    # return best_each_alg

    def _best_algorithms(self, datalist):
        best_algorithms = {}
        for data in datalist:
            # if mean is nan, then skip it
            if np.isnan(data["results"]["mean"]):
                continue
            # if the algorithm is not in the dictionary, add it
            if data["algorithm"] not in best_algorithms:
                best_algorithms[data["algorithm"]] = data
            elif (
                    data["results"]["mean"]
                    > best_algorithms[data["algorithm"]]["results"]["mean"]
                ):
                best_algorithms[data["algorithm"]] = data

        # remove the inner key for each algorithm
        for key, value in best_algorithms.items():
            best_algorithms[key] = value["results"]

        return best_algorithms

    def __number_bad(self, dataset):
        """Returns the number of bad images in the dataset"""
        # make a switch for the dataset
        if dataset == "A":
            return 8
        elif dataset == "B":
            return 13
        elif dataset in ["C", "D"]:
            return 24
        else:
            return 63

    def _print_results(self, best_each_alg_datasetA, variable):
        """Prints the results in a table format"""
        # it is dataset algorithm norm feat    tp/totalbad    accuracy
        print(
            f'{"Dataset":<7} {"Alg":<5} {"Norm":<5} {"Feat":<5} '
            f'{variable:<18} {"std":<10}'
        )
        # add formating to right align the values evenly from left side with 2 decimal places
        for k, v in best_each_alg_datasetA.items():
            print(
                f'{v["dataset"]:<7} {k:<5} {v["norm"]:<5} '
                f'{v["feat"]:<5} {v["mean"]:<18.4f} {v["std"]:<10.4f}'
            )


def sort_dict_by_value(
    d: dict, reverse=True, variable="tp/totalbad", display=True
) -> dict:
    """Sorts a dictionary by its values

    Parameters
    ----------
    d : dict
        The dictionary to sort
    reverse : bool
        Whether to sort in reverse order
    variable : str
        The variable be displayed in the table
    display : bool

    Returns
    -------
    sorted_dict : dict
        The sorted dictionary

    """
    sorted_dict = dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))

    if display:
        # it is dataset algorithm norm feat    tp/totalbad    accuracy
        print(
            f'{"Dataset":<7} {"Alg":<5} {"Norm":<5} {"Feat":<5} '
            f"{variable:<18} {variable:<10}"
        )
        # add formating to right align the values evenly from left side with 2 decimal places
        for k, v in sorted_dict.items():
            print(f"{k:<40} {v:>10.2f}")

    return sorted_dict


# sort dict by value version 2, only has the algorithm and the value
def sort_dict_by_value_v2(
    d: dict, reverse=True, variable="tp/totalbad", display=True
) -> dict:
    """Sorts a dictionary by its values

    Parameters
    ----------
    d : dict
        The dictionary to sort
    reverse : bool
        Whether to sort in reverse order
    variable : str
        The variable be displayed in the table
    display : bool

    Returns
    -------
    sorted_dict : dict
        The sorted dictionary

    """
    sorted_dict = dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))

    if display:
        # it is dataset algorithm norm feat    tp/totalbad    accuracy
        print(f'{"Alg":<25} {variable:<10}')
        # add formating to right align the values evenly from left side with 2 decimal places
        for k, v in sorted_dict.items():
            print(f"{k:<20} {v:>10.2f}")

    return sorted_dict
