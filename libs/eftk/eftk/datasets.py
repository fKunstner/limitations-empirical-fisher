import numpy as np
import os
import sklearn as sk
import sklearn.datasets

DATA_ROOT = os.path.expanduser("~/data")


class Dataset():
    def __init__(self, X, y, X_te=None, y_te=None, split=None):
        provided_test_set = (X_te is not None) and (y_te is not None)
        provided_split = split is not None

        if provided_split and provided_test_set:
            raise ValueError
        if provided_split:
            if isinstance(split, float) and ((0.0 < split) and (split < 1.0)):
                split = np.random.rand(X.shape[0]) < split
            self.X_tr = X[split]
            self.y_tr = y[split]
            self.X_te = X[~split]
            self.y_te = y[~split]
            self.N_te = self.X_te.shape[0]
        else:
            self.X_tr = X
            self.y_tr = y
            if provided_test_set:
                self.X_te = X_te
                self.y_te = y_te
                self.N_te = X_te.shape[0]
            else:
                self.X_te = None
                self.y_te = None
                self.N_te = None
        self.N_tr = self.X_tr.shape[0]
        self.D = self.X_tr.shape[1]

    def split(self, split):
        return Dataset(self.X_tr, self.y_tr, split)

    def add_bias(self):
        self.X_tr = np.append(self.X_tr, np.ones((self.X_tr.shape[0], 1)), axis=1)
        if self.X_te is not None:
            self.X_te = np.append(self.X_te, np.ones((self.X_te.shape[0], 1)), axis=1)
        self.D = self.X_tr.shape[1]
        return self

    def normalize_output(self):
        self.y_tr = self.y_tr - np.mean(self.y_tr)
        self.y_tr /= np.std(self.y_tr)

    def normalize_input(self):
        scaler = sk.preprocessing.StandardScaler()
        self.X_tr = scaler.fit_transform(self.X_tr)
        if self.X_te is not None:
            self.X_te = scaler.transform(self.X_te)

    def subsample(self, N):
        mask = np.isin(np.array(range(self.N_tr)), np.random.choice(self.N_tr, N, replace=False))
        return Dataset(self.X_tr[mask, :], self.y_tr[mask], self.X_te, self.y_te), Dataset(self.X_tr[~mask, :], self.y_tr[~mask])

    def get_train(self):
        return self.X_tr, self.y_tr

    def get_test(self):
        return self.X_te, self.y_te


def load_boston():
    X, y = sk.datasets.load_boston(return_X_y=True)
    return Dataset(X, y)


def load_iris():
    X, y = sk.datasets.load_iris(return_X_y=True)
    return Dataset(X, y)


def load_wine():
    return Dataset(*sk.datasets.load_wine(return_X_y=True))


def load_diabetes():
    X, y = load_libsvm(os.path.join("diabetes", "diabetes.libsvm"))
    X = X.toarray()
    return Dataset(X, (y + 1) / 2)


def load_breast_cancer():
    return Dataset(*sk.datasets.load_breast_cancer(return_X_y=True))


def load_uci(dataset_folder):
    data_folder = os.path.join(DATA_ROOT, dataset_folder)
    data_file = os.path.join(data_folder, "data.txt")
    idx_target = os.path.join(data_folder, "index_target.txt")
    idx_features = os.path.join(data_folder, "index_features.txt")

    data = np.loadtxt(data_file)
    idx_target = np.loadtxt(idx_target, dtype=int).tolist()
    idx_features = np.loadtxt(idx_features, dtype=int).tolist()

    X = data[:, idx_features]
    y = data[:, idx_target]
    return Dataset(X, y)


def load_energy():
    return load_uci("energy")


def load_powerplant():
    return load_uci("power-plant")


def load_yacht():
    return load_uci("yacht")


def load_libsvm(dataset_file):
    path = os.path.join(DATA_ROOT, dataset_file)
    return sk.datasets.load_svmlight_file(path)


def load_australian():
    X, y = load_libsvm(os.path.join("australian", "australian.libsvm"))
    X = X.toarray()
    return Dataset(X, (y + 1) / 2)


def load_a1a():
    X, y = load_libsvm(os.path.join("a1a", "a1a.libsvm"))
    X = X.toarray()
    return Dataset(X, (y + 1) / 2)


name_to_function = {
    "Boston": load_boston,
    "Iris": load_iris,
    "Wine": load_wine,
    "Diabetes": load_diabetes,
    "BreastCancer": load_breast_cancer,
    "Energy": load_energy,
    "Powerplant": load_powerplant,
    "Yacht": load_yacht,
    "Australian": load_australian,
    "a1a": load_a1a,
}


def available_datasets():
    return list(name_to_function.keys())


def loader(dsname):
    return name_to_function[dsname]


def load(dsname):
    return loader(dsname)()
