"""a module that defines PredictionMixin which uses sklearn to make predictions about results"""
# core data
import numpy
import numpy as np
import time
import pandas
import random

# store average and variance in a dict
base_stat = {"avg": 0, "var": 0}


class PredictionMixin:
    train_window: int = 5000  # number of data points to use for training
    prediction_goal_error: float = 0.02  # goal error for training 99%
    trained: bool = False

    _basis: np.ndarray = None  # basis for normalizing  data
    _prediction_models: dict = (
        None  # parm: {'mod':obj,'train_time':float,'N':int,'train_score':scr}
    )
    _prediction_parms: list = None  # list of strings to get from dataframe
    _re_train_frac: float = 2  # retrain when N_data > model.N * _re_train_frac
    _re_train_maxiter: int = 50  # max number of retraining iterations
    _running_error: dict = None  # parm: {'err':float,'N':int}
    _fitted: bool = False
    _training_history = None
    _do_print = False

    _sigma_retrain = 3  # retrain when sigma is greater than this
    _X_prediction_stats: list = None  # list of strings to get from dataframe
    _prediction_records: list = None  # made lazily, but you can replace this

    @property
    def prediction_records(self):
        if self._prediction_records is None:
            self._prediction_records = []
        return self._prediction_records

    @property
    def _prediction_record(self):
        raise NotImplementedError(f"must implement _prediction_record")

    @property
    def basis(self):
        return self._basis

    def add_prediction_record(
        self, record, extra_add=True, mult_sigma=1, target_items=1000
    ):
        """adds a record to the prediction records, and calcultes the average and variance of the data
        :param record: a dict of the record
        :param extra_add: if true, the record is added to the prediction records even if the data is inbounds
        :returns: a boolean indicating if the record was out of bounds of current data (therefor should be added)
        """
        if self._X_prediction_stats is None:
            self._X_prediction_stats = {
                p: base_stat.copy() for p in self._prediction_parms
            }

        N = len(self.prediction_records) + 1
        Pf = max((N / target_items), 0.1)
        J = len(self._prediction_parms)

        # moving average and variance
        out_of_bounds = extra_add  # skip this check
        choice = False
        temp_stat = {}
        for i, parm in enumerate(self._prediction_parms):
            rec_val = record[parm]
            cur_stat = self._X_prediction_stats[parm]
            temp_stat[parm] = cur_stat = cur_stat.copy()
            avg = cur_stat["avg"]
            dev = rec_val - avg
            var = cur_stat["var"]
            std = var**0.5
            std_err = std / N
            # check out of bounds
            # probability of acceptance - narrow distributions are not wanted
            near_zero = abs(rec_val) / max(abs(avg), 1) <= 0.005
            if (
                (not choice or not out_of_bounds)
                and avg != 0
                and std != 0
                and not near_zero
            ):
                g = 3 * min(
                    (abs(avg) - std) / abs(std), 1
                )  # negative when std > avg
                a = min(abs(dev) / std / J, 1)
                # prob accept should be based on difference from average
                prob_deny = np.e**g
                prob_accept = a / Pf
                std_choice = std
                avg_choice = avg
                dev_choice = dev
                choice = random.choices(
                    [True, False], [prob_accept, prob_deny]
                )[0]
            cur_stat["avg"] = avg + dev / N
            cur_stat["var"] = var + (dev**2 - var) / N

        # observe point and predict to get error
        self.observe_and_predict(record)

        # update stats if point accepted
        accepted = choice or out_of_bounds or extra_add
        if accepted:
            for parm, cur_stat in temp_stat.items():
                self._X_prediction_stats[parm] = cur_stat  # reassign (in case)

        # fmt null values
        if not choice:
            std_choice = ""
            avg_choice = ""
            dev_choice = ""
            prob_accept = ""
            prob_deny = ""

        if accepted:
            if self._do_print:
                self.info(
                    f"add record | chance: {choice and not extra_add} | {prob_accept:5.3f} | {prob_deny:5.3f} | {std_choice/avg_choice} | {dev_choice/std_choice} | {avg_choice}"
                )
            self._prediction_records.append(record)

            try:
                self.check_and_retrain(self.prediction_records)
            except Exception as e:
                self.warning(f"error in prediction: {e}")

    def check_out_of_domain(self, record, extra_margin=1, target_items=1000):
        """checks if the record is in bounds of the current data"""
        if self._X_prediction_stats is None:
            return True

        if hasattr(self, "max_rec_parm") and self.max_rec_parm in record:
            # we're not counting extemities
            if record[self.max_rec_parm] > self.max_margin:
                return False

        N = len(self.prediction_records)
        Pf = max((N / target_items), 0.1)
        J = len(self._prediction_parms)

        for i, parm in enumerate(self._prediction_parms):
            rec_val = record[parm]
            cur_stat = self._X_prediction_stats[parm]
            avg = cur_stat["avg"]
            dev = rec_val - avg
            var = max(cur_stat["var"], 1)
            std = var**0.5
            std_err = std / len(self.prediction_records)
            # check out of bounds
            g = 3 * min(
                (abs(avg) - std) / abs(std), 1
            )  # negative when std > avg
            a = min(abs(dev) / std / J, 1)
            near_zero = abs(rec_val) / max(abs(avg), 1) <= 1e-3
            prob_deny = np.e**g
            prob_accept = a * extra_margin / Pf  # relative to prob_deny
            choice = random.choices([True, False], [prob_accept, prob_deny])[0]
            if choice and not near_zero:
                if self._do_print:
                    self.info(
                        f"record oob chance: {choice} | {prob_accept:5.3f} | {prob_deny:5.3f} | {std/abs(avg)} | {dev/std} | {avg} "
                    )
                return True
        return False

    def train_compare(self, df, test_frac=2, train_full=False, min_rec=250):
        """Use the dataframe to train the models, and compare the results to the current models using `train_frac` to divide total samples into training and testing sets, unless `train_full` is set.

        :param df: dataframe to train with
        :param test_frac: N/train_frac will be size of the training window
        :param train_full: boolean to use full training data
        :return: trained models
        """
        if self._prediction_models is None or self._basis is None:
            return {}

        if self._training_history is None:
            train_iter = {}
            self._training_history = []
        else:
            train_iter = {}

        out = {}
        N = len(df)
        window = self.train_window
        if window > (N / test_frac):
            window = max(int(N / test_frac), 25)

        MargRec = max(min_rec / len(self.prediction_records), 1)

        self.info(f"training dataset: {N} | training window: {window}")
        for parm, mod_dict in self._prediction_models.items():
            mod = mod_dict["mod"]
            X = df[self._prediction_parms] / self._basis
            y = df[parm]
            N = len(y)

            stl = time.time()
            weights = self.prediction_weights(df, N)
            if train_full:
                mod.fit(X, y, weights)
                etl = time.time()
                scr = mod.score(X, y, weights) / MargRec
                train_iter[parm] = {"scr": scr, "time": etl - stl, "N": N}
            else:
                mod.fit(*self._subsample_data(X, y, window, weights))
                etl = time.time()
                scr = mod.score(*self._score_data(X, y, weights)) / MargRec
                train_iter[parm] = {"scr": scr, "time": etl - stl, "N": N}

            dt = etl - stl

            self.info(
                f"Prediction: {mod.__class__.__name__}| Score[{parm}] = {scr*100:3.5}% | Training Time: {dt}s"
            )

            out[parm] = {
                "mod": mod,
                "train_time": dt,
                "N": N,
                "train_score": scr,
            }

        self._running_error = {
            parm: {"err": 0.5, "N": 0} for parm in self._prediction_models
        }
        self._prediction_models = out
        self._fitted = True

        # add training item to history
        self._training_history.append(train_iter)

        # do something when trained
        self.training_callback(self._prediction_models)

        return out

    def training_callback(self, models):
        """override to provide a callback when training is complete, such as saving the models"""
        pass

    def prediction_weights(self, df, window):
        return np.ones(min(len(df), window))

    def _subsample_data(self, X, y, window, weights):
        """subsamples the data to the window size"""
        return X.iloc[:window], y.iloc[:window], weights[:window]

    def _score_data(self, X, y, weights):
        """override this, by default, just returns the data to the score"""
        return X, y, weights

    def score_data(self, df):
        """scores a dataframe"""

        if self._prediction_models is None:
            return

        if not self._fitted:
            return

        train_iter = {}
        self._training_history.append(train_iter)

        scores = {}
        for parm, mod_dict in self._prediction_models.items():
            model = mod_dict["mod"]
            X = df[self._prediction_parms] / self._basis
            y = df[parm]
            scr = model.score(X, y)
            scores[parm] = scr
            train_iter[parm] = {"scr": scr, "time": np.nan, "N": len(y)}
            if self._do_print:
                self.info(
                    f"Prediction: {model.__class__.__name__}| Score[{parm}] = {scr*100:3.5}%"
                )
            else:
                self.debug(
                    f"Prediction: {model.__class__.__name__}| Score[{parm}] = {scr*100:3.5}%"
                )

        return scores

    def observe_and_predict(self, row):
        """uses the existing models to predict the row and measure the error"""
        if self._prediction_models is None:
            # self.warning('No models to predict with')
            return

        if self._running_error is None:
            self._running_error = {
                parm: {"err": 0, "N": 0} for parm in self._prediction_models
            }

        if not self._fitted:
            return

        if not all([p in row for p in self._prediction_parms]):
            return

        for parm, mod_dict in self._prediction_models.items():
            model = mod_dict["mod"]
            if parm not in row:
                continue
            X = pandas.DataFrame(
                [{parm: row[parm] for parm in self._prediction_parms}]
            )
            y = row[parm]
            x = abs(model.predict(X))
            if y == 0:
                if x == 0:
                    err = 0
                else:
                    err = 1
            else:
                y = abs(y)
                err = (y - x) / y
            cerr = self._running_error[parm]["err"]
            N = self._running_error[parm]["N"] + 1
            self._running_error[parm]["err"] = cerr + (err - cerr) / N
            self._running_error[parm]["N"] = N

    def check_and_retrain(self, records, min_rec=None):
        """Checks if more data than threshold to train or if error is sufficiently low to ignore retraining, or if more data already exists than window size (no training)"""
        if self._prediction_models is None:
            return

        if not hasattr(self, "min_rec"):
            min_rec = 50
        else:
            min_rec = self.min_rec

        if self.trained:
            if len(records) % min_rec == 0:
                df = self.prediction_dataframe(records)
                self.score_data(df)
            return

        Nrec = len(records)
        if Nrec < min_rec:
            return

        df = self.prediction_dataframe(records)

        if not self._fitted:
            self.train_compare(df)
            return

        # an early check if the data is more than the target or if the error is less than the target
        self.trained = all(
            [
                (1 - mod_dict["train_score"]) <= self.prediction_goal_error
                for mod_dict in self._prediction_models.values()
            ]
        )

        for parm, mod_dict in self._prediction_models.items():
            model = mod_dict["mod"]
            N = mod_dict["N"]
            #
            if (
                Nrec > N * self._re_train_frac
                or (Nrec - N) > self._re_train_maxiter
            ):
                self.train_compare(df)
                return

    def prediction_dataframe(self, records):
        df = pandas.DataFrame(records)._get_numeric_data()
        df[np.isnan(df)] = 0  # zeros baybee
        return df
