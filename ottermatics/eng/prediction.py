"""a module that defines PredictionMixin which uses sklearn to make predictions about results"""
#core data
import numpy
import numpy as np
import time
import pandas

class PredictionMixin:

    _basis: np.ndarray = None #basis for normalizing  data 
    _prediction_models: dict = None #parm: {'mod':obj,'train_time':float,'N':int,'train_score':scr}
    _prediction_parms: list = None #list of strings to get from dataframe
    _re_train_frac: float = 2 #retrain when N_data > model.N * _re_train_frac
    _re_train_maxiter: int = 100 #max number of retraining iterations
    train_window: int = 2500 #number of data points to use for training
    _goal_error: float = 0.05 #goal error for training 99%
    _running_error: dict = None #parm: {'err':float,'N':int}
    _fitted:bool = False
    _training_history = None
    trained: bool = False

    def train_compare(self,df,do_print=True):
        if self._prediction_models is None:
            return {}
        
        if self._training_history is None:
            train_iter = {}
            self._training_history = [train_iter]
        else:
            train_iter = {}
            self._training_history.append(train_iter)

        out = {}
        window = self.train_window
        for parm,mod_dict in self._prediction_models.items():
            mod = mod_dict['mod']
            X = df[self._prediction_parms]/self._basis
            y = df[parm]
            N = len(y)
            stl = time.time()
            mod.fit(X[:window],y[:window])
            etl = time.time()
            scr = mod.score(X,y)
            train_iter[parm] = {'scr':scr,'time':etl-stl,'N':N}
            dt = etl-stl
            if do_print:
                self.info(f'Prediction: {mod.__class__.__name__}| Score[{parm}] = {scr*100:3.5}% | Training Time: {dt}s')

            out[parm] = {'mod':mod,'train_time':dt,'N':N,'train_score':scr}
        
        self._running_error = {parm:{'err':0.5,'N':0} for parm in self._prediction_models}
        self._prediction_models = out
        self._fitted = True
        return out

    def score_data(self,df,do_print=True):
        """scores a dataframe"""

        if self._prediction_models is None:
            return        
        
        if not self._fitted:
            return                
        
        train_iter = {}
        self._training_history.append(train_iter)

        scores = {}        
        for parm,mod_dict in self._prediction_models.items():
            model = mod_dict['mod']
            X = df[self._prediction_parms]/self._basis
            y = df[parm]
            scr = model.score(X,y)
            scores[parm] = scr
            train_iter[parm] = {'scr':scr,'time':np.nan,'N':len(y)}
            if do_print:
                self.info(f'Prediction: {model.__class__.__name__}| Score[{parm}] = {scr*100:3.5}%')

        return scores
    
    def observe_and_predict(self,row):
        """uses the existing models to predict the row and measure the error"""
        if self._prediction_models is None:
            #self.warning('No models to predict with')
            return
        
        if self._running_error is None:
            self._running_error = {parm:{'err':0,'N':0} for parm in self._prediction_models}

#         if not self._fitted:
#             return
#         
#         if not all([p in row for p in self._prediction_parms]):
#             return
# 
#         for parm,mod_dict in self._prediction_models.items():
#             model = mod_dict['mod']
#             if parm not in row:
#                 continue
#             X = pandas.DataFrame([{parm:row[parm] for parm in self._prediction_parms}])
#             y = row[parm]
#             x = abs(model.predict(X))
#             if y == 0:
#                 if x==0:
#                     err = 0
#                 else:
#                     err = 1
#             else:
#                 y = abs(y)
#                 err = (x - y)/y
#             cerr = self._running_error[parm]['err']
#             N = self._running_error[parm]['N'] + 1
#             self._running_error[parm]['err'] = cerr + (err-cerr)/N
#             self._running_error[parm]['N'] = N


    def check_and_retrain(self,records,min_rec=50):
        """Checks if more data than threshold to train or if error is sufficiently low to ignore retraining, or if more data already exists than window size (no training)"""
        if self._prediction_models is None:
            return        

        if self.trained:
            return
        
        Nrec = len(records)
        if Nrec < min_rec:
            return
        
        if Nrec > self.train_window*self._re_train_frac*1.1:
            return 
        
        df = self.prediction_dataframe(records)

        if not self._fitted:
            self.train_compare(df)
            return

        #an early check if the data is more than the target or if the error is less than the target
        self.trained = all([(1-mod_dict['train_score']) <= self._goal_error for mod_dict in self._prediction_models.values()])

        for parm,mod_dict in self._prediction_models.items():
            model = mod_dict['mod']
            N = mod_dict['N']
            #
            if Nrec > N * self._re_train_frac or (Nrec-N) > self._re_train_maxiter:
                self.train_compare(df)
                return

    def prediction_dataframe(self,records):
        df = pandas.DataFrame(records)
        df[np.isnan(df)] = 0 #zeros baybee
        return df