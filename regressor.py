import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from constants import ASSIGNEMENT
from toolbox import load_all_data, get_error_dfs
from predictors import pred_same_last_week, pred_reg_two_weeks, \
                       pred_same_last_year, pred_reg_two_years


class Regressor(BaseEstimator):
    def __init__(self):
        """Load all the data during the init.
        """
        self.coef_first_preds = 1.22
        self.coef_final_preds = 1.16
        self.X_df = load_all_data()
        self.regs = []

    def _fit_simple_predictors_(self, dates):
        """Fit the simple predictor on self.X_df.
        """
        pred = []
        pred.append(pred_same_last_week(dates, self.X_df))
        pred.append(pred_reg_two_weeks(dates, self.X_df))
        pred.append(pred_same_last_year(dates, self.X_df))
        pred.append(pred_reg_two_years(dates, self.X_df))
        return pd.concat(pred, axis=1)

    def preprocess_data(self, df, full=False):
        """Enhance features.
        """
        if full:
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            df['hour'] = df.index.hour
            df['min'] = df.index.minute
            MINHOUR, MAXHOUR = 7, 19
            is_day = (df.index.hour >= MINHOUR) * (df.index.hour < MAXHOUR)
            df['is_day'] = is_day
            is_we = ((df.index.dayofweek == 6)+(df.index.dayofweek == 5)).astype(bool)
            df['is_we'] = is_we
        else:
            df['epoch'] = df.index.astype(int)
        return df

    def fit(self, dates):
        """Fit all the intern predictors.
        """
        concat_pred = self._fit_simple_predictors_(dates)
        regs = []
        features_full = ['Week','Evol_week','Year','Evol_year', 'year', 'month',
                         'day', 'hour' ,'min', 'is_day', 'is_we', 'y_true']
        features = ['Week','Evol_week','Year','Evol_year', 'epoch', 'y_true']
        for assign in ASSIGNEMENT:
            fit_dataset = self.preprocess_data(concat_pred[assign])
            fit_dataset['y_true'] = self.X_df[assign]
            fit_dataset.columns = features
            fit_dataset = fit_dataset[np.isfinite(fit_dataset['y_true'])]
            X_fit = fit_dataset.drop(['y_true'], axis=1)
            y_fit = fit_dataset['y_true']
            reg = GradientBoostingRegressor(n_estimators=1000,                             
                learning_rate=0.2, random_state=42)             
            reg.fit(X_fit, y_fit)
            self.regs.append(reg)

    def predict(self, dates):
        """Create a prediction for the given date.
        """
        concat_pred = self._fit_simple_predictors_(dates) * self.coef_first_preds
        tot_pred = []
        features_full = ['Week','Evol_week','Year','Evol_year', 'year', 'month',
                         'day', 'hour' ,'min', 'is_day', 'is_we']
        features = ['Week','Evol_week','Year','Evol_year', 'epoch']
        for i, assign in enumerate(ASSIGNEMENT):
            pred_dataset = self.preprocess_data(concat_pred[assign])
            pred_dataset.columns = features
            y_pred = self.regs[i].predict(pred_dataset)
            y_pred[y_pred < 0] = 0
            y_pred = y_pred.astype(int)
            tot_pred.append(pd.DataFrame(y_pred, index=dates))
        concat_y_pred = pd.concat(tot_pred, axis=1)
        concat_y_pred *= self.coef_final_preds
        concat_y_pred = pd.DataFrame(concat_y_pred, index=dates)
        concat_y_pred.columns = ASSIGNEMENT
        return concat_y_pred
    
    def gridsearch(self,dates,param_grid, verbose=1):
        """Create a prediction for the given date.
        """
        concat_pred = self._fit_simple_predictors_(dates) * self.coef_first_preds
        features_full = ['Week','Evol_week','Year','Evol_year', 'year', 'month',
                         'day', 'hour' ,'min', 'is_day', 'is_we']
        features = ['Week','Evol_week','Year','Evol_year', 'epoch', 'y_true']        
        scoresgridreg={}
        for i,assign in enumerate(ASSIGNEMENT):
            i+=1
            print "--- Gridsearching %d/28 : %s" % (i, assign)
            fit_dataset = self.preprocess_data(concat_pred[assign])
            fit_dataset['y_true'] = self.X_df[assign]
            fit_dataset.columns = features
            fit_dataset = fit_dataset[np.isfinite(fit_dataset['y_true'])]
            X_fit = fit_dataset.drop(['y_true'], axis=1)
            y_fit = fit_dataset['y_true']
            reg = GradientBoostingRegressor(n_estimators=1000,                             
                learning_rate=0.2, random_state=42)               
            grid = GridSearchCV(reg, param_grid=param_grid, scoring=make_scorer(get_error_dfs), verbose=verbose)
            grid.fit(X_fit, y_fit)
            scoresgridreg[assign]=grid.best_params_, grid.best_score_
        tot_score=0
        print "============================================================================"
        print "                              GRID SEARCH                                   "
        print "----------------------------------------------------------------------------"
        for assign in ASSIGNEMENT:
            print assign,' : ', scoresgridreg[assign][0]
            tot_score+=scoresgridreg[assign][1]
        print "----------------------------------------------------------------------------"
        print  "**************SCORE**************: ",tot_score

