methods = {'SVR': "SVR(**params, gamma='auto')", 'SVC':"SVC(**params, gamma='auto')",
            'XGBR':"xgb.XGBRegressor(**params, verbosity=0, nthread=1)",
            'XGBC':"xgb.XGBClassifier(**params, verbosity=0, nthread=1)",
            'RFR':"RandomForestRegressor(**params)",
            'RFC':"RandomForestClassifier(**params)"}

calculators = {
    'circus':"ChythonCircus(**descriptor_params)",
    'chyline':"ChythonLinear(**descriptor_params)",
    'morgan':"Fingerprinter(fp_type='morgan', **descriptor_params)",
    'morganfeatures':"Fingerprinter(fp_type='morgan', params={'useFeatures':True}, **descriptor_params)",
    'rdkfp':"Fingerprinter(fp_type='rdkfp', **descriptor_params)",
    'rdkfplinear':"Fingerprinter(fp_type='rdkfp', params={'branchedPaths':False}, **descriptor_params)",
    'layered':"Fingerprinter(fp_type='layered', **descriptor_params)",
    'atompairs':"Fingerprinter(fp_type='atompairs', **descriptor_params)",
    'avalon':"Fingerprinter(fp_type='avalon', **descriptor_params)",
    'torsion':"Fingerprinter(fp_type='torsion', **descriptor_params)",
    'mordred2d':"Mordred2DCalculator(**descriptor_params)",
}

def suggest_params(trial, method):
    if method == 'SVR':
        params = { 
            'C': trial.suggest_float('C', 1e-9, 1e9, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
            'coef0': trial.suggest_float('r0', -10, 10)
        }
    elif method == 'SVC':
        params = { 
            'C': trial.suggest_float('C', 1e-9, 1e9, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
            'coef0': trial.suggest_float('r0', -10, 10)
        }   
    elif method == 'LGBMR':
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, step=10),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
        }
    elif method == 'XGBR':
        params = {
            'max_depth': trial.suggest_int("max_depth", 3, 18),
            'eta': trial.suggest_float('eta', 1e-9, 1, log=True),
            'gamma': trial.suggest_float ('gamma', 1,9),
            'reg_alpha' : trial.suggest_int('reg_alpha', 10, 180),
            'reg_lambda' : trial.suggest_float('reg_lambda', 0, 1),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1),
            'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
            'n_estimators': trial.suggest_categorical("n_estimators", [20,50,100,150,200]),
            
        }
    elif method == 'XGBC':
        params = {
            'max_depth': trial.suggest_int("max_depth", 3, 18),
            'eta': trial.suggest_float('eta', 1e-9, 1, log=True),
            'gamma': trial.suggest_float ('gamma', 1,9),
            'reg_alpha' : trial.suggest_int('reg_alpha', 10, 180),
            'reg_lambda' : trial.suggest_float('reg_lambda', 0, 1),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1),
            'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
            'n_estimators': trial.suggest_categorical("n_estimators", [20,50,100,150,200]),
            
        }
    elif method == 'RFR':
        params = {
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_samples' : trial.suggest_categorical('max_samples', [0.2, 0.3, 0.5, 0.7, 0.8, 1]),
            'n_estimators': trial.suggest_categorical("n_estimators", [20,50,100,150,200]),
        }
    elif method == 'RFC':
        params = {
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_samples' : trial.suggest_categorical('max_samples', [0.2, 0.3, 0.5, 0.7, 0.8, 1]),
            'n_estimators': trial.suggest_categorical("n_estimators", [20,50,100,150,200]),
        }

    return params