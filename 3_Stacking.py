import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import random
import joblib
import os

if not os.path.exists('models'):
    os.makedirs('models')

file_path = "D2_dataset.xlsx"
data = pd.read_excel(file_path)

# T1
selected_features_target1 = [
    'MP_A', 'NM_A', 'QN_A', 'CR_A', 'VEN/NC_A', 'RB_A', 'VED_B', 'NM_B', 'EA_B',  # SVR
    'MP_A', 'CR_A', 'PCR_A', 'V_A', 'ED_A', 'NM_B', 'EA_B', 'ED_B',  # ABR
    'SF_A', 'CD_A', 'EG-MB_A', 'MP_A', 'CR_A', 'EP_A', 'CED_B', 'SE_B', 'V_B', 'EVR_B'  # GBR
]

# T2
selected_features_target2 = [
    'CED_A', 'EC_A', 'FEI_A', 'CR_A', 'ED_A', 'SE_B', 'EA_B',  # SVR
    'SF_A', 'CD_A', 'SE_A', 'MP_A', 'CR_A', 'A-O_A', 'CD_B', 'No_B', 'VEN/NC_B',  # ABR
    'CD_A', 'CED_A', 'EG-MB_A', 'EC_A', 'EVR_A', 'CD_B'  # GBR
]

features_target1 = list(set(selected_features_target1))
features_target2 = list(set(selected_features_target2))

X1 = data[features_target1]  
X2 = data[features_target2]  
y1 = data['Target1']  
y2 = data['Target2']  

scaler1 = StandardScaler()
scaler2 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)
X2_scaled = scaler2.fit_transform(X2)

# 保存标准化器
joblib.dump(scaler1, 'models/scaler1.joblib')
joblib.dump(scaler2, 'models/scaler2.joblib')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def objective_target1(trial):
    svr_c = trial.suggest_float('svr_c', 0.1, 10.0)
    svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto', 0.01, 0.1, 1.0])
    gbr_n_estimators = trial.suggest_int('gbr_n_estimators', 50, 200)
    gbr_learning_rate = trial.suggest_float('gbr_learning_rate', 0.01, 0.3)
    gbr_max_depth = trial.suggest_int('gbr_max_depth', 3, 10)
    abr_n_estimators = trial.suggest_int('abr_n_estimators', 50, 200)
    abr_learning_rate = trial.suggest_float('abr_learning_rate', 0.01, 1.0)
    meta_c = trial.suggest_float('meta_c', 0.1, 10.0)
    meta_gamma = trial.suggest_categorical('meta_gamma', ['scale', 'auto', 0.01, 0.1, 1.0])

    svr = SVR(kernel='rbf', C=svr_c, gamma=svr_gamma)
    gbr = GradientBoostingRegressor(n_estimators=gbr_n_estimators, learning_rate=gbr_learning_rate, 
                                    max_depth=gbr_max_depth, random_state=42)
    abr = AdaBoostRegressor(n_estimators=abr_n_estimators, learning_rate=abr_learning_rate, random_state=42)
    meta_model = SVR(kernel='rbf', C=meta_c, gamma=meta_gamma)

    random_seed = np.random.randint(0, 999999)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=random_seed)
    meta_features_train = np.zeros((X1_train.shape[0], 3))
    meta_features_test = np.zeros((X1_test.shape[0], 3))

    base_models = [svr, gbr, abr]
    for i, model in enumerate(base_models):
        model.fit(X1_train, y1_train)
        meta_features_train[:, i] = model.predict(X1_train)
        meta_features_test[:, i] = model.predict(X1_test)

    meta_model.fit(meta_features_train, y1_train)
    y1_pred = meta_model.predict(meta_features_test)
    return rmse(y1_test, y1_pred)

def objective_target2(trial):
    svr_c = trial.suggest_float('svr_c', 0.1, 10.0)
    svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto', 0.01, 0.1, 1.0])
    gbr_n_estimators = trial.suggest_int('gbr_n_estimators', 50, 200)
    gbr_learning_rate = trial.suggest_float('gbr_learning_rate', 0.01, 0.3)
    gbr_max_depth = trial.suggest_int('gbr_max_depth', 3, 10)
    abr_n_estimators = trial.suggest_int('abr_n_estimators', 50, 200)
    abr_learning_rate = trial.suggest_float('abr_learning_rate', 0.01, 1.0)
    meta_c = trial.suggest_float('meta_c', 0.1, 10.0)
    meta_gamma = trial.suggest_categorical('meta_gamma', ['scale', 'auto', 0.01, 0.1, 1.0])

    svr = SVR(kernel='rbf', C=svr_c, gamma=svr_gamma)
    gbr = GradientBoostingRegressor(n_estimators=gbr_n_estimators, learning_rate=gbr_learning_rate, 
                                    max_depth=gbr_max_depth, random_state=42)
    abr = AdaBoostRegressor(n_estimators=abr_n_estimators, learning_rate=abr_learning_rate, random_state=42)
    meta_model = SVR(kernel='rbf', C=meta_c, gamma=meta_gamma)

    random_seed = np.random.randint(0, 999999)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=random_seed)
    meta_features_train = np.zeros((X2_train.shape[0], 3))
    meta_features_test = np.zeros((X2_test.shape[0], 3))

    base_models = [svr, gbr, abr]
    for i, model in enumerate(base_models):
        model.fit(X2_train, y2_train)
        meta_features_train[:, i] = model.predict(X2_train)
        meta_features_test[:, i] = model.predict(X2_test)

    meta_model.fit(meta_features_train, y2_train)
    y2_pred = meta_model.predict(meta_features_test)
    return rmse(y2_test, y2_pred)

study1 = optuna.create_study(direction='minimize')
study1.optimize(objective_target1, n_trials=200)
study2 = optuna.create_study(direction='minimize')
study2.optimize(objective_target2, n_trials=200)

print("\nTarget1 best_para：", study1.best_params)
print("Target1 best_RMSE：", study1.best_value)
print("\nTarget2 best_para：", study2.best_params)
print("Target2 best_RMSE：", study2.best_value)

#best para
svr1 = SVR(kernel='rbf', C=study1.best_params['svr_c'], gamma=study1.best_params['svr_gamma'])
gbr1 = GradientBoostingRegressor(n_estimators=study1.best_params['gbr_n_estimators'], 
                                 learning_rate=study1.best_params['gbr_learning_rate'], 
                                 max_depth=study1.best_params['gbr_max_depth'], random_state=42)
abr1 = AdaBoostRegressor(n_estimators=study1.best_params['abr_n_estimators'], 
                         learning_rate=study1.best_params['abr_learning_rate'], random_state=42)
meta_model1 = SVR(kernel='rbf', C=study1.best_params['meta_c'], gamma=study1.best_params['meta_gamma'])

svr2 = SVR(kernel='rbf', C=study2.best_params['svr_c'], gamma=study2.best_params['svr_gamma'])
gbr2 = GradientBoostingRegressor(n_estimators=study2.best_params['gbr_n_estimators'], 
                                 learning_rate=study2.best_params['gbr_learning_rate'], 
                                 max_depth=study2.best_params['gbr_max_depth'], random_state=42)
abr2 = AdaBoostRegressor(n_estimators=study2.best_params['abr_n_estimators'], 
                         learning_rate=study2.best_params['abr_learning_rate'], random_state=42)
meta_model2 = SVR(kernel='rbf', C=study2.best_params['meta_c'], gamma=study2.best_params['meta_gamma'])

X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)
meta_features_train1 = np.zeros((X1_train.shape[0], 3))
meta_features_test1 = np.zeros((X1_test.shape[0], 3))

base_models1 = [svr1, gbr1, abr1]
for i, model in enumerate(base_models1):
    model.fit(X1_train, y1_train)
    meta_features_train1[:, i] = model.predict(X1_train)
    meta_features_test1[:, i] = model.predict(X1_test)
    #save
    joblib.dump(model, f'models/{["svr1", "gbr1", "abr1"][i]}_model.joblib')

meta_model1.fit(meta_features_train1, y1_train)
joblib.dump(meta_model1, 'models/meta_model1.joblib')


X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)
meta_features_train2 = np.zeros((X2_train.shape[0], 3))
meta_features_test2 = np.zeros((X2_test.shape[0], 3))

base_models2 = [svr2, gbr2, abr2]
for i, model in enumerate(base_models2):
    model.fit(X2_train, y2_train)
    meta_features_train2[:, i] = model.predict(X2_train)
    meta_features_test2[:, i] = model.predict(X2_test)

    joblib.dump(model, f'models/{["svr2", "gbr2", "abr2"][i]}_model.joblib')

meta_model2.fit(meta_features_train2, y2_train)
joblib.dump(meta_model2, 'models/meta_model2.joblib')

y1_train_pred = meta_model1.predict(meta_features_train1)
y1_test_pred = meta_model1.predict(meta_features_test1)
y2_train_pred = meta_model2.predict(meta_features_train2)
y2_test_pred = meta_model2.predict(meta_features_test2)

print("\nTarget1 Evaluation:")
print(f"Train RMSE: {rmse(y1_train, y1_train_pred):.4f}")
print(f"Test RMSE: {rmse(y1_test, y1_test_pred):.4f}")
print(f"Train R²: {r2_score(y1_train, y1_train_pred):.4f}")
print(f"Test R²: {r2_score(y1_test, y1_test_pred):.4f}")

print("\nTarget2 Evaluation:")
print(f"Train RMSE: {rmse(y2_train, y2_train_pred):.4f}")
print(f"Test RMSE: {rmse(y2_test, y2_test_pred):.4f}")
print(f"Train R²: {r2_score(y2_train, y2_train_pred):.4f}")
print(f"Test R²: {r2_score(y2_test, y2_test_pred):.4f}")

print("\nAll models saved to 'models'") 