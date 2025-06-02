import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('generated_formulas_with_features.csv')

# Target1 and Target2 
selected_features_target1 = [
    'MP_A', 'NM_A', 'QN_A', 'CR_A', 'VEN/NC_A', 'RB_A', 'VED_B', 'NM_B', 'EA_B',  # SVR
    'MP_A', 'CR_A', 'PCR_A', 'V_A', 'ED_A', 'NM_B', 'EA_B', 'ED_B',  # ABR
    'SF_A', 'CD_A', 'EG-MB_A', 'MP_A', 'CR_A', 'EP_A', 'CED_B', 'SE_B', 'V_B', 'EVR_B'  # GBR
]

selected_features_target2 = [
    'CED_A', 'EC_A', 'FEI_A', 'CR_A', 'ED_A', 'SE_B', 'EA_B',  # SVR
    'SF_A', 'CD_A', 'SE_A', 'MP_A', 'CR_A', 'A-O_A', 'CD_B', 'No_B', 'VEN/NC_B',  # ABR
    'CD_A', 'CED_A', 'EG-MB_A', 'EC_A', 'EVR_A', 'CD_B'  # GBR
]


features_target1 = list(set(selected_features_target1))
features_target2 = list(set(selected_features_target2))


X1 = data[features_target1]
X2 = data[features_target2]

scaler1 = StandardScaler()
scaler2 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)
X2_scaled = scaler2.fit_transform(X2)


# Target1 
svr1 = joblib.load('models/svr1_model.joblib')
gbr1 = joblib.load('models/gbr1_model.joblib')
abr1 = joblib.load('models/abr1_model.joblib')
meta_model1 = joblib.load('models/meta_model1.joblib')

# Target2 
svr2 = joblib.load('models/svr2_model.joblib')
gbr2 = joblib.load('models/gbr2_model.joblib')
abr2 = joblib.load('models/abr2_model.joblib')
meta_model2 = joblib.load('models/meta_model2.joblib')

meta_features1 = np.zeros((X1_scaled.shape[0], 3))
meta_features2 = np.zeros((X2_scaled.shape[0], 3))

base_models1 = [svr1, gbr1, abr1]
base_models2 = [svr2, gbr2, abr2]

for i, model in enumerate(base_models1):
    meta_features1[:, i] = model.predict(X1_scaled)

for i, model in enumerate(base_models2):
    meta_features2[:, i] = model.predict(X2_scaled)


predictions1 = meta_model1.predict(meta_features1)
predictions2 = meta_model2.predict(meta_features2)

data['Target1_pred_Wrec'] = predictions1
data['Target2_pred_n'] = predictions2

output_file = "generated_formulas_with_predictions.csv"
data.to_csv(output_file, index=False)

print(f"Predicted, results saved to {output_file}")
print(f"Predicted {len(data)} formulas") 