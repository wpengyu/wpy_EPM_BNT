import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler  
from genetic_selection import GeneticSelectionCV  
from multiprocessing import freeze_support  

target_folder = 'T1' 
# or target2
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

def main():

    filename = 'D1_dataset.xlsx'  
    dataset = pd.read_excel(filename)

    data = np.array(dataset)
    X = data[:, 3:]  
    Y = data[:, 1]  # target1:col1, target2:col2

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  

    validation_size = 0.2
    seed = np.random.randint(0, 999999)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    models = {
        'LR': LinearRegression(),
        'LASSO': Lasso(),
        'EN': ElasticNet(),
        'DTR': DecisionTreeRegressor(),
        'KNR': KNeighborsRegressor(),
        'SVR': SVR(),
        'RFR': RandomForestRegressor(),
        'GBR': GradientBoostingRegressor(),
        'ETR': ExtraTreesRegressor(),
        'ABR': AdaBoostRegressor()
    }

    results = []

    #GA
    for name, model in models.items():
        print(f"Training {name} with Genetic Algorithm feature selection...")

        selector = GeneticSelectionCV(model,
                                      cv=5,  
                                      verbose=1,
                                      scoring='neg_mean_squared_error',
                                      max_features=10,  
                                      n_population=50,  # 种群规模
                                      crossover_proba=0.5,  # 交叉概率
                                      mutation_proba=0.2,  # 变异概率
                                      n_generations=40,  
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,  
                                      n_gen_no_change=10,  # 最大代数无变化则停止
                                      caching=True,
                                      n_jobs=-1)

        selector = selector.fit(x_train, y_train)

        selected_features = selector.support_
        print(f"Selected features by GA: {selected_features}")

        x_train_selected = x_train[:, selected_features]
        x_test_selected = x_test[:, selected_features]

        model.fit(x_train_selected, y_train)
        
        #model_eve
          
        train_pred = model.predict(x_train_selected)
        test_pred = model.predict(x_test_selected)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        print(f"{name} - Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"{name} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
        selected_trainingset = pd.DataFrame(np.concatenate([y_train.reshape(-1, 1), x_train_selected], axis=1), 
                                            columns=[dataset.columns[1]] + list(dataset.columns[3:][selected_features]))
        selected_testset = pd.DataFrame(np.concatenate([y_test.reshape(-1, 1), x_test_selected], axis=1), 
                                        columns=[dataset.columns[1]] + list(dataset.columns[3:][selected_features]))

        selected_trainingset.to_excel(f"{target_folder}/{name}-GA-selected_trainingset.xlsx", index=False)
        selected_testset.to_excel(f"{target_folder}/{name}-GA-selected_testset.xlsx", index=False)

        results.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel(f"{target_folder}/model_performance_T1.xlsx", index=False)

    print("Performance saved to 'model_performance_T1.xlsx'.")

if __name__ == '__main__':
    freeze_support()  
    main() 
