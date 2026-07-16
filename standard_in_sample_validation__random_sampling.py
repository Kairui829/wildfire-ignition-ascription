
import time 
import sqlite3 
import pandas as pd 
import numpy as np 
from tensorflow import keras 
from sklearn import ensemble 
from sklearn import neighbors 
from sklearn import linear_model 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
import json
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier



def func_cal_train_data_and_predict(experiment_name = 'default'): 

    flg_list = [
        "USA", "Canada", "Australia", 
        "Russia", "Turkey", "France", 
        "Reference", "Website", "China", 
        "Spain", "Portugal", "Ref_lightning", 
        "Brazil", "Chile", "Indonesia",
        "Korea", "Mexico"
    ]
    
    var_name_met = [
        "WIND", "TEMP", "2m_RH", "CAPE", "PREC", 
        "swvl1_gradient",
        "Lightning_lagover", "Lightning_0-2_mean_density", "Lightning_3-6_mean_density", "Lightning_7-14_mean_density", 
    ] 
    var_name_geo = [
        "LAND"+f"{i+1:02d}" for i in range(17) 
    ] + ["Elevation"]
    
    var_name_soc = [
        "Nighttime_Light", "ROAD", "WEEKEND", "Low_Impact_Land" 
    ] 

    
    var_name_list = var_name_met+var_name_geo+var_name_soc

    df_list = [] 
    for flg in flg_list: 
        file_to_load = "./"+ flg+"_data_2mRH_smgrad.csv"      
        df = pd.read_csv(file_to_load) 
        df["date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
        df = df[df['YEAR']>=2010]
        df = df.dropna(subset=["CAUSE","YEAR"] + var_name_list)    
        df['Region'] = flg 
        df_list.append(df) 

    seed_id = [1, 11, 111, 1111, 11111] 
    n_folds = 5
    fold_seed = 2020
    fold_data_list = []

    def split_into_folds(df, n):
        return [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), n)]

    for df in df_list:
        tmp_human = df[df['CAUSE'] == 0].sample(frac=1, random_state=fold_seed)
        tmp_lightning = df[df['CAUSE'] == 1].sample(frac=1, random_state=fold_seed)

        human_folds = split_into_folds(tmp_human, n_folds)
        lightning_folds = split_into_folds(tmp_lightning, n_folds)
        fold_data_list.append((human_folds, lightning_folds))

    for ff in range(n_folds):
        print("fold:"+f"{ff+1:02d}") 

        human_list_test= [] 
        lightning_list_test = [] 
        human_list_train = [] 
        lightning_list_train= [] 
        
        for human_folds, lightning_folds in fold_data_list:
            human_list_test.append(human_folds[ff])
            lightning_list_test.append(lightning_folds[ff])

            human_list_train.append(
                pd.concat(
                    [fold for ii, fold in enumerate(human_folds) if ii != ff],
                    ignore_index=False,
                )
            )
            lightning_list_train.append(
                pd.concat(
                    [fold for ii, fold in enumerate(lightning_folds) if ii != ff],
                    ignore_index=False,
                )
            )

        train_data_ran = pd.concat(
            [
                pd.concat(human_list_train), 
                pd.concat(lightning_list_train), 
            ]
        )
        test_data = pd.concat(
            [
                pd.concat(human_list_test), 
                pd.concat(lightning_list_test), 
            ]
        )

        df_result = test_data.copy() 
        
        for strategy_flg in ["ran"]:
            train_data = train_data_ran.copy() 
            train_data = train_data.sample(frac=1, random_state=seed_id[ff]).reset_index(drop=True)     

            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_data[var_name_list]) 
            x_test = scaler.transform(test_data[var_name_list])  
            
            y_train = train_data['CAUSE'].values 
            y_test = test_data['CAUSE'].values    
            
            # ====================== nntensor ======================         
            EPOCHS = 64 
            model = keras.models.Sequential()
            model.add(keras.Input(shape=(x_train.shape[1],))) 
            model.add(keras.layers.Dense(8, activation='relu'))
            model.add(keras.layers.Dense(4, activation='relu'))
            model.add(keras.layers.Dense(2, activation='relu'))
            model.add(keras.layers.Dense(1, activation = 'sigmoid'))
            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(
                x_train,                                
                y_train,                              
                epochs=EPOCHS,                            
                batch_size=64,       
                validation_split=0.0,                       
                shuffle=True,                              
                verbose = 0,                                
            )  
           
            output_nntensor_proba = model.predict(x_test, verbose=0).ravel()    
            output_nntensor_pred  = (output_nntensor_proba >= 0.5).astype(int)

            df_result['tensor_' + strategy_flg] = output_nntensor_pred
            df_result['tensor_proba_' + strategy_flg] = output_nntensor_proba
            print("nntensor:", accuracy_score(y_test, output_nntensor_pred))
            
            # ====================== knn ======================  
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=31,
                weights="distance",
                p=1
            )

            clm_knn = clf.fit(x_train, y_train) 

  
            output_knn_proba = clm_knn.predict_proba(x_test)[:, 1]
            output_knn_pred  = (output_knn_proba >= 0.5).astype(int)

            df_result['knn_' + strategy_flg] = output_knn_pred
            df_result['knn_proba_' + strategy_flg] = output_knn_proba
            print("knn:", accuracy_score(y_test, output_knn_pred))
                
            # ====================== logistics ====================== 
            class1 = 0
            class2 = 1
            class_weight = {class1:1,class2:1} 

            clf = linear_model.LogisticRegression(C=0.3, penalty='l2', 
                                                  tol=1e-6, solver='liblinear', 
                                                  max_iter=10000, class_weight = class_weight, random_state=89)  
            clm_logistics = clf.fit(x_train, y_train)


            output_log_proba = clm_logistics.predict_proba(x_test)[:, 1]
            output_log_pred  = (output_log_proba >= 0.5).astype(int)

            df_result['log_' + strategy_flg] = output_log_pred
            df_result['log_proba_' + strategy_flg] = output_log_proba
            print("logistics:", accuracy_score(y_test, output_log_pred))

            # ====================== XGBoost ====================== 
            clf = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.6,
                reg_lambda=5.0,
                reg_alpha=0.0,
                min_child_weight=5,
                gamma=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )

            clm_xgb = clf.fit(x_train, y_train)


            output_xgb_proba = clm_xgb.predict_proba(x_test)[:, 1]
            output_xgb_pred  = (output_xgb_proba >= 0.5).astype(int)

            df_result['xgb_' + strategy_flg] = output_xgb_pred
            df_result['xgb_proba_' + strategy_flg] = output_xgb_proba
            print("XGB:", accuracy_score(y_test, output_xgb_pred))          
            



            # ====================== GBDT ====================== 
            clf = ensemble.GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=400,
                max_depth=3,
                subsample=0.7,
                max_features=0.6,
                min_samples_leaf=10,
                random_state=42
            )

            clm_GBDT = clf.fit(x_train, y_train)


            output_GBDT_proba = clm_GBDT.predict_proba(x_test)[:, 1]
            output_GBDT_pred  = (output_GBDT_proba >= 0.5).astype(int)

            df_result['GBDT_' + strategy_flg] = output_GBDT_pred
            df_result['GBDT_proba_' + strategy_flg] = output_GBDT_proba
            print("GBDT:", accuracy_score(y_test, output_GBDT_pred))  



            # ====================== ridge ======================
            clf = linear_model.RidgeClassifier(alpha=1, solver="auto")

            ridge_cal = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=5)
            ridge_cal.fit(x_train, y_train)


            output_ridge_proba = ridge_cal.predict_proba(x_test)[:, 1]
            output_ridge_pred  = (output_ridge_proba >= 0.5).astype(int)

            df_result['ridge_' + strategy_flg] = output_ridge_pred
            df_result['ridge_proba_' + strategy_flg] = output_ridge_proba
            print("ridge:", accuracy_score(y_test, output_ridge_pred))     
            


            # ====================== forest ====================== 
            clf = ensemble.RandomForestClassifier(
                n_estimators=800,
                max_depth=15,
                max_features="sqrt",
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ) 
            clm_forest = clf.fit(x_train, y_train)

    
            output_forest_proba = clm_forest.predict_proba(x_test)[:, 1]
            output_forest_pred  = (output_forest_proba >= 0.5).astype(int)

            df_result['forest_' + strategy_flg] = output_forest_pred
            df_result['forest_proba_' + strategy_flg] = output_forest_proba
            print("forest:", accuracy_score(y_test, output_forest_pred))
            


            output_all_pred = np.vstack([
                output_nntensor_pred,
                output_knn_pred,
                output_log_pred,
                output_xgb_pred,
                output_GBDT_pred,
                output_ridge_pred,
                output_forest_pred,
            ])  

            tmp = np.sum(output_all_pred, axis=0)  

            vote_hard = np.zeros(len(tmp), dtype=int)
            vote_hard[tmp > 3] = 1     
            vote_hard[tmp <= 3] = 0   

            df_result['VoteHard_' + strategy_flg] = vote_hard
            print("HardVote:", accuracy_score(y_test, vote_hard))


            output_all_proba = np.vstack([
                output_nntensor_proba,
                output_knn_proba,
                output_log_proba,
                output_xgb_proba,
                output_GBDT_proba,
                output_ridge_proba,
                output_forest_proba,
            ])  

            vote_soft_proba = np.mean(output_all_proba, axis=0)  
            vote_soft = (vote_soft_proba >= 0.5).astype(int)

            df_result['VoteSoftProba_' + strategy_flg] = vote_soft_proba
            df_result['VoteSoft_' + strategy_flg] = vote_soft
            print("SoftVote:", accuracy_score(y_test, vote_soft))
 
                
        output_path = Path(f"./model_predict/{experiment_name}")
        output_path.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(output_path / f"Vote_sampling_{ff+1:02d}.csv", index=False) 
                

   
if __name__ == '__main__':
    ex_name = 'standard_validation'
    print('experiment name: ', ex_name)
    print("***beginning***") 
    time00 = time.time() 
    
    func_cal_train_data_and_predict(experiment_name = ex_name)  
    
    time11 = time.time()
    print("***sucessfully done***")
    print("***cost %d mins***" % ((time11-time00)/60.0))          
    
    
