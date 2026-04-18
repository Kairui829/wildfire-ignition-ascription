import time
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn import ensemble, linear_model, neighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_DIR = Path("./data/region_datasets")
MODEL_DIR = Path("./outputs/saved_models")
PREDICTION_DIR = Path("./outputs/model_predict")


def func_cal_train_data_and_predict(experiment_name="default", save_models=True):
    print("***func cal train data and predict***")

    """
    Train and evaluate wildfire ignition classifiers under repeated random splits.

    This function performs five repeated experiments. In each experiment, 20% of
    the samples from each class within each region are randomly selected as the
    test set, and the remaining 80% are used as the candidate training pool.
    Two training-data sampling strategies are then constructed from the training
    pool:

    1. Ratio-controlled sampling ("pro"):
       the human-to-lightning ratio is adjusted toward a predefined target ratio.

    2. Random sampling ("ran"):
       a random subset is drawn with the same sample size as the ratio-controlled
       training set.

    For each sampling strategy, seven base classifiers are trained.
    Predictions from these models are saved individually,
    and both hard-voting and soft-voting ensemble predictions are generated.

    Optionally, trained models, feature names, scalers, and background samples
    for interpretability analysis are saved to disk. Prediction results for
    each repeated split are also written to output files.

    Parameters
    ----------
    experiment_name : str, default='default'
        Name of the experiment. This string is used to organize output
        directories for saved models and prediction files.

    SAVE_MODELS : bool, default=True
        If True, save trained models, feature names, scaler objects, and
        background samples for later interpretation and reproducibility.

    Returns
    -------
    None
        The function writes model outputs and prediction tables to disk
        but does not return any object.
    """


    flg_list = [
        "USA", "Canada", "Australia",
        "Russia", "Turkey", "France",
        "Reference", "Website", "China",
        "Spain", "Portugal", "Ref_lightning",
        "Brazil", "Chile", "Indonesia",
        "Korea", "Mexico",
    ]

    var_name_met = [
        #meteorological 
        "WIND", "TEMP", "2m_RH", "CAPE", "PREC",
        "HCC", "MCC", "LCC", "swvl1", "Lightning_lagover",
        "Lightning_0-2_mean_density", "Lightning_3-6_mean_density", "Lightning_7-14_mean_density",
    ]
    var_name_geo = [
        #geographical 
        f"LAND{i + 1:02d}" for i in range(17)
    ] + ["Elevation"]
    var_name_soc = [
        #socioeconomic 
        "Nighttime_Light", "ROAD", "WEEKEND", "Low_Impact_Land"
    ]
    var_name_list = var_name_met + var_name_geo + var_name_soc

    df_list = []
    for flg in flg_list:
        file_to_load = DATA_DIR / f"{flg}_data_XY_with_MeanLightningDensity.csv"
        df = pd.read_csv(file_to_load)

        # Retain events from 2010 onward and require complete predictors/labels.
        df = df[df["YEAR"] >= 2010]
        df = df.dropna(subset=["CAUSE", "YEAR"] + var_name_list)
        df["Region"] = flg
        df_list.append(df)

    seed_id = [1, 11, 111, 1111, 11111]
    for ff in range(5):
        print("sampling:" + f"{ff + 1:02d}")

        # Build region-wise stratified train/test splits by ignition class.
        # Randomly select 20% of samples as the test set.
        # Use the remaining ~80% of samples as the training pool.
        human_list_test = []
        lightning_list_test = []
        human_list_train = []
        lightning_list_train = []

        for df in df_list:
            tmp_human = df[df["CAUSE"] == 0]
            tmp_lightning = df[df["CAUSE"] == 1]

            tmp_human_test = tmp_human.sample(frac=0.2, random_state=seed_id[ff])
            tmp_lightning_test = tmp_lightning.sample(frac=0.2, random_state=seed_id[ff])

            human_list_test.append(tmp_human_test)
            lightning_list_test.append(tmp_lightning_test)

            tmp_human_train = tmp_human.drop(tmp_human_test.index)
            tmp_lightning_train = tmp_lightning.drop(tmp_lightning_test.index)

            human_list_train.append(tmp_human_train)
            lightning_list_train.append(tmp_lightning_train)

        # Construct two training sets for each split:
        # (1) ratio-constrained sampling and (2) random sampling with matched size.
        ratio_pro = 2.535
        train_data_pro_list = []
        train_data_ran_list = []

        for rr in range(len(human_list_train)):
            df_human = human_list_train[rr]
            df_lightning = lightning_list_train[rr]

            if df_human.shape[0] == 0 or df_lightning.shape[0] == 0:
                train_lightning = df_lightning
                train_human = df_human
            else:
                hl_ratio = df_human.shape[0] / df_lightning.shape[0]
                if hl_ratio > ratio_pro:
                    train_lightning = df_lightning
                    train_human = df_human.sample(frac=ratio_pro / hl_ratio, random_state=1)
                else:
                    train_lightning = df_lightning.sample(frac=hl_ratio / ratio_pro, random_state=1)
                    train_human = df_human

            df_all = pd.concat([df_lightning, df_human])
            train_random = df_all.sample(n=len(train_lightning) + len(train_human), random_state=99)

            train_data_pro_list.append(pd.concat([train_human, train_lightning]))
            train_data_ran_list.append(train_random)

        train_data_pro = pd.concat(train_data_pro_list)
        train_data_ran = pd.concat(train_data_ran_list)
        test_data = pd.concat([
            pd.concat(human_list_test),
            pd.concat(lightning_list_test),
        ])

        df_result = test_data.copy()

        for dd in range(2):
            if dd == 0:
                train_data = train_data_pro.copy()
                strategy_flg = "pro"
            else:
                train_data = train_data_ran.copy()
                strategy_flg = "ran"

            # Shuffle training rows before model fitting.
            train_data = train_data.sample(frac=1, random_state=seed_id[ff]).reset_index(drop=True)

            # Fit the scaler on training data only and apply it to the held-out test set.
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_data[var_name_list])
            x_test = scaler.transform(test_data[var_name_list])
            y_train = train_data["CAUSE"].values
            y_test = test_data["CAUSE"].values

            save_dir = MODEL_DIR / experiment_name / f"sampling_{ff + 1:02d}" / strategy_flg
            if save_models:
                save_dir.mkdir(parents=True, exist_ok=True)

                with open(save_dir / "feature_names.json", "w", encoding="utf-8") as f:
                    json.dump(var_name_list, f, ensure_ascii=False, indent=2)

                joblib.dump(scaler, save_dir / "scaler.joblib")

                bg_n = min(200, len(train_data))
                bg_df = train_data.sample(n=bg_n, random_state=seed_id[ff])
                x_bg_raw = bg_df[var_name_list].values
                x_bg_scaled = scaler.transform(x_bg_raw)

                np.save(save_dir / "X_bg_raw.npy", x_bg_raw)
                np.save(save_dir / "X_bg_scaled.npy", x_bg_scaled)

            # Neural network.
            epochs = 64
            model = keras.models.Sequential()
            model.add(keras.Input(shape=(x_train.shape[1],)))
            model.add(keras.layers.Dense(8, activation="relu"))
            model.add(keras.layers.Dense(4, activation="relu"))
            model.add(keras.layers.Dense(2, activation="relu"))
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=64,
                validation_split=0.0,
                shuffle=True,
                verbose=0,
            )
            if save_models:
                model.save(save_dir / "nn.keras")

            output_nntensor_proba = model.predict(x_test, verbose=0).ravel()
            output_nntensor_pred = (output_nntensor_proba >= 0.5).astype(int)
            df_result[f"tensor_{strategy_flg}"] = output_nntensor_pred
            df_result[f"tensor_proba_{strategy_flg}"] = output_nntensor_proba
            # print("nntensor:", accuracy_score(y_test, output_nntensor_pred))

            # k-nearest neighbors.
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=31,
                weights="distance",
                p=1,
            )
            clm_knn = clf.fit(x_train, y_train)
            if save_models:
                joblib.dump(clm_knn, save_dir / "knn.joblib")

            output_knn_proba = clm_knn.predict_proba(x_test)[:, 1]
            output_knn_pred = (output_knn_proba >= 0.5).astype(int)
            df_result[f"knn_{strategy_flg}"] = output_knn_pred
            df_result[f"knn_proba_{strategy_flg}"] = output_knn_proba
            # print("knn:", accuracy_score(y_test, output_knn_pred))

            # Logistic regression.
            clf = linear_model.LogisticRegression(
                C=0.3,
                penalty="l2",
                tol=1e-6,
                solver="liblinear",
                max_iter=10000,
                class_weight={0: 1, 1: 1},
                random_state=89,
            )
            clm_logistics = clf.fit(x_train, y_train)
            if save_models:
                joblib.dump(clm_logistics, save_dir / "logistic.joblib")

            output_log_proba = clm_logistics.predict_proba(x_test)[:, 1]
            output_log_pred = (output_log_proba >= 0.5).astype(int)
            df_result[f"log_{strategy_flg}"] = output_log_pred
            df_result[f"log_proba_{strategy_flg}"] = output_log_proba
            # print("logistics:", accuracy_score(y_test, output_log_pred))

            # XGBoost.
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
            if save_models:
                joblib.dump(clm_xgb, save_dir / "xgb.joblib")

            output_xgb_proba = clm_xgb.predict_proba(x_test)[:, 1]
            output_xgb_pred = (output_xgb_proba >= 0.5).astype(int)
            df_result[f"xgb_{strategy_flg}"] = output_xgb_pred
            df_result[f"xgb_proba_{strategy_flg}"] = output_xgb_proba
            # print("XGB:", accuracy_score(y_test, output_xgb_pred))

            # Gradient boosting decision tree.
            clf = ensemble.GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=400,
                max_depth=3,
                subsample=0.7,
                max_features=0.6,
                min_samples_leaf=10,
                random_state=42,
            )
            clm_gbdt = clf.fit(x_train, y_train)
            if save_models:
                joblib.dump(clm_gbdt, save_dir / "gbdt.joblib")

            output_gbdt_proba = clm_gbdt.predict_proba(x_test)[:, 1]
            output_gbdt_pred = (output_gbdt_proba >= 0.5).astype(int)
            df_result[f"GBDT_{strategy_flg}"] = output_gbdt_pred
            df_result[f"GBDT_proba_{strategy_flg}"] = output_gbdt_proba
            # print("GBDT:", accuracy_score(y_test, output_gbdt_pred))

            # Ridge classifier with probability calibration.
            clf = linear_model.RidgeClassifier(alpha=1, solver="auto")
            ridge_cal = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=5)
            ridge_cal.fit(x_train, y_train)
            if save_models:
                joblib.dump(ridge_cal, save_dir / "ridge_calibrated.joblib")

            output_ridge_proba = ridge_cal.predict_proba(x_test)[:, 1]
            output_ridge_pred = (output_ridge_proba >= 0.5).astype(int)
            df_result[f"ridge_{strategy_flg}"] = output_ridge_pred
            df_result[f"ridge_proba_{strategy_flg}"] = output_ridge_proba
            # print("ridge:", accuracy_score(y_test, output_ridge_pred))

            # Random forest.
            clf = ensemble.RandomForestClassifier(
                n_estimators=800,
                max_depth=15,
                max_features="sqrt",
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            clm_forest = clf.fit(x_train, y_train)
            if save_models:
                joblib.dump(clm_forest, save_dir / "rf.joblib")

            output_forest_proba = clm_forest.predict_proba(x_test)[:, 1]
            output_forest_pred = (output_forest_proba >= 0.5).astype(int)
            df_result[f"forest_{strategy_flg}"] = output_forest_pred
            df_result[f"forest_proba_{strategy_flg}"] = output_forest_proba
            # print("forest:", accuracy_score(y_test, output_forest_pred))

            # Hard-voting ensemble across the seven base classifiers.
            output_all_pred = np.vstack([
                output_nntensor_pred,
                output_knn_pred,
                output_log_pred,
                output_xgb_pred,
                output_gbdt_pred,
                output_ridge_pred,
                output_forest_pred,
            ])
            tmp = np.sum(output_all_pred, axis=0)
            vote_hard = np.zeros(len(tmp), dtype=int)
            vote_hard[tmp > 3] = 1
            vote_hard[tmp <= 3] = 0
            df_result[f"VoteHard_{strategy_flg}"] = vote_hard
            # print("HardVote:", accuracy_score(y_test, vote_hard))

            # Soft-voting ensemble using the unweighted mean probability.
            output_all_proba = np.vstack([
                output_nntensor_proba,
                output_knn_proba,
                output_log_proba,
                output_xgb_proba,
                output_gbdt_proba,
                output_ridge_proba,
                output_forest_proba,
            ])
            vote_soft_proba = np.mean(output_all_proba, axis=0)
            vote_soft = (vote_soft_proba >= 0.5).astype(int)
            df_result[f"VoteSoftProba_{strategy_flg}"] = vote_soft_proba
            df_result[f"VoteSoft_{strategy_flg}"] = vote_soft
            # print("SoftVote:", accuracy_score(y_test, vote_soft))

        output_path = PREDICTION_DIR / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(output_path / f"Vote_sampling_{ff + 1:02d}.csv", index=False)


if __name__ == "__main__":
    ex_name = "standard_validation"
    print("experiment name: ", ex_name)
    print("***beginning***")
    time00 = time.time()

    func_cal_train_data_and_predict(experiment_name=ex_name, save_models=True)

    time11 = time.time()
    print("***successfully done***")
    print("***cost %d mins***" % ((time11 - time00) / 60.0))
