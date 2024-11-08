# elif parse_args.mode == 'predict' and parse_args.resume_path is not None:
#     printl("Start prediction.", log_path=log_path)
#     printl(f"{'=' * 128}", log_path=log_path)
#     true_classes, predicted_classes, predicted_scores = infer_one(encoder, projection_head, dataloaders["train_loader"], tokenizer, inference_dataloaders["test_loader"], log_path)
#     print(true_classes)
#     print(predicted_classes)
#
#     correct_predictions = sum(1 for true, pred in zip(true_classes, predicted_classes) if true == pred)
#     accuracy = correct_predictions / len(true_classes) if len(true_classes) > 0 else 0
#
#     # 获取所有唯一类别
#     unique_classes = set(true_classes + predicted_classes)
#
#     # 初始化 Precision、Recall 和 F1 Score 列表
#     precision_scores = []
#     recall_scores = []
#     f1_scores = []
#
#     for label in unique_classes:
#         # 计算每个类别的 True Positives、False Positives 和 False Negatives
#         true_positive = sum(
#             1 for true, pred in zip(true_classes, predicted_classes) if true == label and pred == label)
#         predicted_positive = sum(1 for pred in predicted_classes if pred == label)
#         actual_positive = sum(1 for true in true_classes if true == label)
#
#         # 计算 Precision
#         precision = true_positive / predicted_positive if predicted_positive > 0 else 0
#         precision_scores.append(precision)
#
#         # 计算 Recall
#         recall = true_positive / actual_positive if actual_positive > 0 else 0
#         recall_scores.append(recall)
#
#         # 计算 F1 Score
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         f1_scores.append(f1)
#
#     precision = sum(precision_scores) / len(precision_scores)
#     recall = sum(recall_scores) / len(recall_scores)
#     f1 = sum(f1_scores) / len(f1_scores)
#
#     lb = LabelBinarizer()
#     true_binarized = lb.fit_transform(true_classes)
#
#     # 提取所有类别
#     all_classes = sorted(set(true_classes + predicted_classes))
#
#     # 将 predicted_scores 转换为二维数组，顺序匹配 all_classes
#     predicted_scores_array = np.array([
#         [score_dict.get(cls, 0.0) for cls in all_classes]
#         for score_dict in predicted_scores
#     ])
#
#     # 检查 true_binarized 和 predicted_scores_array 的每一列，移除那些没有正负样本的类别
#     valid_indices = []
#     for i in range(true_binarized.shape[1]):
#         if len(np.unique(true_binarized[:, i])) > 1:  # 确保每个类别至少有一个正样本和一个负样本
#             valid_indices.append(i)
#
#     # 对 true_binarized 和 predicted_scores_array 进行筛选
#     true_binarized_filtered = true_binarized[:, valid_indices]
#     predicted_scores_array_filtered = predicted_scores_array[:, valid_indices]
#
#     # 计算 AUC
#     auc = roc_auc_score(true_binarized_filtered, predicted_scores_array_filtered, average="macro",
#                         multi_class="ovr")
#
#     # 显示各项指标
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC: {auc:.4f}" if auc is not None else "AUC: Not applicable")
#     return

















# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, f1_score
# from sklearn.preprocessing import OneHotEncoder
# from rdkit import Chem
# from rdkit.Chem import Descriptors

# # Load train and test datasets
# df_train = pd.read_csv('/content/11785-TCR/dataset/XGB/feature_data_train.csv')
# df_train_ori = pd.read_csv('/content/11785-TCR/dataset/pytdc_new/train2_PyTDC.csv')

# df_test = pd.read_csv('/content/11785-TCR/dataset/XGB/feature_data_test.csv')
# df_test_ori = pd.read_csv('/content/11785-TCR/dataset/pytdc_new/test_PyTDC.csv')

# # Add 'epitope_smi' column from original train and test data
# df_train['epitope_smi'] = df_train_ori['epitope_smi'].values
# df_test['epitope_smi'] = df_test_ori['epitope_smi'].values

# # Function to extract features from SMILES
# def extract_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return pd.Series([None] * 5)  # If SMILES is invalid, return empty values
#     features = {
#         'mol_weight': Descriptors.MolWt(mol),
#         'logP': Descriptors.MolLogP(mol),
#         'tpsa': Descriptors.TPSA(mol),
#         'num_h_donors': Descriptors.NumHDonors(mol),
#         'num_h_acceptors': Descriptors.NumHAcceptors(mol)
#     }
#     return pd.Series(features)

# # Prepare features for both train and test datasets
# def prepare_features(df, use_smi=False, encoder=None):
#     X = df.drop(columns=['x', 'label'])

#     non_numeric_columns = X.select_dtypes(include=['object']).columns
#     if 'y' in non_numeric_columns:
#         # One-hot encode the 'y' column
#         if encoder is None:
#             encoder = OneHotEncoder(sparse_output=False, drop='first')
#             y_encoded = encoder.fit_transform(X[['y']])
#         else:
#             y_encoded = encoder.transform(X[['y']])
#         y_encoded_df = pd.DataFrame(y_encoded, columns=encoder.get_feature_names_out(['y']))

#         # Merge the one-hot encoded 'y' column with other features and drop the original 'y' column
#         X = X.drop(columns=['y'])
#         X = pd.concat([X, y_encoded_df], axis=1)

#     if use_smi:
#         # Apply feature extraction to 'epitope_smi' column
#         X_features = X['epitope_smi'].apply(extract_features)
#         X = pd.concat([X, X_features], axis=1)
#         X = X.drop(columns=['epitope_smi'])

#     return X, encoder

# # Prepare training features
# X_train, encoder = prepare_features(df_train, use_smi=True)
# # Target labels for training
# y_train = df_train['label']

# # Prepare test features
# X_test, _ = prepare_features(df_test, use_smi=True, encoder=encoder)
# # Target labels for testing
# y_test = df_test['label']

# # Initialize and train the XGBoost classifier
# model = XGBClassifier(
#     n_estimators=100,  # Number of base models
#     max_depth=5,       # Maximum depth of trees
#     learning_rate=0.1, # Learning rate
#     random_state=42    # Random seed
# )
# model.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# auroc = roc_auc_score(y_test, y_pred_proba)
# aupr = average_precision_score(y_test, y_pred_proba)
# f1 = f1_score(y_test, y_pred)

# print(f'Accuracy on test data: {accuracy:.2f}')
# print(f'AUROC on test data: {auroc:.2f}')
# print(f'AUPR on test data: {aupr:.2f}')
# print(f'F1 Score on test data: {f1:.2f}')
# print('Classification Report:')
# print(classification_report(y_test, y_pred))