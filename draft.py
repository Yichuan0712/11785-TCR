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