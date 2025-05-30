
      ______   ______   .__   __. .___________..______          ___   .___________.  ______ .______      
     /      | /  __  \  |  \ |  | |           ||   _  \        /   \  |           | /      ||   _  \     
    |  ,----'|  |  |  | |   \|  | `---|  |----`|  |_)  |      /  ^  \ `---|  |----`|  ,----'|  |_)  |    
    |  |     |  |  |  | |  . `  |     |  |     |      /      /  /_\  \    |  |     |  |     |      /     
    |  `----.|  `--'  | |  |\   |     |  |     |  |\  \----./  _____  \   |  |     |  `----.|  |\  \----.
     \______| \______/  |__| \__|     |__|     | _| `._____/__/     \__\  |__|      \______|| _| `._____|
 
# ContraTCR: Predicting TCR-Epitope Binding with Fine-Tuned Protein Language Models and Contrastive Learning

Welcome to **ContraTCR**, a tool designed for training and predicting T-cell receptor (TCR) and epitope binding using contrastive learning techniques. This guide will walk you through the steps required to run the project, from training the model to making predictions. 

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Running the Project](#running-the-project)
    - [Step 1: Train the ESM-2 Model with Projection Head](#step-1-train-the-esm-2-model-with-projection-head)
    - [Step 2: Extract Features Using the Trained Model](#step-2-extract-features-using-the-trained-model)
    - [Step 3: Predict Binding Specificity with Classifiers](#step-3-predict-binding-specificity-with-classifiers)
4. [Extended Explanation](#extended-explanation) 
5. [References](#references)

---

## Introduction

**ContraTCR** leverages contrastive learning to model the interaction between T-cell receptors and epitopes. By training on the modified PyTDC dataset, it aims to predict binding specificity with high accuracy. This guide provides a step-by-step walkthrough for users to:

- Train the ESM-2 model along with the projection head using modified PyTDC data.
- Extract features using the trained model.
- Use the extracted features to train classifiers for binding specificity prediction.


---

## Project Structure

The project consists of several key files and directories:

- `run.py`: Main script to run different modes (`train`, `extract`, `predict`).
- `config/`: Contains predefined configuration files (e.g., `midterm400_clean.yaml`).
- `model.py`: Defines the ESM-2 model and projection head.
- `data.py`: Data loading and preprocessing utilities.
- `train.py`: Training routines for different contrastive modes.
- `extract.py`: Feature extraction functions.
- `xgb.py`: Functions for training and evaluating the XGB model.
- `mlp.py`: Functions for training and evaluating the MLP model.
- `cnn.py`: Functions for training and evaluating the CNN model.
- `result/`: Default directory where results, logs, and checkpoints are saved.


---

## Running the Project

The project operates in three primary modes: `train`, `extract`, and `predict`. Below are detailed instructions for each step, including how to incorporate your own data. If you want to run it on Colab, please refer to `./colab/11785_project_runner.ipynb`

### Step 1: Train the ESM-2 Model with Projection Head

**Description**: Train the ESM-2 model along with a projection head.

**Command**:

```bash
!python ./run.py --config_path './config/default/your_config.yaml' --mode train
```

**Instructions**:

1. **Prepare Your Configuration File**:

   - Copy the example configuration file `midterm400_clean.yaml` and rename it (e.g., `your_config.yaml`).
   - Open `your_config.yaml` and update the hyperparameters as desired. 
   
2. **Run Training**:

   - Execute the command in your terminal or Colab notebook, replacing `your_config.yaml` with the name of your configuration file.
   - This will start the training process using your specified settings.

3. **Monitor Training**:

   - Training logs will be printed to the console and saved in the log directory specified in your configuration file.
   - Model checkpoints will be saved in the checkpoint directory.


---

### Step 2: Extract Features Using the Trained Model

**Description**: Utilize the trained ESM-2 model and projection head to extract features from the dataset.

**Command**:
   ```bash
   !python ./run.py --config_path ./config/default/your_config.yaml --resume_path '/path/to/your/model_checkpoint.pth' --mode extract
   ```

**Instructions**:


1. **Ensure Model Checkpoint Exists**:

   - After training, the model checkpoint should be saved in the checkpoint directory specified in your configuration file.
   - Locate the checkpoint file (e.g., `model_triplet.pth` or a similarly named file).


2. **Run Feature Extraction**:



   - Replace `/path/to/your/model_checkpoint.pth` with the actual path to your model checkpoint.

3. **Verify Extraction**:

   - The extracted features will be saved to the paths within the `result` directory.
   - Ensure that the feature files are generated successfully.



---

### Step 3: Predict Binding Specificity with Classifiers

**Description**: Use the extracted features to train different models, including XGB, MLP, and CNN, and perform binding specificity prediction.


**Command**:

```bash
!python ./run.py --config_path './config/default/your_config.yaml' \
--train_feature_path '/path/to/your/feature_data_train.csv' \
--test_feature_path '/path/to/your/feature_data_test.csv' \
--mode predict
```

**Instructions**:

1. **Locate Extracted Features**:

   - Identify where the feature extraction step saved your feature files.
   - Typically, these are named `feature_data_train.csv` and `feature_data_test.csv`.

2. **Update Paths**:

   - Replace `/path/to/your/feature_data_train.csv` and `/path/to/your/feature_data_test.csv` with the actual paths to your feature files.

3. **Run Prediction**:

   - Execute the command to start training the models and perform predictions.
   - The script will output performance metrics such as precision, recall, F1 score, and ROC AUC.

---

## Extended Explanation

   This repository was created for fullfilment of the term project expectations of Carnegie Mellon University's 11785 - Intro to Deep Learning. 
   For an in-depth explanation of the novelty, intention, and goals of this project please refer to the following:

   - **Final Report**: [Predicting TCR-Epitope Binding with Fine-Tuned Protein Language Models and Contrastive Learning](https://github.com/Yichuan0712/11785-TCR/tree/main/final_report/11785report.pdf)
   - **Video Explanation**: [11785 IDL Group 29 Video Report](https://www.youtube.com/watch?v=CmsnBlBCdzY)

---

## References

- **ESM-2 Model**: [Evolutionary Scale Modeling (ESM) GitHub Repository](https://github.com/facebookresearch/esm)
- **Contrastive Learning**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

