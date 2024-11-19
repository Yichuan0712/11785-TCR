import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from util import printl, printl_file


def cnn_train_and_evaluate(configs, train_csv_path, test_csv_path, use_smi, log_path):
    if configs.dataset != "PyTDC":
        raise NotImplementedError

    # Load train and test datasets
    df_train = pd.read_csv(train_csv_path)
    df_train_ori = pd.read_csv('/content/11785-TCR/dataset/pytdc_new/train2_PyTDC.csv')

    df_test = pd.read_csv(test_csv_path)
    df_test_ori = pd.read_csv('/content/11785-TCR/dataset/pytdc_new/test_PyTDC.csv')

    # Add 'epitope_smi' column from original train and test data
    df_train['epitope_smi'] = df_train_ori['epitope_smi'].values
    df_test['epitope_smi'] = df_test_ori['epitope_smi'].values

    # Function to extract features from SMILES
    def extract_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return pd.Series([0] * 5)  # If SMILES is invalid, return zeros
        features = {
            'mol_weight': Descriptors.MolWt(mol),
            'logP': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol)
        }
        return pd.Series(features)

    # Prepare features for both train and test datasets
    def prepare_features(df, use_smi=False, encoder=None):
        X = df.drop(columns=['x', 'label'])

        non_numeric_columns = X.select_dtypes(include=['object']).columns
        if 'y' in non_numeric_columns:
            # One-hot encode the 'y' column
            if encoder is None:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                y_encoded = encoder.fit_transform(X[['y']])
            else:
                y_encoded = encoder.transform(X[['y']])
            y_encoded_df = pd.DataFrame(y_encoded, columns=encoder.get_feature_names_out(['y']))

            # Merge the one-hot encoded 'y' column with other features and drop the original 'y' column
            X = X.drop(columns=['y'])
            X = pd.concat([X, y_encoded_df], axis=1)

        if use_smi:
            # Apply feature extraction to 'epitope_smi' column
            X_features = X['epitope_smi'].apply(extract_features)
            X = pd.concat([X, X_features], axis=1)
            X = X.drop(columns=['epitope_smi'])
        else:
            X = X.drop(columns=['epitope_smi'])

        return X, encoder

    # Prepare training features
    X_train, encoder = prepare_features(df_train, use_smi=use_smi)
    # Target labels for training
    y_train = df_train['label']

    # Prepare test features
    X_test, _ = prepare_features(df_test, use_smi=use_smi, encoder=encoder)
    # Target labels for testing
    y_test = df_test['label']

    # Handle missing values if any
    # X_train.fillna(0, inplace=True)
    # X_test.fillna(0, inplace=True)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for CNN input
    # CNN expects 3D input: (samples, time steps, features)
    # Here, we treat each feature as a "time step" with one "channel"
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # Build Enhanced CNN model
    model = Sequential([
        # First Convolutional Block
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Second Convolutional Block
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Third Convolutional Block
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Flatten and Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_cnn_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(
        X_train_cnn, y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Make predictions on the test data
    y_pred_proba = model.predict(X_test_cnn).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    printl(f'Accuracy on test data: {accuracy:.2f}', log_path=log_path)
    printl(f'AUROC on test data: {auroc:.2f}', log_path=log_path)
    printl(f'AUPR on test data: {aupr:.2f}', log_path=log_path)
    printl(f'F1 Score on test data: {f1:.2f}', log_path=log_path)
    printl('Classification Report:', log_path=log_path)
    report = classification_report(y_test, y_pred)
    printl(report, log_path=log_path)

