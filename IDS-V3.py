#
#
# # #Code for Training Random Forest, XGBoost and Deep learning Model on CIC-IDS2017 dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE #SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


data_path = "D:/COM711-CyberSecurity/Coursework-2/MachineLearningCSV (2)/MachineLearningCVE"

#Step 1: Load and Merge CSVs
data_files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]


dataframes = [pd.read_csv(os.path.join(data_path, file), low_memory=False) for file in data_files]

# Strip spaces from column names
dataframes = [df.rename(columns=lambda x: x.strip()) for df in dataframes]

# Removing any leading/trailing spaces from column names (like ' Label' → 'Label').
#print to verify 'Label' exists
for i, df in enumerate(dataframes):
    print(f"File {i+1} columns: {df.columns.tolist()}")

df = pd.concat(dataframes, ignore_index=True)
#df = df.sample(frac=0.4, random_state=42) ## Reduce size to 40% for example

# Step 2: Clean and Preprocess
df.replace([np.inf, -np.inf], np.nan, inplace=True)#Convert infinities (inf, -inf) to NaN and Replaces all +infinity and -infinity values with NaN (Not a Number)
df.dropna(axis=1, how='all', inplace=True) #Removes any column where all values are NaN
df.dropna(inplace=True) #Removes any row that contains at least one NaN

# Count and plot - Histogram of Attack Types (Label Counts)
label_counts = df["Label"].value_counts()
plt.figure(figsize=(14, 12))
label_counts.plot(kind='bar', color='steelblue')
plt.title("Histogram of Attack Types (Label Counts)", fontsize=16)
plt.xlabel("Attack Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Step 3: Encode labels
label_encoder = LabelEncoder() #convert categorical string labels (like "BENIGN", "DDoS", "Bot") into numeric values (like 0, 1, 2, etc.).
df['Label'] = label_encoder.fit_transform(df['Label'])#fit() tells the encoder to learn all unique label values.transform() converts those labels into numbers.
class_names = label_encoder.classes_
joblib.dump(label_encoder, "label_encoder.joblib")
#See encoded class index-to-label mapping
for idx, label in enumerate(label_encoder.classes_):
    print(f"{idx}: {label}")

# Step 4: Feature Selection
df = df.select_dtypes(include=['int64', 'float64']) #It filters out only numerical columns from the entire DataFrame df.Only columns with data types int64 or float64 are kept.
X = df.drop('Label', axis=1)#It creates a new DataFrame X which contains all columns except the 'Label' column. axis=1 means it’s dropping a column (not a row).
y = df['Label']#This extracts the 'Label' column from df and assigns it to y. y is now your target variable for prediction
features = X.columns.tolist()
joblib.dump(features, "features_list.joblib")

# Step 5: Scale & Split
scaler = StandardScaler() #StandardScaler standardizes features by removing the mean and scaling to unit variance.
X_scaled = scaler.fit_transform(X)#This fits the scaler to your dataset X and transforms it in one step. All features in X_scaled now have a mean of 0 and standard deviation of 1.
#This splits your dataset into: Training Set: 70%, Test Set: 30%'.
# random_state=42-Fixes randomness for reproducibility
#stratify=y Ensures class distribution is preserved in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
joblib.dump(scaler, "scaler.joblib")
counts = Counter(y)
print("Whole Data Count :", counts)
print("To be trained Data Count, used in Whole Data without SMOTE:", Counter(y_train))

# Applying SMOTE for Undersampling 'Bening' class and oversampling all other classes
counts = Counter(y)
print("Whole data Count:", counts)
# Determine new class 0 size
class_0_current = counts[0]
class_0_target = class_0_current - 2041196
# Build under-sampling strategy
under_strategy = {0: class_0_target}  # Only class 0 will be downsampled
undersampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
X_final, y_final = undersampler.fit_resample(X, y)
print("Data Count After undersampling Benign Class:", Counter(y_final))

custom_strategy = {
    1 : 230124,
    2 : 230124,
    3 : 230124,
    4 : 230124,
    5 : 230124,
    6 : 230124,
    7 : 230124,
    8 : 230124,
    9 : 230124,
    10 : 230124,
    11 : 230124,
    12 : 230124,
    13 : 230124,
    14 : 230124
}
# target_count = 200000  # target count for all classes
# class_dist = Counter(y_train)
#
# custom_strategy = {
#     cls: target_count for cls, count in class_dist.items() if count < target_count
# }
smote = SMOTE(random_state=42, sampling_strategy= custom_strategy, k_neighbors=10) #SMOTE, by default, uses k_neighbors=5, meaning it needs at least 6 samples in each minority class to generate synthetic ones. The error:.. means you have a class with only 2 samples, so SMOTE can’t find enough neighbors to generate new data for it.
X_smote, y_smote = smote.fit_resample(X_final, y_final)
print("Data Count After SMOTE:", Counter(y_smote))
# Apply Standard SCALAR
smoteScaler = StandardScaler() #StandardScaler standardizes features by removing the mean and scaling to unit variance
X_smote_scaled = smoteScaler.fit_transform(X_smote)
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote_scaled, y_smote, test_size=0.3, random_state=42, stratify=y_smote)
print("Data Count After Splitting SMOTE Data in 70/30:", Counter(y_train_smote))
joblib.dump(smoteScaler, "SMOTEScaler.joblib")

# Count and plot - Histogram of Attack Types (Label Counts) after using SMOTE
plt.figure(figsize=(12, 6))
sns.countplot(x=y_train_smote)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Attack Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion matrix plotting function
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Step 6: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#n_estimators=100: The model will use 100 decision trees in the forest.
#More trees usually increase accuracy (but also training time).
#random_state=42: This sets a seed for reproducibility.
#Helps ensure the same results every time you run the code.
rf_model.fit(X_train, y_train) #Trains the model on labeled data. Builds 100 decision trees using bootstrapped subsets of data.
rf_pred = rf_model.predict(X_test) #Applies the model to test data to make predictions
print(" Random Forest Report without SMOTE:\n", classification_report(y_test, rf_pred))
#Plot confusion matrix heatmap from Random Forest
plot_confusion(y_test, rf_pred, "Random Forest Confusion Matrix without SMOTE")
#Plot Feature Importance from Random Forest
importances = rf_model.feature_importances_  # raw importance values
feature_names = X.columns  # feature names from the dataset
indices = np.argsort(importances)[::-1] # descending order
plt.figure(figsize=(12, 13))
plt.title("Feature Importances - Random Forest", fontsize=16)
plt.barh(range(len(indices)), importances[indices], align="center", color='royalblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
plt.gca().invert_yaxis()  # highest at top
plt.xlabel("Relative Importance", fontsize=12)
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
# save Rf Model
joblib.dump(rf_model, "rf_model.joblib")

#Random Forest Model Using SMOTE
smoteRF_model = RandomForestClassifier(n_estimators=100, random_state=42)
smoteRF_model.fit(X_train_smote, y_train_smote)
smoteRF_pred = smoteRF_model.predict(X_test_smote) #Applies the model to test data to make predictions
print(" Random Forest Report with Using SMOTE:\n", classification_report(y_test_smote, smoteRF_pred))
# save smoteRf_Model
joblib.dump(smoteRF_model, "smoteRF_model.joblib")
plot_confusion(y_test_smote, smoteRF_pred, "Random Forest Confusion Matrix After applying SMOTE ")
# save smoteRf_Model



# Step 7: XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42) #use_label_encoder=False:Disables the older label encoding mechanism
# instance of an XGBoost classifier model. Tells the model to use multi-class log loss (cross-entropy) to evaluate performance during training.
xgb_model.fit(X_train, y_train) #Trains the model on labeled data.
xgb_pred = xgb_model.predict(X_test)#Applies the model to test data to make predictions
print("XGBoost Report without SMOTE:\n", classification_report(y_test, xgb_pred))
#Plot confusion matrix heatmap from XGBoost Model
plot_confusion(y_test, xgb_pred, "XGBoost Confusion Matrix without SMOTE")
# save xgb Model to File
joblib.dump(xgb_model, "xgb_model.joblib")

#XGBoost Forest Model Using SMOTE
smoteXGB_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
smoteXGB_model.fit(X_train_smote, y_train_smote)
smoteXGB_pred = smoteXGB_model.predict(X_test_smote) #Applies the model to test data to make predictions
print("XGBoost Report After Applying SMOTE:\n", classification_report(y_test_smote, smoteXGB_pred))
#Plot confusion matrix heatmap from XGBoost Model after applying SMOTE
plot_confusion(y_test_smote, smoteXGB_pred, "XGBoost Confusion Matrix with SMOTE")
# save xgb Model to File
joblib.dump(smoteXGB_model, "smoteXGB_model.joblib")

# Step 8: Deep Learning (Keras)
# y_train_cat = to_categorical(y_train) #Converts the training labels to catagorical values
# y_train_smote_cat = to_categorical(y_train_smote) #For SMOTE
# y_test_cat = to_categorical(y_test) #Converts the test labels to catagorical values
# y_test_smote_cat = to_categorical(y_test_smote)

dl_model = Sequential([
    Input(shape=(X_train.shape[1],)), #Takes input features — this equals the number of features per sample, like packet length, IAT, etc.
    Dense(128, activation='relu'), #First hidden layer with 128 neurons and ReLU activation (helps model learn nonlinear patterns).
    Dropout(0.3),                  #Randomly drops 30% of the neurons during training to prevent overfitting.
    Dense(64, activation='relu'),  #Second hidden layer, smaller — forces model to compress and learn deeper representations.
    Dropout(0.2),                  #Another  regularization layer with 20% dropout.
    #Dense(y_train_cat.shape[1], activation='softmax') #Output layer. The number of units equals the number of attack classes.
    Dense(len(class_names), activation='softmax')
])                                 #softmax: ensures the output is a probability distribution over classes.
# compile the model
dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Adaptive learning optimizer — fast and efficient.
#loss='categorical_crossentropy':Used for multi-class classification when labels are one-hot encoded (like y_train_cat)
#metrics=['accuracy']: To track how often predictions match actual labels.
# fit/Train the model
#dl_model.fit(X_train, y_train_cat, epochs=200, batch_size=128, verbose=1)
dl_model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1)
#Trains the model for 200 epochs.
#Uses 128 samples per training batch.
# Evaluate DL model
#loss, accuracy = dl_model.evaluate(X_test, y_test_cat) #Evaluates performance on unseen test data.
loss, accuracy = dl_model.evaluate(X_test, y_test) #Evaluates performance on unseen test data.
print(f"Deep Learning Accuracy: {accuracy:.4f}")
# DL predictions and confusion matrix
dl_pred = dl_model.predict(X_test).argmax(axis=1) #Gets predicted probabilities.
#.argmax(axis=1): Converts probabilities into class labels.
print(classification_report(y_test, dl_pred))
# precision = precision_score(y_test, dl_pred, average='weighted', zero_division=0)
# recall = recall_score(y_test, dl_pred, average='weighted', zero_division=0)
# f1 = f1_score(y_test, dl_pred, average='weighted', zero_division=0)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# The error you're getting is due to how precision_score, recall_score, and f1_score are being used on a multiclass classification problem.
# By default, these functions assume a binary classification task (average='binary'). Since your task involves multiple attack classes, you must specify a suitable average method.
#Plot confusion matrix heatmap from Deep Learning Model
# Alternatively, you can also use average='macro' or average='micro' depending on your evaluation needs:
# 'macro': Treats all classes equally.
# 'weighted': Accounts for class imbalance (recommended in your case).
# 'micro': Calculates metrics globally by counting total true positives, etc.
plot_confusion(y_test, dl_pred, "Deep Learning Confusion Matrix")
# save DL Model to file
dl_model.save('dl_model.keras')
#Saving all three Models and the passing the Input Network traffic features

#Smote DL Model
smoteDL_model = Sequential([
    Input(shape=(X_train_smote.shape[1],)), #Takes input features — this equals the number of features per sample, like packet length, IAT, etc.
    Dense(128, activation='relu'), #First hidden layer with 128 neurons and ReLU activation (helps model learn nonlinear patterns).
    Dropout(0.3),                  #Randomly drops 30% of the neurons during training to prevent overfitting.
    Dense(64, activation='relu'),  #Second hidden layer, smaller — forces model to compress and learn deeper representations.
    Dropout(0.2),                  #Another  regularization layer with 20% dropout.
    #Dense(y_train_smote_cat.shape[1], activation='softmax') #Output layer. The number of units equals the number of attack classes.
    Dense(len(class_names), activation='softmax')
])
smoteDL_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#smoteDL_model.fit(X_train_smote, y_train_smote_cat, epochs=200, batch_size=128, verbose=1)
smoteDL_model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=128, verbose=1)
#loss2, accuracy = smoteDL_model.evaluate(X_test_smote, y_test_smote_cat) #Evaluates performance on unseen test data.
loss2, accuracy = smoteDL_model.evaluate(X_test_smote, y_test_smote) #Evaluates performance on unseen test data.
print(f"Deep Learning Accuracy Using SMOTE: {accuracy:.4f}")
smoteDL_pred = smoteDL_model.predict(X_test_smote).argmax(axis=1) #Gets predicted probabilities.
print(classification_report(y_test_smote, smoteDL_pred))
# precision_Smote = precision_score(y_test_smote, smoteDL_pred, average='weighted', zero_division=0)
# recall_Smote = recall_score(y_test_smote, smoteDL_pred, average='weighted', zero_division=0)
# f1_Smote = f1_score(y_test_smote, smoteDL_pred, average='weighted', zero_division=0)
# print("Precision_Smote:", precision_Smote)
# print("Recall_Smote:", recall_Smote)
# print("F1_Smote:", f1_Smote)
plot_confusion(y_test_smote, smoteDL_pred, "Deep Learning Confusion Matrix after applying SMOTE  ")
smoteDL_model.save('smoteDL_model.keras')


#Load all saved Models
from tensorflow.keras.models import load_model
dl_model_pred = load_model('dl_model.keras')
xgb_model_pred = joblib.load("xgb_model.joblib")
rf_model_pred = joblib.load("rf_model.joblib")

#Input for predictions
#DDOS Attack
final_Input = np.array([[80.0, 1293792.0, 3.0, 7.0, 26.0, 11607.0, 20.0, 0.0, 8.666666667, 10.26320288,
           5840.0, 0.0, 1658.142857, 2137.29708, 8991.398927, 7.72921768, 143754.6667, 430865.8067,
           1292730.0, 2.0, 747.0, 373.5, 523.9661249, 744.0, 3.0, 1293746.0, 215624.3333, 527671.9348,
           1292730.0, 2.0, 0.0, 0.0, 0.0, 0.0, 72.0, 152.0, 2.318765304, 5.410452376, 0.0, 5840.0,
           1057.545455, 1853.437529, 3435230.673, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0,
           1163.3, 8.666666667, 1658.142857, 72.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 26.0, 7.0,
           11607.0, 8192.0, 229.0, 2.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

input_scaled = scaler.transform(final_Input)


#For Random Forest and XGBoost, predict() already returns the class ind ex directly (e.g., 2 for DDoS).
#This makes the model incorrectly label the input as class 0 (BENIGN) even though it predicted class 2 (DDoS).
#So, no need np.argmax() for them.
#RF predictions
rf_pred = rf_model_pred.predict(input_scaled)
# Extract the predicted class index
predicted_class_index = rf_pred[0]
# Decode the index into the actual class label (optional)
predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
# Print result
print("Predicted probabilities:", rf_pred)
print("Predicted class index:", predicted_class_index)
print("Predicted label:", predicted_class_label)

#XGBoost predictions
xgb_pred = xgb_model_pred.predict(input_scaled)
# Extract the predicted class index
predicted_class_index = xgb_pred[0]
# Decode the index into the actual class label (optional)
predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
# Print result
print("Predicted probabilities:", xgb_pred)
print("Predicted class index:", predicted_class_index)
print("Predicted label:", predicted_class_label)


#DL predictions
dl_pred = dl_model_pred.predict(input_scaled)
# Extract the predicted class index
predicted_class_index = np.argmax(dl_pred)
# Decode the index into the actual class label (optional)
predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
# Print result
print("Predicted probabilities:", dl_pred)
print("Predicted class index:", predicted_class_index)
print("Predicted label:", predicted_class_label)


##################------------------------------------------------------------------------------------------------------------------------------------
#Code for Monitoring CICFlowMeter output for detecting the live intrusions in daily logs
#Now I’ll run the monitoring code. A simulated network flow file will be dropped into the folder.
# As soon as it's detected, the system will read it, classify it using all models,
# and if an attack is detected, you’ll see an alert printed in real-time on the terminal.

import time
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Setup logging
logging.basicConfig(filename='alerts.log', level=logging.INFO, format='%(asctime)s - %(message)s')


class FlowCSVHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.csv'):
            return

        file_path = event.src_path
        print(f"\nNew flow file detected: {file_path}")
        # Wait a bit for file writing to complete
        time.sleep(2)

        # Retry loop until file is not in use
        for _ in range(5):
            try:
                # Try opening the file
                with open(file_path, 'r') as f:
                    f.read()
                break  # Success
            except PermissionError:
                print("⏳ File is still being written... waiting")
                time.sleep(2)
        else:
            print("File could not be accessed after multiple attempts.")
            return
         # Load all saved models
        scaler = joblib.load("scaler.joblib")
        smoteScaler = joblib.load("SMOTEScaler.joblib")
        expected_features = joblib.load("features_list.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        rf_model_pred = joblib.load("rf_model.joblib")
        xgb_model_pred = joblib.load("xgb_model.joblib")
        dl_model_pred = load_model('dl_model.keras')
        smoteRF_model_pred = joblib.load("smoteRF_model.joblib")
        smoteXGB_model_pred = joblib.load("smoteXGB_model.joblib")
        smoteDL_pred = load_model('smoteDL_model.keras')

        seen_rows = 0
        while True:
            try:
                df_new = pd.read_csv(file_path)
                df_new.columns = df_new.columns.str.strip() #Strip spaces from column names
                df_new = df_new.select_dtypes(include=['int64', 'float64'])

                if 'Label' in df_new.columns:
                    df_new.drop('Label', axis=1, inplace=True)

                # Ensuring columns match exactly
                df_new = df_new.reindex(columns=expected_features, fill_value=0)

                # Droping missing or irrelevant columns
                df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_new.dropna(axis=1, how='all', inplace=True)
                df_new.dropna(inplace=True)

                # Skip previously seen rows
                new_data = df_new.iloc[seen_rows:]
                if new_data.empty:
                    time.sleep(2)
                    continue

                for index, row in new_data.iterrows():
                    # input_row = row.values.reshape(1, -1)
                    # input_scaled = scaler.transform(input_row)
                    input_row = pd.DataFrame([row.to_dict()])
                    input_row = input_row.reindex(columns=expected_features, fill_value=0)
                    input_scaled = scaler.transform(input_row.values)
                    smoteInput = smoteScaler.transform(input_row.values)

                    # Random Forest
                    rf_preds = rf_model_pred.predict(input_scaled)[0]
                    rf_label = label_encoder.inverse_transform([rf_preds])[0]

                    #SMOTE Random Forest
                    smoteRF_preds = smoteRF_model_pred.predict(smoteInput)[0]
                    smoteRF_label = label_encoder.inverse_transform([smoteRF_preds])[0]

                    # XGBoost
                    xgb_pred = xgb_model_pred.predict(input_scaled)[0]
                    xgb_label = label_encoder.inverse_transform([xgb_pred])[0]

                    # SMOTE XGBoost
                    smoteXGB_preds = smoteXGB_model_pred.predict(smoteInput)[0]
                    smoteXGB_label = label_encoder.inverse_transform([smoteXGB_preds])[0]


                    # # Deep Learning
                    # dl_pred_index = np.argmax(dl_model_pred.predict(input_scaled))
                    dl_pred_probs = dl_model_pred.predict(input_scaled)
                    dl_pred_index = np.argmax(dl_pred_probs, axis=1)[0]
                    dl_label = label_encoder.inverse_transform([dl_pred_index])[0]


                    # # SMOTE Deep Learning
                    # smoteDL_preds_index = np.argmax(smoteDL_pred.predict(smoteInput))[0]
                    smoteDL_pred_probs = smoteDL_pred.predict(smoteInput)
                    smoteDL_preds_index = np.argmax(smoteDL_pred_probs, axis=1)[0]
                    smoteDL_label = label_encoder.inverse_transform([smoteDL_preds_index])[0]

                    # ALERT if attack
                    is_attack = rf_label != "BENIGN" or xgb_label != "BENIGN" or dl_label != "BENIGN"
                    timestamp = datetime.now().strftime('%H:%M:%S')

                    print(f"\n [{timestamp}] New Flow Detected")
                    print(f"    RF : {rf_label}")
                    print(f"    XGB: {xgb_label}")
                    print(f"    DL : {dl_label}")
                    print(f"    Smote_RF : {smoteRF_label}")
                    print(f"    Smote_XGB: {smoteXGB_label}")
                    print(f"    Smote_DL : {smoteDL_label}")

                    if is_attack:
                        print("ALERT: Suspicious activity detected!\n")

                seen_rows = len(df_new)

            except Exception as e:
                print(f"Error processing file: {e}")
            time.sleep(2)

# Monitor this directory
csv_watch_dir = "D:/COM711-CyberSecurity/Coursework-2/MachineLearningCSV (2)/Daily_Network_Logs"
event_handler = FlowCSVHandler()
observer = Observer()
observer.schedule(event_handler, path=csv_watch_dir, recursive=False)
observer.start()

try:
    print(f"Monitoring folder: {csv_watch_dir} for new flows...\n")
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    observer.stop()
observer.join()

