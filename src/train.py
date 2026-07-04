import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from preprocess import preprocess
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import joblib



# Read the train.csv
# Then, we extract features of bidders present inside train.csv
train_df = pd.read_csv("./data/train.csv")
features = preprocess(train_df)

# Merge with outcome column
train_data = features.merge(train_df[['bidder_id', 'outcome']], on='bidder_id')

# Generate the X and y variables accordingly
X = train_data.drop(columns=['bidder_id', 'outcome']).values
y = train_data['outcome'].values

# Train/validation split
# Test size for training is 80%, so, it is around 1,600 of data whereas for validation size is 20% and have 400 data
# Make sure to stratify, so that the number of bots are same across both testing set and validation set
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Class weights
# Since there are class imbalances, computing class_weights is important 
# so that penalty of misclassifying bot as human is bigger penalty
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Feedforward Neural Network
# We have 5 layers, 4 hidden layers and 1 output layer
# Dropout of 0.4 for layer 1-3, and dropout 0.3 for layer 4
# Batchnorm after layer 1-3
# use activation function GELU instead of ReLU, in my experimentation, it performs better
# Output layer uses sigmoid function of course
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='gelu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='gelu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='gelu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='gelu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Use Adam optimizer as it is the standard
# Use Binary Crossentropy, which is the standard for binary classification tasks
# Use AUC metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC']
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=1
)

# Evaluate with validation set
y_pred_proba = model.predict(X_val).flatten()
auc = roc_auc_score(y_val, y_pred_proba)
print(f"\nValidation AUC: {auc:.4f}")



# Calculate metrics
val_auc = roc_auc_score(y_val, y_pred_proba)
val_aupr = average_precision_score(y_val, y_pred_proba)

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {val_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR (AUPR = {val_aupr:.3f})')
plt.axhline(y=sum(y_val)/len(y_val), color='k', linestyle='--', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.savefig('model_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"ROC AUC: {val_auc:.4f}")
print(f"AUPR: {val_aupr:.4f}")



# Finding best threshold
# It is rarely the case that the threshold will be 0.5
# So, after training, we tried to find the best threshold that maximize the f1 score
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_proba > threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")


# Use best threshold to calculate the classification report and confusion matrix
y_pred_binary = (y_pred_proba > best_threshold).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_pred_binary))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred_binary))


# Save
# We will be saving multiple files:
# - The FNN model
# - The scaler file, which is fit to the training data sample
# - The best threshold as well 
model.save('./models/FNN.keras')
joblib.dump(scaler, './models/scaler.pkl')
joblib.dump(best_threshold, './models/threshold.pkl')
