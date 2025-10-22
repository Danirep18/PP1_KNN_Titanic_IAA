"""
Daniel Muñoz Barragán - 308486
KNN algorithm for titanic survivors prediction (As part of the first partial project for the course PP1)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from collections import Counter

# --- Data Loading and Preprocessing ---
# Note: The file path is replaced with the local file name "train.csv"
data = pd.read_csv(r'C:\Users\sonic\OneDrive\Documents\IA Algorithms\PP1_Titanic_KNN\train.csv',sep=',')
print("--- Quick View of the Original Dataset ---")
print(data.head(5))
print("\n--- General Information ---")
print(data.info())
print(f"\nDataset Dimensions: {data.shape}")
print(f"Dataset Columns: {data.columns.tolist()}")

# Data imputation and cleaning
data['Age'].fillna(data['Age'].median(), inplace=True) # Filling missing Age values with the median age
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True) # Filling missing Embarked values with the mode
data['Fare'].fillna(data['Fare'].median(), inplace=True) # Filling missing Fare values with the median

# One-hot encoding for 'Sex' and 'Embarked' columns
data_train = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Selection of key features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = data_train[features].values # Converts to numpy array
y = data_train['Survived'].values # Converts to numpy array



#Implementation of Data Split (train_test_split) ---


def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Divides the data into test and train sets.

    X: Features (numpy array)
    y: Target variable (numpy array)
    test_size: Size of the test array (float)
    random_state: Seed for reproducibility (int)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Creates an array of shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split the indexes for test and train
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Applies the split to X and y
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Uses the custom function to split the data
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Distance and KNN


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points (numpy arrays).
    """
    return np.sqrt(np.sum((point1 - point2)**2))

class CustomKNearestNeighbors:
    """
    Custom implementation of the k-NN classifier from scratch.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        'Group fitting' in k-NN: stores the training data.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _predict_single(self, x_test):
        """
        Predicts the label for a single test instance.
        """
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = euclidean_distance(x_test, x_train)
            distances.append((dist, self.y_train[i]))
        
        # Orders by distance and selects the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.n_neighbors]
        
        # Obtains the neighbor labels
        neighbor_labels = [neighbor[1] for neighbor in neighbors]
        
        # Finds the most common label among neighbors (voting)
        most_common = Counter(neighbor_labels).most_common(1)
        
        return most_common[0][0]

    def predict(self, X_test):
        """
        Predicts the labels for the test set.
        """
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        Calculates the accuracy of the model.
        """
        predictions = self.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        return accuracy
    
# Initialize and fit the model 
k_value_for_evaluation = 11
knn_model = CustomKNearestNeighbors(n_neighbors=k_value_for_evaluation)
knn_model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = knn_model.predict(X_test)
accuracy = knn_model.score(X_test, y_test)

print(f"\n--- Prediction Results (k={k_value_for_evaluation}) ---")
print(f"Custom k-NN Accuracy (k={k_value_for_evaluation}): {accuracy:.4f}")

# Elbow method

def custom_elbow_method_knn(X_train, y_train, X_test, y_test, max_k=20):
    """
    Calculates the error (1 - accuracy) for different k values
    """
    errors = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        knn = CustomKNearestNeighbors(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        error = 1 - score
        errors.append(error)
        
    return list(k_range), errors

# Execute elbow method
k_values, k_errors = custom_elbow_method_knn(X_train, y_train, X_test, y_test, max_k=15)

# --- Data Preparation for Demographic Analysis (Moved Here to Prevent KeyError: 'AgeGroup') ---
# Create Age Range Categories (CRITICAL FIX)
bins = [0, 12, 18, 40, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle-Age', 'Elder']
data_train['AgeGroup'] = pd.cut(data_train['Age'], bins=bins, labels=labels, right=False)

# Re-label Pclass (Requested: 1, 2, 3 -> High, Medium, Low)
class_mapping = {1: 'High', 2: 'Medium', 3: 'Low'}
data_train['TicketClass'] = data_train['Pclass'].map(class_mapping)

# Select only survivors using the new ticket class column
survivor_data_modified = data_train[data_train['Survived'] == 1].copy()
# Convert Sex_male (0/1) to readable labels
survivor_data_modified['Sex'] = survivor_data_modified['Sex_male'].apply(lambda x: 'Male' if x == 1 else 'Female')
# ---------------------------------------------------------------------------------------------

# Outputs and Visualizations
print("\n" + "="*50)
print("--- ORIGINAL DATA ANALYSIS (data['Survived']) ---")
total_passengers = len(data)
survivors = data['Survived'].sum()
non_survivors = total_passengers - survivors
survival_rate = (survivors / total_passengers) * 100

print(f"Total Passengers in the training set: {total_passengers}")
print(f"People who Survived the Titanic: {survivors} ({survival_rate:.2f}%)")
print(f"People who Did NOT Survive: {non_survivors} ({100 - survival_rate:.2f}%)")
print("\n--- Survival Rate by Gender ---")
survival_by_sex = data.groupby('Sex')['Survived'].mean() * 100
print(survival_by_sex)
print("="*50)


# Metrics Implementation (No changes here)
def custom_classification_metrics(y_true, y_pred, pos_label=1):
    """Calculates and prints the confusion matrix, precision, recall, and F1-score."""
    
    # Confusion Matrix Calculations
    TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
    TN = np.sum((y_true != pos_label) & (y_pred != pos_label))
    FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
    FN = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    # Metrics Calculations
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Console Results
    print("\n--- CLASSIFICATION METRICS (POSITIVE CLASS: 1 - SURVIVED) ---")
    print(f"True Positives (TP - Survived and Predicted Survived): {TP}")
    print(f"True Negatives (TN - Did Not Survive and Predicted Did Not Survive): {TN}")
    print(f"False Positives (FP - Did Not Survive and Predicted Survived): {FP}")
    print(f"False Negatives (FN - Survived and Predicted Did Not Survive): {FN}")
    print("-" * 50)
    print(f"Precision: {precision:.4f} (Of all predicted '1's, how many are actually '1's)")
    print(f"Recall (Sensitivity): {recall:.4f} (Of all actual '1's, how many were detected)")
    print(f"F1-Score: {f1_score:.4f} (Harmonic mean of Precision and Recall)")
    
    return np.array([[TN, FP], [FN, TP]]) # Confusion Matrix [[0,0, 0,1], [1,0, 1,1]]

conf_matrix = custom_classification_metrics(y_test, y_pred, pos_label=1)


# Plots

# --- Enhanced Elbow Method Visualization with Polynomial Curve Fitting ---
from scipy.optimize import curve_fit

# Polynomial function of degree 2
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

# Curve fitting
params, _ = curve_fit(poly2, k_values, k_errors)

# Dense line for smooth curve
x_fit = np.linspace(min(k_values), max(k_values), 300)
y_fit = poly2(x_fit, *params)

# Optimal k determination (minimum error from original points)
optimal_k = k_value_for_evaluation
optimal_error = k_errors[optimal_k - 1]

# Plot
plt.figure(figsize=(10, 6))

# Blue line with markers (observed errors)
plt.plot(k_values, k_errors, color='royalblue', linestyle='-', linewidth=2, marker='o',
          markersize=7, label='Observed Errors')

# Smoothed polynomial fit line
plt.plot(x_fit, y_fit, '-', color='crimson', linewidth=2.2, label='Polynomial Fit (Degree 2)')

# Resalts optimal point
plt.scatter(optimal_k, optimal_error, color='limegreen', s=180, edgecolors='black', zorder=6,
             label=f'Optimal k = {optimal_k}')
plt.axvline(x=optimal_k, color='limegreen', linestyle='--', linewidth=2, alpha=0.8)

# Optimal point annotation
plt.annotate(f'k = {optimal_k}\nError = {optimal_error:.3f}',
             xy=(optimal_k, optimal_error),
             xytext=(optimal_k + 1.3, optimal_error + 0.012),
             arrowprops=dict(facecolor='limegreen', arrowstyle='->', lw=1.5),
             fontsize=11, color='black', fontweight='bold')

# Chart aesthetics
plt.title('Elbow Method with Polynomial Curve Fitting (k-NN)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Error Rate (1 - Accuracy)', fontsize=12)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.show()


# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
             xticklabels=['Predicted Not Survive (0)', 'Predicted Survive (1)'], 
             yticklabels=['Actual Not Survive (0)', 'Actual Survive (1)'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix (k={knn_model.n_neighbors})')
plt.show()


# Actual vs Predicted Survivors Plot (Test Set)
df_results = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
# Use Melt for Seaborn
df_melted = pd.melt(df_results.reset_index(), id_vars='index', value_vars=['Actual', 'Prediction'], 
                     var_name='Type', value_name='Survived')

plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Type', data=df_melted, palette='viridis')
plt.title('Survivor Comparison: Actual vs Predicted (Test Set)')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Passenger Count')
plt.legend(title='Data Type')
plt.show()

print("\n" + "="*50)
print("--- TOTAL SURVIVAL AND DEMOGRAPHIC ANALYSIS (Modified) ---")

# --- Visualization 1: Total Passengers vs. Total Survivors (Requested) ---

total_passengers = len(data_train)
survivors = data_train['Survived'].sum()
non_survivors = total_passengers - survivors

survival_counts = pd.DataFrame({
    'Condition': ['Total Passengers', 'Survived', 'Did Not Survive'],
    'Count': [total_passengers, survivors, non_survivors]
})
survival_counts['Percentage'] = (survival_counts['Count'] / total_passengers) * 100

plt.figure(figsize=(9, 6))
# Fix for Seaborn Deprecation Warning: Explicitly set hue='Condition' and legend=False
sns.barplot(x='Condition', y='Count', data=survival_counts, 
            hue='Condition', # Assign x to hue
            palette=['grey', 'green', 'red'], 
            order=['Total Passengers', 'Survived', 'Did Not Survive'],
            legend=False) # Suppress legend

plt.title('Total Passengers and Survivors on the Titanic', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('Number of People')


for index, row in survival_counts.iterrows():
    # Adjusted position for text annotation
    plt.text(row.name, row.Count + (total_passengers * 0.015), 
             f"{row.Count}\n({row.Percentage:.1f}%)", 
             color='black', ha="center", fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- Demographic Analysis Printout ---

print("\n--- Survivors by Sex, Age Group, and Ticket Class (Re-labeled) ---")
# Use the correct column names for grouping
survival_summary = data_train.groupby(['Sex_male', 'AgeGroup', 'TicketClass']).agg(
    Total_Passengers=('Survived', 'size'), 
    Survivors=('Survived', 'sum')
)
print(survival_summary)


# --- Visualization 2: Survivors by Sex and Age Group (Facet by Re-labeled Class) ---

# Loop to facet by the new class label ('High', 'Medium', 'Low')
for class_level in ['High', 'Medium', 'Low']:
    plt.figure(figsize=(10, 6))
    class_subset = survivor_data_modified[survivor_data_modified['TicketClass'] == class_level]
    sns.countplot(
        data=class_subset,
        x='AgeGroup',
        hue='Sex',
        palette='viridis',
        order=['Child', 'Teen', 'Adult', 'Middle-Age', 'Elder'] # Ensure consistent order
    )
    plt.title(f'Survivors by Sex and Age Group (Ticket Class: {class_level})', fontsize=13, fontweight='bold')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Survivors')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Sex')
    plt.tight_layout()
    plt.show()