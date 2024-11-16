## Human Activity Recognition
### **Objective and Goals:**
The primary objective of this project is to classify human activity based on sensor data collected from wearable devices. The goal is to predict the activity (such as walking, sitting, standing, etc.) based on features derived from accelerometer and gyroscope data. This will involve preprocessing the data, building machine learning models, performing hyperparameter tuning, and evaluating model performance.

The specific goals of the project include:

1. **Data Preprocessing**:
   - Load the raw data (train and test datasets).
   - Clean and transform the data, handling issues such as missing values, duplicates, and column naming conventions.
   - Perform feature selection if needed.
   - Create a dataset suitable for model training.

2. **Data Exploration**:
   - Understand the distribution of the data and class labels.
   - Visualize the activity distribution across subjects.
   - Check for any data imbalance.

3. **Modeling**:
   - Train machine learning models for activity classification.
   - Evaluate models using metrics such as accuracy, confusion matrix, and classification report.
   - Fine-tune the models using techniques like grid search.

4. **Model Evaluation**:
   - Assess models on the test dataset.
   - Visualize the confusion matrix and classification reports.
   - Interpret the results and select the best model based on performance.

5. **Model Selection and Comparison**:
   - Compare different models (e.g., Logistic Regression, Support Vector Classifiers) to determine the best model for this problem.

### **Models Used:**
1. **Logistic Regression**:
   - Logistic Regression is a simple yet effective model for binary and multiclass classification problems.
   - Grid Search was used to optimize hyperparameters such as the regularization parameter `C` and the penalty type (`l1` or `l2`).

2. **Linear Support Vector Classifier (LinearSVC)**:
   - LinearSVC is a variant of SVM (Support Vector Machine) that is suited for linear classification problems. It can handle multiple classes via the "one-vs-rest" approach.
   - The `C` parameter was tuned through Grid Search to find the best value for the model's performance.

3. **Support Vector Classifier (SVC)**:
   - The standard SVC model is used for classification and includes different kernel options, which can be explored further. A grid search can also be applied to tune the kernel and other parameters.


### **Working Process:**

1. **Data Loading and Preprocessing**:
   - Load the features and the datasets (training and testing) from files.
   - Clean the columns, handling naming conventions and any formatting issues.
   - Add the `subject` and `Activity` columns to the dataset.
   - Convert the target labels to readable activity names (e.g., WALKING, SITTING, etc.).
   - Check for duplicates and missing values, handling them accordingly (e.g., by filling missing values with the mean).

2. **Data Visualization**:
   - Explore the data visually, including:
     - Count plot for each subject and activity type.
     - Visualizing the distribution of features and activity classes to understand the dataset better.
   - Check for any data imbalance issues using class counts.

3. **Data Splitting**:
   - Split the data into features (`X_train`, `X_test`) and target labels (`y_train`, `y_test`).
   - Perform a check on the shape of the data after splitting.

4. **Training and Model Evaluation**:
   - **Logistic Regression**: Using Grid Search to tune hyperparameters such as `C` and `penalty` to select the best combination.
   - **LinearSVC**: Similar hyperparameter tuning using Grid Search to select the best `C` value.
   - Train and test the models on the training and testing datasets.
   - Evaluate the models using performance metrics like accuracy, confusion matrix, and classification report.
   - Plot confusion matrices for better visual interpretation of the results.

5. **Hyperparameter Tuning**:
   - Grid Search is applied to optimize model parameters, such as the regularization parameter `C` for Logistic Regression and Linear SVC.
   - Perform model fitting and cross-validation to find the optimal model configuration.

6. **Performance Evaluation**:
   - Calculate and print the confusion matrix and classification report for the final model.
   - Visualize the confusion matrix to understand where the model is making mistakes (i.e., which activities are being confused).
   - Choose the best-performing model based on accuracy, confusion matrix, and classification report metrics.

### **Steps Summary:**

1. **Load the Data**: Load training and testing datasets, and assign column names from the `features.txt` file.
2. **Preprocess the Data**: Handle missing values and duplicate rows, perform feature cleaning.
3. **Visualize the Data**: Use visualizations (count plots, t-SNE, etc.) to understand the dataset's structure.
4. **Train Models**: Train models like Logistic Regression and LinearSVC using Grid Search for hyperparameter tuning.
5. **Evaluate Models**: Use metrics such as accuracy, confusion matrix, and classification report for evaluation.
6. **Tune Hyperparameters**: Use Grid Search to tune parameters and improve the models.
7. **Final Model Selection**: Compare the models and select the one with the best performance.
8. **Visualization and Reporting**: Generate confusion matrices, classification reports, and plots for better model interpretation.


