#!/usr/bin/env python
# coding: utf-8

# In[39]:


get_ipython().system('pip install pandas')


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rcParams['figure.figsize']=(20, 8)
import numpy as np


# In[41]:


hyp_data = pd.read_csv('GHdata.csv')


# In[42]:


hyp_data


# In[43]:


hyp_data.shape


# In[44]:


hyp_data.info


# In[45]:


#missing values check
hyp_data.isnull().sum()


# In[46]:


hyp_data.plot(kind='box')
plt.show()


# In[47]:


#Encoding
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

age=enc.fit_transform(hyp_data['Age'])
gravida=enc.fit_transform(hyp_data['Gravida'])
para=enc.fit_transform(hyp_data['Para'])
temp=enc.fit_transform(hyp_data['Temp'])
pulse=enc.fit_transform(hyp_data['Pulse'])
resp=enc.fit_transform(hyp_data['Respiration'])
bp=enc.fit_transform(hyp_data['BP'])
hb=enc.fit_transform(hyp_data['HB'])
hgt=enc.fit_transform(hyp_data['HGT'])
weight=enc.fit_transform(hyp_data['Weight'])


hyp_data['Age']=age
hyp_data['Gravida']=gravida
hyp_data['Para']=para
hyp_data['Temp']=temp
hyp_data['Pulse']=pulse
hyp_data['Respiration']=resp
hyp_data['BP']=bp
hyp_data['HB']=hb
hyp_data['HGT']=hgt
hyp_data['Weight']=weight


# In[48]:


#Quotient-- X---train_X, test_X 80/20

#Answer--Y---train_Y, test_Y
# Drop the Gestational_Hypertension because this is the target not a feature
X=hyp_data.drop('Gestational_Hypertension', axis=1)

X.head()



# In[49]:


Y=hyp_data['Gestational_Hypertension']
Y


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler
std=StandardScaler()

X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)


# In[51]:


X_train_std


# In[52]:


X_test_std


# In[53]:


##1. Evaluate Decision Tree Classifier (for Classification)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instantiate and train the Decision Tree classifier
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_tree = tree_classifier.predict(X_test)

# Compute classification evaluation metrics
accuracy_tree = accuracy_score(Y_test, Y_pred_tree)
precision_tree = precision_score(Y_test, Y_pred_tree)
recall_tree = recall_score(Y_test, Y_pred_tree)
f1_tree = f1_score(Y_test, Y_pred_tree)
conf_matrix_tree = confusion_matrix(Y_test, Y_pred_tree)

# Display classification evaluation metrics
print("Decision Tree Classifier Metrics:")
print("Accuracy:", accuracy_tree)
print("Precision:", precision_tree)
print("Recall:", recall_tree)
print("F1 Score:", f1_tree)
#print("Confusion Matrix:\n", conf_matrix_tree)


# In[ ]:





# In[54]:


# 2Logistic Reggression

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Instantiate and train the Logistic Regression classifier
logreg_classifier = LogisticRegression(random_state=42,  max_iter=1000)
logreg_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_classifier = logreg_classifier.predict(X_test)

# Compute classification evaluation metrics
accuracy_classifierLR = accuracy_score(Y_test, Y_pred_classifier)
precision_classifierLR = precision_score(Y_test, Y_pred_classifier)
recall_classifierLR = recall_score(Y_test, Y_pred_classifier)
f1_classifierLR = f1_score(Y_test, Y_pred_classifier)
conf_matrix_classifier = confusion_matrix(Y_test, Y_pred_classifier)

# Display classification evaluation metrics
print("Logistic Regression Classifier Metrics:")
print("Accuracy:", accuracy_classifierLR)
print("Precision:", precision_classifierLR)
print("Recall:", recall_classifierLR)
print("F1 Score:", f1_classifierLR)
#print("Confusion Matrix:\n", conf_matrix_classifier)



# In[ ]:





# In[55]:


#3 KNN-K-Nearest Neighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instantiate and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Example: using 5 neighbors
knn_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_classifier = knn_classifier.predict(X_test)

# Compute classification evaluation metrics
accuracy_classifierKKN = accuracy_score(Y_test, Y_pred_classifier)
precision_classifierKKN = precision_score(Y_test, Y_pred_classifier)
recall_classifierKKN = recall_score(Y_test, Y_pred_classifier)
f1_classifierKKN = f1_score(Y_test, Y_pred_classifier)
conf_matrix_classifier = confusion_matrix(Y_test, Y_pred_classifier)

# Display classification evaluation metrics
print("K-Nearest Neighbors (KNN) Classifier Metrics:")
print("Accuracy:", accuracy_classifierKKN)
print("Precision:", precision_classifierKKN)
print("Recall:", recall_classifierKKN)
print("F1 Score:", f1_classifierKKN)
#print("Confusion Matrix:\n", conf_matrix_classifier)



# In[ ]:





# In[ ]:





# In[56]:


# 4 Random Forest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instantiate and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=70, random_state=42)  # Example: using 100 trees
rf_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_classifier = rf_classifier.predict(X_test)

# Compute classification evaluation metrics
accuracy_classifierRF = accuracy_score(Y_test, Y_pred_classifier)
precision_classifierRF = precision_score(Y_test, Y_pred_classifier)
recall_classifierRF = recall_score(Y_test, Y_pred_classifier)
f1_classifierRF = f1_score(Y_test, Y_pred_classifier)
conf_matrix_classifier = confusion_matrix(Y_test, Y_pred_classifier)

# Display classification evaluation metrics
print("Random Forest Classifier Metrics:")
print("Accuracy:", accuracy_classifierRF)
print("Precision:", precision_classifierRF)
print("Recall:", recall_classifierRF)
print("F1 Score:", f1_classifierRF)
#print("Confusion Matrix:\n", conf_matrix_classifier)



# In[ ]:





# In[ ]:






# In[57]:


#########
#5 SVM-Support Vector Machine

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Assuming X and Y are defined
# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instantiate and train the SVM classifier with class weights
svm_classifier = SVC(kernel='rbf', class_weight='balanced', random_state=42)  # Example: using radial basis function (RBF) kernel
svm_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_classifier = svm_classifier.predict(X_test)

# Compute classification evaluation metrics with zero_division set
accuracy_classifierSVM = accuracy_score(Y_test, Y_pred_classifier)
precision_classifierSVM = precision_score(Y_test, Y_pred_classifier, zero_division=1)
recall_classifierSVM = recall_score(Y_test, Y_pred_classifier)
f1_classifierSVM = f1_score(Y_test, Y_pred_classifier)
conf_matrix_classifier = confusion_matrix(Y_test, Y_pred_classifier)

# Display classification evaluation metrics
print("SVM Classifier Metrics:")
print("Accuracy:", accuracy_classifierSVM)
print("Precision:", precision_classifierSVM)
print("Recall:", recall_classifierSVM)
print("F1 Score:", f1_classifierSVM)
print("Confusion Matrix:\n", conf_matrix_classifier)


# In[58]:


#6 Decision Jungle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instantiate and train the AdaBoostClassifier
adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)  # Example: using 50 estimators
adaboost_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = adaboost_classifier.predict(X_test)

# Compute classification evaluation metrics
accuracyDJ = accuracy_score(Y_test, Y_pred)
precisionDJ = precision_score(Y_test, Y_pred)
recallDJ = recall_score(Y_test, Y_pred)
f1DJ = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Display classification evaluation metrics
print("Decision Jungle Metrics:")
print("Accuracy:", accuracyDJ)
print("Precision:", precisionDJ)
print("Recall:", recallDJ)
print("F1 Score:", f1DJ)
#print("Confusion Matrix:\n", conf_matrix)


# In[ ]:





# In[59]:


# 7 AP-Averaged Perceptron
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix,recall_score, f1_score
from sklearn.metrics import r2_score
perc=Perceptron(max_iter=1000, random_state=43) # You can adjust hyperparameters as needed
perc.fit(X_train_std, Y_train)
Y_pred=perc.predict(X_test_std)
accuracy_perc=accuracy_score(Y_test,Y_pred)
precision_AP = precision_score(Y_test, Y_pred)


recall_AP = recall_score(Y_test, Y_pred)
f1_AP = f1_score(Y_test, Y_pred)

# Step 4: Calculate R-squared (Pearson Correlation Coefficient for regression)
correlation_coefficient8 = np.corrcoef(Y_test, Y_pred)[0, 1]
r_squared = r2_score(Y_test, Y_pred)
#print("Pearson Correlation Coefficient:", correlation_coefficient8)

# Optional: Confusion Matrix
conf_matrix = confusion_matrix(Y_test,Y_pred)

#print("Confusion Matrix:\n", conf_matrix)

# Step 5: Print or Use the Results
print("Averaged Perceptron Metrics:")
print("Accuracy:", accuracy_perc)
print("Precision:", precision_AP)
print("R-squared:", r_squared)

print("F1 Score:", f1_AP)
print("Recall:", recall_AP)


# In[ ]:





# In[ ]:





# In[60]:


########
#8  Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming X and Y are defined
# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate and train the MLPClassifier (Neural Network) with increased iterations
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, solver='adam', learning_rate_init=0.001)  # Example: 2 hidden layers with 100 and 50 neurons
mlp_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = mlp_classifier.predict(X_test)

# Compute classification evaluation metrics with zero_division set
accuracyNN = accuracy_score(Y_test, Y_pred)
precisionNN = precision_score(Y_test, Y_pred, zero_division=1)
recallNN = recall_score(Y_test, Y_pred)
f1NN = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Display classification evaluation metrics
print("Neural Network (MLPClassifier) Metrics:")
print("Accuracy:", accuracyNN)
print("Precision:", precisionNN)
print("Recall:", recallNN)
print("F1 Score:", f1NN)
print("Confusion Matrix:\n", conf_matrix)



# In[61]:


#9 Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instantiate and train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = nb_classifier.predict(X_test)

# Compute classification evaluation metrics
accuracyNB = accuracy_score(Y_test, Y_pred)
precisionNB = precision_score(Y_test, Y_pred)
recallNB = recall_score(Y_test, Y_pred)
f1NB = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Display classification evaluation metrics
print("Multinomial Naive Bayes Classifier Metrics:")
print("Accuracy:", accuracyNB)
print("Precision:", precisionNB)
print("Recall:", recallNB)
print("F1 Score:", f1NB)
#print("Confusion Matrix:\n", conf_matrix)






#################################################################################################



# In[ ]:





# In[62]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define algorithms and their corresponding metrics
algorithms = [
    "Decision Tree",
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Random Forest",
    "Support Vector Machine",
    "Decision Jungle",
   "Averaged Perceptron",
    "Neural Network (MLPClassifier)",
    "Multinomial Naive Bayes"
]

accuracy_scores = [ accuracy_tree, accuracy_classifierLR, accuracy_classifierKKN, accuracy_classifierRF,accuracy_classifierSVM, accuracyDJ, accuracy_perc, accuracyNN, accuracyNB ]
precision_scores = [precision_tree, precision_classifierLR, precision_classifierKKN, precision_classifierRF, precision_classifierSVM, precisionDJ, precision_AP, precisionNN, precisionNB ]
recall_scores = [recall_tree, recall_classifierLR, recall_classifierKKN, recall_classifierRF, recall_classifierSVM, recallDJ, recall_AP, recallNN, recallNB]
f1_scores = [f1_tree, f1_classifierLR, f1_classifierKKN, f1_classifierRF, f1_classifierSVM,  f1DJ, f1_AP, f1NN, f1NB ]

# Create subplots for each metric
plt.figure(figsize=(12, 10))

# Accuracy subplot
plt.subplot(2, 2, 1)
sns.barplot(x=accuracy_scores, y=algorithms, palette="viridis")
plt.title("Accuracy Scores")
plt.xlabel("Accuracy")
plt.ylabel("Algorithm")

# Precision subplot
plt.subplot(2, 2, 2)
sns.barplot(x=precision_scores, y=algorithms, palette="magma")
plt.title("Precision Scores")
plt.xlabel("Precision")
plt.ylabel("Algorithm")

# Recall subplot
plt.subplot(2, 2, 3)
sns.barplot(x=recall_scores, y=algorithms, palette="plasma")
plt.title("Recall Scores")
plt.xlabel("Recall")
plt.ylabel("Algorithm")

# F1 Score subplot
plt.subplot(2, 2, 4)
sns.barplot(x=f1_scores, y=algorithms, palette="inferno")
plt.title("F1 Scores")
plt.xlabel("F1 Score")
plt.ylabel("Algorithm")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[ ]:




