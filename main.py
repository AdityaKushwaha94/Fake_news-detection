#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Specify file paths for the datasets
file_path1 = 'C:\\Users\\Aditya\\OneDrive\\Desktop\\interview_imp_question\\fake.csv'
file_path2 = 'C:\\Users\\Aditya\\OneDrive\\Desktop\\interview_imp_question\\true.csv'

# Read the first dataset into a pandas DataFrame (df1)
df1 = pd.read_csv(file_path1)

# Read the second dataset into another pandas DataFrame (df2)
df2 = pd.read_csv(file_path2)

# Display the first few rows of df1
print("First few rows of df1:")
print(df1.head())

# Display the first few rows of df2
print("\nFirst few rows of df2:")
print(df2.head())


# In[5]:


df1.head()


# In[18]:


df1['label'] = 0
df2['label'] = 1
concatenated_df = pd.concat([df1, df2], axis=0)


# In[19]:


df=concatenated_df


# In[20]:


df.head()


# In[21]:


from sklearn.model_selection import train_test_split



# In[22]:


x=df['text']
y=df['label']#to 0 and 1 if fake 0 if true then 1.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000000)
X_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
X_test_tfidf = tfidf_vectorizer.transform(x_test)


# In[26]:


from sklearn.linear_model import LogisticRegression

# Define and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test_tfidf)


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on the testing TF-IDF vectors
y_pred = model.predict(X_test_tfidf)

# Evaluate the model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[33]:


# Example: Make predictions on new news article text
new_articles = input("Enter the news: ")  # Get input as a single string

# Transform the new article into a list containing the single article text
new_articles_list = [new_articles]
new_articles_tfidf = tfidf_vectorizer.transform(new_articles_list)
predicted_labels = model.predict(new_articles_tfidf)

print("Predicted Labels for New Article:")
for article, label in zip(new_articles_list, predicted_labels):
    predicted_label = 'True' if label == 1 else 'Fake'
    print(f"Article: '{article}' --> Predicted Label: {predicted_label}")


# In[32]:


from sklearn.metrics import classification_report

# Evaluate model performance on the training data
y_train_pred = model.predict(X_train_tfidf)
print("Training Set Performance:")
print(classification_report(y_train, y_train_pred))


# In[ ]:




