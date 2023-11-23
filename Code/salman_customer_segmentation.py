#%%
#1. Setup - mainly importing packages
import os
from datetime import datetime  

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
# %%
# 2. Load Data
df = pd.read_csv('train.csv')
# %%
# Inspect data
df.keys()

# %%
df.info() 

# object = id,job_type,marital,education,default,housing_loan,personal_loan, communication_type, month, prev_campaign_outcome.

# float = customer_age, balance, last_contact_duration, num_contacts_in_campaign, days_since_prev_campaign_contact

# int = day_of_month, num_contacts_prev_campaign, term_deposit_subscribed

# %%
df.describe()

#%%
# %%
df.drop(columns=['id'], inplace=True)
#%%
df.hist(figsize=(20,20), edgecolor='black')
plt.show()

# %%
df.isnull().sum()

#%% drop 'days_since_prev_campaign_contact' column due to too many NAs
df = df.drop(columns='days_since_prev_campaign_contact')


#%%
df.isnull().sum()
#%%
# Fill null value
df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
df['marital'].fillna('Unknown', inplace=True)
df['balance'].fillna(df['balance'].median(), inplace=True)
df['personal_loan'].fillna('Unknown', inplace=True)
df['last_contact_duration'].fillna(df['last_contact_duration'].median(), inplace=True)
df['num_contacts_in_campaign'].fillna(df['num_contacts_in_campaign'].median(), inplace=True)

# %% double check
df.isnull().sum()

#%%
df.duplicated().sum()   #no duplicated

#%%
num_cols = ['customer_age', 'balance', 'day_of_month', 'last_contact_duration', 'num_contacts_in_campaign', 'num_contacts_prev_campaign']
df[num_cols]

#%%
cat_cols = list(df.drop(columns=num_cols).columns)

df[cat_cols] = df[cat_cols].fillna(method='ffill', axis=0)

df.isnull().sum()
#%%
# Correlation
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

sns.heatmap(df[num_cols].corr(), 
            annot=True, 
            ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show()

#%%
df.describe()

#%%
numerical_df = df.select_dtypes(include=['float64','int64'])
numerical_df.head()

#%%
categorical_df=df.select_dtypes(include='object')
categorical_df.head()

# %%
def count_plot(df,feature):
    sns.set(color_codes = 'Blue', style="whitegrid")
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context(rc = {'patch.linewidth': 0.0})
    fig = plt.subplots(figsize=(10,3))
    sns.countplot(x=feature, data=df, color = 'steelblue') # countplot
    plt.show()

#%%
df.hist(figsize=(8,8), edgecolor='black')
plt.show()

#%%
for cat_col in categorical_df.columns:
    if cat_col in ['job_type','marital','education', 'default','housing_loan','personal_loan']:
        count_plot(df,cat_col)

# %%
# Preprocessing
oe = OrdinalEncoder()
df[cat_cols[0: -1]] = oe.fit_transform(df[cat_cols[0: -1]])
df

# %%
oe.categories_

# %%
X = df.drop(columns='term_deposit_subscribed')
X

# %%
y = df[['term_deposit_subscribed']]
y

# %%
ohe = OneHotEncoder(sparse=False)

y_encoded = ohe.fit_transform(y)
y_encoded

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=13)

#%%
# Create a Sequential model
model = Sequential()

# Add input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Add hidden layers
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Add output layer
model.add(Dense(y_train.shape[1], activation='softmax'))

#%%

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# TensorBoard callback for logging training process

PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = TensorBoard(logpath)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, tensorboard]
)

#%%
# Display the model summary
model.summary()

#%%
plot_model(model, show_shapes=True, show_layer_names=(True))

#%%
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")

# %%
model.save(os.path.join('models','classify_v1.h5'))

# %%
