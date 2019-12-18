# coding=utf8

'''
version: December 18 2019, 03:56 PM
Last revision: December 18, 2019 05:52 PM

Author : Chao-Hsuan Ke
'''

'''
Reference

https://www.kaggle.com/sanikamal/text-classification-with-python-and-keras
https://keras.io/visualization/
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python#svm
'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# import classification
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt


#dataset folder
filepath_dict = {'yelp':   'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\data\\yelp_labelled.txt',
                 'amazon': 'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\data\\amazon_cells_labelled.txt',
                 'imdb':   'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\data\\imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
#print(df.iloc[0])
#print(df.iloc[1])

# data decoding
sentences = ['John likes ice cream', 'John hates chocolate.']
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
#print(vectorizer.vocabulary_)
#print(vectorizer.transform(sentences).toarray())

# data split
df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# classification (Logistic Regression)
#classifier = LogisticRegression()
#classifier.fit(X_train, y_train)
#score = classifier.score(X_test, y_test)
#print("Accuracy:", score)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Keras classification
input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#print(model.summary())

history = model.fit(X_train, y_train,
                    epochs = 100,
                    verbose = False,
                    validation_data = (X_test, y_test),
                    batch_size= 20)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
#print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
#print("Testing Accuracy:  {:.4f}".format(accuracy))

# plotting
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
#    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
#    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

#plot_history(history)
