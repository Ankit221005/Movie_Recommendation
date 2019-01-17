
# coding: utf-8

# In[116]:


from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000) 
#The argument num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded.
'''The variables train_data and test_data are lists of reviews; each review is a list of
word indices (encoding a sequence of words). train_labels and test_labels are
lists of 0s and 1s, where 0 stands for negative and 1 stands for positive'''


# In[117]:


#you’re restricting yourself to the top 10,000 most frequent words, no word index will exceed 10,000
print(max([max(sequence) for sequence in train_data]))
print(train_labels)
train_data.shape


# In[118]:


word_index = imdb.get_word_index() #word_index is a dictionary mapping words to an integer index.
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()]) #Reverses it, mapping integer indices to words
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]]) #Decodes the review. Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown


# In[119]:


import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences ), dimension ))
    for i,sequence in enumerate( sequences):
        results[i , sequence]=1
    return results
X_train=vectorize_sequences(train_data)
X_test=vectorize_sequences(test_data)


# In[120]:


Y_train=np.asarray(train_labels).astype('float32')
Y_test=np.asarray(test_labels).astype('float32')


# In[121]:


Y_train.shape


# In[122]:


from keras import models 
from keras import layers


# In[123]:


model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


# In[124]:


X_val=X_train[:10000]
partial_X_train=X_train[10000:]
Y_val=Y_train[:10000]
partial_Y_train=Y_train[10000:]


# In[125]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


# In[126]:


History=model.fit(partial_X_train,partial_Y_train,epochs=10,batch_size=512,validation_data=(X_val,Y_val))


# In[127]:


History_dict=History.history
'''model.fit() returns a History object. This object has a member
history, which is a dictionary containing data about everything that happened
during training'''
History_dict.keys()


# In[128]:


import matplotlib.pyplot as plt
loss_values=History_dict['loss']
val_loss=History_dict['val_loss']
acc=History_dict['acc']
val_acc=History_dict['val_acc']


# In[129]:


len(History_dict['loss'])


# In[130]:


epochs=range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo' , label='Training loss') #bo is for blue dot
plt.plot(epochs, val_loss, 'b' , label='Validation loss') #b is for solid blue line
plt.title('Training and validation sets')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[131]:


plt.clf() #clear the figures
plt.plot(epochs, acc, 'bo' , label='Training accuracy') #bo is for blue dot
plt.plot(epochs, val_acc, 'b' , label='Validation accuracy') #b is for solid blue line
plt.title('Training and validation sets')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[132]:


results=model.evaluate(X_test,Y_test)


# In[133]:


results


# In[134]:


print('Loss :' +str(results[0]))
print('Accuracy :' +str(results[1]))

