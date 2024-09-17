# Spam Email Detection Using LSTM and NLP

This project involves detecting spam emails using a deep learning model (LSTM) and NLP techniques. The model is trained using textual data from a dataset of emails and is built in Google Colab using TensorFlow and Keras.


## Overview
The goal of this project is to classify emails as either "Spam" or "Non-Spam" using a deep learning model. The key steps involve exploratory data analysis (EDA), text preprocessing, model building using LSTM, and performance evaluation.

## Dataset
The dataset contains 5,171 emails labeled as spam or non-spam. It is loaded into a pandas DataFrame, and it consists of two columns:
- `text`: The email content.
- `spam`: Label indicating whether the email is spam (1) or non-spam (0).



## Libraries and Dependencies
The following libraries are required for running the project:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `nltk`
- `wordcloud`
- `sklearn`

Install them using:

```bash
pip install numpy pandas matplotlib seaborn tensorflow nltk wordcloud scikit-learn
```

## Data Preprocessing
1. **Text Cleaning:**
   - Removal of stop words, punctuation, and unwanted text.
   - Performed using `nltk` for stop words and Python’s string library for punctuations.
   
2. **Downsampling:**
   - Since the dataset is imbalanced (more non-spam than spam emails), downsampling is used to create a balanced dataset.

3. **Tokenization:**
   - Convert text data to sequences of token IDs using the `Tokenizer` from Keras.

4. **Padding:**
   - Padding the tokenized sequences to ensure they all have the same length for feeding into the model.

## Model Architecture
The model consists of:
- **Embedding Layer:** Converts words into dense vectors of fixed size.
- **LSTM Layer:** Captures sequential patterns in the text data.
- **Dense Layers:** Fully connected layers with ReLU and Sigmoid activations.
- **Output Layer:** A single unit with a sigmoid activation function for binary classification.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=100),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Training the Model
The model is trained using binary cross-entropy loss, Adam optimizer, and evaluated using accuracy metrics. EarlyStopping and ReduceLROnPlateau callbacks are used to optimize the training process.

```python
history = model.fit(train_sequences, train_Y, validation_data=(test_sequences, test_Y), epochs=20, batch_size=32, callbacks=[lr, es])
```

## Model Evaluation
The model achieved:
- **Test Accuracy:** 0.6934270858764648
- **Test Loss:** 0.48494982719421387

Model performance is visualized using accuracy and loss graphs.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

## Results
- The model successfully identifies spam emails with an accuracy of **0.6934270858764648**.
- The model shows good generalization as seen in the minimal difference between training and validation accuracy.

