import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

def load_and_tokenize_data(file_path, sequence_length=50):
    """
    Load preprocessed text and tokenize it into sequences
    
    Args:
        file_path (str): Path to preprocessed text file
        sequence_length (int): Length of input sequences
        
    Returns:
        tuple: (tokenizer, X, y, total_words)
    """
    with open('preprocessed_shakespeare.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Tokenize the text at word level
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    
    # Create input sequences
    input_sequences = []
    lines = text.split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    # Pad sequences and create predictors and label
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    return tokenizer, X, y, total_words, max_sequence_len

def build_model(total_words, max_sequence_len):
    """
    Build LSTM model for text generation
    
    Args:
        total_words (int): Total number of unique words
        max_sequence_len (int): Maximum sequence length
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        LSTM(150, return_sequences=True),
        LSTM(100),
        Dense(total_words, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X, y, epochs=100, batch_size=128):
    """
    Train the model with early stopping
    
    Args:
        model: Keras model
        X: Training data
        y: Training labels
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        history: Training history
    """
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Train the model
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    # Load and preprocess data
    tokenizer, X, y, total_words, max_sequence_len = load_and_tokenize_data(
        "preprocessed_shakespeare.txt"
    )
    
    # Build model
    model = build_model(total_words, max_sequence_len)
    
    # Train model
    history = train_model(model, X, y)
    
    # Save model and tokenizer
    model.save("lstm_model.h5")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    print("Model training complete. Model and tokenizer saved.")