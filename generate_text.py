import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random

def load_model_and_tokenizer():
    
    model = tf.keras.models.load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    
    max_sequence_len = model.input_shape[1]
    
    return model, tokenizer, max_sequence_len

def sample_predictions(preds, temperature=1.0):
    
    preds = np.asarray(preds).astype('float64')
    
    if temperature == 0:
        return np.argmax(preds)
    
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, num_words, temperature=0.7):
    
    model, tokenizer, max_sequence_len = load_model_and_tokenizer()
    
    generated_text = seed_text
    
    for _ in range(num_words):
        
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        
        token_list = pad_sequences(
            [token_list], 
            maxlen=max_sequence_len-1, 
            padding='pre'
        )
        
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        
        predicted_index = sample_predictions(predicted_probs, temperature)
        
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
                
        # Update seed text and generated text
        seed_text += " " + output_word
        generated_text += " " + output_word
    
    return generated_text

if __name__ == "__main__":
    # Example usage
    seeds = [
        "the king hath",
        "to be or not to be",
        "i lve your",
        
    ]
    
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        generated = generate_text(seed, num_words=20, temperature=0.7)
        print("Generated text:")
        print(generated)