import string

# Load the dataset
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Convert text to lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Save the preprocessed text
with open('preprocessed_shakespeare.txt', 'w', encoding='utf-8') as file:
    file.write(text)