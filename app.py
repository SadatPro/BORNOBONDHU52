import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import time
import re
import os
import csv
from functools import lru_cache

# ==========================================
# Core Logic: Group Average Based Checker
# ==========================================
class GroupAveragedSpellChecker:
    def __init__(self, n_gram_range=(2, 4)): 
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=n_gram_range)
        self.tfidf_matrix = None
        self.root_to_indices = {} 
        self.valid_roots_set = set() 
        self.is_fitted = False

    def load_and_train(self, csv_file_path):
        print("Loading dictionary and training BornoBondhu52 Model...")
        try:
            # Check if file exists to avoid errors later
            if not os.path.exists(csv_file_path):
                print(f"Error: File {csv_file_path} not found.")
                return

            df = pd.read_csv(csv_file_path)
            corpus = []
            current_index = 0
            
            for _, row in df.iterrows():
                # Handle potential non-string data
                root_word = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                
                if not root_word or root_word.lower() == 'nan':
                    continue

                if root_word not in self.root_to_indices:
                    self.root_to_indices[root_word] = []
                    self.valid_roots_set.add(root_word) 

                # Add Root
                corpus.append(root_word)
                self.root_to_indices[root_word].append(current_index)
                current_index += 1
                
                # Add Misspellings (iterate over remaining columns)
                for col_idx in range(1, len(df.columns)):
                    val = row.iloc[col_idx]
                    if pd.notna(val) and str(val).strip() != "":
                        corpus.append(str(val).strip())
                        self.root_to_indices[root_word].append(current_index)
                        current_index += 1

            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.is_fitted = True
            print(f"Model Ready. Vocab size: {len(corpus)}")
            
        except Exception as e:
            print(f"Error loading file: {e}")

    @lru_cache(maxsize=2000)
    def suggest(self, input_word, top_n=5, threshold=0.15):
        if not self.is_fitted: return []
        
        # 1. Exact match check
        if input_word in self.valid_roots_set:
            return [] 

        input_vec = self.vectorizer.transform([input_word])
        all_scores = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        
        candidates = []
        for root, indices in self.root_to_indices.items():
            group_scores = all_scores[indices]
            avg_score = np.mean(group_scores)
            
            if avg_score > threshold:
                candidates.append({
                    'root_word': root,
                    'score': float(avg_score)
                })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]

# ==========================================
# Web Server Logic
# ==========================================
app = Flask(__name__)
checker = GroupAveragedSpellChecker()

# Config
CSV_PATH = 'cleaned_merged_final.csv' 

if os.path.exists(CSV_PATH):
    checker.load_and_train(CSV_PATH)
else:
    print(f"WARNING: {CSV_PATH} not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/check', methods=['POST'])
def check_text():
    if not checker.is_fitted:
        return jsonify({'error': 'Model not trained'}), 500

    data = request.json
    text = data.get('text', '')
    
    tokens = re.split(r'(\s+|[।,?!.\-])', text)
    
    response_data = []
    corrected_tokens = []
    
    for token in tokens:
        stripped_token = token.strip()
        
        if not stripped_token or not any(c.isalpha() for c in stripped_token):
            response_data.append({'word': token, 'type': 'text'})
            corrected_tokens.append(token)
            continue

        suggestions = checker.suggest(stripped_token, top_n=5)
        
        if suggestions:
            response_data.append({
                'word': token, 
                'type': 'error',
                'suggestions': [s['root_word'] for s in suggestions],
                'scores': [s['score'] for s in suggestions]
            })
            
            # Auto-correct Logic
            best_match = suggestions[0]
            if best_match['score'] > 0.6: 
                prefix = token[:len(token)-len(stripped_token)] if token.startswith(' ') else ''
                suffix = token[len(stripped_token):] if len(token) > len(stripped_token) else ''
                corrected_tokens.append(prefix + best_match['root_word'] + suffix)
            else:
                corrected_tokens.append(token)
                
        else:
            response_data.append({'word': token, 'type': 'text'})
            corrected_tokens.append(token)

    return jsonify({
        'tokens': response_data,
        'corrected_text': "".join(corrected_tokens)
    })

@app.route('/api/search_word', methods=['POST'])
def search_word():
    """Check if a word exists in the dictionary."""
    data = request.json
    word = data.get('word', '').strip()
    
    exists = word in checker.valid_roots_set
    return jsonify({'word': word, 'exists': exists})

@app.route('/api/add_word', methods=['POST'])
def add_word():
    data = request.json
    new_word = data.get('word', '').strip()
    
    if not new_word:
        return jsonify({'error': 'No word provided'}), 400
        
    try:
        # 1. Update In-Memory Set immediately
        checker.valid_roots_set.add(new_word)
        # Also ensure it has an entry in indices so future checks don't crash if re-indexed
        if new_word not in checker.root_to_indices:
            checker.root_to_indices[new_word] = []
    
        # 2. Append to CSV using standard CSV module (Safer than Pandas append)
        if os.path.exists(CSV_PATH):
            # Read header first to know how many columns to pad
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                num_columns = len(header)
        else:
            # Fallback if file doesn't exist (creates new file with 1 column)
            header = ['word']
            num_columns = 1
            with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # Prepare row data: [Word, Empty, Empty, ...]
        row_data = [''] * num_columns
        row_data[0] = new_word
        
        # Append mode 'a' with newline='' handles line breaks correctly
        with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        return jsonify({'success': True, 'message': f'Added "{new_word}" to dictionary.'})
        
    except Exception as e:
        print(f"Error appending to CSV: {e}")
        return jsonify({'error': 'Failed to save to file'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)