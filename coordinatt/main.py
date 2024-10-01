import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna as opt
import opencompass as oc

import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertTokenizer


# Attention Model Definition
class AttentionModel(nn.Module):
    def __init__(self, emb_dim):
        super(AttentionModel, self).__init__()
        self.W_Q = nn.Linear(emb_dim, emb_dim)
        self.W_K = nn.Linear(emb_dim, emb_dim)
        self.W_V = nn.Linear(emb_dim, emb_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_embedding, sentence_embeddings):
        Q = self.W_Q(query_embedding)
        K = self.W_K(sentence_embeddings)
        V = self.W_V(sentence_embeddings)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# VectorStore class to store BERT embeddings
class VectorStore_CS:
    def __init__(self):
        self.vector_data = {}  # A dictionary to store vectors

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the store.
        Args:
            vector_id (str or int): A unique identifier for the vector.
            vector (numpy.ndarray): The vector data to be stored.
        """
        self.vector_data[vector_id] = vector

    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector using brute-force cosine similarity.
        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.
        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.
        """
        query_vector = np.squeeze(query_vector)  # Remove extra dimensions
        results = []

        for vector_id, vector in self.vector_data.items():
            vector = np.squeeze(vector)  # Remove extra dimensions
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
class VectorStore_Euc:
    def __init__(self):
        self.vector_data = {}

    def add_vector(self, vector_id, vector):
        self.vector_data[vector_id] = vector

    def find_similar_vectors(self, query_vector, num_results: int):
      query_vector = np.squeeze(query_vector)  # Remove extra dimensions
      results = []

      for vector_id, vector in self.vector_data.items():
          vector = np.squeeze(vector)  # Remove extra dimensions
          euclidian = np.linalg.norm(query_vector - vector)
          results.append((vector_id, euclidian))

      # Sort by similarity in ascending order
      results.sort(key=lambda x: x[1], reverse=False)

      # Return the top N results
      return results[:num_results]
class VectorStore_Man:
    def __init__(self):
        self.vector_data = {}

    def add_vector(self, vector_id, vector):
        self.vector_data[vector_id] = vector

    def find_similar_vectors(self, query_vector, num_results: int):
      query_vector = np.squeeze(query_vector)  # Remove extra dimensions
      results = []

      for vector_id, vector in self.vector_data.items():
          vector = np.squeeze(vector)  # Remove extra dimensions
          manhattan = np.absolute(query_vector - vector)
          results.append((vector_id, manhattan))

      ## Sort by similarity in ascending order
      results.sort(key=lambda x: x[1], reverse=False)

      # Return the top N results
      return results[:num_results]

# VectorStore class to store and retrieve embeddings
class VectorStore_Att:
    def __init__(self):
        self.vector_data = {}

    def add_vector(self, vector_id, vector):
        self.vector_data[vector_id] = vector

    def get_vectors(self):
        return self.vector_data

class VectorStore_DIEM:
    def __init__(self, vM, vm):
        self.vector_data = {}
        self.vM = vM
        self.vm = vm

    def add_vector(self, vector_id, vector):
        """Armazena um vetor na base de dados."""
        self.vector_data[vector_id] = vector

    def find_similar_vectors(self, query_vector, num_results=5):
        """Encontra os vetores mais similares usando a métrica DIEM."""
        query_vector = np.squeeze(query_vector)

        results = []
        for vector_id, vector in self.vector_data.items():
            vector = np.squeeze(vector)

            # Usando DIEM para calcular a similaridade
            similarity = diem(query_vector, vector, self.vM, self.vm)
            results.append((vector_id, similarity))

        # Ordenando por similaridade DIEM (menor distância = mais similar)
        results.sort(key=lambda x: x[1])
        return results[:num_results]

# Function to get BERT embedding for a sentence
def get_embedding(text, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token's embedding
    return embedding

def training(model, X, y, epochs):
    criterion = nn.Adam()  # You could use other losses like contrastive or triplet loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0017)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output, _ = model(X)
        loss = criterion(output, y)  # Compare model output to target labels

        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
# Function to query vector DB using attention mechanism
def query_vector_db(query, attention_model, vector_db, tokenizer, bert_model):
    # Get the query embedding
    query_embedding = get_embedding(query, bert_model, tokenizer)
    query_embedding = query_embedding.float().unsqueeze(0)  # Add batch dimension and ensure it's float32

    # Apply attention mechanism between query and stored embeddings
    db_tensor = vector_db.float()  # Ensure embeddings are float32
    output, attention_weights = attention_model(query_embedding, db_tensor)  # Pass both query and sentence embeddings

    return output, attention_weights

# Function to evaluate retrieval using attention mechanism
def evaluate_retrieval(attention_model, query, vector_db, tokenizer, bert_model):
    output, attention_weights = query_vector_db(query, attention_model, vector_db, tokenizer, bert_model)

    # Cosine similarity for comparison
    query_embedding = get_embedding(query, bert_model, tokenizer)
    cosine_similarities = torch.cosine_similarity(query_embedding.float(), vector_db.float())

    print("Attention-based Retrieval Output:", output)
    print("\nCosine Similarities:", cosine_similarities)

# Function to evaluate and display retrieval
def evaluate_retrieval(attention_model, query, vector_db, tokenizer, bert_model, sentences):
    output, attention_weights = query_vector_db(query, attention_model, vector_db, tokenizer, bert_model)

    # Cosine similarity for comparison
    query_embedding = get_embedding(query, bert_model, tokenizer)
    cosine_similarities = torch.cosine_similarity(query_embedding.float(), vector_db.float())

    # Sort sentences by similarity score (Cosine Similarity)
    sorted_indices = torch.argsort(cosine_similarities, descending=True)

    print(f"Query Sentence: {query}\n")
    print("Similar Sentences:")
    for idx in sorted_indices:
        sentence = sentences[idx]
        similarity = cosine_similarities[idx].item()
        print(f"{sentence}: Similarity = {similarity:.4f}")

def calcula_sigma_n(
    n: int,
    vM: float,
    vm:float,
    num_samples: int =10000): -> float

    """
    Calcula o desvio padrão da distância euclidiana para a dimensão n
    usando simulação.

    Args:
      n: Número de dimensões.
      vM: Valor máximo dos elementos do vetor.
      vm: Valor mínimo dos elementos do vetor.
      num_samples: Número de amostras para a simulação (opcional).

    Returns:
      O desvio padrão da distância euclidiana para a dimensão n.
    """
    vetores_a = np.random.uniform(vm, vM, size=(num_samples, n))
    vetores_b = np.random.uniform(vm, vM, size=(num_samples, n))
    distancias = np.linalg.norm(vetores_a - vetores_b, axis=1)
    return np.std(distancias)
 
def diem(
    a: np.ndarray,
    b: np.ndarray,
    vM: float,
    vm: float):
    """
    Calcula o DIEM entre dois vetores a e b.

    Args:
      a: O primeiro vetor.
      b: O segundo vetor.
      vM: Valor máximo dos elementos do vetor.
      vm: Valor mínimo dos elementos do vetor.

    Returns:
      O valor DIEM entre os vetores a e b.
    """
    n = len(a)
    sigma_n = calcula_sigma_n(n, vM, vm)
    distancia_euclidiana = np.linalg.norm(a - b)
    # Cálculo do DIEM
    diem_value = (vM - vm) / (sigma_n ** 2) * (distancia_euclidiana - np.sqrt(n * ((vM - vm) ** 2) / 6))
    return diem_value
       
# Definindo o modelo utilizado
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Example sentences for validation
sentences = [
    "The First Industrial Revolution began in Britain in the late 18th century.",
    "The invention of the steam engine was a key development during the First Industrial Revolution.",
    "Factories and mechanized production transformed industries like textiles and iron.",
    "The First Industrial Revolution marked a shift from manual labor to machine-based manufacturing.",
    "Railways expanded rapidly during the First Industrial Revolution, connecting distant regions.",
    "Steam power was used to run factories, locomotives, and ships, accelerating global trade.",
    "The Industrial Revolution had a profound impact on urbanization and the growth of cities.",
    "New inventions like the spinning jenny and the power loom revolutionized textile production.",
    "Child labor and poor working conditions were common in early industrial factories.",
    "The First Industrial Revolution laid the foundation for modern economies and industrial practices.",
    "Bananananananananananananodnsaubdyiqwofnquohr. What were the main inventions during the First Industrial Revolution?",
    "What were the main inventions during the First Industrial Revolution?",
    "Pizza barbecue paella"
]

# Defining a query sentence
query_sentence = "What were the main inventions during the First Industrial Revolution?" 


# Establish a VectorStore instance
vector_store = VectorStore_CS()

# Generate and store BERT embeddings in VectorStore
for sentence in sentences:
    embedding = get_embedding(sentence, model, tokenizer)
    vector_store.add_vector(sentence, embedding)

# Generate BERT embedding for the query sentence
query_vector = get_embedding(query_sentence, model, tokenizer)

# Find similar sentences using BERT embeddings
similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=12)

# Display similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences - With CS:")

for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
    
# Establish a VectorStore instance
vector_store_Euc = VectorStore_Euc()

# Generate and store BERT embeddings in VectorStore
for sentence in sentences:
    embedding = get_embedding(sentence, model, tokenizer)
    vector_store_Euc.add_vector(sentence, embedding)

# Generate BERT embedding for the query sentence
query_vector = get_embedding(query_sentence, model, tokenizer)

# Find similar sentences using BERT embeddings
similar_sentences = vector_store_Euc.find_similar_vectors(query_vector, num_results=12)

# Display similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences: - Euc")

for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
    
vector_store_att = VectorStore_Att()
# Generate embeddings for each sentence
embeddings = torch.cat([get_embedding(sentence, model, tokenizer) for sentence in sentences])

# Initialize the Attention Model
embed_dim = embeddings.shape[1]  # Dimensionality of BERT embeddings (768)
attention_model = AttentionModel(embed_dim)

evaluate_retrieval(attention_model, query_sentence, embeddings, tokenizer, model, sentences)

# Initialize the vector store with DIEM settings
vector_store_DIEM = VectorStore_DIEM(vM=1.0, vm=0.0)

# Assuming get_embedding function and BERT model are already defined
# Generate and store BERT embeddings in VectorStore
for sentence in sentences:
    embedding = get_embedding(sentence, model, tokenizer)
    vector_store_DIEM.add_vector(sentence, embedding)

# Generate BERT embedding for the query sentence
query_sentence = "What were the main inventions during the First Industrial Revolution?"
query_vector = get_embedding(query_sentence, model, tokenizer)

# Find similar sentences using BERT embeddings
similar_sentences = vector_store_DIEM.find_similar_vectors(query_vector, num_results=12)

# Display similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences:")

for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")