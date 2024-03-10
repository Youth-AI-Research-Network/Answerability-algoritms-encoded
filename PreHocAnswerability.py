import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def random_sample(dataset, x):

    return np.random.choice(dataset, x, replace=False)

def embed(data):

    return np.random.rand(len(data))

def pre_hoc_answerability_estimator(query, dataset):

    sample_size = min(10, len(dataset))
    sample_data = random_sample(dataset, sample_size)
    
    # Compute embeddings for the sampled data
    sample_embeddings = [embed(d) for d in sample_data]
    average_embedding = np.mean(sample_embeddings, axis=0)
    
    query_embedding = embed(query)
    
    # calcs cosine similarity of query and avg samle embedding
    similarity = cosine_similarity([query_embedding], [average_embedding])[0][0] 
    
    # Normalizes similarity score
    normalized_similarity = normalize([[similarity]])[0][0]
    

    forecasted_answerability = normalized_similarity > 0.5
    
    return forecasted_answerability

dummy_dataset = [ #array of vectors
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5]
]  

query = [0.2, 0.3, 0.4]  # Example query  
forecasted_answerability = pre_hoc_answerability_estimator(query, dummy_dataset)
print("Forecasted answerability:", forecasted_answerability)