import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def calculate_distance(embedded_query, embedded_source):
    return np.linalg.norm(embedded_query - embedded_source)

def post_hoc_answerability(query, sources, output, baseline=None):
    query_source_distances = [calculate_distance(query, source) for source in sources]
    output_source_distances = [calculate_distance(output, source) for source in sources]
    source_similarities = [cosine_similarity([query_source_distances[i]], [output_source_distances[i]])[0][0] for i in range(len(sources))]
    
    alpha = 0.2
    weighted_avg = source_similarities[0]
    for i in range(1, len(source_similarities)):
        weighted_avg = alpha * source_similarities[i] + (1 - alpha) * weighted_avg
    
    matching_sources = sum(1 for d, e in zip(query_source_distances, output_source_distances) if d == e)
    multiplier = 1 + 0.1 * matching_sources
    
    if baseline is not None:
        normalized_avg = normalize([[weighted_avg * multiplier]], norm='l2', axis=1, copy=True, return_norm=False)[0][0]
    else:
        normalized_avg = weighted_avg * multiplier
    
    return normalized_avg

# Example usage:
query = np.array([0.2, 0.3, 0.4])  # Query vector
sources = [  # Source vectors list
    np.array([0.1, 0.2, 0.3]),
    np.array([0.4, 0.5, 0.6]),
    np.array([0.7, 0.8, 0.9]),
    np.array([1.0, 1.1, 1.2]),
    np.array([1.3, 1.4, 1.5])
]
output = np.array([0.3, 0.4, 0.5])  # Output vector

normalized_avg_similarity = post_hoc_answerability(query, sources, output)  # Calculate normalized average similarity
print("Normalized average similarity:", normalized_avg_similarity)  # Print result
