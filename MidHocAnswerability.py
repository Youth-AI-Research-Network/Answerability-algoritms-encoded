import numpy as np
from sklearn.preprocessing import normalize

def calculate_distance(embedded_query, embedded_source):

    # Placeholder implementation of distance calcs
    return np.linalg.norm(embedded_query - embedded_source)

def assess_answerability(query, sources, alpha=0.2, baseline=None):

    # Calculate distances between the query and each source
    distances = [calculate_distance(query, source) for source in sources]
    # Initialize weighted average with the distance to the first source
    weighted_avg = distances[0]
    # Calculate weighted average of distances
    for i in range(1, len(distances)):
        weighted_avg = alpha * distances[i] + (1 - alpha) * weighted_avg
    # Normalize the weighted average
    if baseline is not None:
        normalized_avg = normalize([[weighted_avg]], norm='l2', axis=1, copy=True, return_norm=False)[0][0]
    else:
        normalized_avg = weighted_avg
    
    return normalized_avg

# Example usage:
query = np.array([0.2, 0.3, 0.4]) 
sources = [
    np.array([0.1, 0.2, 0.3]),
    np.array([0.4, 0.5, 0.6]),
    np.array([0.7, 0.8, 0.9]),
    np.array([1.0, 1.1, 1.2]),
    np.array([1.3, 1.4, 1.5])
]  

normalized_avg_distance = assess_answerability(query, sources)
print("Normalized average distance:", normalized_avg_distance)