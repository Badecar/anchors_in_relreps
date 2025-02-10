import numpy as np
from scipy.spatial.distance import pdist

def objective_function(embeddings, anchors, Coverage_weight=1, diversity_weight=1, exponent=0.5):
    def diversity(embeddings, anchors):
        return (1/len(embeddings)) * sum([min([
                                        abs(pdist([embedding, anchor], metric="cosine")) for anchor in anchors])
                                        for embedding in embeddings])
    
    def coverage(anchors, exponent):
        dists = pdist(anchors, metric="cosine")
        return sum(abs(dist)**exponent for dist in dists)
    
    return (diversity_weight * diversity(embeddings, anchors) - Coverage_weight * coverage(anchors, exponent))[0]
