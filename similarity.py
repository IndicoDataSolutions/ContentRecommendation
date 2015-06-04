from sklearn.externals import joblib
from scipy.spatial.distance import cdist
import numpy as np

print "Loading data"
posts = joblib.load('data/posts.jl')
post_tags = joblib.load('data/post_tags.jl')
features = np.asarray([tags.values() for tags in post_tags])

def find_nearest(array, value, n=3):
    """
    Return the indices of the most similar documents
    """
    idxs = cdist(array, np.asarray([value]), metric='cosine').argsort(axis=0)
    return idxs[:n]

if __name__ == "__main__":
    print "Finding pairs"

    for i in range(5):
        idxs = find_nearest(features, features[i])

        print "Original:"
        print posts[idxs[0]]

        for idx in idxs[1:]:
            print "Similar:\n-----------------"
            print  posts[idx]
