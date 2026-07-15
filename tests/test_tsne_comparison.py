import numpy as np
from src.unsupervised.pca import PCA
from sklearn.manifold import TSNE


def test_tsne_and_pca_integration():
    X = np.random.randn(50, 10)

    # PCA test
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (50, 2)

    # t-SNE test
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    X_tsne = tsne.fit_transform(X)
    assert X_tsne.shape == (50, 2)
