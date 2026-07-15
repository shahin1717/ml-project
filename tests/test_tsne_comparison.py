import numpy as np
from src.unsupervised.pca import PCA
from sklearn.manifold import TSNE
from src.unsupervised.tsne_comparison import run_tsne_comparison


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


def test_run_tsne_comparison():
    # Test execution on a tiny MNIST-like subset with plotting covered
    X_pca, X_tsne = run_tsne_comparison(n_samples=15, save_plot=True)
    assert X_pca.shape == (15, 2)
    assert X_tsne.shape == (15, 2)


