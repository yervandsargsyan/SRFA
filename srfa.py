import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def srfa_(X, y, eps=1e-8, random_state=42):
    np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    if n_features < n_samples:
        n_fictitious = n_samples - n_features
        X_fict = eps * np.eye(n_samples)[:, :n_fictitious]
        X_aug = np.hstack([X, X_fict])
    elif n_features > n_samples:
        n_fictitious = n_features - n_samples
        X_fict = eps * np.eye(n_features)[:n_samples, :]
        X_aug = np.hstack([X, X_fict])
    else:
        X_aug = X  

    w_aug = np.linalg.inv(X_aug) @ y

    w_real = w_aug[:n_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    y_train_pred = X_train @ w_real
    y_test_pred = X_test @ w_real

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return w_real, y_train_pred, y_test_pred

# -------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_features = 1000, 600
    X = np.random.rand(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    srfa(X, y)
