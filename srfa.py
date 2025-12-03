import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def srfa(data, test_size=0.2, n_fictitious=5, eps=1e-8, alpha_real=1e-5, alpha_fictitious=1e3, random_state=42):
    """
    SRFA pipeline: builds stabilized regression, evaluates on train/test
    
    Parameters:
        data : tuple (X, y)
            X : array-like, shape (n_samples, n_features)
            y : array-like, shape (n_samples,)
        test_size : float
            Fraction of data for testing
        n_fictitious : int
            Number of fictitious features
        eps : float
            Small magnitude for fictitious features
        alpha_real : float
            Regularization for real features
        alpha_fictitious : float
            Strong regularization for fictitious features
        random_state : int
            For reproducibility
    
    Returns:
        dict with coefficients, train/test predictions and R^2
    """
    X, y = data
    X = np.asarray(X)
    y = np.asarray(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    n_samples, n_features = X_train.shape

    # Step 1: Add fictitious features
    X_fict = eps * np.eye(n_samples)[:, :n_fictitious]
    X_aug = np.hstack([X_train, X_fict])

    # Step 2: Regularization
    alphas = np.array([alpha_real]*n_features + [alpha_fictitious]*n_fictitious)
    reg_matrix = np.diag(alphas)

    # Step 3: Solve augmented system
    w_aug = np.linalg.inv(X_aug.T @ X_aug + reg_matrix) @ (X_aug.T @ y_train)

    # Step 4: Original coefficients
    w_original = w_aug[:n_features]

    # Step 5: Predictions
    y_train_pred = X_train @ w_original
    y_test_pred = X_test @ w_original

    # Step 6: Metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print("Stabilized coefficients:", w_original)
    print(f"R² train: {r2_train:.5f}, R² test: {r2_test:.5f}, MSE test: {mse_test:.5f}")

    return {
        "coefficients": w_original,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mse_test": mse_test
    }


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 100
    X = np.random.rand(n_samples, 3)
    true_coef = np.array([2.0, -1.5, 0.5])
    y = X @ true_coef + np.random.normal(0, 0.1, size=n_samples)

    results = srfa((X, y))
