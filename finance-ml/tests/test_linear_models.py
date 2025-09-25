import numpy as np
import pytest
from my_package.linear_models import LinearRegression

def test_fit_predict_with_intercept():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression(use_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Vérifie que les prédictions sont proches des vraies valeurs
    assert np.allclose(y, y_pred, atol=1e-8)
    # Vérifie les coefficients
    assert np.isclose(model.intercept_, 1.0, atol=1e-8)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-8)


def test_fit_predict_without_intercept():
    # Données : y = 3x (pas d'intercept)
    X = np.array([[1], [2], [3]])
    y = np.array([3, 6, 9])

    model = LinearRegression(use_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert np.allclose(y, y_pred, atol=1e-8)
    assert model.intercept_ == 0.0
    assert np.isclose(model.coef_[0], 3.0, atol=1e-8)


def test_predict_new_data():
    # y = 2x + 5
    X = np.array([[1], [2], [3]])
    y = np.array([7, 9, 11])

    model = LinearRegression(use_intercept=True)
    model.fit(X, y)

    # Nouvelle donnée
    X_new = np.array([[10]])
    y_pred = model.predict(X_new)

    assert np.isclose(y_pred[0], 25.0, atol=1e-8)


