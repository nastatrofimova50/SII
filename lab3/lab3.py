import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


def init_data():
    combined_cycle_power_plant = fetch_ucirepo(id=294)
    print(combined_cycle_power_plant.variables)
    return combined_cycle_power_plant.data.features, combined_cycle_power_plant.data.targets


def separator(x, y):
    np.random.seed(42)
    selection = np.random.permutation(len(x))
    size = int(len(x) * 0.2)
    testing_selection = selection[:size]
    training_selection = selection[size:]
    x_training, x_testing = x.iloc[training_selection], x.iloc[testing_selection]
    y_training, y_testing = y.iloc[training_selection], y.iloc[testing_selection]

    return x_training, x_testing, y_training, y_testing


def main():
    x, y = init_data()
    x_training, x_testing, y_training, y_testing = separator(x, y)

    regressor = LinearRegression().fit(x_training, y_training)

    y_pred = regressor.predict(x_testing)
    print(f"Mean squared error: {mean_squared_error(y_testing, y_pred):.2f}")
    print(f"Coefficient of determination: {r2_score(y_testing, y_pred):.2f}")

    training_errors = []
    testing_errors = []
    degrees = range(1, 5)

    for degree in degrees:
        print(f'poly {degree}')

        polynom = PolynomialFeatures(degree=degree)
        x_training_poly = polynom.fit_transform(x_training)
        x_testing_poly = polynom.transform(x_testing)

        model = LinearRegression()
        model.fit(x_training_poly, y_training)

        y_training_pred = model.predict(x_training_poly)
        y_testing_pred = model.predict(x_testing_poly)

        training_errors.append(mean_squared_error(y_training, y_training_pred))
        testing_errors.append(mean_squared_error(y_testing, y_testing_pred))

    print('Polynomial Features graphics')

    plt.figure()
    plt.plot(degrees, training_errors, label='Training error')
    plt.plot(degrees, testing_errors, label='Testing error')
    plt.xlabel('Degree of polynomial')
    plt.ylabel('Mean Squared Error')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()

    del training_errors, testing_errors, degrees

    # Регуляризация
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

    training_errors = []
    testing_errors = []

    for alpha in alphas:
        print(f'alpha = {alpha}')
        regular = Ridge(alpha=alpha)
        regular.fit(x_training, y_training)

        y_training_pred = regular.predict(x_training)
        y_testing_pred = regular.predict(x_testing)

        training_errors.append(mean_squared_error(y_training, y_training_pred))
        testing_errors.append(mean_squared_error(y_testing, y_testing_pred))

    print('Graphics')
    plt.figure()
    plt.plot(np.log10(alphas), training_errors, label='Training error')
    plt.plot(np.log10(alphas), testing_errors, label='Testing error')
    plt.xlabel('log(alpha)')
    plt.ylabel('Mean Squared Error')
    plt.title('Regression')
    plt.legend()
    plt.show()


main()