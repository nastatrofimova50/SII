import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def init_data():
    spambase = fetch_ucirepo(id=94)
    print(spambase.variables)
    return spambase.data.features, spambase.data.targets.squeeze()


def separator(X, y, size):
    x_first, x_second, y_first, y_second = train_test_split(X, y, test_size=size, random_state=42)

    return x_first, x_second, y_first, y_second

def main():

    X, y = init_data()

    X_train, X_temp, y_train, y_temp = separator(X, y, 0.3)
    X_val, X_test, y_val, y_test = separator(X_temp, y_temp, 0.5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    perceptron = Perceptron(tol=1e-3, random_state=0)
    perceptron.fit(X_train_scaled, y_train)
    y_pred_perceptron = perceptron.predict(X_test_scaled)

    mlp = MLPClassifier(random_state=1, solver='lbfgs', learning_rate_init=0.001, early_stopping=True,
                        hidden_layer_sizes=(20, 20), max_iter=2000, tol=1e-5)
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)

    perceptron_accuracy = accuracy_score(y_test, y_pred_perceptron)
    print(f"Accuracy Perceptron on tests data: {perceptron_accuracy:.4f}")

    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
    print(f"Accuracy MLPClassifier on tests data: {mlp_accuracy:.4f}")

    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    regularization = [0.0001, 0.001, 0.01, 0.1]
    optimizers = ['lbfgs', 'sgd', 'adam']

    results = []

    for lr in learning_rates:
        for reg in regularization:
            for opt in optimizers:
                mlp = MLPClassifier(hidden_layer_sizes=(50, 50), solver=opt, alpha=reg, learning_rate_init=lr,
                                    max_iter=2000, random_state=1, early_stopping=True, tol=1e-5)
                mlp.fit(X_train_scaled, y_train)
                y_val_pred = mlp.predict(X_val_scaled)
                acc = accuracy_score(y_val, y_val_pred)
                results.append((lr, reg, opt, acc))

    results_df = pd.DataFrame(results, columns=['Learning Rate', 'Regularization', 'Optimizer', 'Accuracy'])

    plt.figure(figsize=(12, 6))
    for opt in optimizers:
        subset = results_df[results_df['Optimizer'] == opt]
        plt.plot(subset['Learning Rate'], subset['Accuracy'], marker='o', label=f'Optimizer: {opt}')

    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Learning Rate for Different Optimizers')
    plt.legend()
    plt.grid()
    plt.show()

    best_params = results_df.loc[results_df['Accuracy'].idxmax()]
    print("Best parameters:")
    print(best_params)

    best_mlp = MLPClassifier(hidden_layer_sizes=(50, 50), solver=best_params['Optimizer'],
                             alpha=best_params['Regularization'], learning_rate_init=best_params['Learning Rate'],
                             max_iter=500, random_state=1)
    best_mlp.fit(X_train_scaled, y_train)
    y_test_pred = best_mlp.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy on tests data with best parameters: {final_accuracy:.4f}")

main()