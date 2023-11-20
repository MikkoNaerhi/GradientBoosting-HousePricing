from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Load the California housing dataset.

    Returns:
    --------
        tuple: A tuple containing the feature matrix (X) and target vector (y).
    """
    data = fetch_california_housing(as_frame=True)
    X = data['data']
    y = data['target']
    
    return (X,y)


def calc_and_plot_corr_mat(X:pd.DataFrame) -> None:
    """ Calculate and plot the correlation matrix of a DataFrame.

    Parameters:
    -----------
        X (pd.DataFrame): The DataFrame for which the correlation matrix is calculated.
    """
    x_cov = X.corr()
    sns.heatmap(data=x_cov, annot=True)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def main():
    """ Main function to run the analysis on the California housing dataset.
    """
    X,y = load_data()

    # Split data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Calculate and plot the correlation matrix
    calc_and_plot_corr_mat(X=X)

    # Drop column 'AveRooms' due to high correlation with column 'AveBedrms'
    X.drop(columns='AveRooms', inplace=True)

    # Initiate the model
    model = GradientBoostingRegressor()

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_scores = -cv_scores  # Convert to positive values
    print("Cross-validated scores:", cv_scores)
    print("Mean MSE:", cv_scores.mean())

    # Fit and evaluate the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    print("Test MSE:", loss)


if __name__=='__main__':
    main()