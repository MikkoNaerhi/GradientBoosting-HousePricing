from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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

def perform_hyperparameter_optimization(
    model:GradientBoostingRegressor,
    X_train:pd.DataFrame,
    y_train:pd.Series,
    param_grid:dict,
    cv_num:int=5
) -> GradientBoostingRegressor:
    """ Perform hyperparameter optimization for GradientBoostingRegressor using GridSearchCV.

    Parameters:
    -----------
        model: GradientBoostingRegressor with default hyperparameters.
        X_train: The training feature dataset.
        y_train: The target values for the training dataset.
        param_grid : A dictionary where keys are hyperparameters and values are lists of parameter settings to try as values.
        cv_num : The number of folds to use for cross-validation.

    Returns:
    --------
        GradientBoostingRegressor: The GradientBoostingRegressor model fitted with the best set of hyperparameters found.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_num, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    return GradientBoostingRegressor(**best_params)

def main():
    """ Main function to run the analysis on the California housing dataset.
    """
    # Number of folds in cross-validation
    cv_num = 5

    plot_corr_matrix = True
    opt_hyperparams = False

    # Define the grid of hyperparameters to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6]
    }

    X,y = load_data()

    # Split data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Calculate and plot the correlation matrix
    if plot_corr_matrix:
        calc_and_plot_corr_mat(X=X)

    # Drop column 'AveRooms' due to high correlation with column 'AveBedrms'
    X.drop(columns='AveRooms', inplace=True)

    # Initiate the model with default params
    model = GradientBoostingRegressor()

    if opt_hyperparams:
        best_model = perform_hyperparameter_optimization(
            model=model,
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid,
            cv_num=cv_num
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred)
        print("Test MSE with optimized hyperparameters:", loss)
    else:
        cv_scores = cross_val_score(estimator=model, X=X_train, y=y_train, scoring='neg_mean_squared_error', cv=cv_num, n_jobs=-1)
        cv_scores = -cv_scores
        print("Cross-validated scores:", cv_scores)
        print("Test MSE with default hyperparameters:", cv_scores.mean())

if __name__=='__main__':
    main()