from Bayesian_regression_main import GET_CLEAN_DATA
from astroML.linear_model import PolynomialRegression
import matplotlib.pyplot as plt
import numpy as np

galaxy_log10_mass, ug_color, velocity_dispersion = GET_CLEAN_DATA()

# write a code for cross-validation only later!!!! This is not a k-fold cross validation
# include training and cross validation errors!!! Remember to think about the fold size, the indices, and of course the concatenation of the training data for this approach
# Actually, do the k-fold cross validation tomorrow!!!

class cross_validation(object):

    def __init__(self, feature_data: list[float,...], prediction_data: list[float, ...], model_degree: int, string_name: str = None) -> None:
        self.feature_data = feature_data
        self.prediction_data = prediction_data
        self.model_degree = model_degree
        self.string_name = string_name

    
    def cross_validation(self):

        length_of_feature_data = len(self.feature_data)
        rng = np.random.default_rng(42)
        permuted_indices = rng.permutation(length_of_feature_data)
        
        permuted_feature_data = self.feature_data[permuted_indices]
        permuted_prediction_data = self.prediction_data[permuted_indices]

        array_for_train_size = np.linspace(0.1, 1, 10) # Includes the last values "1"
        train_sizes = (length_of_feature_data * array_for_train_size).astype(int)

        train_rmse_error = []
        cv_rmse_error = []

        for i, train_size in enumerate(train_sizes):

            X_train = (permuted_feature_data[: train_size])[:, None]
            X_test = (permuted_feature_data[train_size :])[:, None]
            y_train = permuted_prediction_data[: train_size]
            y_test = permuted_prediction_data[train_size :]

            if len(X_test) < 2: # this is a very important step in this code!!!
                print(f' The cross validation data is too small for the {i}th train. Skipping... ')
                continue

            model = PolynomialRegression(degree=self.model_degree).fit(X_train, y_train)

            # code for cross_validation plus the cv_error
            cross_validation_pred = model.predict(X_test)
            cv_rmse = np.sqrt(np.mean((y_test - cross_validation_pred)**2)) # note the difference here
            cv_rmse_error.append(cv_rmse)

            # code for training evaluation plus the training_error
            training_pred = model.predict(X_train)
            training_rmse = np.sqrt(np.mean((y_train - training_pred)**2)) # note the difference here
            train_rmse_error.append(training_rmse)
        
        # calculate/monitor the mean and the spread of the training error to evaluate the models
        mean_train_rmse_of_models = np.mean(train_rmse_error)
        mean_cv_rmse_of_models = np.mean(cv_rmse_error)

        return train_sizes[:len(train_rmse_error)], train_rmse_error, cv_rmse_error, mean_train_rmse_of_models, mean_cv_rmse_of_models

    def plot_results(self):

        data_to_train = self.feature_data[:, None]
        prediction_data = self.prediction_data

        model = PolynomialRegression(degree=self.model_degree).fit(data_to_train, prediction_data) # best fitting model

        figure, ax_main = plt.subplots(figsize=(8.5, 8))

        mass_range = np.linspace(min(self.feature_data), max(self.feature_data), 100)
        ug_color_pred = model.predict(mass_range[:, None])
        colors = model.predict(data_to_train)

        
        # plot the results
        scatter = ax_main.scatter(self.feature_data, self.prediction_data, c=colors, cmap='viridis') # of course! Why not???

        ax_main.plot(mass_range, ug_color_pred, color="black", linestyle="--", label=f' {self.string_name} fit', linewidth=3.5)
        ax_main.set_xlabel(" log10(Mass of Galaxy)")
        ax_main.set_ylabel(" UG Color")
        ax_main.set_title(f' {self.string_name} Regressive fit')
        ax_main.legend(loc="best")

        colorbar = plt.colorbar(scatter, label="log10(Mass of Galaxy)", orientation="vertical")
        colorbar.remove() # removes the side color panel but leaves the coloring on the datapoints! Awesome!
        ax_main.grid(True)

        figure.tight_layout()
        plt.show()
        


if __name__ == "__main__":

    linear_model = cross_validation(galaxy_log10_mass, ug_color, 1, string_name = "Linear")
    quadratic_model = cross_validation(galaxy_log10_mass, ug_color, 2, string_name = "Quadratic")
    cubic_model = cross_validation(galaxy_log10_mass, ug_color, 3, string_name = "Cubic")
    quartic_model = cross_validation(galaxy_log10_mass, ug_color, 4, string_name = "Quartic")
    quintic_model = cross_validation(galaxy_log10_mass, ug_color, 5, string_name = "Quintic")
    sextic_or_hexic_model = cross_validation(galaxy_log10_mass, ug_color, 5, string_name = "Sextic/Hexic")



    # code for learning curve curve plots
    model_names = ['Linear', "Quadratic", "Cubic", "Quartic", "Quintic", "Sextic/Hexic"]
    model_container = [linear_model, quadratic_model, cubic_model, quartic_model, quintic_model, sextic_or_hexic_model]

    figure, ax_main = plt.subplots(len(model_container),1, figsize=(10,7.5), sharex=True)

    for i, model in enumerate(model_container):

        print(f' Plotting model {i+1} ')
        train_sizes, train_rmse_error, cv_rmse_error, mean_train_rmse_of_models, mean_cv_rmse_of_models = model.cross_validation()

        ax_main[i].plot(train_sizes, train_rmse_error, color='green', linestyle="-", label="training error")
        ax_main[i].plot(train_sizes, cv_rmse_error, color="red", linestyle="--", label="cv error")
        ax_main[i].set_ylabel(" RMSE ")
        ax_main[i].legend(loc="best")
        # add grid if needed
    
    ax_main[0].set_title(f' Learning Curve ')
    ax_main[-1].set_xlabel(" Training set sizes")
    figure.tight_layout()
    plt.show()

    for models in model_container:
        models.plot_results()

    



    


