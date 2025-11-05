from Bayesian_regression_main import GET_CLEAN_DATA
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist as mat_hist # this is my attempt to get bins="freedman" to work since as it seemed the python interpretter was having trouble differentiating matplotlib's hist and the astropy.visualization.hist()
from astropy.visualization import hist
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import train_test_split # might need to split data into training and testing data
# from sklearn.model_selection import StandardScaler # might need to standardize the data
# from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture


galaxy_log10_mass, u_g_color, velocity_dispersion = GET_CLEAN_DATA()



class probability_density_estimation(object):

    def __init__(self, data: list[float,...]) -> None:
        self.data = data


    def probability_density_estimation_with_kde(self, data_name: str) -> None:

        data_to_study = self.data[:, None]

        bandwidth_array = np.logspace(-2,0,20) # search for a better bandwidth space

        grid_search = GridSearchCV(KernelDensity(), {"bandwidth": bandwidth_array}, cv=5, n_jobs=-2) # use all but one CPU Bad for debugging!
        grid_search.fit(data_to_study) # fit our data to the model

        optimal_bandwidth = grid_search.best_params_["bandwidth"]
        best_score = grid_search.best_score_ # cross validation score(log likelihood)

        kde_model = grid_search.best_estimator_ # this is the probability density function estimated by the GridSearchCV that best fits the data

        # we now create an array of data we would like to estimate their probabilities with our model!
        data_to_estimate_their_density = np.linspace(min(self.data), max(self.data), 500)
        transformed_version_of_data_to_estimate_their_density = data_to_estimate_their_density[:, None] # this transforms it into a form acceptable to scikit learn

        log_probabilities_of_data = kde_model.score_samples(transformed_version_of_data_to_estimate_their_density)
        # it is important we convert to log_probabilities first because these values are quite small and are not well stored by computers
        # However, by taking the logs of these values, we get a much smaller value which can be more efficiently stored by computers.
        # note that their is a big difference between "score_samples" and "score_sample"
        probabilities = np.exp(log_probabilities_of_data)

        # now, we add plots of the histogram and the probability_estimations

        figure, ax_main = plt.subplots(2, 1, figsize=(10,9.1), sharex= True)

        ax_main[0].hist(self.data, bins="scott", density=True, histtype="stepfilled", label=f"Histogram for {data_name}", color="black")
        ax_main[0].legend(loc='best')

        ax_main[1].plot(data_to_estimate_their_density, probabilities, label=f"KDE for {data_name}, bandwidth = {optimal_bandwidth:.4}")
        ax_main[1].fill_between(data_to_estimate_their_density, probabilities, alpha=0.5, color='black')
        ax_main[1].set_xlabel(f"{data_name}") # this is sharedf by the histogram and the plot
        ax_main[1].set_ylabel("Probability/Density")
        ax_main[1].legend(loc="best")

        figure.tight_layout()
        plt.show()
        # I could also decide to write the GridSearchCV code differently from this. That is, I could decide to keep the KDE and the GridSearchCV functions separate. But, that would be quite inefficient.

        print(f' \n =========KDE Results for {data_name}=========')
        print(f' The best bandwidth for the KDE of {data_name} is {optimal_bandwidth:.4}.')
        print(f' The best cross-validation score(log-likelihood) is {best_score}.')
        print(f' =========End of results for {data_name}========= \n')


    # next thing is to test/compare different bandwidths.
    # write this down here!!!


    def probability_density_estimation_with_gmm(self, data_name: str) -> None:

        data_to_study = self.data[:, None]

        number_of_gaussian_components = np.arange(1, 20) # better preffered to np.linspace() for this particular function  # np.linspace() is preffered when we care about the inclusion of the endpoint

        gaussian_models = [GaussianMixture(components, random_state=42).fit(data_to_study) for components in number_of_gaussian_components] # this is called a python list comprehension

        # to choose the best model, we need to evaluate the BIC and choose the model with the lowest BIC value since the lower the better
        gaussian_model_bics = [model_i.bic(data_to_study) for model_i in gaussian_models]

        # now, we find the index of the model with the least BIC value
        index_of_optimal_gaussian_mixture_model = np.argmin(gaussian_model_bics)
        optimal_number_of_gaussian_components = number_of_gaussian_components[index_of_optimal_gaussian_mixture_model] 
        optimal_gaussian_mixture_model = gaussian_models[index_of_optimal_gaussian_mixture_model]
        bic_for_optimal_gaussian_mixture_model = gaussian_model_bics[index_of_optimal_gaussian_mixture_model]

        # Now, we create an array of data to estimate their densities(probabilities)
        data_to_estimate_their_density = np.linspace(min(self.data), max(self.data), 500)
        transformed_data_to_estimate_their_density = data_to_estimate_their_density[:, None]

        log_probabilities = optimal_gaussian_mixture_model.score_samples(transformed_data_to_estimate_their_density)
        probabilities = np.exp(log_probabilities)

        # now, we have happily(not exactly) reached best part of the this stuff
        # add the code for the visualization ot the individual pdfs here
        probability_of_data_points_belonging_to_any_gaussian_component = optimal_gaussian_mixture_model.predict_proba(transformed_data_to_estimate_their_density)
        pdf_for_indvidual_gaussian_component = probability_of_data_points_belonging_to_any_gaussian_component * probabilities[:, None] # now, you wanna plot this

        figure, ax_main = plt.subplots(2,1, figsize=(10,9.1), sharex=True)

        ax_main[0].hist(self.data, bins="scott", histtype="stepfilled", label=f"Histogram for {data_name}", color="black")
        ax_main[0].legend(loc="best")

        ax_main[1].plot(data_to_estimate_their_density, probabilities, label=f"GMM estimate for {data_name}, {optimal_number_of_gaussian_components} components")
        ax_main[1].plot(data_to_estimate_their_density,pdf_for_indvidual_gaussian_component, linestyle="--", color="green")
        ax_main[1].fill_between(data_to_estimate_their_density, probabilities, color="black")
        ax_main[1].set_xlabel(f"{data_name}") # this is the shared axis
        ax_main[1].set_ylabel('Probability/Density')
        ax_main[1].legend(loc="best")

        figure.tight_layout()
        plt.show()

        print(f" \n =========GMM Results for {data_name}=========")
        print(f' The optimal number of components is {optimal_number_of_gaussian_components}.')
        print(f' The BIC for this Gaussian Mixture Model is {bic_for_optimal_gaussian_mixture_model:.6}.')
        print(f' =========End of results for {data_name}========= \n')

        # print the discoveries about the Gaussian components

        for values in range(optimal_number_of_gaussian_components):
            weight = optimal_gaussian_mixture_model.weights_[values] # Fraction of Galaxies in this components
            mean = optimal_gaussian_mixture_model.means_[values,0]
            std = optimal_gaussian_mixture_model.covariances_[values,0,0]

            if values == 0:
                print(f"=========Discoveries from the Gaussian Components for {data_name}=========")
                
            print(f' Group {values + 1} contains {weight:.4%} of galaxies, has mean {mean:.4}, and standard deviation {std:.4} ')

            if values == optimal_number_of_gaussian_components-1:
                print("=========En of results=========")


#  Check whether the number of components for UG-Color and log_mass are equivalent.

if __name__ == "__main__":
    first_data_to_estimate_pdf_for = galaxy_log10_mass[:]
    second_data_to_estimate_pdf_for = u_g_color[:]
    third_data_to_estimate_pdf_for = velocity_dispersion[:]

    # initialization of the pdf-estimator objects
    pdf_estimator_for_log_mass = probability_density_estimation(first_data_to_estimate_pdf_for)
    pdf_estimator_for_log_mass.probability_density_estimation_with_kde("log(Mass of Galaxies)")
    pdf_estimator_for_log_mass.probability_density_estimation_with_gmm("log(Mass of Galaxies)")

    pdf_estimator_for_ug_color = probability_density_estimation(second_data_to_estimate_pdf_for)
    pdf_estimator_for_ug_color.probability_density_estimation_with_kde("UG colors of Galaxies")
    pdf_estimator_for_ug_color.probability_density_estimation_with_gmm("UG colors of Galaxies")

    pdf_estimator_for_velocity_dispersion = probability_density_estimation(third_data_to_estimate_pdf_for)
    pdf_estimator_for_velocity_dispersion.probability_density_estimation_with_kde("Velocity Dispersion of Galaxies")
    pdf_estimator_for_velocity_dispersion.probability_density_estimation_with_gmm("Velocity Dispersion of Galaxies")



## Implement the recent LINEAR REGRESSION as well because you need to know how to do these things!
## Apply for the Germany thing!!!