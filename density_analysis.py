from Bayesian_regression_main import GET_CLEAN_DATA
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from astropy.visualization import hist
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import train_test_split # might need to split data into training and testing data
# from sklearn.model_selection import StandardScaler # might need to standardize the data
# from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture


galaxy_log10_mass, u_g_color, velocity_dispersion = GET_CLEAN_DATA()


def probability_density_estimation_with_kde(data: list[float,...], data_name: str) -> None:

    data_to_study = data[:, None]

    bandwidth_array = np.logspace(-2,0,20) # search for a better bandwidth space

    grid_search = GridSearchCV(KernelDensity(), {"bandwidth": bandwidth_array}, cv=5, n_jobs=-2) # use all but one CPU Bad for debugging!
    grid_search.fit(data_to_study) # fit our data to the model

    optimal_bandwidth = grid_search.best_params_["bandwidth"]
    best_score = grid_search.best_score_
    print(f' The best bandwidth for the KDE of {data_name} is {optimal_bandwidth}.')
    print(f' The best cross-validation score(log-likelihood) is {best_score}.')

    kde_model = grid_search.best_estimator_ # this is the probability density function estimated by the GridSearchCV that best fits the data

    # we now create an array of data we would like to estimate their probabilities with our model!
    data_to_estimate_their_density = np.linspace(min(data), max(data), 500)
    transformed_version_of_data_to_estimate_their_density = data_to_estimate_their_density[:, None] # this transforms it into a form acceptable to scikit learn

    log_probabilities_of_data = kde_model.score_samples(transformed_version_of_data_to_estimate_their_density)
    # it is important we convert to log_probabilities first because these values are quite small and are not well stored by computers
    # However, by taking the logs of these values, we get a much smaller value which can be more efficiently stored by computers.
    # note that their is a big difference between "score_samples" and "score_sample"
    probabilities = np.exp(log_probabilities_of_data)

    # now, we add plots of the histogram and the probability_estimations

    figure, ax_main = plt.subplots(2, 1, figsize=(10,9.1), sharex= True)

    ax_main[0].hist(data, bins="scott", density=True, histtype="stepfilled", label=f"Histogram of {data_name}")
    ax_main[0].legend(loc='best')

    ax_main[1].plot(data_to_estimate_their_density, probabilities, label=f"KDE for {data_name}")
    ax_main[1].fill_between(data_to_estimate_their_density, probabilities, alpha=0.5, color='red')
    ax_main[1].set_xlabel("Data") # this is sharedf by the histogram and the plot
    ax_main[1].set_ylabel("Probability")
    ax_main[1].legend(loc="best")

    figure.tight_layout()
    plt.show()
    # I could also decide to write the GridSearchCV code differently from this. That is, I could decide to keep the KDE and the GridSearchCV functions separate. But, that would be quite inefficient.



# next thing is to test/compare different bandwidths.
# write this down here!!!






# This code is highly recommended !!!

# this is with the Gaussian Mixture Model !!!
# # LINE 9: Print what we discovered about each group
# print(f"\n DISCOVERED GALAXY GROUPS:")
# for i in range(best_n):
#     weight = best_gmm.weights_[i]  # Fraction of galaxies in this group
#     mean = best_gmm.means_[i, 0]   # Average mass of this group
#     std = np.sqrt(best_gmm.covariances_[i, 0, 0])  # Spread of masses
    
#     print(f"Group {i+1}: {weight:.1%} of galaxies, " +
#             f"mean mass = {mean:.2f}, spread = {std:.2f}")


if __name__ == "__main__":
    first_data_to_estimate_pdf_for = galaxy_log10_mass[:]
    second_data_to_estimate_pdf_for = u_g_color[:]
    third_data_to_estimate_pdf_for = velocity_dispersion[:]

    # visualizations
    probability_density_estimation_with_kde(first_data_to_estimate_pdf_for,"log_mass_of_galaxies")
    probability_density_estimation_with_kde(second_data_to_estimate_pdf_for,"UG colors of Galaxies")
    probability_density_estimation_with_kde(third_data_to_estimate_pdf_for, "Velocity Dispersion of Galaxies")









### Implement kde, knn(best for classication) and GMM in your codes!!!


















# def demonstrate_nearest_neighbor_density():

#     # Create sample data - some crowded areas, some empty
#     np.random.seed(42)
#     crowded_data = np.random.normal(2, 0.3, 100)      # Dense cluster
#     sparse_data = np.random.normal(6, 0.8, 30)        # Sparse cluster
#     data = np.concatenate([crowded_data, sparse_data])
    
#     # Reshape for scikit-learn
#     data_2d = data.reshape(-1, 1)
    
#     # Choose k (number of neighbors)
#     k = 5
    
#     # Fit nearest neighbors model
#     nbrs = NearestNeighbors(n_neighbors=k + 1)  # +1 because point is its own neighbor
#     nbrs.fit(data_2d)
    
#     # Create points where we want to estimate density
#     x_range = np.linspace(0, 8, 200).reshape(-1, 1)
    
#     # For each point, find distance to k-th nearest neighbor
#     distances, indices = nbrs.kneighbors(x_range)
#     kth_distances = distances[:, k]  # Distance to k-th neighbor (0-indexed)
    
#     # Calculate density: k / (N * volume)
#     # For 1D: volume = 2 * distance (radius to diameter)
#     densities = k/(len(data) * 2 * kth_distances)
    
#     # Plotting
#     plt.figure(figsize=(15, 10))
    
#     # Plot 1: Show the data points
#     plt.subplot(2, 2, 1)
#     plt.scatter(data, np.zeros_like(data), alpha=0.6, s=50, label='Data points')
#     plt.title('OUR DATA: Some crowded areas, some empty areas')
#     plt.xlabel('Value')
#     plt.ylabel('(Just showing data layout)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Plot 2: Show k-nearest neighbor distances
#     plt.subplot(2, 2, 2)
#     plt.plot(x_range, kth_distances, 'green', linewidth=2)
#     plt.title(f'DISTANCE to {k}-TH NEAREST NEIGHBOR')
#     plt.xlabel('Position (x)')
#     plt.ylabel(f'Distance to {k}-th neighbor')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 3: Show the calculated density
#     plt.subplot(2, 2, 3)
#     plt.plot(x_range, densities, 'red', linewidth=2)
#     plt.title(f'K-NN DENSITY ESTIMATE (k={k})')
#     plt.xlabel('Position (x)')
#     plt.ylabel('Density')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 4: Compare with histogram
#     plt.subplot(2, 2, 4)
#     plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram')
#     plt.plot(x_range, densities, 'red', linewidth=2, label=f'K-NN Density (k={k})')
#     plt.title('COMPARISON: K-NN vs Histogram')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     return data, x_range, densities

# Run the demonstration


