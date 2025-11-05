from Bayesian_regression_main import GET_CLEAN_DATA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans

galaxy_log10_mass, filtered_u_g_color, filtered_velocity_dispersion = GET_CLEAN_DATA()

data_to_analyse = np.vstack([galaxy_log10_mass,filtered_u_g_color,filtered_velocity_dispersion]).T
print(data_to_analyse)



# implement K-Means clustering to estimate the number of galaxy cluster in your data
# number_of_components = np.arange(1,30)
# kmeans_models = [KMeans(n_clusters=values, random_state=42).fit() for values in number_of_components]
# print(kmeans_models)

class cluster_analysis(object):

    def __init__(self, data):
        self.data = data

    def kmeans_cluster_analysis(self) -> None:
        # if data.shape() No need to write this since it's much better to fit all columns simultaneously.
        data_to_analyse = self.data

        number_of_components_for_analysis = np.arange(2,30) # I start at 2 becuase silhouette analysis requies at least 2 clusters and I want to plot both results. This ensures the results have equal dimensions for plotting
        kmeans_models = [KMeans(n_clusters=values, random_state=42).fit(data_to_analyse) for values in number_of_components_for_analysis]
        # print(kmeans_models)
        inertia_ = [model.inertia_ for model in kmeans_models]

        # remember using inertia relies on visually recording the elbow point
        # Now, implement silhouette score
        silhouette_scores = [] #I find it more readable to write the code for silhoutte scores in this manner so I can track every detail especially the confusing "labels that needs to be used here" 

        for values in number_of_components_for_analysis:
            kmeans_model = KMeans(n_clusters=values, random_state=42).fit(data_to_analyse)
            label = kmeans_model.labels_
            score = silhouette_score(data_to_analyse,label)
            silhouette_scores.append(score)
        
        figure, ax_main = plt.subplots(2,1, figsize=(7.5,9), sharex=True)

        ax_main[0].plot(number_of_components_for_analysis, inertia_, "ro--")
        ax_main[0].set_ylabel(" Inertia")
        ax_main[0].set_title(" Inertia Method for KMeans Model Selection")

        ax_main[1].plot(number_of_components_for_analysis, silhouette_scores, "ro--")
        ax_main[1].set_ylabel(" Silhouette Scores")
        ax_main[1].set_title(" Inertia Method for KMeans Model Selection")
        
        ax_main[1].set_xlabel(" Number of clusters")
        plt.show()

    def gmm_analysis(self):

        data_to_analyse = self.data

        gmm_model = GMM(n_components=7, random_state=42).fit(data_to_analyse)
        labels_for_color = gmm_model.predict(data_to_analyse)

        figure, ax_main = plt.subplots(figsize=(7.5,9))
        # figure = plt.figure(figsize=(9.5,10)) # For the 3D plot including the filtered velocity dispersion
        # ax_main = figure.add_subplot(111, projection="3d")

        ax_main.scatter(data_to_analyse[:,0], data_to_analyse[:,1], data_to_analyse[:,2], c=labels_for_color )
        ax_main.set_ylabel(" Filtered UG Color")
        ax_main.set_xlabel(" log10(Mass of Galaxy) ")
        ax_main.set_zlabel(" Filtered Velocity Dispersion")
        plt.show()



if __name__ == "__main__":

    cluster_test = cluster_analysis(data_to_analyse)
    cluster_test.kmeans_cluster_analysis()
    cluster_test.gmm_analysis()



