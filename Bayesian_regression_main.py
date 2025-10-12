import numpy as np
from astropy.table import Table
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

galaxy_photometric_url_path: str = "Skyserver_spectro.csv"

# galaxy_photometric_data: list[int] = Table.read(galaxy_photometric_url_path)

# we try reading with different options
try:
    # we let Astropy auto-detect the format
    galaxy_photometric_data: list[float] = Table.read(galaxy_photometric_url_path, format='ascii')
    
except:
    try:
        # or we xplicitly specify CSV format
        galaxy_photometric_data: list[float] = Table.read(galaxy_photometric_url_path, format='csv')
        
    except:
        # or we use pandas to read and convert to Astropy Table
        import pandas as pd
        df: list[float] = pd.read_csv(galaxy_photometric_url_path)
        galaxy_photometric_data: list[float] = Table.from_pandas(df)

# We check that what we got is correct
# print(f"Successfully loaded table with {len(galaxy_photometric_data)} rows")
# print("Columns:", galaxy_photometric_data.columns)

galaxy_u_mag: list[float] = galaxy_photometric_data['modelMag_u']
galaxy_g_mag: list[float] = galaxy_photometric_data['modelMag_g'] # Note that g_mag is very close to v_mag. So, we use it here!
galaxy_velocity_dispersion: list[float] = galaxy_photometric_data['velDisp']
galaxy_distance_info: list[float] = galaxy_photometric_data['redshift']

galaxy_u_g_color: list[float] = galaxy_u_mag - galaxy_g_mag

# We use the relation between mass and velocity dispersion provided by the virial theorem \n
# to estimate the mass of the galaxies
# We're assuming that the velocity dispersion is always positive since we'll be taking the log of the numbers
velocity_dispersion_mask: list[bool] = ~np.isnan(galaxy_velocity_dispersion) & (galaxy_velocity_dispersion > 0)
filtered_u_g_color: list[float] = galaxy_u_g_color[velocity_dispersion_mask]
filtered_velocity_dispersion: list[float] = galaxy_velocity_dispersion[velocity_dispersion_mask]
# print(filtered_velocity_dispersion)
galaxy_log10_mass: list[float] =  2*np.log10(filtered_velocity_dispersion) - np.log10(6.673) + 11
# print(galaxy_log10_mass)

# if 'modelMagErr_u' not in galaxy_photometric_data.columns:
#     print(' Sorry, not found.')
#     print(galaxy_photometric_data.columns)

# fig = plt.figure(figsize=(10,8.5))
# ax = fig.add_subplot(1,1,1)

# ax.hist(galaxy_g_mag, bins= int(np.floor(0.5*np.sqrt(len(galaxy_g_mag)))), color='red')
# plt.show()

class maximum_likelihood_estimation_object(object):

    def __init__(self, initial_guess: list[float], colors: list[float], mass: list[float]) -> None:
        self.initial_guess = initial_guess
        self.colors = colors
        self.mass = mass

    def log_likelihood_estimation_function(self, parameters):
        slope, intercept, sigma_int = parameters
        predicted_fit = slope*self.mass + intercept

        return -np.sum(norm.logpdf(self.colors,predicted_fit,sigma_int))

    def maximum_likelihood_estimation(self) -> list[float]:

        maximized_paremeters = minimize(self.log_likelihood_estimation_function, self.initial_guess, method="Nelder-Mead")

        slope, intercept, sigma_int = maximized_paremeters.x

        return slope,intercept

    def frequentist_line_fit(self):

        slope, intercept = self.maximum_likelihood_estimation()

        # x_fit = self.colors
        # y_fit = slope * self.mass + intercept
        mass_range = np.linspace(np.min(self.mass), np.max(self.mass), 500)
        color_fit = intercept + slope * mass_range

        figure = plt.figure(figsize=(10,8.7))
        ax_main = figure.add_subplot(1,1,1)

        # Plot data points
        ax_main.scatter(self.mass, self.colors, alpha=0.5, s=10, label='Galaxies')
        
        # Plot fitted line
        ax_main.plot(mass_range, color_fit, 'r-', linewidth=2,label=f'Fit: color = {intercept:.2f} + {slope:.2f}*mass')

        # ax_main.plot(x_fit, y_fit, ls='--', color='red', label=' U-V color against log10(Mass of Galaxy)')
        # ax_main.set_xlim()
        # ax_main.set_ylim()
        ax_main.set_xlabel('Log10(M_galaxy)')
        ax_main.set_ylabel('(U-V) color')
        ax_main.set_title('Frequentist Fitting of Line to Data')
        ax_main.legend(loc='best')
        
        return figure, ax_main


if __name__ == '__main__':
    initial_guess = [0.5,1.0,0.15]
    desired_fit = maximum_likelihood_estimation_object(initial_guess, filtered_u_g_color, galaxy_log10_mass)

    slope, intercept = desired_fit.maximum_likelihood_estimation()
    print(f' The best fit slope and intercept are {slope:.4} and {intercept:.4} respectively')
    desired_fig, desired_axis = desired_fit.frequentist_line_fit()
    plt.show()