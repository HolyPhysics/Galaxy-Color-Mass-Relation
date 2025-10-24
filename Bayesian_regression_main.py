import numpy as np
from astropy.table import Table
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import emcee
import corner

galaxy_photometric_url_path: str = "Skyserver_spectro.csv"

# galaxy_photometric_data: list[int] = Table.read(galaxy_photometric_url_path)

# we try reading with different options
try:
    # we let Astropy auto-detect the format
    galaxy_photometric_data: Table = Table.read(galaxy_photometric_url_path, format='ascii')
    
except Exception as err1:

    print(f' {type(err1).__name__} occured.')

    try:
        # or we explicitly specify CSV format
        galaxy_photometric_data: Table = Table.read(galaxy_photometric_url_path, format='csv')
        
    except Exception as err2:
        # or we use pandas to read and convert to Astropy Table
        print(f' {type(err2).__name__} occured.')

        import pandas as pd
        pandas_file: pd.DataFrame = pd.read_csv(galaxy_photometric_url_path)
        galaxy_photometric_data: Table = Table.from_pandas(pandas_file)

# We check that what we got is correct
# print(f"Successfully loaded table with {len(galaxy_photometric_data)} rows")
# print("Columns:", galaxy_photometric_data.columns)

galaxy_u_mag: list[float,...] = galaxy_photometric_data['modelMag_u']
galaxy_g_mag: list[float,...] = galaxy_photometric_data['modelMag_g'] # Note that g_mag is very close to v_mag. So, we use it here!
galaxy_velocity_dispersion: list[float] = galaxy_photometric_data['velDisp']
galaxy_distance_info: list[float,...] = galaxy_photometric_data['redshift']

galaxy_u_g_color: list[float,...] = galaxy_u_mag - galaxy_g_mag # ... indicates that this can be of variable length
print(np.mean(galaxy_u_g_color))

# We use the relation between mass and velocity dispersion provided by the virial theorem \n
# to estimate the mass of the galaxies
# We're assuming that the velocity dispersion is always positive since we'll be taking the log of the numbers
velocity_dispersion_mask: list[bool,...] = ~np.isnan(galaxy_velocity_dispersion) & (galaxy_velocity_dispersion > 0)
filtered_u_g_color: list[float,...] = galaxy_u_g_color[velocity_dispersion_mask]
filtered_velocity_dispersion: list[float,...] = galaxy_velocity_dispersion[velocity_dispersion_mask]
# print(filtered_velocity_dispersion)
galaxy_log10_mass: list[float,...] =  2*np.log10(filtered_velocity_dispersion) - np.log10(6.673) + 11

# === ADD THIS LINE TO CENTER YOUR DATA ===
print(np.mean(galaxy_log10_mass))
galaxy_log10_mass = galaxy_log10_mass - np.mean(galaxy_log10_mass)
# print(galaxy_log10_mass)

# if 'modelMagErr_u' not in galaxy_photometric_data.columns:
#     print(' Sorry, not found.')
#     print(galaxy_photometric_data.columns)

# fig = plt.figure(figsize=(10,8.5))
# ax = fig.add_subplot(1,1,1)

# ax.hist(galaxy_g_mag, bins= int(np.floor(0.5*np.sqrt(len(galaxy_g_mag)))), color='red')
# plt.show()

class Bayesian_Regressoion_for_Galaxy_Mass_Color_Fit(object):

    def __init__(self, initial_guess: list[float], colors: list[float], mass: list[float]) -> None:
        self.initial_guess: list[float] = initial_guess
        self.colors: list[float] = colors
        self.mass: list[float] = mass

        # a much more efficient way to store the samples and avoid running MCMC in every single call just to make a plot
        self.samples_with_chain = None
        self.samples_without_chain = None

    def log_likelihood_estimation_function(self, parameters: list) -> float:
        slope, intercept, sigma_int = parameters
        predicted_fit: list[float] = slope*self.mass + intercept

        return np.sum(norm.logpdf(self.colors,predicted_fit,sigma_int))

    def maximum_likelihood_estimation(self) -> tuple[float]:

        def log_likelihood(parameters):

            return -self.log_likelihood_estimation_function(parameters)

        maximized_paremeters: OptimizeResult = minimize(log_likelihood, self.initial_guess, method="Nelder-Mead")

        slope, intercept, sigma_int = maximized_paremeters.x

        return slope, intercept

    def frequentist_line_fit(self, mass_mean: float = 14.355933029780603):

        slope, intercept = self.maximum_likelihood_estimation()

        # x_fit = self.colors
        # y_fit = slope * self.mass + intercept
        mass_range: list[float] = np.linspace(np.min(self.mass + mass_mean ), np.max(self.mass + mass_mean ), 500)
        # print(mass_range)
        color_fit: list[float] = intercept + slope * (mass_range - mass_mean) 

        figure = plt.figure(figsize=(10,8.7))
        ax_main = figure.add_subplot(1,1,1)

        # Plot data points
        ax_main.scatter(self.mass + mass_mean, self.colors, alpha=0.5, s=10, label='Galaxies')
        
        # Plot fitted line
        ax_main.plot(mass_range, color_fit, 'r-', linewidth=2,label=f'Fit: color = {intercept:.2f} + {slope:.2f}*mass')

        # ax_main.plot(x_fit, y_fit, ls='--', color='red', label=' U-V color against log10(Mass of Galaxy)')
        # ax_main.set_xlim()
        # ax_main.set_ylim()
        ax_main.set_xlabel('Log10(M_galaxy)')
        ax_main.set_ylabel('(U-V) color')
        ax_main.set_title('Frequentist Fitting of Line to Data')
        ax_main.grid() # I added the grid for so the plot'll be nicer
        ax_main.legend(loc='best')
        
        return figure, ax_main

    def log_prior_probability(self, parameters: list) -> float:

        slope, intercept, sigma_int = parameters

        #sigma_int > 0 since this is a scatter and scatter > 0 to be meaningful

        if sigma_int <= 0:
            return -np.inf

        # I'm using weakly informative priors for all these
        prior_slope: float = norm.logpdf(slope, 0,1)
        prior_intercept: float = norm.logpdf(intercept,0,1)
        prior_sigma_int: float = norm.logpdf(sigma_int, 0, 1)

        return prior_slope + prior_intercept + prior_sigma_int

    def log_posterior_probability(self, parameters) -> float:

        return self.log_prior_probability(parameters) + self.log_likelihood_estimation_function(parameters)

    ### testing the model works

    def testing_Bayesian_setup(self) -> None:

        test_parameter: list = self.initial_guess

        likelihood_value: float = self.log_likelihood_estimation_function(test_parameter)
        prior_value: float = self.log_prior_probability(test_parameter)
        posterior_value: float = self.log_posterior_probability(test_parameter)
        # print(f' Likelihood is: {likelihood_value}')
        # print(f' Prior is: {prior_value}')
        # print(f' Posterior is: {posterior_value}')

        #test with bad parameters to see the limit of the code
        bad_test_parameter: list[float] = [0,1,-1]
        bad_posterior: float = self.log_posterior_probability(bad_test_parameter)
        # print(f' Bad posterior should have -inf since the param {bad_test_parameter} is bad: {bad_posterior}')

    # counterpart of the mle for the frequentist approach!
    def maximum_a_posteriori_function(self) -> list[float]:

        def negative_log_posterior(parameters: list[float]) -> float:

            return  -self.log_posterior_probability(parameters)

        map_parameters: OptimizeResult = minimize(negative_log_posterior, self.initial_guess, method="Nelder-Mead")
 
        # Cache to avoid recomputation
        # if hasattr(self, '_map_cache'):
        #     return self._map_cache

        if map_parameters.success:

            map_slope, map_intercept, map_sigma_int = map_parameters.x

            # print(map_slope, map_intercept, map_sigma_int)

            # we now compare with the mle value
            mle_slope, mle_intercept = self.maximum_likelihood_estimation()

            # print(f' map slope {map_slope} vs mle slope {mle_slope}')
            # print(f' map intercept {map_intercept} vs mle intercept {mle_intercept}')

            #if the map results are in good agreement, then choose them over the mle counterpartss
            return map_parameters.x
        else:
            # print(f' maximization of map failed', map_parameters.message)

            return None

    def create_contour_grid(self, number_of_points: int = 50) -> tuple:

        shift_value_slope: float = 0.05      # Reduced from 0.15
        shift_value_intercept: float = 0.1   # Reduced from 0.15

        # we shall do this with the map value! of course, we can equally do this with 
        # the mle value. I just wanted to test this out the map value. It's my 
        # first time using this

        map_slope, map_intercept, map_sigma_int = self.maximum_a_posteriori_function()

        # we are only able to visualize 2D data and since we're mostly interested in the 
        # slope and the intercept, we'll have sigma_int fixed

        fixed_sigma_int: float = map_sigma_int
        #now, we create grid lines for slope and intercepts
        slope_range: list[float] = np.linspace(map_slope - shift_value_slope, map_slope + shift_value_slope, number_of_points)
        intercept_range: list[float] = np.linspace(map_intercept - shift_value_intercept, map_intercept + shift_value_intercept, number_of_points)

        return slope_range, intercept_range, fixed_sigma_int


    def evaluate_pdf_on_grid(self) -> list[float]:

        slope_range, intercept_range, fixed_sigma_int = self.create_contour_grid()

        # we initialize 2D areay to which will store the posterior probability

        posterior_probability_container: list[float] = np.zeros( ( len(slope_range), len(intercept_range) ))

        log_posterior_probability_container: list[float] = np.zeros( ( len(slope_range), len(intercept_range) ))


        # loop over every possible combination of slope and intercept, compute the probability
        # and return a value
        for i, slope_value in enumerate(slope_range):
            for j, intercept_value in enumerate(intercept_range):

                parameters: list[float] = [slope_value, intercept_value, fixed_sigma_int]

                # By a mile, this is the most imperative step
                # log_posterior_probability: float = self.log_posterior_probability(parameters)

                # convert to actual probabilitty value(exponentiate) and store in posterior_probability_container
                log_posterior_probability_container[i,j] = self.log_posterior_probability(parameters)
        
        max_log_posterior = np.max(log_posterior_probability_container)
        # print(f' log_posterior_max is {max_log_posterior} ')
 

        posterior_probability_container = np.exp(log_posterior_probability_container - max_log_posterior ) 

        # print(posterior_probability_container)

        return posterior_probability_container


    def calculate_contour_levels_for_sigmas_range(self) -> tuple:

        posterior_probability_container = self.evaluate_pdf_on_grid()

        # flatten probability array using the memeory efficient np.ravel() function
        # we didn't bother to call np.ravel since posterior_probability_container is a numpy array
        flattened_probability = posterior_probability_container.ravel()
        
        # we sort in descending order to get the highest probability regions first
        sorted_probability = np.sort(flattened_probability)[::-1]

        #cumulative mass counts how much total probability we've reached

        cumulative_mass = np.cumsum(sorted_probability) / np.sum(sorted_probability) # the ensures the probabilities are normalized
        # Note to your future self(good to remember). Should also use this maneuver when dealing with probabilities involving machine learning models.
        # we now find thresholds for when we've reached our desired probability coverage

        level_of_1sigma = sorted_probability[ np.searchsorted(cumulative_mass, 0.393) ]
        level_of_2sigma = sorted_probability[ np.searchsorted(cumulative_mass, 0.865) ]
        level_of_3sigma = sorted_probability[ np.searchsorted(cumulative_mass, 0.989) ]

        return level_of_1sigma, level_of_2sigma, level_of_3sigma


    def posterior_probability_contour_plot(self):

        # we combine all the steps we've been doing separately here:
        # create grid, evaluate pdf, calculate contour levels

        slope_range, intercept_range, fixed_sigma_int = self.create_contour_grid()

        posterior_probability_container = self.evaluate_pdf_on_grid()
        
        level_of_1sigma, level_of_2sigma, level_of_3sigma = self.calculate_contour_levels_for_sigmas_range()

        figure = plt.figure(figsize=(10,9))

        ax_contour = figure.add_subplot(1,1,1)

        # Note: we need to transpose posterior_2d because of how contour() expects data
        levels = [level_of_3sigma, level_of_2sigma, level_of_1sigma]
        colors = ['red', 'blue', 'green']
        linewidths = [1,2,3]
        contour_plot = ax_contour.contour(slope_range, intercept_range,posterior_probability_container,levels=levels,colors=colors,linewidths=linewidths)

        #add contour label
        ax_contour.clabel(contour_plot, inline=True, fontsize=10, fmt={level_of_1sigma: '1σ (39%)', level_of_2sigma: '2σ (87%)', level_of_3sigma: '3σ (99%)'})

        # mark the MAP values

        map_slope, map_intercept, _ = self.maximum_a_posteriori_function()
        ax_contour.plot(map_slope, map_intercept,'ro', markersize=8,label="MAP Estimate")

        ax_contour.set_xlabel("Slope")
        ax_contour.set_ylabel("Intercept")
        ax_contour.set_ylim(1.59,1.81)
        # ax_contour.set_ylim(-4.5,-4.4)
        # ax_contour.set_xlim(-1,1)
        ax_contour.set_title("2D Posterior: Slope vs Intercept\n" + f'(σ_int fixed at {fixed_sigma_int:.3f})', fontsize=14)
        ax_contour.grid()
        ax_contour.legend()

        return figure, ax_contour

    
    def bayesian_mcmc_initiator(self, number_of_steps: int =2000, burning_point: int = 100, number_of_walkers: int = 50, number_of_dimensions: int = 3) -> tuple[list[float]]:
        #proper bayesian fit using mcmc sampling
        #define posterior function for MCMC # This wraps the existing log posteriorfunction for MCMC

        def mcmc_log_posterior(parameters: list[float]) -> float:

            return self.log_posterior_probability(parameters)

        #initialize sampler

        initial_parameters: list[float] = self.maximum_a_posteriori_function() # smart starting point # Very advisable

        # set up MCMC for running
        initializer: list[array] = initial_parameters + 1e-4*np.random.randn(number_of_walkers,number_of_dimensions) # CREATE 32 WALKERS, EACH STARTING NEAR OUR MAP ESTIMATE
        # 32 walkers = 32 parallel chains exploring simultaneously
        # 1e-4 * random: small random perturbations so they start slightly differently
        # print(initializer)

        sampler = emcee.EnsembleSampler(number_of_walkers, number_of_dimensions, mcmc_log_posterior) #CREATE THE MCMC ENGINE:
        # number_of_walkers: how many explorers (50)
        # number_of_dimensions: dimensions of parameter space (3)  
        # mcmc_log_posterior: the probability landscape to explore
        sampler.run_mcmc(initializer,number_of_steps, progress=True) # RUN THE SAMPLER FOR n_steps ITERATIONS
        # progress=True shows a nice progress bar

        # Now, we extract the samples
        self.samples_with_chain = sampler.get_chain()
        self.samples_without_chain = sampler.get_chain(discard=burning_point,thin=15, flat=True) # CLEAN UP THE RAW SAMPLES:
        # discard=100: remove first 100 steps (BURN-IN - before chains stabilize)
        # thin=15: keep every 15th sample (reduce correlation between steps)
        # flat=True: combine all 32 walkers into one array
        return self.samples_with_chain, self.samples_without_chain

        
    def plot_mcmc_trace(self, burning_point: int =100) -> tuple:

        # Get the raw chains (including burn-in)
        chains = self.samples_with_chain[:] # the shape of this chains is of the form: (n_steps, n_walkers, n_dim)
        
        parameter_names = ['Slope', 'Intercept', 'Sigma_int']
        figure, ax_main = plt.subplots(3,1, figsize=(10,9))

        for i in range(3):  # for each parameters
            ax_current = ax_main[i]

            #plot all walkers for this parameter
            # for walkers in range(chains.shape[1]):
            #     ax_current.plot(chains[:,walkers,i],"-k", alpha=0.4)

            ax_current.plot(chains[:,:,i], "-k", alpha=0.4) # this works because chains[:,:,i] returns a 2d array of iterations and nwalkers
            # add burn-in cutoff line
            ax_current.axvline(burning_point,color='red', linestyle='--', label=' Burn-in Cutoff')
            ax_current.set_ylabel(parameter_names[i])
            ax_current.grid()
        
        ax_main[0].set_title(' MCMC Trace Plots - Check for Convergence ')
        ax_main[2].set_xlabel(' Number of Steps ')
        ax_main[0].legend()

        return figure, ax_main[0], ax_main[1], ax_main[2]


    def plot_mcmc_corner(self):
        # helps in visualizing the posterior
        # we use the python package called corner
        samples = self.samples_without_chain[:]
        labels: list[str] = ['Slope', 'Intercept', 'Sigma_int']
        # what does truths do??
        truths=[np.mean(samples[:,0]), np.mean(samples[:,1]), np.mean(samples[:,2])]

        corner_figure = corner.corner(samples, labels=labels, truths=truths, show_titles=True)

        return corner_figure

    def plot_bayesian_fit(self,  mass_mean = 14.355933029780603 ): 

        # import sample data
        samples = self.samples_without_chain[:]

        # Extract parameter samples
        sample_slopes = samples[:,0]
        sample_intercepts = samples[:,1]

        #create mass range
        mass_range = np.linspace(min(self.mass + mass_mean ), max(self.mass + mass_mean ),500) # setting everything baclk to their original scale!!!

        #create predictive line # plot 100 random lines from the posterior

        number_of_lines = 100
        random_line_indices = np.random.choice(len(sample_slopes), number_of_lines, replace=False)

        figure = plt.figure(figsize=(10,9))

        ax_main = figure.add_subplot(1,1,1)

        #Plot data points
        ax_main.scatter(self.mass + mass_mean, self.colors, alpha=0.5, s=10, label='Galaxies')

        #plot random lines from posterior

        for lines in random_line_indices:
            color_fit: list[float] = sample_intercepts[lines] + sample_slopes[lines] * (mass_range - mass_mean)
            ax_main.plot(mass_range, color_fit, 'r-', linewidth=1)

        # Plot mean line
        mean_line = np.mean(sample_intercepts) + np.mean(sample_slopes) * (mass_range - mass_mean)
        ax_main.plot(mass_range, mean_line, color="black", linewidth=4, label='Bayesian Mean')

        ax_main.set_xlabel('Log10(M_galaxy) [centered]')
        ax_main.set_ylabel('(U-V) color')
        ax_main.set_title('Bayesian Fitting of Line to Data with Posterior Uncertainty')
        ax_main.grid() # I added the grid for so the plot'll be nicer
        ax_main.legend(loc='best')
        
        return figure, ax_main

if __name__ == '__main__':
    initial_guess: list[float] = [0.5,1.0,0.2]
    desired_fit = Bayesian_Regressoion_for_Galaxy_Mass_Color_Fit(initial_guess, filtered_u_g_color, galaxy_log10_mass)
    slope, intercept = desired_fit.maximum_likelihood_estimation()
    # print(f' The best fit slope and intercept are {slope:.4} and {intercept:.4} respectively \n')
    desired_fig, desired_axis = desired_fit.frequentist_line_fit()
    plt.show()

    desired_fit.testing_Bayesian_setup()
    # print(desired_fit.maximum_a_posteriori_function())

    contour_figure, contour_ax = desired_fit.posterior_probability_contour_plot()
    plt.show()

    # Run MCMC
    desired_fit.bayesian_mcmc_initiator()

    # Visualize everything!
    # figure, ax_main[0], ax_main[1], ax_main[2] = desired_fit.plot_mcmc_trace() # Check convergence
    desired_fit.plot_mcmc_trace() # Check convergence
    plt.show()
    desired_fit.plot_mcmc_corner()  # Beautiful posterior
    plt.show()
    desired_fit.plot_bayesian_fit()  # Final fit
    plt.show()


    