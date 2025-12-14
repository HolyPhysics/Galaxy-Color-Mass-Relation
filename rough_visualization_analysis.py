import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file_path = "Skyserver_Spectro.csv"

try:
    data_container: Table = Table.read(file_path, format="ascii")
except Exception as rough_error:
    print(f' {type(rough_error).__name__} occured!')

    data_container: Table = Table.read(file_path, format="csv")


ra_container: list[float, ...] = data_container["ra"]
dec_container: list[float, ...] = data_container['dec']
redshift_container: list[float, ...] = data_container["redshift"]

def ra_dec_sketcher() -> None:

    figure, ax_main = plt.subplots(figsize=(8.5,9.4))
    ax_main.scatter(ra_container, dec_container, c=redshift_container, s=60)

    ax_main.set_xlim(ax_main.get_xlim()[::-1])

    ax_main.set_xlabel(" Right Ascension(RA)")
    ax_main.set_ylabel(" Declination(Dec)")
    ax_main.legend(loc="best")

    figure.tight_layout()

    plt.show()

ra_dec_sketcher()  #This does a very similar job to the pairplot below.



def a_much_faster_visualization() -> None:
    data_to_visualize: pd.DataFrame = pd.DataFrame({
        "Right Ascension(RA)": ra_container,
        "Declination(Dec)": dec_container,
        "Redshift(z)": redshift_container
    })

    print(data_to_visualize)
    sns.pairplot(data_to_visualize)
    plt.show()


a_much_faster_visualization() # This shows all the possible relationships between the different features of the data
# It appears there ought to be about 6-8 galaxy clusters in the data and the two visualizations are in complete agreement???( if forgot what I wrote initially).

