"""
This file contains functions to load the dataset and the NIR spectra used in the IKPLS
tests. The dataset originates from the articles by Dreier et al. and Engstrøm et al.
and is publicly available on GitHub. The dataset consists of 26617 rows and 11 columns.
The columns represent ground truth values for 8 different grain varieties, protein,
moisture, and an assignment to a dataset split. The NIR spectra consist of 26617
near-infrared spectra with 102 wavelength channels. The spectra are transformed from
reflectance to pseudo absorbance.

Dreier et al.:
https://journals.sagepub.com/doi/abs/10.1177/09670335221078356?journalCode=jnsa

Engstrøm et al.:
https://openaccess.thecvf.com/content/ICCV2023W/CVPPA/html/Engstrom_Improving_Deep_Learning_on_Hyperspectral_Images_of_Grain_by_Incorporating_ICCVW_2023_paper.html

Dataset repository: https://github.com/Sm00thix/IKPLSTestData

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import io
from urllib.request import urlopen

import numpy as np
import pandas as pd

GITHUB_DATADIR = "https://raw.githubusercontent.com/Sm00thix/IKPLSTestData/main/data/"


def load_csv():
    """
    Loads a csv-file with 26617 rows and 11 columns. The columns represent ground truth
    values for 8 different grain varieties, protein, moisture, and an assignment to a
    dataset split.
    """
    csv_url = GITHUB_DATADIR + "dataset.csv"
    columns = [
        "Rye_Midsummer",
        "Wheat_H1",
        "Wheat_H3",
        "Wheat_H4",
        "Wheat_H5",
        "Wheat_Halland",
        "Wheat_Oland",
        "Wheat_Spelt",
        "Moisture",
        "Protein",
        "split",
    ]
    csv = pd.read_csv(csv_url, usecols=columns)
    csv = csv.astype(np.float64)
    return csv


def load_spectra():
    """
    Loads 26617 near-infrared (NIR) spectra with 102 wavelength channels and transforms
    them from reflectance to pseudo absorbance.
    """
    spectra_url = GITHUB_DATADIR + "spectra.npz"
    with urlopen(spectra_url) as resp:
        resp_byte_array = resp.read()
    byte_contents = io.BytesIO(resp_byte_array)
    npz_arr = np.load(byte_contents)
    spectra = np.vstack([npz_arr[k] for k in npz_arr.keys()])
    spectra = spectra.astype(np.float64)
    spectra = -np.log10(spectra)
    return spectra
