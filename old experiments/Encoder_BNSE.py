from mogptk.init import BNSE
import mogptk

import numpy as np
import torch
torch.manual_seed(1);

from sklearn.datasets import fetch_openml

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data['year']
    m = ml_data.data['month']
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs

x, y = load_mauna_loa_atmospheric_co2()
data = mogptk.Data(x, y, name='Mauna Loa')
w, _,_, mu_real, mu_imag, var_real, var_imag =BNSE(data.X,data.Y)
print(mu_real)
print(mu_imag)