import csv
import numpy as np
import matplotlib.pyplot as plt
import time_series_preprocessor as tsp
timeseries = tsp.load_series('dataset-pm25.csv')
print(timeseries)

plt.figure()
plt.plot(timeseries)
plt.title('Normalized time series')
plt.xlabel('Fecha')
plt.ylabel('Material particulado')
plt.legend(loc='upper left')
plt.show()


def split_data(data, percent_train):
	num_rows = len(data)    
	train_data, test_data = [], []    
	for idx, row in enumerate(data):
		if idx < num_rows * percent_train:
			train_data.append(row)
		else:
			test_data.append(row)
	return train_data, test_data

