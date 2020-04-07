import csv
import numpy as np

x = np.zeros(1682)
row_names = ['movie_id', 'movie_url']
with open('movie_url.csv', 'r', newline='') as in_csv:
	reader = csv.DictReader(in_csv, fieldnames=row_names, delimiter=',')
	for row in reader:
		movie_id = row['movie_id']
		x[int(movie_id)-1] = 1
		
for i in range(1682):
	if x[i] == 0:
		print(i+1)