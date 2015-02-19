"""
Kamil Orlowski/C11743571
AI2 Assignment 1
16/02/15
"""

import os
import numpy as np
import csv

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_lines(f):
	for line in f:
		line = line.strip()
		if(line):
			yield line

def get_array(g):
	return np.array(list(g))

def get_features(filename):
	if os.path.isfile(filename):
		lines = read_lines(open(filename))
		return get_array(lines)
	else:
		raise Exception(filename + 'does not exist')

def get_dataset(filename):
	reader = csv.reader(open(filename), skipinitialspace=True)
	return get_array(reader)

def get_is_cont(dataset, features):
	is_cont = {}
	for f in range(len(features)):
		for i in range(len(dataset)):
			is_cont[f] = is_number(dataset[i][f])
			break
		if f not in is_cont:
			is_cont[f] = False
	return is_cont

cont_columns = ['min','1st quartile','mean','median', '3rd quartile','max', 'std', 'count', 'cardinality', 'missing %']
cont_columns = ['count', 'missing %', 'cardinality', 'min','1st quartile','mean','median', '3rd quartile','max', 'std']

def report_cont_feature(dataset, f):
	feature = dict()
	values = np.array(dataset[:, f], dtype='float')
	non_zero_values = values[np.nonzero(values)]
	
	feature[cont_columns[0]] = len(non_zero_values)
	feature[cont_columns[1]] = (len(values) - len(non_zero_values)) * 100 / len(values)
	feature[cont_columns[2]] = len(np.unique(non_zero_values))
	feature[cont_columns[3]] = np.min(non_zero_values)
	feature[cont_columns[4]] = np.percentile(non_zero_values, 25)
	feature[cont_columns[5]] = np.mean(non_zero_values)
	feature[cont_columns[6]] = np.median(non_zero_values)
	feature[cont_columns[7]] = np.percentile(non_zero_values, 75)
	feature[cont_columns[8]] = np.max(non_zero_values)
	feature[cont_columns[9]] = np.std(non_zero_values)

	return feature

def get_modes(values):
	index = np.column_stack((range(len(values[0])), values[1]))
	mode_index = np.argsort(index[:, 1])
	values = np.column_stack((values[0],values[1]))
	modes = values[index[mode_index[-2:],0]]
	return modes[::-1]

cat_columns = ['count', 'missing %', 'cardinality', 'mode','mode count','mode %','2nd mode','2nd mode count', '2nd mode %']

def report_cat_feature(dataset, f):
	feature = dict()
	values = np.array(dataset[:, f])
	non_empty_values = values[np.where(values != '?')]
	unique_values = np.unique(values, return_counts=True)
	modes = get_modes(unique_values)

	feature[cat_columns[0]] = len(non_empty_values)
	feature[cat_columns[1]] = (len(values) - len(non_empty_values)) * 100 / len(values)
	feature[cat_columns[2]] = len(unique_values[0])	
	feature[cat_columns[3]] = modes[0,0]
	feature[cat_columns[4]] = modes[0,1]
	feature[cat_columns[5]] = int(modes[0,1]) * 100 / len(values)
	feature[cat_columns[6]] = modes[1,0]
	feature[cat_columns[7]] = modes[1,1]
	feature[cat_columns[8]] = int(modes[1,1]) * 100 / len(values)

	return feature


def write_report(features, report, columns, path):
    writer = csv.writer(open(path, 'wb'))
    headings = ['FEATURENAME']
    headings.extend(list(columns))
    writer.writerow(headings)

    for feature in features:
    	row = [feature]
    	for col in columns:
    		row.append(report[feature][col])
    	writer.writerow(row)

def write_reports(dataset, features, is_cont, cont_path, cat_path):
	cont = dict()
	cat = dict()
	cont_features = list()
	cat_features = list()

	for f in range(len(features)):
		if is_cont[f]:
			cont[features[f]] = report_cont_feature(dataset, f)
			cont_features.append(features[f])
		else:
			cat[features[f]] = report_cat_feature(dataset, f)
			cat_features.append(features[f])

	write_report(cont_features, cont, cont_columns, cont_path)
	write_report(cat_features, cat, cat_columns, cat_path)

features_path = './data/featurenames.txt'
dataset_path = './data/DataSet.txt'
cont_path = './data/c11743571CONT.csv'
cat_path = './data/c11743571CAT.csv'
features = get_features(features_path)
dataset = get_dataset(dataset_path)
is_cont = get_is_cont(dataset, features)
write_reports(dataset, features, is_cont, cont_path, cat_path)
