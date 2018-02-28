#svm_predictor.py
import csv
import re
import numpy as np
from sklearn.svm import SVR
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os
import copy
import xlrd
import xlsx2csv_m
import csv

file_path_file = "./file_path.txt"
log_st = ""
def get_data(filename, last_x = 75, front_days= 15, price_ind = 2):
	dates = []
	prices = []
	test_dates = []
	test_prices = []

	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		next(csvFileReader)
		next(csvFileReader)
		'''next(csvFileReader)
		next(csvFileReader)
		next(csvFileReader)
		'''


		for row in csvFileReader:
			if row[1] is not '' and row[price_ind]:
				l = row[1].split('/')
				if len(l)<3:
					l = row[1].split('-')
					if len(l) < 3:
						print("wrong date format")
						exit(2)
				for ind in range(len(l)):
					item = l[ind]
					if len(item)==1:
						l[ind] = '0'+l[ind]
				l = [l[2]]+[l[0]]+[l[1]]
				dbg = ''.join(l)
				try:
					prices.append(float(row[price_ind]))
					dates.append(int(''.join(l)))
					#print(row[1])
				except ValueError as err:

					print(err)
					break
				finally:
					pass
		num_lis = [i for i in range(len(dates))]
		num_lis.sort(key=lambda x: dates[x])
		dates = [ dates[i] for i in range(len(num_lis))]
		prices = [ prices[i] for i in range(len(num_lis))]
		test_dates = dates[-front_days:]
		test_prices = prices[-front_days:]
		dates = dates[-last_x-front_days: -front_days]
		prices = prices[ -last_x-front_days: -front_days]

	return dates, prices, test_dates, test_prices


def test(predicted_price, test_dates, test_prices):

	tp = np.array(test_prices)
	tp = np.reshape(tp, (tp.shape[0],1))

	err = (predicted_price - tp)**2

	err = np.sum(err, axis = 0)

	return err

def predict_prices(dates, prices, front_days = 15, last_x = 75, price_ind = 2):
	dates = copy.deepcopy(dates)
	dates = np.reshape( dates, (len(dates), 1))


	dates_num = [i for i in range(len(dates))]
	dates_num_ar = np.array(dates_num)
	dates_num_ar = dates_num_ar.reshape((dates_num_ar.shape[0], 1))
	if not os.path.isfile("./" + "pickle_jar/svrs_"+str(last_x)+"_"+str(price_ind)+".pickle"):
		svr_lin = SVR(kernel='linear', C=1e3)
		svr_poly = SVR(kernel='poly', C=1e3, degree=2)
		svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		svr_lin.fit(dates_num_ar, prices)
		svr_poly.fit(dates_num_ar, prices)
		svr_rbf.fit(dates_num_ar, prices)
		fs = open( "./" + "pickle_jar/svrs_"+str(last_x)+"_"+str(price_ind)+".pickle", "wb")
		tup = (svr_lin, svr_poly, svr_rbf)
		pickle.dump( tup , fs)
		fs.close()
	else:
		fs = open("./" + "pickle_jar/svrs_" + str(last_x) + "_" + str(price_ind) + ".pickle", "rb")

		tup = pickle.load( fs)
		(svr_lin, svr_poly, svr_rbf) = tup
		fs.close()
	plt.plot(dates_num_ar, prices, color='red', label='Data')
	plt.plot(dates_num_ar,svr_rbf.predict(dates_num_ar) , color='black', label='RBF_model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Machine training')
	plt.legend()
	plt.show()
	lis = []
	last_x = len(prices)
	for i in range(front_days):
		#print([svr_rbf.predict(last_x+i), svr_lin.predict(last_x+i), svr_poly.predict(last_x+i) ])
		lis.append(np.concatenate((np.concatenate((svr_rbf.predict(last_x+i), svr_lin.predict(last_x+i))), svr_poly.predict(last_x+i)) ))

	q = np.array(lis)
	fileo = open("./log_folder/log_predictions_price_ind_" + str(price_ind) + "_last_days_"+str(last_x)+ "_front_days_" + str(front_days) + ".csv", "w+")
	st = ""
	global log_st
	log_st = "./log_folder/log_predictions_price_ind_" + str(price_ind)+"_last_days_"+str(last_x)+ "_front_days_" + str(front_days) + ".csv"
	for row in q:
		for col in row:
			st+=","+str(col)+" "
		st+="\n"
	fileo.write(st)
	fileo.close()
	return q

def main_result(price_ind, mod_num):

	lish = []
	last_x_list = [55, 65, 75]
	front_days_list = [1, 2, 3, 5, 15]
	for last_x in last_x_list:
		lis = []
		for front_days in front_days_list:

			dates, prices, test_dates, test_prices = get_data('WMA.csv', last_x=last_x, front_days=front_days, price_ind=price_ind)
			dates = np.reshape(dates, (len(dates), 1))

			dates_num = [i for i in range(len(dates))]
			dates_num_ar = np.array(dates_num)
			dates_num_ar = dates_num_ar.reshape((dates_num_ar.shape[0], 1))
			fs = open("./" + "pickle_jar/svrs_" + str(last_x) + "_" + str(price_ind) + ".pickle", "rb")

			tup = pickle.load(fs)
			(svr_lin, svr_poly, svr_rbf) = tup
			fs.close()


			last_x = len(prices)

			lisp = []
			for i in range(front_days):
				# print([svr_rbf.predict(last_x+i), svr_lin.predict(last_x+i), svr_poly.predict(last_x+i) ])
				lisp.append(np.concatenate(
					(np.concatenate((svr_rbf.predict(last_x + i), svr_lin.predict(last_x + i))), svr_poly.predict(last_x + i))))
			p = np.array(lisp)
			tp = test(p, test_dates, test_prices)
			lis+=[tp[mod_num]]
		lish.append(lis)
	q = np.array(lish)
	fileo = open("./log_folder/log_error_price_ind_" + str(price_ind) + "_mod_num_" + str(mod_num) + ".csv", "w+")
	st = ""
	for row in q:
		for col in row:
			st+=","+str(col)+" "
		st+="\n"
	fileo.write(st)
	fileo.close()
	return q
	#p = test(predicted_price)
def main_graph():
	import sys
	global file_path_file
	last_x = int(sys.argv[1])
	front_days = int(sys.argv[2])
	price_ind = int(sys.argv[3])
	with open(file_path_file, 'r+') as file_ob:
		data_set = file_ob.read().rstrip().lstrip()

	#data_set = sys.argv[4]
	print(data_set, type(data_set))
	st_lis = re.split('(\w+).(\w+)$', data_set)
	if (st_lis[2] == 'xlsx' ):
		obj = xlsx2csv_m.Xlsx2csv(data_set)
		str_lis = re.split('(\w+).\w+$', data_set)
		obj.convert(str_lis[0] + 'converted_' + str_lis[1] + '.csv')
		data_set = str_lis[0] + 'converted_' + str_lis[1] + '.csv'
	elif (st_lis[2] != 'csv' ):
		print("The file must be xlsx or csv type")
		exit(1)
	dates, prices, test_dates, test_prices=get_data(data_set, last_x=last_x, front_days = front_days, price_ind = price_ind)
	predicted_price = predict_prices(dates, prices, front_days = front_days, last_x=last_x,  price_ind = price_ind)
	test_dates_num = np.arange(len(test_dates))
	plt.plot(test_dates_num, test_prices, color = 'red', label = 'Data')
	plt.plot(test_dates_num, predicted_price[:,0], color = 'black', label = 'RBF_model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Machine')
	plt.legend()
	plt.show()
	print()
	return predicted_price
	#print(predicted_price)



def main(last_x, front_days, price_ind):
	import sys

	dates, prices, test_dates, test_prices=get_data('WMA.csv', last_x=last_x, front_days = front_days, price_ind = price_ind)
	predicted_price = predict_prices(dates, prices, front_days = front_days, last_x=last_x,  price_ind = price_ind)
	#print(predicted_price)
	p = test(predicted_price, test_dates, test_prices)
	fileo = open("./log_folder/log_"+str(last_x)+"_"+str(front_days)+"_"+str(price_ind)+".txt", "a+")
	st = ""
	for i in p:
		st += str(i)+" "
	st+="\n"
	fileo.write(st)
	fileo.close()
	print(p)
if __name__ == '__main__':
	print(main_graph())