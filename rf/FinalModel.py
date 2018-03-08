from tkinter import *
from tkinter import filedialog
import math
import csv
import os
file_path = "WMA.csv"
import re
file_path_file = "file_path.txt"
#import dummy
def run():
	global param1, param2
	st_lis = re.split('(\w+.\w+)$', sys.argv[0])



	if st_lis[0] is not '':
		os.chdir(st_lis[0])
	print(st_lis[0])
	with open(file_path_file, 'w+') as file_ob:
		file_ob.write(file_path)

	#os.system('cd ..\\vENV\\bin\\')
	try:
		r = os.system('python random_forest.py 75 '+str(param1.get())+ ' ' + str(int(param2.get())+2))
		if r:
			q = os.system(
				'python3 random_forest.py 75 ' + str(param1.get()) + ' ' + str(int(param2.get()) + 2))
			if q:
				print("upgrade python or if already there alias it with one of python or python3")
				exit(-1)
	except OSError:
		print("upgrade python")
		exit(-1)

def browse():

	root = Tk()
	root.withdraw()
	file = filedialog.askopenfile(parent=root, mode='rb', title='Choose a file')
	print(file.name)
	if file is not None:
		data = file.read()
		file.close()
	else:
		print("invalid file")
	global file_path
	file_path = str(file.name)
	print(type(file_path))


def quit():
	root.destroy()
	exit(1)


root = Tk()
root.title("SVM")
top = Frame(root)
top.pack(side='top')

hwframe = Frame(top)
hwframe.pack(side='top')
font = 'times 18 bold'
hwtext = Label(hwframe, text='SVM Model', font=font)
hwtext.pack(side='top', pady=20)

rframe = Frame(top)
rframe.pack(side='top', padx=10, pady=20)

Label1 = Label(rframe, text = "Number of days")
Label2 = Label(rframe, text = "Price_ind OHLC")

Label1.grid(row=0, sticky=E)
Label2.grid(row=1, sticky=E)

param_1 = IntVar()
param_2 = IntVar()

param_1.set(1)
param_2.set(0)

param1 = Entry(rframe, textvariable=param_1)
param2 = Entry(rframe, textvariable=param_2)

param1.grid(row=0, column=1, sticky=E, padx=40)
param2.grid(row=1, column=1, sticky=E, padx=40)

dummy = Label(rframe, text="")
dummy.grid(row=3, sticky=E)

button1 = Button(rframe, text="Choose File", command=browse)
button1.grid(row=5, column=0)
button2 = Button(rframe, text="   Run   ", command=run)
button2.grid(row=5, column=1)
button1 = None
button3 = Button(rframe, text="   Quit    ", command=quit)
button3.grid(row=5, column=2)


root.mainloop()