from tkinter import *
from tkinter import filedialog
import os
file_path = "WMA.csv"
def function():
	root = Tk()
	root.withdraw()
	file = filedialog.askopenfile(parent=root, mode='rb', title='Choose a file')
	print(file.name)
	#print(len(file))
	#print(file.__dir__())
	if file != None:
		data = file.read()
		file.close()
		#print("I got %d bytes from this file." % len(data))
	else:
		print("invalid file")
	global file_path
	file_path = str(file.name)
	print(type(file_path))


root = Tk()

root.geometry('300x300')
root.title("SVM")
Label1 = Label(root, text = "predict_days_ahead")
Label2 = Label(root, text = "price_ind OHLC")

param_1 = IntVar()
param_2 = IntVar()

param_1.set(1)
param_2.set(0)


param1 = Entry(root, textvariable=param_1)
param2 = Entry(root, textvariable=param_2)

Label1.grid(row=0, sticky=E)
Label2.grid(row=1, sticky=E)


param1.grid(row=0, column=1)
param2.grid(row=1, column=1)

def run():
	global param1, param2

	os.system('/usr/bin/python3.5 svm_predictor.py 75 '+str(param1.get())+ ' ' + str(int(param2.get())+2) +' '+ file_path)
	# print(Entry1.get())

def quit():
	root.destroy()
	exit(1)
button1 = Button(root,text="Browse", command=function)
button1.grid(row=3, columnspan=2)
button2 = Button(root,text="Run", command=run)
button2.grid(row=4, columnspan=3)
button1 = None
button3 = Button(root,text="quit", command=quit)
button3.grid(row=5, columnspan=3)
root.mainloop()

input()