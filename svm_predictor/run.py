import os
dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)
os.chdir(dir_path)

#os.system('pwd')
os.system("python3 FinalModel.py")