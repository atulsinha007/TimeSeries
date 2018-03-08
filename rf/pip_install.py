import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
os.chdir(dir_path)
os.system("pip3 install -r requirements.txt" )