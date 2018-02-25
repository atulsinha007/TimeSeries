import sys

import os.path
from cx_Freeze import setup, Executable

'''pip install cx_Freeze
  python setup.py build
'''

includes = []
include_files = [r"C:\Users\Atul Sinha\AppData\Local\Programs\Python\Python36-32\DLLs\tcl86t.dll",
				 r"C:\Users\Atul Sinha\AppData\Local\Programs\Python\Python36-32\DLLs\tk86t.dll"]
os.environ['TCL_LIBRARY'] = r'C:\Users\Atul Sinha\AppData\Local\Programs\Python\Python36-32\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Atul Sinha\AppData\Local\Programs\Python\Python36-32\tcl\tk8.6'

options={"build_exe": {"includes": includes, "include_files": include_files}}

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
# os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
# os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

# options = {
# 	'build_exe': {
# 		'include_files':[
# 			os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
# 			os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
# 		 ],
# 	},
# }

setup(
	name = "Final Model",
	version = "3.6",
	description = "Any Description you like",
	executables = [Executable("FinalModel.py", base = "Win32GUI")])
