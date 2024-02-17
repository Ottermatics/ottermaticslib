# Install datastores libraries
import os, pathlib, sys, subprocess

this_file = pathlib.Path(__file__)
this_dir = this_file.parent
req_file = os.path.join(this_dir, "datastores_requirements.txt")
#try:
import engforge.datastores.data

#FIXME: works on every import error including this modules imports (circular buffer issue)
# except ImportError as e:
#     print(f"got import error {e}")
# 
#     answer = input("type `CONFIRM` to install datastore requirements:")
#     if answer.strip() == "CONFIRM":
#         cmd = f'{sys.executable} -m pip install -r "{req_file}"'
#         print(f"running: {cmd}")
#         os.system(cmd)
# 
#     else:
#         print("ok fine, enjoy your error then")
