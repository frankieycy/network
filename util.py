import os,glob

progressBars = ['-','\\','|','/']

def setUpFolder(folder, format=None):
    # create folder if unexisting or clear folder
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if format: files = glob.glob(os.path.join(folder,format))
    else: files = glob.glob(os.path.join(folder,'*'))
    for f in files:
        os.remove(f)
