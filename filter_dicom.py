import os, sys
import glob

def median(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0

def get_filesize(filen):
  statinfo = os.stat(filen)
  return str(statinfo.st_size).replace("L","")

def filter_outlier_dcms(patient, filelist):
  med = median(filelist.values())
  if not os.path.exists(patient+"/outlier_dcms"):
    os.makedirs(patient+"/outlier_dcms")
  for f in filelist.keys():
    if filelist[f] > med+med*.2:  #move all files 20% bigger than median
      os.rename(f, patient+"/outlier_dcms/"+os.path.basename(f))



root_dir = sys.argv[1]
curr_patient = ""
curr_filelist = {}

for dcm in glob.iglob('{}/*/1/*.dcm'.format(root_dir)):
     rt, patient, subfolder, filename = dcm.split("/")
     if curr_patient != "{}/{}".format(rt,patient):
       #calculate outlier images and move them
       if curr_patient != "":
         filter_outlier_dcms(curr_patient, curr_filelist)
       curr_patient = "{}/{}".format(rt,patient)
       curr_filelist = {}

     curr_filelist[dcm] = int(get_filesize(dcm))

if len(curr_filelist.keys()) > 0 and curr_patient != "":
     filter_outlier_dcms(curr_patient, curr_filelist)
