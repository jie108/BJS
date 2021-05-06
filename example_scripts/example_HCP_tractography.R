###HCP data analysis: Step 2 (cont'd):  tractography
#take the peak detection results as input

##need to first install Rpackage:
# install.packages("devtools")
devtools::install_github("vic-dragon/dmri.tracking")
#If the above code does not work, try the following code
remotes::install_github("vic-dragon/dmri.tracking@main")

##load packages
library(R.matlab)
library(rgl)
library(dmri.tracking)

data_path = '/your/path/to/BJS/data/'
##load peak detection results from "example_HCP.py"
file_name= paste0(data_path,"peak.mat")
peak_result= readMat(file_name) #read matlab data into R

##format the peak detection results for the tracking algorithm
v.obj<-NULL
v.obj$vec<-peak_result$vec
v.obj$loc<-peak_result$loc
v.obj$map<-c(peak_result$map)
v.obj$rmap<-c(peak_result$rmap)
v.obj$n.fiber<-c(peak_result$n.fiber)
v.obj$n.fiber2<-c(peak_result$n.fiber2)
v.obj$braingrid<-peak_result$braingrid
v.obj$xgrid.sp<-peak_result$xgrid.sp
v.obj$ygrid.sp<-peak_result$ygrid.sp
v.obj$zgrid.sp<-peak_result$zgrid.sp


## Apply Tracking Algorithm
# It takaes 20 min on 2020 mac 16inch
# If you want to skip this step, you can load the attached Rdata file (tracts_SLF_L.Rdata)

tracts <- v.track(v.obj,  max.line=500, elim.thres=10)
# elim.tresh:  return indices of tracks of at least elim.thres length: use this information for quicker plotting

#save(tracts, file=paste0(data_path,'tracts_SLF_L.Rdata'))
#load(paste0(data_path,'tracts_SLF_L.Rdata'))

###Streamline Selection based on predefined streamline selection masks

rmap = temp$rmap

seed = rmap[as.vector(peak_result$seed)]
target = rmap[as.vector(peak_result$target)]

iind_store = c()
for(iind in (tracts$sorted.iinds[tracts$sorted.update.ind])){
  cond1 = (sum(tracts$tracks1[[iind]]$iinds %in% seed) + sum(tracts$tracks2[[iind]]$iinds %in% seed)>0)
  cond2 = (sum(tracts$tracks1[[iind]]$iinds %in% target) + sum(tracts$tracks2[[iind]]$iinds %in% target)>0)

  if (cond1*cond2 > 0){
    print(iind)
    iind_store<-c(iind_store, iind)
  }
}

## plot the tractography results (the selected streamlines)
open3d()
for (iind in iind_store){
  cat(iind,"\n")
  # plot
  tractography(tracts$tracks1[[iind]]$inloc, tracts$tracks1[[iind]]$dir)
  tractography(tracts$tracks2[[iind]]$inloc, tracts$tracks2[[iind]]$dir)
}
par3d(windowRect = c(0, 0, 700, 700))
load(paste0(data_path,'view_left_slf.Rdata'))
rgl.viewpoint(scale=c(1,1,1),zoom=0.7,userMatrix = view_M)
rgl.snapshot(paste0(data_path,'slf_l'), fmt='png')

## Feature extraction (e.g. number of fiber longer than 10mm)
length(iind_store) # number of streamlines
summary(tracts$lens[iind_store])  # Summary of tracts lengths for the selected streamlines
