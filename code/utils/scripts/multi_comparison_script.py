
"""
Purpose:
-----------------------------------------------------------------------------------
We seek the activated voxel positionsi through multi-comparison of beta values across
subjects

Step
-----------------------------------------------------------------------------------
1. calculate the mean of each single beta values across subject and plot them
2. calculate the variance of each single beta values across subject and plot them
3. calculate the t-stat of each single beta values across subject and plot them
4. calculate the p-value of each single betav values across subject and plot them
"""


import sys, os
##sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
sys.path.append(os.path.join(os.path.dirname('__file__'), "./"))
import numpy as np
from glm import *
import nibabel as nib
from matplotlib import colors
import matplotlib.pyplot as plt
#from scipy.stats import sem
from smoothing import *
from plot_mosaic import * 
from scipy.stats import t as t_dist
#from visualization import *


nice_cmap_values = np.loadtxt('actc.txt')
nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')
dirs = ['../../../txt_output/multi_beta','../../../fig/multi_beta']
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
project_path='../../../'

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)


#need for plotting mask
#TODO: change the location path
template = nib.load(project_path+\
           'data/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii')
template_data = template.get_data()
img = nib.load(project_path+'data/ds005_2/sub001/model/model001/task001_run002.feat/' + \
 'masked_filtered_func_data_mni.nii.gz')

task = dict()
gain = dict()
loss = dict()

#load all of them
for x in range(1,17):
	task[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_task.txt')

for x in range(1,17):
	gain[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_gain.txt')

for x in range(1,17):
	loss[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_loss.txt')

# for x in range(1,17):
# 	dist[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_dist.txt')


##################################### MEAN plot #########################################

#calculate mean and plot (let's try for task)
task_sum = task[1]
for x in range(2,17):
	task_sum +=task[x]

task_mean = task_sum/16
task_mean_reshape = task_mean.reshape(91,109,91)


gain_sum = gain[1]
for x in range(2,17):
	gain_sum +=gain[x]

gain_mean = gain_sum/16
gain_mean_reshape = gain_mean.reshape(91,109,91)


loss_sum = gain[1]
for x in range(2,17):
	loss_sum +=loss[x]

loss_mean = loss_sum/16
loss_mean_reshape = loss_mean.reshape(91,109,91)


plt.title('In brain activated voxels - \nmean across 16 subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_mean_reshape, transpose=False), cmap='seismic', alpha=1, vmin=task_mean_reshape.min(), vmax= task_mean_reshape.max())
plt.colorbar()
plt.savefig(dirs[1]+'/mean_task.png')
plt.clf()

plt.title('In brain activated voxels - \nmean across 16 subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_mean_reshape, transpose=False), cmap='seismic', alpha=1, vmin=gain_mean_reshape.min(), vmax= gain_mean_reshape.max())
plt.colorbar()
plt.savefig(dirs[1]+'/mean_gain.png')
plt.clf()

plt.title('In brain activated voxels - \nmean across 16 subjects on LOSS condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(loss_mean_reshape, transpose=False), cmap='seismic', alpha=1, vmin=loss_mean_reshape.min(), vmax= loss_mean_reshape.max())
plt.colorbar()
plt.savefig(dirs[1]+'/mean_loss.png')
plt.clf()


##################################### SD plot #########################################

#calculate variance and plot
stdlst = []
for x in range(1,17):
	stdlst.append(task[x])

stdarray = np.array(stdlst)
task_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
task_std_reshape = task_std.reshape(91,109,91)

stdlst = []
for x in range(1,17):
	stdlst.append(gain[x])

stdarray = np.array(stdlst)
gain_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
gain_std_reshape = gain_std.reshape(91,109,91)

stdlst = []
for x in range(1,17):
	stdlst.append(loss[x])

stdarray = np.array(stdlst)
loss_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
loss_std_reshape = loss_std.reshape(91,109,91)

plt.title('In brain activated voxels - \nStandard Deviation across 16 subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_std_reshape, transpose=False), cmap='seismic', alpha=1, vmin=task_std_reshape.min(), vmax= task_std_reshape.max())
plt.colorbar()
plt.savefig(dirs[1]+'/std_task.png')
plt.clf()

plt.title('In brain activated voxels - \nStandard Deviation across 16 subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_std_reshape, transpose=False), cmap='seismic', alpha=1, vmin=gain_std_reshape.min(), vmax= gain_std_reshape.max())
plt.colorbar()
plt.savefig(dirs[1]+'/std_gain.png')
plt.clf()

plt.title('In brain activated voxels - \nStandard Deviation across 16 subjects on LOSS condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(loss_std_reshape, transpose=False), cmap='seismic', alpha=1, vmin=loss_std_reshape.min(), vmax= loss_std_reshape.max())
plt.colorbar()
plt.savefig(dirs[1]+'/std_loss.png')
plt.clf()


##################################### stat plot #########################################


#calculate t-stat and plot
task_tstat = task_mean/(task_std/np.sqrt(15))
task_tstat = np.nan_to_num(task_tstat)
task_tstat_reshape = task_tstat.reshape(91,109,91)

gain_tstat = gain_mean/(gain_std/np.sqrt(15))
gain_tstat = np.nan_to_num(gain_tstat)
gain_tstat_reshape = gain_tstat.reshape(91,109,91)

loss_tstat = loss_mean/(loss_std/np.sqrt(15))
loss_tstat = np.nan_to_num(loss_tstat)
loss_tstat_reshape = loss_tstat.reshape(91,109,91)


plt.title('In brain activated voxels - \nT-statistics across 16 subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_tstat_reshape, transpose=False), cmap='seismic', alpha=1)
plt.colorbar()
plt.savefig(dirs[1]+'/tstat_task.png')
plt.clf()

plt.title('In brain activated voxels - \nT-statistics across 16 subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_tstat_reshape, transpose=False), cmap='seismic', alpha=1)
plt.colorbar()
plt.savefig(dirs[1]+'/tstat_gain.png')
plt.clf()

plt.title('In brain activated voxels - \nT-statistics across 16 subjects on LOSS condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(loss_tstat_reshape, transpose=False), cmap='seismic', alpha=1)
plt.colorbar()
plt.savefig(dirs[1]+'/tstat_loss.png')
plt.clf()



##################################### P-value plot #########################################

#calculate p-value and plot
task_pval = t_dist.cdf(abs(task_tstat), 15)
task_pval_reshape = task_pval.reshape(91,109,91)
task_pval_reshape = 1-task_pval_reshape

gain_pval = t_dist.cdf(abs(gain_tstat), 15)
gain_pval_reshape = gain_pval.reshape(91,109,91)
gain_pval_reshape = 1-gain_pval_reshape

loss_pval = t_dist.cdf(abs(loss_tstat), 15)
loss_pval_reshape = loss_pval.reshape(91,109,91)
loss_pval_reshape = 1-loss_pval_reshape
p_value_thres = 0.05/(91*109*91)

plt.title('In brain activated voxels - \nP-value across 16 subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_pval_reshape, transpose=False), cmap='Greys', alpha=1)
plt.colorbar()
plt.savefig(dirs[1]+'/pval_task.png')
plt.clf()

plt.title('In brain activated voxels - \nP-value across 16 subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_pval_reshape, transpose=False), cmap='Greys', alpha=1)
plt.colorbar()
plt.savefig(dirs[1]+'/pval_gain.png')
plt.clf()





