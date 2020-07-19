# %%
import openpiv.tools
import openpiv.process
import openpiv.preprocess
import openpiv.validation
import openpiv.filters
import openpiv.scaling

# %%
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %matplotlib inline
# import pandas as pd
import os
# import cv2s
# from scipy import ndimage

# %%
winsize = 32  
searchsize = 64  
overlap = 16  
dt = 462.44*10**(-6)

# %%
# for i in range(200,300,100):
i = 200

File1 = 'frame%05d.png' % i
ii = i+1
File2 = 'frame%05d.png' %(ii)
print('Frames ', i, ' and ', ii)

img_a  = openpiv.tools.imread( File1)
img_b  = openpiv.tools.imread( File2)

frame_a = img_a[130:400,200:1200]
frame_b = img_b[130:400,200:1200]

#rotation angle in degree
# frame_a = ndimage.rotate(frame_a, -1)
# frame_b = ndimage.rotate(frame_b, -1)

# %%
plt.imshow(img_a,cmap=cm.gray)

# %%
plt.imshow(frame_a,cmap=cm.gray)

# %%
plt.imshow(frame_b,cmap=cm.gray)

# %%
u0, v0, sig2noise = openpiv.process.extended_search_area_piv( frame_a.astype(np.int32),
                                                             frame_b.astype(np.int32), 
                                                         window_size=winsize, 
                                                         overlap=overlap, 
                                                         dt=dt, 
                                                         search_area_size=searchsize, 
                                                         sig2noise_method='peak2peak' )
x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )


# Scales 
u0 = u0/80*2*0.001
v0 = v0/235*8.225*0.001
x = x/80*2*0.001
y = y/235*8.225*0.001


# u0, v0, mask = openpiv.validation.global_val(u0,v0,(-0.1,0.1),(-0.3,0))


fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.quiver(x,y,u0,v0,
           angles='xy', scale_units='xy', scale = 160,
           headlength = 3, headwidth = 2, headaxislength = 3, pivot = 'mid')
plt.axis('equal')
plt.clim(0,0.06)
# plt.colorbar(orientation='horizontal')
pic = 'Run20_1_%05d.png' % i
plt.savefig(pic, dpi=600, facecolor='w', edgecolor='w')
plt.show()
plt.close()

# %%
u1, v1, mask = openpiv.validation.sig2noise_val( u0, v0, sig2noise, threshold =1.3)
u2, v2 = openpiv.filters.replace_outliers( u1, v1, method='localmean', max_iter=100, kernel_size=1)

fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.quiver(x,y,u2,v2,
           angles='xy', scale_units='xy', scale = 160,
           headlength = 3, headwidth = 2, headaxislength = 3, pivot = 'mid')
plt.axis('equal')
plt.clim(0,0.06)
# plt.colorbar(orientation='horizontal')
pic = 'Run20_2_%05d.png' % i
plt.savefig(pic, dpi=600, facecolor='w', edgecolor='w')
plt.show()
plt.close()

# %% [raw]
# U = pd.DataFrame(data = u2)
# V = pd.DataFrame(data = v2)
# Ux = U.sub(U.mean(axis=1),axis=0)
# Vy = V.sub(V.mean(axis=1),axis=0)
#
# print(U.mean(axis=1))
# print(V.mean(axis=1))
#
# fig=plt.figure(figsize=(6, 3), dpi= 80, facecolor='w', edgecolor='k')
# plt.quiver(x,y,Ux,Vy,
#            (Ux**2+Vy**2)**0.5,cmap = 'jet',
#            angles='xy', scale_units='xy', scale=160,
#            headlength = 3, headwidth = 2, headaxislength = 3, pivot = 'mid')
# # plt.axis('equal')
# plt.clim(0,0.06)
# plt.colorbar(orientation='horizontal')
# plt.streamplot(x,y,Ux,Vy,
#                density=[2,1.5],
#                color = 'black')
# plt.xlim(0,0.025)
# plt.ylim(0,0.0085)
# pic = 'PIVFIG/Run20_Stream_%05d.png' %i
# plt.savefig(pic, dpi=600, facecolor='w', edgecolor='w')
# # plt.show()
# plt.close()
#
#
# File3 = 'PIVFILE/Run20_%05d.txt' %i
# openpiv.tools.save(x, y, u2, v2, mask, File3)
# # openpiv.tools.display_vector_field(File3,window_size=32, scaling_factor = 10)

# %%
