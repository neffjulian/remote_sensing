from pathlib import Path

import numpy as np
from skimage.transform import resize
from PIL import Image

# def fine_reg(im1,imref,r=10,alpha=0.2,na=5,b=0):
#     # FINE REGISTRATION of multiband images im1 to imref (nans allowed).
#     # im1 and imref must be float numpy arrays of the same size, with bands on the 3rd dimension.
#     # due to translation/rotation non-mapped pixels present nans.
#     # PARAMETERS: (number of tries=(r+1)^(2)*na+1)
#     # r=10 # shift radius in pixels (be careful, cpu time can explode)
#     # alpha=0.2 # rotation degree range (both clockwise and counterclockwise)
#     # na=5 # number of degree steps for rotation (the no-rotation case is added)
#     # b=0 # band to use to make the registration (indicated in the 3rd dimension)
   
#     im1tmp=np.copy(im1[:,:,b]) # choose b-th band to find the transformation vector
#     im1tmp[np.isnan(im1tmp)]=0
#     imreftmp=np.copy(imref[:,:,b])
#     imreftmp[np.isnan(imreftmp)]=0
#     a=np.append(np.linspace(-alpha,alpha,na),0) # array of angle shifts
#     R=np.empty([len(a),4])*np.nan # array of correlation values and trans vectors
#     d=0 # counter

#     for k in a:

#         # first rotation
#         # can=rotate(im1tmp,k,reshape=False,cval=0,order=0)

#         # then correlation
#         # C=fftconvolve(can[::-1,::-1],imreftmp)
#         N=np.array(np.shape(can))-1
#         trans=np.array(np.unravel_index(np.argmax(C),np.shape(C)))-N
#         # store params and compute correlation
#         R[d,0:3]=[trans[0],trans[1],k]
#         # R[d,3]=np.max(C)
#         d=d+1
#         print("\r%.2f%s" % ((d/len(R)*100),"%"),end='')

#     # best transformation
#     best_ind=np.argmax(R[:,3])
#     nbest=R[best_ind,0].astype('int')
#     mbest=R[best_ind,1].astype('int')
#     abest=R[best_ind,2]
 
#     # APPLY THE TRANSFORMATION (missing data as nan)
#     # first rotation
#     # im1_reg=rotate(im1,abest,reshape=False,cval=np.nan,order=0)
   
#     # then translation
#     im1_reg=np.roll(im1,(nbest,mbest),axis=(0,1))
#     if mbest>0:
#         im1_reg[:,:mbest,:]=np.nan
#     elif mbest<0:
#         im1_reg[:,mbest:,:]=np.nan
#     if nbest>0:
#         im1_reg[:nbest,:,:]=np.nan
#     elif nbest<0:
#         im1_reg[nbest:,:,:]=np.nan

#     #store trans vector
#     trans=R[best_ind,:3]
#     return im1_reg, trans # output:  transformed image, transformation vector

# def co_register_images(year: str, month: str, index: int):

def get_RGB_s2(s2_data: np.ndarray) -> np.ndarray:
    return np.concatenate((s2_data['B02'], s2_data['B03'], s2_data['B04']), axis=0)

def get_RGB_ps(ps_data: np.ndarray) -> np.ndarray:
    return np.transpose(ps_data[:,:,:3], (2, 0, 1))

def co_register_images():
    planet_image = Path('images/planet_scope/22_april/0000/20220428_101547_22_2426\data.npy')
    sentinel_image = Path('images/sentinel_2/22_apr/0000/data.npz')

    planet_data = np.load(planet_image)
    sentinel_data = np.load(sentinel_image)

    s2_red = np.squeeze(sentinel_data['B04'])
    s2_blue = np.squeeze(sentinel_data['B03'])
    s2_green = np.squeeze(sentinel_data['B02'])

    s2_rgb = (np.dstack((s2_red, s2_green, s2_blue)) * 2).astype(np.uint8)
    print(np.shape(s2_rgb))
    img = Image.fromarray(s2_rgb)
    img.show()


    reshaped_s2 = get_RGB_s2(sentinel_data)
    reshaped_ps = get_RGB_ps(planet_data)

    # reshaped_s2 = np.transpose(reshaped_s2[:, :, :], (1, 2, 0))
    # image = Image.fromarray(reshaped_s2, 'RGB')
    # image.show()

    # downscaled_ps = np.resize(reshaped_ps, np.shape(reshaped_s2))

    # downscaled_ps = np.transpose(downscaled_ps[:, :, :], (1, 2, 0))
    # print(np.shape(downscaled_ps))

    # image = Image.fromarray(downscaled_ps.astype('uint8'), 'RGB')

    # print(downscaled_ps)

    # image.show()
if __name__ == "__main__":
    
    co_register_images()