import numpy as np
from numpy import fft
from PIL import Image

# Takes in a matrix and produces a new matrix like P, except that the number of rows and
# columns are both multiples of 15.
# Note that P is n x m x 3.
# Rows and columns of 0 are added to the "end" of P, if needed
def pad15(P):
    dim = P.shape

    rpad15 = 15 - dim[0] % 15
    cpad15 = 15 - dim[1] % 15

    if rpad15 > 0 or cpad15 > 0:
        Q = np.pad(P, pad_width=((0, rpad15), (0,cpad15), (0,0)), mode='constant', constant_values=0)
    return Q
def process_block(B, tol):
    num_zeros = 0
    FB = np.fft.fft2(B)
    for row in FB:
        for num in row:
            if num <tol:
                num = 0
                num_zeros += 1
    B1 = np.fft.ifft2(FB)

    B1 = np.real(B1)
    B1 = np.round(B1)
    B1 = np.clip(B1, 0,255)
    B1 = B1.astype(np.uint8)

    return [B1, num_zeros]

def trim(image):
    # Check if any channel has non-zero values
    non_zero = np.any(image != 0, axis=2)
    
    # Find non-zero rows/columns
    rows = np.any(non_zero, axis=1)
    cols = np.any(non_zero, axis=0)
    
    # Get bounds
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return image[rmin:rmax+1, cmin:cmax+1, :]

def compress_image(P,tol):
    numrows,numcols,colours = P.shape
    compressedP = np.zeros((numrows,numcols,3), dtype=np.uint8)
    padded_P = pad15(P)
    num_zeros = 0
    p_rows,p_cols,colours = padded_P.shape
    for colour in range(3):
        rows_of_15 = p_rows // 15
        cols_of_15 = p_cols // 15
        for row in range(rows_of_15 -1):
            for col in range(cols_of_15 - 1):
                block = P[row * 15: (row + 1) * 15, col * 15:(col+1) * 15,colour]
                processed_info = process_block(block,tol)
                compressedP[15*row:15*row+15,15*col:15*col+15,colour] = processed_info[0]
                num_zeros += processed_info[1]

    compressedP = trim(compressedP)
    compression_rate = num_zeros / (p_rows * p_cols * 3) * 100
    return [compressedP, compression_rate]

    
img = Image.open(r'C:/Users/vince/Documents/CS371/A4/jerma.jpg')
img_arr = np.array(img)
cimg_info = compress_image(img_arr,29)

cimg = Image.fromarray(cimg_info[0])

print("Compression Rate: %", cimg_info[1])




        