import numpy as np
from numpy import fft
import matplotlib.pyplot as plt 
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
    mask = np.abs(FB) < tol
    FB[mask] = 0
    num_zeros = np.sum(mask)
    
    B1 = np.fft.ifft2(FB)
    B1 = np.real(B1)
    B1 = np.round(B1)
    B1 = np.clip(B1, 0, 255)
    B1 = B1.astype(np.uint8)
    
    return B1, num_zeros

def compress_image(P, tol):
    numrows, numcols, colours = P.shape
    padded_P = pad15(P)
    compressedP = np.zeros_like(padded_P, dtype=np.uint8)
    num_zeros = 0
    p_rows, p_cols, colours = padded_P.shape
    
    for colour in range(3):
        rows_of_15 = p_rows // 15
        cols_of_15 = p_cols // 15
        for row in range(rows_of_15):
            for col in range(cols_of_15):
                block = padded_P[row*15:(row+1)*15, col*15:(col+1)*15, colour]
                processed_block, block_zeros = process_block(block, tol)
                compressedP[row*15:(row+1)*15, col*15:(col+1)*15, colour] = processed_block
                num_zeros += block_zeros
    
    # Trim back to original size or actual content
    if numrows == p_rows and numcols == p_cols:
        compressedP = compressedP[:numrows, :numcols, :]
    else:
        compressedP = trim(compressedP)
        
    compression_rate = num_zeros / (p_rows * p_cols * 3) * 100
    return compressedP, compression_rate

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

def plot_images(image_file):
    plt.figure(figsize=(15, 10))
    
    # Original Image
    P = np.array(Image.open(image_file))
    plt.subplot(2, 2, 1)
    plt.imshow(P)
    plt.title('Original Image')
    plt.axis('off')
    
    # 50% Compression
    plt.subplot(2, 2, 2)
    tol50 = 21  
    cP50, comp50 = compress_image(P, tol50)
    plt.imshow(cP50)
    plt.title(f'Compression Rate: {comp50:.1f}%')
    plt.axis('off')
    
    # 80% Compression
    plt.subplot(2, 2, 3)
    tol80 = 105  
    cP80, comp80 = compress_image(P, tol80)
    plt.imshow(cP80)
    plt.title(f'Compression Rate: {comp80:.1f}%')
    plt.axis('off')
    
    # 95% Compression
    plt.subplot(2, 2, 4)
    tol95 = 5000
    cP95, comp95 = compress_image(P, tol95)
    plt.imshow(cP95)
    plt.title(f'Compression Rate: {comp95:.1f}%')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

image_path = "C:/Users/vince/Documents/CS371/image_compression/jerma.jpeg"
img = Image.open(image_path)
img_arr = np.array(img)

plot_images(image_path)

"""
inf = compress_image(img_arr,105)
cimg = Image.fromarray(inf[0])
cimg.show()
print(inf[1])

"""




        