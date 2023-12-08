# MIT License

# Copyright (c) 2023 wliang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Wenyu Liang
# Date: 2023-12-05
# Contact: liangwy@salus-bio.com
import os
import scanpy as sc
import scselpy as scS 
import matplotlib.pyplot as plt
import cv2
import heapq
import numpy as np
import matplotlib
import time
import argparse
import xopen
import warnings
from scipy.interpolate import splprep, splev
warnings.filterwarnings('ignore')

description="Example: python seg.py -p simple_grids/YL1025E1new_E1_b400 -o result -b 400"

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', type=str,help='path to the matrix')
parser.add_argument('-o', '--outpath', type=str,help='output path', default='./')
parser.add_argument('-t', '--threshold', type=int, default=-1, help='threshold for binarization(default 90)')
parser.add_argument('-b', '--bin', type=str, help='bin size (40|100|400)', required=True)
parser.add_argument('-s', '--size', type=int, default=-1, help='marker size of the plot: don\'t change it unless you are not satisfied with the default size')
parser.add_argument('-m', '--smooth', type=bool, default=False, help='whether to smooth the image(default False)')
args = parser.parse_args()

adata = sc.read_10x_mtx(args.path, var_names='gene_symbols', cache=True)
adata.var_names_make_unique()
x = []
y = []
with xopen.xopen(args.path + '/spatial.txt.gz', 'rt') as sp:
    for line in sp:
        x.append(int(line.strip().split(' ')[1]))
        y.append(int(line.strip().split(' ')[2]))
spatial_coordinates = np.column_stack((x, y))
adata.obsm['spatial'] = spatial_coordinates

if args.outpath.endswith('/'):
    outpath = args.outpath
else:
    outpath = args.outpath + '/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

if args.bin == '100':
    marker_size = 15
    threshold = 90
    factor = 0.0005
elif args.bin == '400':
    marker_size = 100
    threshold = 90
    factor = 0.000001
elif args.bin == '40':
    factor = 0.0001#0.0008
    marker_size = 15
    threshold = 100
else:
    raise ValueError("Invalid bin size: Only 40 or 100 or 400 are supported.")

if args.size not in [-1, 15, 100]:
    marker_size = args.size
if threshold not in [-1, 90, 100]:
    threshold = args.threshold

# print("Threshold: ", threshold)
# print("Marker size: ", marker_size)
# print("Factor: ", factor)
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
total_counts = adata.var['total_counts'].sum()
 
# sc.pl.embedding(adata, basis='spatial', color='total_counts', title='total_counts', color_map="RdYlBu_r",s=20, save = "test.png", show=False)
ax = sc.pl.embedding(adata, basis='spatial', color='total_counts', #n_genes_by_counts
                color_map="RdYlBu_r", s=marker_size, show=False)

# Remove axis and frame
plt.axis('off')
ax.set_title('')
# Save the figure without a frame
plt.savefig(outpath + "temp.png", dpi=600)

# Close the plot
plt.close()

# Read the image
img = cv2.imread(outpath + "temp.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if args.smooth:
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    #gray = cv2.bilateralFilter(gray, 41, 250, 250)

    #-------------------------------------grey erosion-------------------------------------#
    kernel_size = 21
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray = cv2.erode(gray, kernel, iterations = 1)
    gray = cv2.dilate(gray, kernel,iterations = 1)
    gray = cv2.erode(gray, kernel, iterations = 1)
    gray = cv2.dilate(gray, kernel,iterations = 1)
    #-------------------------------------grey erosion-------------------------------------#

# Threshold the image
ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

if args.smooth:
    #-------------------------------------binary erosion-------------------------------------#
    kernel_size = 21
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.erode(binary, kernel, iterations = 1)
    binary = cv2.dilate(binary, kernel,iterations = 1)
    binary = cv2.erode(binary, kernel, iterations = 1)
    binary = cv2.dilate(binary, kernel,iterations = 1)
    #-------------------------------------binary erosion-------------------------------------#

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 3. Find the third largest contour
if len(contours) >= 3:
    three_largest_contours = heapq.nlargest(3, contours, key=cv2.contourArea)
    largest_contour = three_largest_contours[2]
else:
    raise ValueError("Not enough contours found to extract the third largest one.")
 
# 4. Approximate contours
if args.smooth:
    epsilon = factor * cv2.arcLength(largest_contour, True)   
    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)  
    contour_points = largest_contour.squeeze()

    # Fit a B-spline curve
    tck, u = splprep(contour_points.T, u=None, s=0.0, per=1)

    # Evaluate the B-spline curve at 100 points
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    combined_points = np.vstack((x_new, y_new)).T  # This stacks them and then transposes the result

    # Step 2: Reshape to the format (n, 1, 2)
    largest_contour = combined_points.reshape(-1, 1, 2).astype(np.int32)
#-------------------------------------smooth-------------------------------------#

def preprocess(adata):
    adata_fake = adata.copy()
    spatial_data = adata_fake.obsm['spatial']
    np.random.seed(int(time.time()))
    # Generate 10 unique random indices from the range of the number of rows in spatial_data
    random_indices = np.random.choice(spatial_data.shape[0], size=3, replace=False)

    # Use the indices to select rows
    barcode_coords = spatial_data[random_indices, :]
    sorted_indices = np.argsort(barcode_coords[:, 0])
    #barcode_coords.sort(axis=0)
    barcode_coords = barcode_coords[sorted_indices]
    custom_cmap = matplotlib.colors.ListedColormap([(0,0,0),(1,1,1)], name='custom')
    # Plot the embedding with the barcode coordinates marked
    ax = sc.pl.embedding(adata_fake, basis='spatial', color='n_genes_by_counts', #total_counts
                    color_map=custom_cmap, s=marker_size, show=False)

    # Highlight the four known barcode positions
    # plt.scatter(barcode_coords[:, 0], barcode_coords[:, 1], c='#00FF00', s=50)  # Use a distinct color and size
    plt.scatter(barcode_coords[:, 0], barcode_coords[:, 1], c=[(254/255, 254/255, 254/255)], s=15, 
                 marker='*')

    # Add a legend to help identify the barcodes
    #plt.legend()
    plt.axis('off')
    ax.set_title('')
    # Save the figure
    plt.savefig(outpath + "highlighted_barcodes.png", dpi=600)

    # Close the plot
    plt.close()

    img = cv2.imread(outpath + 'highlighted_barcodes.png')

    white_pixels = np.all(img == [255, 255, 255], axis=-1)
    # Set those pixels to black (0, 0, 0)
    img[white_pixels] = [0, 0, 0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, binary = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return barcode_coords, binary

def process_image(binary):
    imContours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    barcode_contour = []
    for i in range(len(imContours)):
        ratio = np.sqrt(cv2.contourArea(imContours[i])) / cv2.arcLength(imContours[i], True)
        if 680<cv2.contourArea(imContours[i])<690:
            barcode_contour.append(imContours[i])
   
    barcode_pixel_coords = []
    for cnt in barcode_contour:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            barcode_pixel_coords.append((cx, cy))
    return np.array(barcode_pixel_coords)

def CheckCollinear(p1, p2, p3):
    if p1[0] == p2[0] == p3[0]:
        return True
    elif p1[1] == p2[1] == p3[1]:
        return True
    elif (p1[0] - p2[0]) * (p1[1] - p3[1]) == (p1[0] - p3[0]) * (p1[1] - p2[1]):
        return True
    else:
        return False

count = 0
while count < 10:
    barcode_coords, _binary = preprocess(adata)
    barcode_pixel_coords = process_image(_binary)
    if barcode_pixel_coords.shape == (3,2) and not CheckCollinear(barcode_pixel_coords[0], barcode_pixel_coords[1], barcode_pixel_coords[2]):
        break   
    count += 1
    if count == 10:
        raise ValueError("You need to do the segmentation manually!")
    
#barcode_pixel_coords.sort(axis=0)
sorted_indices = np.argsort(barcode_pixel_coords[:, 0])

# sort the array by the first column
barcode_pixel_coords = barcode_pixel_coords[sorted_indices]
barcode_pixel_coords = barcode_pixel_coords.astype(np.float32)
barcode_coords = barcode_coords.astype(np.float32)
assert(barcode_pixel_coords.shape == barcode_coords.shape == (3,2))
# Compute the affine transformation matrix
affine_matrix = cv2.getAffineTransform(barcode_pixel_coords, barcode_coords)
contour_ = cv2.transform(np.array([largest_contour.squeeze()]), affine_matrix)[0]
#contour_ = cv2.transform(cp_barcode_pixel_coords.reshape(1,10,2), affine_matrix)[0]
#print(affine_matrix)
mock_dict = {'embedding': [contour_.tolist()]}
mock_dict['embedding'][0] = [tuple(inner_list) for inner_list in mock_dict['embedding'][0]]
scS.pl.embedding(adata, basis="spatial", color="total_counts", s=marker_size, skip_float_check = True, mock=mock_dict, save = outpath + "seg.png") # color_map="RdYlBu_r"

adata.obs['in_tissue'] = adata.obs['REMAP_1'].apply(lambda x: 1 if x == '1' else 0)
adata_in_tissue = adata[adata.obs['in_tissue'] == 1]
adata.write(outpath + "adata_original.h5ad")
adata_in_tissue.write(outpath + "adata_in_tissue.h5ad")


ax = sc.pl.embedding(adata_in_tissue, basis="spatial", color="total_counts", s=marker_size,
                     color_map="RdYlBu_r", title="total_counts", show=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.tick_params(axis='x')
ax.tick_params(axis='y')

ax.set_title('Total Counts')
plt.savefig(outpath+"total_counts.png", dpi=600)
plt.show()

ax = sc.pl.embedding(adata_in_tissue, basis="spatial", color="n_genes_by_counts", s=marker_size,
                     color_map="RdYlBu_r", title="n_genes_by_counts", show=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('N_Features') 
plt.savefig(outpath+"n_genes_by_counts.png", dpi=600) 
plt.show()

prefix = args.path[:-1].split('/')[-1] if args.path.endswith('/') else args.path.split('/')[-1]  
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'],
             jitter=0, multi_panel=True, save= prefix + '.png')
#move the file from figures to outpath
os.rename(f'figures/violin{prefix}.png',outpath+'violin.png')

sc.pp.calculate_qc_metrics(adata_in_tissue, percent_top=None, log1p=False, inplace=True)
total_counts = float(adata.var['total_counts'].sum())
valid_total_counts = float(adata_in_tissue.var['total_counts'].sum())
rate = str(valid_total_counts/total_counts)
#print("Valid total counts: ", valid_total_counts)
with open(outpath+'stat.txt','w') as f:
    f.write(rate)