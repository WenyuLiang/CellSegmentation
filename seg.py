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
warnings.filterwarnings('ignore')

description="Example: python seg.py -p simple_grids/YL1025E1new_E1_b400 -o result -b 400"

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', type=str,help='path to the matrix')
parser.add_argument('-o', '--outpath', type=str,help='output path', default='./')
parser.add_argument('-t', '--threshold', type=int, default=90, help='threshold for binarization')
parser.add_argument('-b', '--bin', type=str, default='100', help='bin size (100 or 400)')
parser.add_argument('-s', '--size', type=int, default=15, help='marker size of the plot: don\'t change it unless you are not satisfied with the default size')
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
    marker_size = args.size
elif args.bin == '400':
    marker_size = 100
else:
    raise ValueError("Invalid bin size: Only 100 or 400 are supported.")

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
total_counts = adata.var['total_counts'].sum()
 
# sc.pl.embedding(adata, basis='spatial', color='total_counts', title='total_counts', color_map="RdYlBu_r",s=20, save = "test.png", show=False)
ax = sc.pl.embedding(adata, basis='spatial', color='total_counts',
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

# Threshold the image
ret, binary = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) >= 3:
    three_largest_contours = heapq.nlargest(3, contours, key=cv2.contourArea)
    largest_contour = three_largest_contours[2]
else:
    raise ValueError("Not enough contours found to extract the third largest one.")

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
    ax = sc.pl.embedding(adata_fake, basis='spatial', color='total_counts',
                    color_map=custom_cmap, s=marker_size, show=False)

    # Highlight the four known barcode positions
    # plt.scatter(barcode_coords[:, 0], barcode_coords[:, 1], c='#00FF00', s=50)  # Use a distinct color and size
    plt.scatter(barcode_coords[:, 0], barcode_coords[:, 1], c=[(254/255, 254/255, 254/255)], s=15, 
                 marker='*')

    # Add a legend to help identify the barcodes
    plt.legend()
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

    # Define size constraints
    divMaxSize = 0.180
    divMinSize = 0.120
    # Create a window for display
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    # Loop through the contours
    #print("Number of contours: ", len(imContours))
    barcode_contour = []
    for i in range(len(imContours)):
        ratio = np.sqrt(cv2.contourArea(imContours[i])) / cv2.arcLength(imContours[i], True)
        if 680<cv2.contourArea(imContours[i])<690:
            barcode_contour.append(imContours[i])
            #print("I'm a star!", cv2.contourArea(imContours[i]), ratio)
   
    barcode_pixel_coords = []
    for cnt in barcode_contour:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            barcode_pixel_coords.append((cx, cy))
    return np.array(barcode_pixel_coords)

count = 0
while count < 10:
    barcode_coords, _binary = preprocess(adata)
    barcode_pixel_coords = process_image(_binary)
    if barcode_pixel_coords.shape == (3,2):
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