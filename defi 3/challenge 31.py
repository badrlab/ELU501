import matplotlib.pyplot as plt
"""import struct
input_filename="C:/Users/user/Desktop/population-density-map.bmp"

bmp = open(input_filename, 'rb') # open a binary file
print('-- First part of the header, information about the file (14 bytes)')
print('Type:', bmp.read(2).decode())
print('Size: %s' % struct.unpack('I', bmp.read(4)))
print('Reserved 1: %s' % struct.unpack('H', bmp.read(2)))
print('Reserved 2: %s' % struct.unpack('H', bmp.read(2)))
offset=struct.unpack('I', bmp.read(4))
print('Image start after Offset: %s' % offset)

print('-- Second part of the header, DIB header, bitmap information header (varying size)')
print('The size of this DIB Header Size: %s' % struct.unpack('I', bmp.read(4)))
print('Width: %s' % struct.unpack('I', bmp.read(4)))
print('Height: %s' % struct.unpack('I', bmp.read(4)))
print('Colour Planes: %s' % struct.unpack('H', bmp.read(2)))
pixel_size=struct.unpack('H', bmp.read(2))
print('Bits per Pixel: %s' % pixel_size)
print('Compression Method: %s' % struct.unpack('I', bmp.read(4)))
print('Raw Image Size: %s' % struct.unpack('I', bmp.read(4)))
print('Horizontal Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Vertical Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Number of Colours: %s' % struct.unpack('I', bmp.read(4)))
print('Important Colours: %s' % struct.unpack('I', bmp.read(4)))

# At this step, we have read 14+40 bytes
# As offset[0] = 54, from now, we will read the BMP content
# You have to read each pixel now, and do what you have to do
# First pixel is bottom-left, and last one top-right
# .........
bmp.close()"""


# Another method to parse a BMP image
# To manipulate imageIf you want to work with image data in Python, 
# numpy is the best way to store and manipulate arrays of pixels. 
# You can use the Python Imaging Library (PIL) to read and write data 
# to standard file formats.

# Use PIL module to read file
# http://pillow.readthedocs.io/en/latest/
from PIL import Image
import numpy as np
input_filename="D:/Downloads/defi 3/population-density-map.bmp"
im = Image.open(input_filename)

# This modules gives useful informations
width=im.size[0]
heigth=im.size[1]
colors = im.getcolors(width*heigth)
print(im.getcolors(width*heigth))
print('Nb of different colors: %d' % len(colors))
# To plot an histogram

def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]))

#plt.show()
# We have 32 different colors in this image
# We can see that we have "only" 91189 black pixels able to stop zombies 
# but we have a large majority of dark ones slowing their progression

# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(im) 

print(p.shape)
# a result (3510, 4830, 3) means (rows, columns, color channels)
# where 3510 is the height and 4830 the width

# to get the Red value of pixel on row 3 and column 59
p[3,59][0]

# How to get the coordinates of the green and red pixels where 
# (0,0) is top-left and (width-1, height-1) is bottom-right
# In numpy array, notice that the first dimension is the height, 
# and the second dimension is the width. That is because, for a numpy array, 
# the first axis represents rows (our classical coord y), 
# and the second represents columns (our classical x).

# First method
# Here is a double loop (careful, O(n²) complexity) to parse the pixels from
# (0,0) top-left and (heigth-1, width-1) is bottom-right
"""for y in range(heigth):
    for x in range(width):
        # p[y,x] is the coord (x,y), x the colum, and y the line
        # As an exemple, we search for the green and red pixels
        # p[y,x] is an array with 3 values
        # We test if there is a complete match between the 3 values 
        # from both arrays p[y,x] and np.array([0,255,0])
        # to detect green pixels
        if (p[y,x] == np.array([0,255,0])).all():
            print("Coordinates (x,y) of the green pixel: (%s,%s)" % (str(x),str(y)))
            # Coordinates (x,y) of the green pixel: (4426,2108)
        if (p[y,x] == np.array([255,0,0])).all():
            print("Coordinates (x,y) of the red pixel: (%s,%s)" % (str(x),str(y)))
            # Coordinates (x,y) of the red pixel: (669,1306)"""

# Here is a more efficient method to get the location of the green and red pixels
mask = np.all(p == (0, 255, 0), axis=-1)
z = np.transpose(np.where(mask))
print("Coordinates (x,y) of the green pixel: (%d,%d)" % (z[0][1],z[0][0]))
mask = np.all(p == (255, 0, 0), axis=-1)
z = np.transpose(np.where(mask))
print("Coordinates (x,y) of the red pixel: (%d,%d)" % (z[0][1],z[0][0]))


# Now we have the source and the target positions of our zombies
# we could convert our RGB image into greyscale image to manipulate
# only 1 value for the color and deduce more easily the density of
# population
grayim = im.convert("L")
#grayim.show()
#colors = grayim.getcolors(width*heigth)
print('Nb of different colors: %d' % len(colors))
# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(grayim) 
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
plt.hist(p.ravel())
#plt.show()

# from gray colors to density
density = p/255.0
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
plt.hist(density.ravel())
#plt.show()

# We can use the gray 2D array density to create our graph
# Gray colors density[y,x] range now from 0 (black) to 1 (white)
# density[0,0] is top-left pixel density
# and density[heigth-1,width-1] is bottom-right pixel




######################################################################################################
M=[colors[i][1] for i in range(len(colors))]
M=M[:len(M)-3]


def vitesse_zombie(pixel):
    if pixel==(16,16,16):
        return 1/24
    a=0
    for i in range(len(M)):
        if pixel==M[i]:
            a=i
    return 1-(23*a)/(24*29)

####################################################################################################
import numpy as np
import matplotlib.pyplot as plt    
#%matplotlib inline
'''rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = 12, 8

beta = 1
gamma = 0

def euler_step(u, f, dt):
    return u + dt * f(u)

def f(u):
    S = u[0]
    I = u[1]
    #R = u[2]
    
    new = np.array([-beta*(S[1:-1, 1:-1]*I[1:-1, 1:-1] + S[0:-2, 1:-1]*I[0:-2, 1:-1] + S[2:, 1:-1]*I[2:, 1:-1] + S[1:-1, 0:-2]*I[1:-1, 0:-2] + S[1:-1, 2:]*I[1:-1, 2:]),
                     beta*(S[1:-1, 1:-1]*I[1:-1, 1:-1] + S[0:-2, 1:-1]*I[0:-2, 1:-1] + S[2:, 1:-1]*I[2:, 1:-1] + S[1:-1, 0:-2]*I[1:-1, 0:-2] + S[1:-1, 2:]*I[1:-1, 2:]) - gamma*I[1:-1, 1:-1]])
    
    padding = np.zeros_like(u)
    padding[:,1:-1,1:-1] = new
    padding[0][padding[0] < 0] = 0
    padding[0][padding[0] > 255] = 255
    padding[1][padding[1] < 0] = 0
    padding[1][padding[1] > 255] = 255
    #padding[2][padding[2] < 0] = 0
    #padding[2][padding[2] > 255] = 255
    
    return padding



S_0 = np.asarray(im)[:,:,1]
I_0 = np.zeros_like(S_0)
I_0[2108,4426] = 1 # patient zero

R_0 = []

# On introduit une boucle while np.asarray(im)[1306,669] == [255,0,0] : on incrémente T par le même chiffre associé à la vitesse
dt = 1                          # time increment
""""T = 900                       # final time
N = int(T/dt) + 1               # number of time-steps
t = np.linspace(0.0, T, N)      # time discretization"""

# initialize the array containing the solution for each time-step
u = np.empty((1, 2, S_0.shape[0], S_0.shape[1]))
u[0][0] = S_0
u[0][1] = I_0
#u[0][2] = R_0


import matplotlib.cm as cm
theCM = cm.get_cmap("Reds")
theCM._init()
alphas = np.abs(np.linspace(0, 1, theCM.N))
theCM._lut[:-3,-1] = alphas

from images2gif import writeGif

keyFrames = []
frames = 60.0
i=0
while list(np.asarray(im)[1306,669]) == [255,0,0]:
    imgplot = plt.imshow(im, vmin=0, vmax=255)
    imgplot.set_interpolation("nearest")
    imgplot = plt.imshow(u[0][1], vmin=0, cmap=theCM)
    imgplot.set_interpolation("nearest")
    filename = "outbreak" + str(i) + ".png"
    plt.savefig(filename)
    keyFrames.append(filename)
    i+=1
    u[0] = euler_step(u[0], f, dt)




"""for i in range(0, N-1, int(N/frames)):
    imgplot = plt.imshow(img, vmin=0, vmax=255)
    imgplot.set_interpolation("nearest")
    imgplot = plt.imshow(u[i][1], vmin=0, cmap=theCM)
    imgplot.set_interpolation("nearest")
    filename = "outbreak" + str(i) + ".png"
    plt.savefig(filename)
    keyFrames.append(filename)"""

images = [Image.open(fn) for fn in keyFrames]
gifFilename = "outbreak.gif"
writeGif(gifFilename, images, duration=0.3)
plt.clf()'''


######################################################################################################
M=[colors[i][1] for i in range(len(colors))]
M=M[:len(M)-3]


"""def vitesse_zombie(pixel):
    if pixel==(16,16,16):
        return 1/24
    a=0
    for i in range(len(M)):
        if pixel==M[i]:
            a=i
    return 1-(23*a)/(24*29)
"""
####################################################################################################
def f(i,j):
    return i*4830+j



#######################################################################"""


image=np.array(im)
# matrice contient (3510) éléments , chaque élément est une liste de (4830) listes, et chacune de ces listes contient 3 chifres
import networkx as nx
def generate_matrix(imag):
    G=nx.Graph()
    for i in range(3510):
        for j in range(4830):
            N=f(i,j)
            G.add_node(N)
    for i in range(3510):
        for j in range(4830):
            if i!=0 and i!=3509 and j!=0 and j!=4829:
                G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if i==0 and j!=0 and j!=4829:
                G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if i==3509 and j!=0 and j!=4829:
                G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if  j==0 and i!=0 and i!=3509:
                G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if  j==4829 and i!=0 and i!=3509:
                G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if i==0 and j==0:
                G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if i==0 and j==4829:
                G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if i==3509 and j==0:
                G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            if i==3509 and j==4829:
                G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
                G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
    return G


nx.astar_path(generate_matrix(image),f(2108,4426),f(1306,669),'weight')


