import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import networkx as nx

input_filename="population-density-map2.bmp"
im = Image.open(input_filename)
width=im.size[0]
heigth=im.size[1]
colors = im.getcolors(width*heigth)

def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]))

p = np.array(im) 
mask = np.all(p == (0, 255, 0), axis=-1)
z = np.transpose(np.where(mask))
mask = np.all(p == (255, 0, 0), axis=-1)
z = np.transpose(np.where(mask))
grayim = im.convert("L")
p = np.array(grayim) 
plt.hist(p.ravel())
density = p/255.0
plt.hist(density.ravel())
M=[colors[i][1] for i in range(len(colors))]
M=sorted(M) #sorting the colors in order to remove the colors (16,16,16) and the start and finish points 
M.pop(0)
M.pop(-2)
M.pop(0)
image=np.array(im)

#define the speed of the zombie based on population density 
def vitesse_zombie(pixel):
    if pixel==(16,16,16):
        return 1/24
    if pixel==(255,0,0)or pixel==(0,255,0):
        return 1
    a=0
    for i in range(len(M)):
        if pixel==M[i]:
            a=len(M)-i-1
    return 1-(23*a)/(24*28)

#calculate the time required to arrive 
def timer(l,imag):
    a=0
    for elt in l:
        i=elt // 3949
        j=elt % 3949
        a+=1/vitesse_zombie(tuple(imag[i,j]))
    return print(str(a/24)+' jours')

#test weither the zombies will die or not if they follow the path entered in the parameters
def test(L,imag):
    c=0
    for i in range (len(L)):
        x=L[i] // 3949
        y=L[i] % 3949
        if tuple(imag[x,y])==(16,16,16) or tuple(imag[x,y])==(24,24,24):
            c+=1
        else: c=0
        if c==10 : return print('YOU DIED')
    return print('YOU KILLED EVERYONE')

#transform the coordinates to get a single value to store, we can get the coordinates back via division with remainder
def f(i,j):
    return i*width+j

#create a weighted graph by speed 
def generate_matrix(imag):
    G=nx.Graph()
    for i in range(heigth):
        for j in range(width):
            N=f(i,j)
            G.add_node(N)
            
    for Node in G:
        i=Node//width
        j=Node%width
        if i!=0 and i!=heigth-1 and j!=0 and j!=width-1:
            G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if i==0 and j!=0 and j!=width-1:
            G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if i==heigth-1 and j!=0 and j!=width-1:
            G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if  j==0 and i!=0 and i!=heigth-1:
            G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if  j==width-1 and i!=0 and i!=heigth-1:
            G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if i==0 and j==0:
            G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if i==0 and j==width-1:
            G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i+1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if i==heigth-1 and j==0:
            G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j+1),weight=1/vitesse_zombie(tuple(imag[i,j])))
        if i==heigth-1 and j==width-1:
            G.add_edge(f(i,j),f(i-1,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i-1,j),weight=1/vitesse_zombie(tuple(imag[i,j])))
            G.add_edge(f(i,j),f(i,j-1),weight=1/vitesse_zombie(tuple(imag[i,j])))
    return G

grph = generate_matrix(image)
path=nx.astar_path(grph,f(1035,3810),f(503,53),'weight') #generating the path 
test(path,image) #testing said path
timer(path,image) #time required for the zombies to arrive to Brest

#downloading all the grb files

import urllib2
from datetime import datetime

def get_files(): 
	month='200403'
	day='20040303'
		                                                          
	gfl='gflsanl_3_'
	hour=['_0000_000.grb','_0600_000.grb','_1200_000.grb','_1800_000.grb']
	url='https://nomads.ncdc.noaa.gov/data/gfsanl/'

	for i in range((2017-2004+1)*366*8):
		urlf=url+month+'/'+day+'/'+gfl+day+hour[i%4]
		response = urllib2.urlopen(urlf)
		html = response.read()
		day=str(datetime.datetime.strptime(day,'%Y%m%d').date() + datetime.timedelta(days=1))
		month=day[:5]
	return print('done')

#this doesnt seem possible since each day has 4 files, one year would mean 365*4 with roughly 22MB per file the ressources available at the school arent enough to handle that load

#longitude and latitude to pixel x and y

import math

def get_x(width, lng):
    return int(round(math.fmod((width * (180.0 + lng) / 360.0), (1.5 * width))))

def get_y(width, height, lat):
    lat_rad = lat * math.pi / 180.0
    merc = 0.5 * math.log( (1 + math.sin(lat_rad)) / (1 - math.sin(lat_rad)) )
    return int(round((height / 2) - (width * merc / (2 * math.pi))))

