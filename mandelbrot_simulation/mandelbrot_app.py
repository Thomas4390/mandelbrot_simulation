import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import imageio
import base64



st.title("Mandelbrot Simulation")

st.sidebar.title("Settings")
n = st.sidebar.slider("Number of iterations", min_value=1, max_value=60, value=15)
zoom_factor = st.sidebar.slider("Zoom factor", min_value=1, max_value=10, value=2)
res = st.sidebar.slider("Resolution of image", min_value=50, max_value=500, value=200)
x = st.sidebar.number_input("Choose a x coordinate to zoom in:", value=-0.235125, step=0.01)
y = st.sidebar.number_input("Choose a y coordinate to zoom in:", value=0.827215, step=0.01)
max_iter = st.sidebar.slider("Number of max iteration before divergence", min_value=20, max_value=50, value=20)
    
def diverge(c:complex, max_iter=max_iter)->int:
    c = complex(*c)
    z = 0

    for i in range(max_iter):
        z = np.power(z, 2) + c

        if z.real + z.imag >= 4:
            return i
    return 0


def make_grid(bbox, res=res)->np.ndarray:

    x_min, x_max, y_min, y_max = bbox
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    xx, yy = np.meshgrid(x, y)
    coords = np.dstack([xx.flatten(), yy.flatten()])[0]

    return coords

# ToDo : Remove the ticks from the image
# ToDo : Add a button to save the gif
# ToDo : Add a menu to choose a cmap from predefined ones. 
# (https://matplotlib.org/examples/color/colormaps_reference.html) 


def make_mandelbrot(coords:np.ndarray, div, plotting=True, filename=False)->np.ndarray:

    mb = np.array([div(c) for c in coords])
    res = np.sqrt(coords.shape[0]).astype(int)
    mb = np.reshape(mb, (res, res))

    if plotting:
        plt.figure(figsize=(7, 7))
        plt.imshow(mb, cmap='gnuplot2')
        

        if filename:
            plt.savefig(f"figures/{filename}.png")

    return mb


def zoom(bbox: tuple, p, zoom_factor=zoom_factor) -> tuple:
    p = x, y
    zoom_factor *= 2
    x_min, x_max, y_min, y_max = bbox
    width = (x_max - x_min) / zoom_factor
    height = (y_max - y_min) / zoom_factor

    return x-width, x+width, y-height, y+height

def simulation(p, n=n, zoom_factor=zoom_factor, max_iter=max_iter):
    
    bbox = (-2.1, 1, -1.3, 1.3)

    for i in range(n):
        bbox = zoom(bbox, p, zoom_factor=zoom_factor)
        coords = make_grid(bbox)
        div = partial(diverge, max_iter=max_iter + 5*i)
        filename = f"mb_zoom_{i * zoom_factor}"
        make_mandelbrot(coords, div, filename=filename)
    


def sort_filenime_by_number():

    filenames = os.scandir('figures')
    filenames_list = [fn.name for fn in filenames if fn.name != '.DS_Store']
    return sorted(filenames_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))


# ToDo : Create a function that create the folder 'figures' if it doesn't exist
# and delete it after. This way we don't have to delete the folder everytime we run the app.

def make_gif(filenames_list):
    images = []
    for filename in filenames_list:
        images.append(imageio.v2.imread(f"figures/{filename}"))
    imageio.mimsave(f"zoom.gif", images)

def read_gif():
        file_ = open("zoom.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )

if __name__ == '__main__':
    simulation((x, y))
    filenames_list = sort_filenime_by_number()
    make_gif(filenames_list)
    read_gif()