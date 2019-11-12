import cv2
import matplotlib.pyplot as plt
import numpy as np

def enable_zoom_with_mouse_wheel(ax,base_scale = 1.5):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    #return the function
    return zoom_fun

def click_bees(ax, display_instructions = True):

    n = -1  # wait for manual termination
    show_clicks = True
    timeout = 0
    mouse_add = 1  # left click adds points
    mouse_pop = 3  # right click removes points
    mouse_stop = None  # middle click stops input; if None, rely on 'return' key to terminate when done clicking
    # The keyboard can also be used to select points in case your mouse does not have one or more of the buttons.
    # The delete and backspace keys act like right clicking (i.e., remove last point), the enter key terminates input
    # and any other key (not already used by the window manager) selects a point.

    if display_instructions:
        print('\n\nUSE MOUSE TO MARK CENTER OF TEMPLATE\n\n',
              '\tLEFT CLICK:   select petiole of bee\n'
              '\tRIGHT CLICK:  delete point\n'
              '\tENTER:        exit\n',
              '\tSCROLL WHEEL: zoom in/out\n\n'
              '\tNOTE: the first click is sometimes ignored\n\n')

    plt.sca(ax)
    pts = plt.ginput(n=n, timeout=0, show_clicks=show_clicks, mouse_add=mouse_add, mouse_pop=mouse_pop, mouse_stop=mouse_stop)
    return pts

class LineDrawer(object):
    lines = []
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y)
        ax.figure.canvas.draw()

        self.lines.append(line)

FID = '/Users/Jake/Documents/Python/trackingBees/exampleFrameAndTemplate/wholeFrame.png'
img = cv2.imread(FID, cv2.IMREAD_GRAYSCALE)
print(img)

plt.imshow(img)
ax = plt.gca()
scale = 1.5
f = enable_zoom_with_mouse_wheel(ax, base_scale=scale)
pts = np.array(click_bees(ax))
print(pts)

# Loop over data points; create box from errors at each point
for x, y in zip(xdata, ydata, xerror.T, yerror.T):
    rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
    errorboxes.append(rect)

# Create patch collection with specified colour/alpha
pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                     edgecolor=edgecolor)

# Add collection to axes
ax.add_collection(pc)

# now loop through array of points, draw square roi centered on bee pts, crop images with roi, save templates to file





#plt.show()


