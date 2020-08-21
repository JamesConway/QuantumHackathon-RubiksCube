import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from projection import Quaternion, project_points
import random
import qrubik

"""

----------------------
2D Cube
----------------------

sticker indices:
       ┌──┬──┐
       │ 0│ 1│
       ├──┼──┤
       │ 2│ 3│
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │ 4│ 5│ 6│ 7│ 8│ 9│10│11│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │12│13│14│15│16│17│18│19│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │20│21│
       ├──┼──┤
       │22│23│
       └──┴──┘



face colors:
    ┌──┐
    │ 0│
 ┌──┼──┼──┬──┐
 │ 1│ 2│ 3│ 4│
 └──┼──┼──┴──┘
    │ 5│
    └──┘

moves:
[ U , U2, R ,  R2, F ,F2]

'''




----------------------
3D cube
----------------------

Sticker Representation 


Each face is represented by a length [5, 3] array:

  [v1, v2, v3, v4, v1]

Each face element is represented by a length [9, 3] array:

  [v1a, v1b, v2a, v2b, v3a, v3b, v4a, v4b, v1a]

In both cases, the first point is repeated to close the polygon.

Each face also has a centroid, with the face number appended
at the end in order to sort correctly using lexsort.
The centroid is equal to sum_i[vi].

Colors are accounted for using color indices and a look-up table.

With all faces in an NxNxN cube, then, we have three arrays:

  centroids.shape = (6 * N * N, 4)
  faces.shape = (6 * N * N, 5, 3)
  face_elements.shape = (6 * N * N, 9, 3)
  colors.shape = (6 * N * N,)

The canonical order is found by doing

  ind = np.lexsort(centroids.T)

After any rotation, this can be used to quickly restore the cube to
canonical position.
"""


        


class Cube:
    """Rubik's Cube Representation"""
    # define some attribues
    default_init_state=np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5])
 
    # move indices
    moveInds = { \
    "U": 0, "U2": 1 , "R": 2, "R2": 3, "F": 4, "F2": 5\
    }

    # move definitions
    moveDefs = np.array([ \
    [  2,  0,  3,  1,  6,  7,  8,  9, 10, 11,  4,  5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], \
     [  3,  2,  1,  0,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], \
    [  0,  7,  2, 15,  4,  5,  6, 21, 16,  8,  3, 11, 12, 13, 14, 23, 17,  9,  1, 19, 20, 18, 22, 10], \
    [  0, 21,  2, 23,  4,  5,  6, 18, 17, 16, 15, 11, 12, 13, 14, 10,  9,  8,  7, 19, 20,  1, 22,  3], \
    [  0,  1, 13,  5,  4, 20, 14,  6,  2,  9, 10, 11, 12, 21, 15,  7,  3, 17, 18, 19, 16,  8, 22, 23], \
    [  0,  1, 21, 20,  4, 16, 15, 14, 13,  9, 10, 11, 12,  8,  7,  6,  5, 17, 18, 19,  3,  2, 22, 23], \
    ])

    # piece definitions
    pieceDefs = np.array([ \
    [  0, 11,  4], \
    [  2,  5,  6], \
    [  3,  7,  8], \
    [  1,  9, 10], \
    [ 20, 14, 13], \
    [ 21, 15, 15], \
    [ 23, 18, 17], \
    ])

    default_plastic_color = 'black'
    default_face_colors = ["w", "#ffcf00",
                           "#00008f", "#009f0f",
                           "#ff6f00", "#cf0000",
                           "gray", "none"]
    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)
    face_element_width = 0.9
    face_element_margin = 0.5 * (1. - face_element_width)
    face_element_thickness = 0.001
    (d1, d2, d3) = (1 - face_element_margin,
                    1 - 2 * face_element_margin,
                    1 + face_element_thickness)
    base_face_element = np.array([[d1, d2, d3], [d2, d1, d3],
                             [-d2, d1, d3], [-d1, d2, d3],
                             [-d1, -d2, d3], [-d2, -d1, d3],
                             [d2, -d1, d3], [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face_centroid = np.array([[0, 0, 1]])
    base_face_element_centroid = np.array([[0, 0, 1 + face_element_thickness]])
    # apply a move to a state
    def doMove(self, move):
        return self.state[self.moveDefs[self.moveInds[m]]]

    # apply a string sequence of moves to a state
    def doAlgStr(self,alg):
        moves = alg.split(" ")
        for m in moves:
            if m in self.moveInds:
                self.state = self.state[self.moveDefs[self.moveInds[m]]]
        self._printCube()


    # print state of the cube
    def _printCube(self):
        print("      ┌──┬──┐")
        print("      │ {}│ {}│".format(self.state[0], self.state[1]))
        print("      ├──┼──┤")
        print("      │ {}│ {}│".format(self.state[2], self.state[3]))
        print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(self.state[4], self.state[5], self.state[6], self.state[7], self.state[8], self.state[9], self.state[10], self.state[11]))
        print("├──┼──┼──┼──┼──┼──┼──┼──┤")
        print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(self.state[12], self.state[13], self.state[14], self.state[15], self.state[16], self.state[17], self.state[18], self.state[19]))
        print("└──┴──┼──┼──┼──┴──┴──┴──┘")
        print("      │ {}│ {}│".format(self.state[20], self.state[21]))
        print("      ├──┼──┤")
        print("      │ {}│ {}│".format(self.state[22], self.state[23]))
        print("      └──┴──┘")


    # Define rotation angles and axes for the six sides of the cube
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(np.eye(3)[0], theta)
    for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(np.eye(3)[1], theta)
    for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    # define face movements
    facesdict = dict(F=z, B=-z,
                     R=x, L=-x,
                     U=y, D=-y)

    def __init__(self, N=2, plastic_color=None, face_colors=None):
        self.N = N
        if plastic_color is None:
            self.plastic_color = self.default_plastic_color
        else:
            self.plastic_color = plastic_color

        if face_colors is None:
            self.face_colors = self.default_face_colors
        else:
            self.face_colors = face_colors

        self.state = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5])
        #self.state = np.array(['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010', '01011', '01100', '01101', '01110', '01111', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111'])
        
        self._move_list = []
        self._solution = []
        self._initialize_arrays()

    def _initialize_arrays(self):
        # initialize centroids, faces, and face elements.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2. / self.N
        translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                                   -1 + (j + 0.5) * cubie_width, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Create arrays for centroids, faces, face elements, and colors
        face_centroids = []
        faces = []
        face_element_centroids = []
        face_elements = []
        colors = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            M = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face
                             + translations, M.T)
            face_elements_t = np.dot(factor * self.base_face_element
                                + translations, M.T)
            face_centroids_t = np.dot(self.base_face_centroid
                                      + translations, M.T)
            face_element_centroids_t = np.dot(self.base_face_element_centroid
                                         + translations, M.T)
            colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3),
                                          colors_i[:, None]])
            face_element_centroids_t = face_element_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            face_elements.append(face_elements_t)
            face_element_centroids.append(face_element_centroids_t)
            colors.append(colors_i)

        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._face_element_centroids = np.vstack(face_element_centroids)
        self._face_elements = np.vstack(face_elements)
        self._colors = np.concatenate(colors)

        self._sort_faces()

    def _sort_faces(self):
        # use lexsort on the centroids to put faces in a standard order.
        ind = np.lexsort(self._face_centroids.T)
        self._face_centroids = self._face_centroids[ind]
        self._face_element_centroids = self._face_element_centroids[ind]
        self._face_elements = self._face_elements[ind]
        self._colors = self._colors[ind]
        self._faces = self._faces[ind]

    def rotate_face(self, f, n=1, layer=0):
        self.doAlgStr(f)
        """Rotate Face"""
        if layer < 0 or layer >= self.N:
            raise ValueError('layer should be between 0 and N-1')

        try:
            f_last, n_last, layer_last = self._move_list[-1]
        except:
            f_last, n_last, layer_last = None, None, None

        if (f == f_last) and (layer == layer_last):
            ntot = (n_last + n) % 4
            if abs(ntot - 4) < abs(ntot):
                ntot = ntot - 4
            if np.allclose(ntot, 0):
                self._move_list = self._move_list[:-1]
            else:
                self._move_list[-1] = (f, ntot, layer)
        else:
            self._move_list.append(f)
        
        v = self.facesdict[f]
        r = Quaternion.from_v_theta(v, n * np.pi / 2)
        M = r.as_rotation_matrix()

        proj = np.dot(self._face_centroids[:, :3], v)
        cubie_width = 2. / self.N
        flag = ((proj > 0.9 - (layer + 1) * cubie_width) &
                (proj < 1.1 - layer * cubie_width))

        for x in [self._face_elements, self._face_element_centroids,
                  self._faces]:
            x[flag] = np.dot(x[flag], M.T)
        self._face_centroids[flag, :3] = np.dot(self._face_centroids[flag, :3],
                                                M.T)

    def draw_interactive(self):
        fig = plt.figure(figsize=(5, 5))
        fig.add_axes(InteractiveCube(self))
        return fig


class InteractiveCube(plt.Axes):
    def __init__(self, cube=None,
                 interactive=True,
                 view=(0, 0, 10),
                 fig=None, rect=[0, 0.16, 1, 0.84],
                 **kwargs):
        if cube is None:
            self.cube = Cube(3)
        elif isinstance(cube, Cube):
            self.cube = cube
        else:
            self.cube = Cube(cube)

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1, -1, 0),
                                                  -np.pi / 6)

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # add some defaults, and draw axes
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-2.0, 2.0)),
                           ylim=kwargs.get('ylim', (-2.0, 2.0)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteractiveCube, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Internal state variable
        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._event_xy = None  # store xy position of mouse event

        self._digit_flags = np.zeros(10, dtype=bool)  # digits 0-9 pressed

        self._current_rot = self._start_rot  #current rotation state
        self._face_polys = None
        self._face_element_polys = None

        self._draw_cube()

        # connect some GUI events
        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event',
                                       self._key_press)

        self._initialize_widgets()

        # write some instructions
        self.figure.text(0.05, 0.10,
                         "Press U/R/F keys to turn faces",
                         size=10)

    def _initialize_widgets(self):
        self._ax_reset = self.figure.add_axes([0.75, 0.12, 0.2, 0.075])
        self._btn_reset = widgets.Button(self._ax_reset, 'Reset View')
        self._btn_reset.on_clicked(self._reset_view)

        self._ax_solve = self.figure.add_axes([0.55, 0.12, 0.2, 0.075])
        self._btn_solve = widgets.Button(self._ax_solve, 'Solve Cube')
        self._btn_solve.on_clicked(self._solve_cube)

        self._ax_rando = self.figure.add_axes([0.55, 0.05, 0.4, 0.075])
        self._btn_rando = widgets.Button(self._ax_rando, 'Randomize Cube')
        self._btn_rando.on_clicked(self._randomize_cube)


    def _project(self, pts):
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self):
        face_elements = self._project(self.cube._face_elements)[:, :, :2]
        faces = self._project(self.cube._faces)[:, :, :2]
        face_centroids = self._project(self.cube._face_centroids[:, :3])
        face_element_centroids = self._project(self.cube._face_element_centroids[:, :3])

        plastic_color = self.cube.plastic_color
        colors = np.asarray(self.cube.face_colors)[self.cube._colors]
        face_zorders = -face_centroids[:, 2]
        face_element_zorders = -face_element_centroids[:, 2]

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._face_element_polys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plastic_color,
                                 zorder=face_zorders[i])
                sp = plt.Polygon(face_elements[i], facecolor=colors[i],
                                 zorder=face_element_zorders[i])

                self._face_polys.append(fp)
                self._face_element_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # subsequent call: update the polygon objects
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._face_element_polys[i].set_xy(face_elements[i])
                self._face_element_polys[i].set_zorder(face_element_zorders[i])
                self._face_element_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def rotate(self, rot):
        self._current_rot = self._current_rot * rot

    def rotate_face(self, face, turns=1, layer=0, steps=1):
        if not np.allclose(turns, 0):
            for i in range(steps):
                self.cube.rotate_face(face, turns * 1. / steps,
                                      layer=layer)
                self._draw_cube()

    def _reset_view(self, *args):
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._draw_cube()

    def _solve_cube(self, *args):
        print("---------------------")
        print("--SOLVER INITIATED---")
        print("---------------------")
        move_list = self.cube._move_list[:]
        # traverse in the string
        # pass move state to Q solver
        # pass move state to Q solver
        if(len(move_list)==1):
            cu_state = '' 
            cu_state  += "qrubik.PERM_"+move_list[0]+"1"  
        elif(len(move_list)>1):
            cu_state = '' 
            cu_state  += "qrubik.PERM_"+move_list[0]+"1"  
            for ele in move_list[1:]:
                cu_state  += "+qrubik.PERM_"+ele+"1"
        # return string   

        # return string   
        #print(cu_state)
        cube_init = qrubik.cube_conf_init(eval(cu_state))
        counts = qrubik.simulate_experiment(cube_init)
        qrubik.plot_histogram(counts) 
        
        moves = qrubik.interpret_counts_for_gui(counts).split(" ")
        print("---------------------")
        print("--SOLVE MOVE LIST---")
        print("-------", end = '')
        print(' '.join(moves), end = '')
        print("--------")
        print("---------------------")
        for m in moves:
            if(m=='U'):
                self.rotate_face('U', 1)
                self._draw_cube()
            elif(m=='U2'):
                self.rotate_face('U', 1)
                self.rotate_face('U', 1)
                self._draw_cube()
            elif(m=='R'):
                self.rotate_face('R', 1)
                self._draw_cube()     
            elif(m=='R2'):
                self.rotate_face('R', 1)
                self.rotate_face('R', 1)
                self._draw_cube()
            elif(m=='F'):
                self.rotate_face('F', 1) 
                self._draw_cube()    
            elif(m=='F2'):
                self.rotate_face('F', 1)
                self.rotate_face('F', 1)
                self._draw_cube()
        self.cube._move_list = []
        self._draw_cube()
        print("---------------------")
        print("--SOLVER COMPLETE---")
        print("--------------------")

    def _randomize_cube(self, *args):
        num_moves = random.randint(1,20)
        for move in range(num_moves):
            self.rotate_face(random.choice(['U','R','F']), 1)
        self._draw_cube()
        print(self.cube._move_list[:])
        

    def _key_press(self, event):
        """Handler for key press events"""
        if event.key.isdigit():
            self._digit_flags[int(event.key)] = 1
        elif event.key == 'right':
            self.rotate(Quaternion.from_v_theta(self._ax_LR,
                                                5 * self._step_LR))
        elif event.key == 'left':
            self.rotate(Quaternion.from_v_theta(self._ax_LR,
                                                -5 * self._step_LR))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                5 * self._step_UD))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                -5 * self._step_UD))
        elif event.key.upper() in 'URF':
            self.rotate_face(event.key.upper(), 1)
        self._draw_cube()



    def _mouse_press(self, event):
        """Handler for mouse button press"""
        self._event_xy = (event.x, event.y)
        if event.button == 1:
            self._button1 = True
        elif event.button == 3:
            self._button2 = True

    def _mouse_release(self, event):
        """Handler for mouse button release"""
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event):
        """Handler for mouse motion"""
        if self._button1 or self._button2:
            dx = event.x - self._event_xy[0]
            dy = event.y - self._event_xy[1]
            self._event_xy = (event.x, event.y)

            if self._button1:
                rot1 = Quaternion.from_v_theta(self._ax_UD,
                                               self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(self._ax_LR,
                                               self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()

if __name__ == '__main__':
    N=2
    c = Cube(N)
    c._printCube()
    c.draw_interactive()
    
    plt.show()
