import gym
from gym.spaces.box import Box
import numpy as np
from geometry import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()


class TrackEnv(gym.Env):
    def __init__(self, track_file,
                 nb_sensors=3,
                 sensor_limit=0.3,
                 max_car_acceleration=0.01,
                 max_car_speed=0.01,
                 max_car_deviation_angle=np.pi / 10,
                 car_width=0.015,
                 car_length=0.03,
                 max_clock_time=1000):
        """
        init the track with the file in map_file
        observation = distance given by each sensor + car speed : vector of size nb_sensors + 2
        action = accélération et direction d'accélération.
        La voiture peut accélérer et choisir une direction dans laquelle accélérer
        sensor_limit est la distance maximale que peut détecter le senseur

        # track est un tableau numpy (2*n*2) ou track[0] est une liste des coordonnées des points du bord gauche de la piste
        # track[1] est la liste des coordonnées des points du bord droit de la piste
        # les points peuvent être espacés : on trace les droites qui relient chaque point pour former la piste.
        #  track[:,i:i+1,:] est un quadrilatère. On peut voir la piste comme une liste de quadrilatère consécutifs.
        # Le but est que la voiture avance successivement dans les quadrilatères et aille le plus loin possible.
        # Les quadrilatères sont indexés par i. On garde à chaque itération l'index du quadrilatère dans lequel la voiture est
        # dans une variable car_check_point
        The track width is normalized at 0.1.
        le premier point et dernier point doivent être le même pour la track (circuit fermé)
        """
        ## LOADING TRACK
        self.track = np.load(track_file)
        if (self.track[:, -1, :] != self.track[:, 0, :]).any():
            a = self.track
            self.track = np.zeros((2, self.track.shape[1] + 1, 2))
            self.track[:, :-1, :] = a
            self.track[:, -1, :] = a[0]

        ##  ENV PARAMETERS
        self.nb_checkpoints = self.track.shape[1] - 1
        self.clock_time = 0
        self.max_clock_time = max_clock_time
        self.ax = None
        self.init_figure = False

        ## CAR FIXED PARAMETERS
        assert nb_sensors >= 2, 'Car must have more than 2 sensors per side.'
        self.nb_sensors = nb_sensors  # each side of the car has nb_sensors. at least 2.
        self.sensor_limit = sensor_limit
        self.car_width = car_width
        self.car_length = car_length
        self.max_car_acceleration = max_car_acceleration
        self.max_car_speed = max_car_speed
        self.max_car_deviation_angle = max_car_deviation_angle  # the car deviates from tout droit by a deviation angle.
        self.car_bounds = [np.array([-car_width, car_length]) / 2,
                           np.array([car_width, car_length]) / 2,
                           np.array([car_width, -car_length]) / 2,
                           np.array([-car_width, -car_length]) / 2]

        ## CAR PARAMETERS
        self.car_position = np.zeros(2)
        self.car_speed = np.zeros(2)  # la voiture est toujours orientée dans le sens de sa vitesse.
        self.car_checkpoint = 0  # to keep track of where the car is on the track
        self.real_car_bounds = None
        self.sensors = [{'position': self.car_bounds[0], 'angle': (np.cos(t), np.sin(t))}
                        for t in np.linspace(np.pi, np.pi/2, self.nb_sensors)] + \
                        [{'position': self.car_bounds[1], 'angle': (np.cos(t), np.sin(t))}
                         for t in np.linspace(np.pi/2, 0, self.nb_sensors)]
        self.sensor_data = []

        ## FOR GYM COMPATIBILITY
        self.observation_space = Box(low=np.array(2 * nb_sensors * [0] + 2 * [-max_car_speed]),
                                     high=np.array(2 * nb_sensors * [sensor_limit] + 2 * [max_car_speed]))
        self.action_space = Box(low=np.array([-max_car_acceleration, - max_car_deviation_angle]),
                                high=np.array([max_car_acceleration, max_car_deviation_angle]))

    def get_quad(self, checkpoint_index):
        # check_point index is computed modulo the number of checkpoints
        # quad[i][j] : i track gauche ou droite, j premier point ou dernier point
        idx = checkpoint_index % self.nb_checkpoints
        return self.track[:, idx: idx + 2, :]

    def sense(self, p, v):
        """
        :return: max distance that we can drive starting from point towards direction direction
        without crashing into a wall of the track.
                """
        j = 0
        while True:
            quad = self.get_quad(self.car_checkpoint + j)
            p0, p1, p2, p3 = quad[0, 0], quad[0, 1], quad[1, 1], quad[1, 0]
            if direct_order(p0 - p, v, p3 - p):
                j -= 1
            elif direct_order(p2 - p, v, p1 - p):
                j += 1
            else:
                a1l, a2l = intersection_distance(p, v, p0, p1)
                a1r, a2r = intersection_distance(p, v, p2, p3)
                if 0 <= a2l <= 1 and a1l > 0:
                    return min(a1l, self.sensor_limit)
                elif 0 <= a2r <= 1 and a1r > 0:
                    return min(a1r, self.sensor_limit)
                return None

    def observation(self):
        self.sensor_data = []
        for sensor in self.sensors:
            p = self.car_position + rotate(sensor['position'], self.car_speed)
            a = rotate(sensor['angle'], self.car_speed)
            self.sensor_data.append(self.sense(p, a))
        return self.sensor_data + list(self.car_speed)

    def step(self, action):
        self.clock_time += 1
        (acceleration, car_deviation_angle) = action
        v, max_v = self.car_speed, self.max_car_speed
        self.car_speed = normalize((1 + acceleration) * v + car_deviation_angle * rotate90(v)) * max_v
        self.car_position += self.car_speed

        # updating car checkpoint (la voiture peut reculer d'un quadrilatère ou avancer d'un quadrilatère
        current_quad = self.get_quad(self.car_checkpoint)
        previous_quad = self.get_quad(self.car_checkpoint - 1)
        next_quad = self.get_quad(self.car_checkpoint + 1)
        reward = 0
        if is_in_quad(self.car_position, previous_quad):
            self.car_checkpoint -= 1
            reward = -1
        elif is_in_quad(self.car_position, next_quad):
            self.car_checkpoint += 1
            reward = 1

        new_state = self.observation()
        done = (self.clock_time > self.max_clock_time) or (None in new_state)
        info = {'time': self.clock_time}  # we don't need this now
        if None in new_state:
            reward = -10
        return new_state, reward, done, info

    def render(self, mode='human'):
        # Rendu graphique.
        # uses car position and car speed to get orientation.
        if not self.init_figure:
            # init figure
            self.init_figure = True
            fig = plt.figure()
            fig.canvas.set_window_title('TrackMaster')
            self.ax = fig.add_subplot(111)
            plt.axis('off')
            self.ax.plot(self.track[0, :, 0], self.track[0, :, 1], color='b')
            self.ax.plot(self.track[1, :, 0], self.track[1, :, 1], color='b')
        else:
            #  keep track only
            self.ax.lines = self.ax.lines[:2]
        # draw new car
        real_car_bounds = np.array([self.car_position + rotate(self.car_bounds[i%4], self.car_speed) for i in range(5)])
        self.ax.plot(real_car_bounds[:,0], real_car_bounds[:, 1], color='green')

        # draw sensor
        if None not in self.sensor_data:
            for sensor, d in zip(self.sensors, self.sensor_data):
                p1 = self.car_position + rotate(sensor['position'], self.car_speed)
                p2 = p1 + d*rotate(sensor['angle'], self.car_speed)
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='grey', linestyle='dashed')
        plt.pause(1e-3)
        plt.draw()

    def reset(self):
        self.clock_time = 0
        self.car_checkpoint = np.random.randint(self.nb_checkpoints)
        quad = self.get_quad(self.car_checkpoint)
        self.car_position = (quad[0, 0] + quad[0, 1] + quad[1, 0] + quad[1, 1]) / 4 + np.random.rand(2)*0.01
        car_direction = (quad[0, 1] + quad[1, 1] - quad[0, 0] - quad[1, 0]) / 2 + np.random.rand(2)*0.01
        self.car_speed = normalize(car_direction)*self.max_car_speed*np.random.rand()
        return self.observation()

    def close(self):
        return
