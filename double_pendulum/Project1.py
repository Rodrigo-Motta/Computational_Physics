import numpy as np
from numba import jit, int64, float64
import matplotlib.pyplot as plt
# from numba.experimental import jitclass

@jit(nopython=True)
def equations1(x1, v1, x2, v2, m1, m2, l1, l2):
    g = 9.8
    m1 = m1
    m2 = m2

    dx1dt = (((v1) / (m1 * l1**2)) - ((v2 * np.cos(x1 - x2)) / (m1 * l1 * l2)))
    dx2dt = (((v2) / (m2 * l2**2)) - ((v1 * np.cos(x1 - x2)) / (m1 * l1 * l2)))

    dv1dt = ((-m1 * g * l1 * np.sin(x1)) - ((v1 * v2 * np.sin(x1 - x2)) / (m1 * l1 * l2)))
    dv2dt = ((-m2 * g * l2 * np.sin(x2)) + ((v1 * v2 * np.sin(x1 - x2)) / (m1 * l1 * l2)))

    return dx1dt, dv1dt, dx2dt, dv2dt

@jit(nopython=True)
def equations2(x1, v1, x2, v2, m1, m2, l1, l2):
    g = 9.8
    m1 = m1
    m2 = m2

    dx1dt = ( (l2*v1) - l1*v2*np.cos(x1 - x2)) / (((l1**2)*l2)*(m1 + m2*(np.sin(x1 - x2)**2)))
    dx2dt = ((-l2*v1*np.cos(x1 - x2)) +
             (l1*(1 + (m1/m2))*v2)) / ((l1*(l2**2))*(m1 + m2*(np.sin(x1-x2)**2)))

    A = (v1*v2*np.sin(x1 - x2)) / ((l1*l2)*(m1 + m2*(np.sin(x1-x2)**2)))
    B = ( (((l2**2)*m2*(v1**2)) + (l1**2)*(m1 + m2)*(v2**2)
          - l1*l2*m2*v1*v2*np.cos(x1 - x2)) / \
        (2*(l1**2)*(l2**2)*(m1 + m2*(np.sin(x1 - x2)**2))**2) )*np.sin(2*(x1 - x2))

    dv1dt = (-(m1+m2)*g*l1*np.sin(x1) - A + B)
    dv2dt = (-(m2*g*l2*np.sin(x2)) + A - B)

    return dx1dt, dv1dt, dx2dt, dv2dt

@jit(nopython=True)
def RK2(data, m1, m2, l1, l2, dt, equations):

    for n in range(len(data[0,:]) - 1):

        if data[0, n] < -np.pi:
            data[0, n] = data[0, n] + 2 * np.pi
        if data[0, n] > np.pi:
            data[0, n] = data[0, n] - 2 * np.pi

        if data[2, n] < -np.pi:
            data[2, n] = data[2, n] + 2 * np.pi
        if data[2, n] > np.pi:
            data[2, n] = data[2, n] - 2 * np.pi

        # K1
        k1x1, k1v1, k1x2, k1v2 = equations(data[0, n], data[1, n], data[2, n], data[3, n], m1, m2, l1, l2)

        # Meio passo

        x_1 = data[0, n] + (k1x1*dt)/2
        v_1 = data[1, n] + (k1v1*dt)/2
        x_2 = data[2, n] + (k1x2*dt)/2
        v_2 = data[3, n] + (k1v2*dt)/2

        # K2
        k2x1, k2v1, k2x2, k2v2 = equations(x_1, v_1, x_2, v_2, m1, m2, l1, l2)

        # Atualização
        data[0, n + 1] = data[0, n] + (k2x1*dt)
        data[1, n + 1] = data[1, n] + (k2v1*dt)
        data[2, n + 1] = data[2, n] + (k2x2*dt)
        data[3, n + 1] = data[3, n] + (k2v2*dt)

    return data

# spec = [
#     ('m1', int64),
#     ('m2', int64),
#     ('l1', int64),
#     ('l2', int64),# a simple scalar field
#     ('data', float64[:, :]),# an array field
# ]

#@jitclass(spec)
class double_pendulum:

    def __init__(self, m1, m2, l1, l2):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.data = np.zeros((4, 1))


    def system(self, x10, v10, x20, v20, Tf, dt, equations="m1>>m2"):

        tvec = np.arange(0.0, Tf + dt, dt)
        data = np.zeros((4, np.size(tvec)))  # 0=x1, 1=v1, 2=x2, 3=v2
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        # condicoes iniciais

        data[0, 0] = x10
        data[1, 0] = v10
        data[2, 0] = x20
        data[3, 0] = v20

        if equations == "m1>>m2":
            self.data = RK2(data, self.m1, self.m2, self.l1, self.l2, dt, equations1)
        if equations == "m1=m2":
            self.data = RK2(data, self.m1, self.m2, self.l1, self.l2, dt, equations2)

    def poicare_section(self, phase_space='m2'):
        section = dict({'theta' : np.array([]), 'ptheta' : np.array([])})

        if phase_space == 'm2':
            for n in range(self.data[0, :].size-1):
                if self.data[0,n] < 0 and self.data[0,n + 1] > 0:
                    section['theta'] = np.append(section['theta'], self.data[2,n])
                    section['ptheta'] = np.append(section['ptheta'], self.data[3,n])
            return section


        if phase_space == 'm1':
            for n in range(self.data[0, :].size-1):
                if self.data[2,n] < 0 and self.data[2,n + 1] > 0:
                    section['theta'] = np.append(section['theta'], self.data[0,n])
                    section['ptheta'] = np.append(section['ptheta'], self.data[1,n])
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # ax.plot(section['theta'], section['ptheta'], '.', color='black', markersize=0.5)
            # plt.show()
            return section
    #
    #     # TA ERRADO!
    #     mask = np.array([])
    #     if phase_space=='m2':
    #
    #         for i in range(self.data[0,:].size):
    #         mask = np.append([((self.data[2,:] < 0) & (self.data[3,:] > 0))], mask)
    #         section = self.data[:,(mask > 0)]
    #         fig = plt.plot(section[0, :], section[1, :], '.', color='black', markersize=0.5)
    #         plt.show()
    #     if phase_space =='m1':
    #         mask = np.append([((self.data[0,:] < 0) & (self.data[1,:] > 0))], mask)
    #         section = self.data[:,(mask > 0)]
    #         fig = plt.plot(section[2, :], section[3, :], '.', color='black', markersize=0.5)
    #         plt.show()
    #     return fig



