import numpy as np
import matplotlib.pyplot as plt
from numba import jit,prange

import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Random initial state
def initial_state(L,string):
    # ----- Criando o estado inicial ----
    ### (int, string) -> array LxL
    # ----- input ------
    # L : tamanho da grad
    # string : grade aleatoria ou alinhada
    # ----- output ------
    # grade feita

    if string == "aligned":
            state = np.full((L,L), 1,dtype=float)
    elif string == "random":
        state = 2 * np.random.randint(2, size=(L,L)) - 1
    else:
        return print("write aligned or random")
    return state

# Monte Carlo algorithm
@jit(nopython=True,fastmath=True,nogil=True)
def MC_met(config,beta,J,h):
    # ----- Calculando se deve ou não girar o spin ----
    ### (array LxL, float, float, float) -> array LxL
    # ------ input -------
    # config : grade
    # beta : inverso da temperatura
    # J : constante de acoplamento
    # h : intesidade do campo magnetico externo
    # ------ output -------
    # grade atualizada ou mantida

    L = len(config)
    a = np.random.randint(0, L)
    b = np.random.randint(0, L)
    sigma =  config[a, b]
    neighbors = J*(config[(a+1)%L, b] + config[a, (b+1)%L] + config[(a-1)%L, b] + config[a, (b-1)%L]) + h
    del_E = 2*sigma*neighbors
    if del_E < 0:
        sigma *= -1
    elif np.random.rand() < np.exp(-del_E*beta):
        sigma *= -1
    config[a, b] = sigma
    return config

@jit(nopython=True,fastmath=True,parallel=True)
def mag(config):
    ### ----- Calculo da magnetizacao ----
    # (array Lxl) ---> float
    # ----- input ------
    # config : grade
    # ----- output -----
    # magnetizacao total
    return np.sum(config)

# Energia total
@jit(nopython=True,fastmath=True,nogil=True)
def Total_Energy(config, J,h):
    ### ----- Calculo da energia total ----
    # (array LxL, float, float) -> (float)
    # ---- input ----
    # config : grade
    # J : constante de acoplamento
    # h : intensidade do campo magnético
    # ---- output ----
    # energia total

    L = len(config)
    total_energy = 0
    for i in range(L):
        for j in range(L):
            S = config[i,j]
            nb = J*(config[(i+1)%L, j] + config[i, (j+1)%L] + config[(i-1)%L, j] + config[i, (j-1)%L])+ h
            total_energy += -nb * S
    return (J*total_energy/4) # energia total


@jit(nopython=True, fastmath=True, nogil=True)
def temporalseries(T, config, iterations, iterations_fluc, fluctuations, J, h):
    ### ------ Realizacao das varreduras -------
    # (float, array LxL, int, int, int, float, float, int) -> (array LxL, float, float)
    # ------- input --------
    # T : temperatura
    # config : grade
    # iterations : iteracoes para alcancar o equilibrio termico antes de fazer varreduras
    # iterations_fluc: iteracoes em uma varredura
    # fluctuations : numero de varreduras
    # J : constante de acoplamento
    # h : intesidade do campo magnetico externo
    # ------- output -------
    # config atualizada
    # energia por spin
    # magnetizacao por spin

    beta = 1 / T
    mag_sum = 0
    ene_sum = 0

    # equilibrio termico
    for i in range(iterations):
        config = MC_met(config, beta, J, h)

    # varreduras
    for z in range(fluctuations):

        for i in range(iterations_fluc):
            config = MC_met(config, beta, J, h)
        mag_sum += mag(config)
        ene_sum += Total_Energy(config, J, h)

    return config, ene_sum / (fluctuations * iterations_fluc), mag_sum / (fluctuations * iterations_fluc)

### ------------- Rodando simulacoes ---------------------------------------

T_1 = np.linspace(1.5, 2.21, 10)
T_2 = np.linspace(2.21, 2.5, 15)
T_3 = np.linspace(2.5, 3.5, 5)
Temps = np.hstack((T_1, T_2, T_3)).ravel()  # temperaturas

N = [20, 40]  # tamanhos
J = 1  # constante de acoplamento
h = 0  # intesidade do campo mag externo

energy_data = np.zeros((len(N), len(Temps)))  # guardar as informacoes da magnetizacao
mag_data = np.zeros((len(N), len(Temps)))  # guardar as informcaoes da energia

# loop dos tamahos
for i in range(len(N)):

    config = initial_state(int(N[i]), "aligned")

    # loop das temperaturas
    for t in range(len(Temps)):
        config, energy_data[i, t], mag_data[i, t] = temporalseries(Temps[t], config, N[i] ** 2, N[i] * N[i], 1000, J, h)

# -------------------------------------------------------------------------------

# ------- Plot da magnetizacao media por temperatura -----
# Todos os plots feitos seguiram essa mesma estrutura, a diferenca
# foi apenas de mudar o array que era feito o loop

# plt.figure(2,figsize=(15,9),tight_layout=True)
# j = 0
# h = 0
# colors = plt.cm.jet(np.linspace(0,1,len(Temps)))
# for i in range(len(N)):
#     plt.plot(Temps, mag_data[i,:], color=colors[t], marker="s",label=r"$h={},j={},N={}$".format(h,j,N[i]), linewidth=0.7,markersize=7.5)
#
# plt.axvline(2.26, color="green", label=r"Temperatura Crítica")
# plt.plot(np.linspace(0.1,2.26,100000), g(np.linspace(0.1,2.26,100000)),color='purple', label='Solução analítica de Onsager h=0')
# plt.legend(prop={'size' : 20})
# plt.xlabel(r'$T$')
# plt.ylabel(r'$\left\langle m \right\rangle$')
# plt.tight_layout()
# plt.grid()
# #plt.savefig("Magnetization_per_site_HvsT.png")
# plt.show()

# ------------------------------------------------------------------------------

# ----------- ajuste do expo critico ------------------------------------------
## Utiliza a funcao curve_fit do scipy que utiliza MMQ para estimar os params.
from scipy.optimize import curve_fit

# funcao a ser ajustada
def func(x,a,b):
    return a*(1 - x/2.3)**b

# pontos antes do ponto critico
tpos = np.where((2.26 > Temps))[0]
xdata = Temps[tpos]
ydata = mag_data[:,tpos][0]

# ---------- plot do ajuste -----------------------------------
# plt.figure(figsize=(8,6), dpi=100)
# plt.plot(xdata, ydata, color='black', marker="s",label=r"${}x{}$".format(int(N[i]),int(N[i])), linewidth=0.3,markersize=7.5)
# popt, pcov = curve_fit(func, xdata, ydata, p0=[1,.1])

# plt.plot(xdata, func(xdata, *popt), 'r-', linewidth=0.9 , label=r"$f(x)=a(1-T/T_c)^b$")
# plt.legend(prop={'size' : 20})
# plt.xlabel(r'$\frac{k_B T}{J}$')
# plt.ylabel(r'$\left\langle m \right\rangle$')
# plt.grid()

# --------------------------------------------------------------------

# ------------------ PCA --------------------------------------------
# PCA para verificar ha transicao de fase e onde

@jit(nopython=True, fastmath=True, nogil=True)
def Matrix_X(Temps, config, iterations, J):
    # ------- Criando a matrix X -------
    # (array T, array LxL, int, float) -> array LxL
    # --------- input ---------
    # Temps : array com as temperaturas
    # config : grade
    # iterations : iteracoes para o equilibrio termico
    # J: constante de acoplamento
    # ------ output ----------
    # array X com as simulacoes para fazer o PCA
    X = np.zeros((len(Temps), len(config) * len(config)))

    for t in range(len(Temps)):

        beta = 1 / Temps[t]

        # equilibrio termico
        for i in range(iterations):
            config = MC_met(config, beta, J)

        X[t, :] = np.reshape(config, (len(config) * len(config)))

    return X

J = 1     # J
n = 50    # tamanho da grade
iterations = (n*n)*n    # Iterations to thermal equilibrium
T_1 = np.linspace(1.6,2.21,100)
T_2 = np.linspace(2.21,2.4,100)
T_3 = np.linspace(2.5,2.9,100)
Temps = np.hstack((T_1,T_2,T_3 )).ravel() # array das temperaturas

config = initial_state(n,"aligned")

X = Matrix_X(Temps, config,iterations,J)

from sklearn.decomposition import PCA

# PCA
pca = PCA()
XPCA = pca.fit_transform(X)

# ------- Plot da primeira e segunda componente ----------------------------
# plt.figure(figsize=(12,6),dpi=100)
#
# plt.scatter(XPCA[:,0], XPCA[:,1],c=Temps, cmap='viridis', alpha=0.5)#,edgecolors='black', linewidth=.5, s=50)
#
# plt.colorbar()
# plt.ylabel(r'$P_2$')
# plt.xlabel(r'$P_1$')
# plt.grid()
#
# plt.show()
# ------------------------------------------------------------------
# --------------- plot da variance explica por componente pca -------
# plt.figure(figsize=(10,5), dpi=100)
# plt.plot(pca.explained_variance_ratio_)
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# -------------------------------------------------------------------