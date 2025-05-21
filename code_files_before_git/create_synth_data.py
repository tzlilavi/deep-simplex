import numpy as np
import torch as nn
import matplotlib.pyplot as plt
import CFG
import random
from numpy import linalg as LA
import pickle
from tqdm import tqdm
import time
import gzip
import joblib
from functions import plot_P_speakers
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
CFG.set_mode('supervised')


np.random.seed(int(time.time()))
def plot_simplex2D_3(P, TH=0.5, type='P', title='P simplex'):

    plt.scatter(P[:, 0], P[:, 1], color='orange', label='All points')


    plt.title(title)
    plt.xlabel(type + '0')
    plt.ylabel(type + '1')
    plt.show()

### Try out
def synthesize_data_playground(method='ro-uni', noise_dim=1):
    J = 2
    L = 500
    K = 1000
    if method == 'ro-uni':
        Ro = np.sort(np.random.uniform(size=(L, J - 1 + noise_dim)))
        P = np.zeros((L, J + noise_dim))
        P[:, 0] = Ro[:, 0]
        P[:, 1:-1] = Ro[:, 1:] - Ro[:, 0:-1]
        P[:, -1] = 1 - Ro[:, -1]
        H = np.random.normal(size=(K, J))

    elif method == 'dirichlet':
        # alpha = np.random.lognormal(mean=0, sigma=0.3, size=(J + noise_dim))
        # P = np.random.dirichlet(alpha, size=L)

        alpha = np.random.lognormal(mean=0, sigma=0.3, size=(J))
        P = np.random.dirichlet(alpha, size=L)
        noise = np.random.uniform(size=L)
        P = np.column_stack((P, noise))
        P = P / P.sum(axis=1, keepdims=True) ## Normalize sum to 1
        n_clusters = np.random.randint(L / 50, L / 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=CFG.seed0)

        sorted_indices = np.argsort(kmeans.fit_predict(P))
        P = P[sorted_indices]


        ### Dirichlet sources with uniform noise
        # alpha = np.ones(J)
        # P_sources = np.random.dirichlet(alpha, size=L)
        # noise = np.random.uniform(size=L)
        # P = np.column_stack((P, noise))
        # P = P / P.sum(axis=1, keepdims=True) ## Normalize sum to 1
        # plt.scatter(P[:, 0], P[:, 1], color='orange', label='All points')
        # plt.show()

        # (speakers, plot_name, title, noise) = (P[0:100, :CFG.Q], 'model_P_speakers', 'model_P_speakers', P[0:100, -1])
        # plot_P_speakers(speakers, plot_name, CFG.figs_dir, title=title, noise=noise, save_flag=False,
        #                 show_flag=False, need_fig=False)
        # plt.tight_layout()
        # plt.show()


        H = np.random.normal(size=(K, J))


    elif method == 'identical-sources-dirichlet':
        alpha = np.ones(J + noise_dim)
        P = np.random.dirichlet(alpha, size=L)
        h = np.random.normal(size=J)
        H = np.array(K * [h])


    P_cumsum = np.cumsum(P, axis=1)
    if noise_dim == 1:
        H = np.concatenate((H, 1000 * np.ones(shape=(K, noise_dim))), axis=1)

    # Generate random values for each (l, k) to determine which column to choose
    random_values = np.random.rand(L, K)

    # Determine the chosen indices based on the random values and cumulative probabilities
    chosen_indices = np.argmax(random_values[:, :, None] < P_cumsum[:, None, :], axis=2)

    # Use advanced indexing to fill A based on chosen indices
    A = H[np.arange(K)[:, None], chosen_indices.T].T

    # Replace 1000 with normal samples
    A[A == 1000] = np.random.normal(size=np.sum(A == 1000))

    ## Build estimated W input using A
    W = np.dot(A, A.T) / K

    E_values, E_vectors = LA.eig(W)

    U = E_vectors[:, :J + 1]

    plot_simplex2D_3(mat=U, TH=1000, type='U_est', title='U EVD from estimated W')

    plt.scatter(U[:, 0], U[:, 1], color='orange', label='All points')


    plt.title('Noisy U EVD from estimated W')
    plt.xlabel('u0')
    plt.ylabel('u1')
    plt.show()  ###



## Create the synthesized data
def create_W_P(J, L, K, noise_dim=1):
    np.random.seed(int(time.time()))
    ## Create real P
    Ro = np.sort(np.random.uniform(size=(L, J - 1 + noise_dim)))
    P = np.zeros((L, J + noise_dim))
    P[:, 0] = Ro[:, 0]
    P[:, 1:-1] = Ro[:, 1:] - Ro[:, 0:-1]
    P[:, -1] = 1 - Ro[:, -1]

    ## Create Real W
    W_real = np.dot(P, P.T)
    W_real[range(L), range(L)] = 1

    ## Build A matrix using P
    H = np.random.normal(size=(K, J))
    if noise_dim == 1:
        H = np.concatenate((H, 1000 * np.ones(shape=(K, noise_dim))), axis=1)
    A = np.zeros((L, K))

    for l in range(0, L):
        for k in range(0, K):

            A[l, k] = np.random.choice(H[k, :], p=P[l, :])
            if A[l, k] == 1000:
                A[l, k] = np.random.normal()

    ## Build estimated W input using A
    W = np.dot(A, A.T) / K

    # W = nn.from_numpy(W)
    # P = nn.from_numpy(P)
    return W, P, W_real

def create_W_P_efficient(J, L, K, method='ro-uni', noise_dim=1, augment_simplex=0, seed=42):
    np.random.seed(seed)
    if method == 'ro-uni':
        Ro = np.sort(np.random.uniform(size=(L, J - 1 + noise_dim)))
        P = np.zeros((L, J + noise_dim))
        P[:, 0] = Ro[:, 0]
        P[:, 1:-1] = Ro[:, 1:] - Ro[:, 0:-1]
        P[:, -1] = 1 - Ro[:, -1]
        H = np.random.normal(size=(K, J))

    elif method == 'dirichlet':

        alpha = np.random.lognormal(mean=0, sigma=0.3, size=(J + noise_dim))
        P = np.random.dirichlet(alpha, size=L)
        H = np.random.normal(size=(K, J))

    elif method == 'dirichlet_Kmeans':

        alpha = np.random.lognormal(mean=0, sigma=0.3, size=(J + noise_dim))
        P = np.random.dirichlet(alpha, size=L)

        n_clusters = np.random.randint(L / 80, L / 40)
        kmeans = KMeans(n_clusters=n_clusters, random_state=CFG.seed0)

        sorted_indices = np.argsort(kmeans.fit_predict(P))
        P = P[sorted_indices]

        H = np.random.normal(size=(K, J))


    elif method == 'dirichlet_identical_sources':
        alpha = np.random.lognormal(mean=0, sigma=0.3, size=(J + noise_dim))
        P = np.random.dirichlet(alpha, size=L)
        h = np.random.normal(size=J)
        H = np.array(K * [h])


    if augment_simplex==1 and np.random.uniform()>0.5:
        speaker_augment = np.random.randint(J)
        num_max_points = 15
        argmax_points = np.argsort(P[:, speaker_augment])[-num_max_points:]
        P[argmax_points, :] = np.random.dirichlet(alpha, size=num_max_points)

    ## Create Real W
    W_real = np.dot(P, P.T)
    W_real[range(L), range(L)] = 1

    P_cumsum = np.cumsum(P, axis=1)

    if noise_dim == 1:
        H = np.concatenate((H, 1000 * np.ones(shape=(K, noise_dim))), axis=1)


    # Generate random values for each (l, k) to determine which column to choose
    random_values = np.random.rand(L, K)

    # Determine the chosen indices based on the random values and cumulative probabilities
    chosen_indices = np.argmax(random_values[:, :, None] < P_cumsum[:, None, :], axis=2)

    # Use advanced indexing to fill A based on chosen indices
    A = H[np.arange(K)[:, None], chosen_indices.T].T

    # Replace 1000 with normal samples
    A[A == 1000] = np.random.normal(size=np.sum(A == 1000))

    ## Build estimated W input using A
    W = np.dot(A, A.T) / K


    # _, E0 = np.linalg.eig(W)

    # plt.scatter(P[:, 0], P[:, 1], color='orange', label='All points')
    # plt.show()
    # plt.scatter(E0[:, 0], E0[:, 1], color='orange', label='All points')
    # plt.show()
    return W, P, W_real



def create_data_file(J, num_samples, method, noise_dim=1, augmented=0):
    import time
    L = CFG.N_frames
    K = CFG.H_freqbands


    data_dict = {'Ws': [], 'Ps': []}
    Ws = []
    start = time.time()
    base_seed = int(time.time())
    print(f"Creating dataset, J = {J}, L = {L}, noise = {noise_dim}, method: {method} for {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        seed = base_seed + i
        W, P, W_real = create_W_P_efficient(J, L, K, method=method, noise_dim=noise_dim, augment_simplex=augmented, seed=seed)
        data_dict['Ws'].append(W)
        data_dict['Ps'].append(P)


        # data_dict['Ws_real'].append(W_real)
    #

    # filename = f'{CFG.synth_data_path}/DataDict_J{J}_L{L}_noise{noise_dim}_dirichlet_identical_sources{num_samples}.joblib'
    filename = f'{CFG.synth_data_path}/DataDict_J{J}_L{L}_noise{noise_dim}_{method}{num_samples}_augmented{augmented}.joblib'

    joblib.dump(data_dict, filename, compress=('zlib', 3))

    time = time.time() - start
    print(f"Created data file with {num_samples} samples, in {time} seconds")


# create_W_P_efficient(CFG.Q, 200, CFG.H_freqbands, method='dirichlet', noise_dim=0, augment_simplex=True)
# create_data_file(J=3, num_samples=20000, method='dirichlet', noise_dim=0, augmented=0)












    #


