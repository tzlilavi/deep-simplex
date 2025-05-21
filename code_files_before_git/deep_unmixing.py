import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from itertools import product
import random
import matplotlib.pyplot as plt
import os
import json
import torchvision.models as models
import torch.nn.init as init


from MiSiCNet import MiSiCNet2
from Transformer_model import AutoEncoder
torch.manual_seed(42)


class FirstModelLoss(nn.Module):
    def __init__(self, target_matrix, factor=0.8, center_factor=1e-4, weight_decay=1e-8):
        super(FirstModelLoss, self).__init__()
        self.name = 'SAD'
        self.target_matrix = target_matrix
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.factor = factor

        self.center_factor = center_factor
        self.weight_decay = weight_decay
        self.losses = []

    def forward(self, out, model_parameters):
        dim = self.target_matrix.size(0)  # L
        out_product = torch.matmul(out[:,:,:-1], torch.transpose(out[:,:,:-1], dim0=1, dim1=2))  # LxL
        out_product = (1 - torch.eye(dim)) * out_product + torch.eye(dim)
        target = torch.unsqueeze(self.target_matrix,dim=0)

        output_prod = torch.squeeze(out_product, dim=0)
        column_losses = []

        for col1, col2 in zip(output_prod.t(), self.target_matrix.t()):
            cos_similarity = F.cosine_similarity(col1, col2, dim=0)
            column_loss = torch.acos(torch.clamp(cos_similarity, -1.0, 1.0))
            column_losses.append(column_loss)
        loss_SAD = torch.sum(torch.stack(column_losses))
        loss_RE = self.mse_loss(output_prod.t(), self.target_matrix.t())


        m = torch.mean(self.target_matrix, dim=1)  # Lx1
        m_1 = torch.ger(m, torch.ones(out.size(2)))  # Lx1 * 1xJ = LxJ
        loss_center = torch.sum((out - m_1)**2)

        # loss = loss_SAD * 5e3 + loss_RE * 5e-2 + loss_center * self.center_factor
        loss = loss_SAD + loss_center * self.center_factor
        self.losses.append(loss.item())
        return loss

    def plot_loss(self):
        plt.plot(self.losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Over Time')
        plt.show()

def run(model, input_matrix, W, num_epochs, lr, param_search, center_factor, plot_flag=False):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)

    loss_function = FirstModelLoss(W, center_factor)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    for epoch in range(num_epochs):
        model.train()

        output = model(input_matrix)

        loss = loss_function(output, model.parameters())
        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
        optimizer.step()


        scheduler.step()

        if not param_search:
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


    if plot_flag:
        loss_function.plot_loss()

    return {'loss': loss_function.name, 'loss_value': round(loss.item(), 4), 'output_mat': output.detach().numpy().squeeze(0)}


class QinvU_estimator(nn.Module):
    def __init__(self, weight_init, dropout=0):
        super(QinvU_estimator, self).__init__()
        self.fc = nn.Linear(weight_init.shape[0],weight_init.shape[1])
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            self.fc.weight.copy_(weight_init)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x



class TwoModelsLoss(nn.Module):
    def __init__(self, target_matrix, center_factor, SAD_L2_factor, factor=0.35, weight_decay=1e-8):
        super(TwoModelsLoss, self).__init__()
        self.name = '2loss_SAD_L2_SAD2_center2'
        self.target_matrix = target_matrix

        self.factor = factor

        self.center_factor = center_factor

        self.SAD_L2_factor = SAD_L2_factor
        self.weight_decay = weight_decay
        self.model1_losses = []
        self.model2_center_losses = []
        self.models_diff_losses = []
        self.losses = []

    def model1_loss(self, out, model_parameters):
            L = self.target_matrix.size(0)  # L
            out_product = torch.matmul(out[:,:,:-1], torch.transpose(out[:,:,:-1], dim0=1, dim1=2))  # LxL
            out_product = (1 - torch.eye(L)) * out_product + torch.eye(L)
            target = torch.unsqueeze(self.target_matrix,dim=0)

            loss_MSE = F.mse_loss(out_product, target)
            loss_L2 = torch.sum((out_product - target)**2)

            tensor1_log_softmax = F.log_softmax(target, dim=1)
            tensor2_softmax = F.softmax(out_product, dim=1)

            loss_klv = F.kl_div(tensor1_log_softmax, tensor2_softmax, reduction='batchmean')

            output_prod = torch.squeeze(out_product, dim=0)
            column_losses = []


            for col1, col2 in zip(output_prod.t(), self.target_matrix.t()):
                cos_similarity = F.cosine_similarity(col1, col2, dim=0)
                column_loss = torch.acos(torch.clamp(cos_similarity, -1.0, 1.0))
                column_losses.append(column_loss)
            loss_SAD = torch.sum(torch.stack(column_losses))

            J = out.size(2)
            ones_J = torch.ones(J)

            m = torch.mean(self.target_matrix, dim=1)  # Lx1
            m_1 = torch.ger(m, ones_J)  # Lx1 * 1xJ = LxJ
            loss_center = torch.sum((out - m_1)**2)

            I_j = torch.eye(J)
            ones_mean = ones_J/J

            TV_diff = I_j - (1/J) * torch.matmul(ones_mean, ones_mean.t())

            loss_reg = torch.sum((torch.matmul(out, TV_diff) - m_1)**2) #+ loss_center

            R_diag = -torch.eye(L)
            I_diff = R_diag[:-1, 1:] + torch.eye(L - 1)
            R_diag[:-1, 1:] = I_diff
            R_mat = R_diag[:-1, :]
            R_decay = torch.sum(torch.matmul(R_mat, out)) ** 2


            # weight_decay_loss = 0.0
            # for param in model_parameters:
            #     weight_decay_loss += torch.sum(torch.square(param))



            loss = loss_SAD + self.center_factor * loss_reg*(10**-4) + R_decay * self.center_factor *10**5

            return loss

    def model2_center_loss(self, u, model2):
        m = torch.mean(u, dim=0)  # 1xJ

        Q = torch.linalg.inv(model2.fc.weight)
        m_1 = torch.ger(m, torch.ones(Q.shape[0]))  # Jx1 * 1xJ = JxJ

        return torch.sum((Q - m_1) ** 2) * self.center_factor


    def forward(self, out1, out2, model1_parameters, model2_parameters, model2, U, epoch):
        L = self.target_matrix.size(0)  # L
        loss_model1 = self.model1_loss(out1, model1_parameters)
        model2_center_loss = self.model2_center_loss(U, model2)


        out1 = torch.squeeze(out1, dim=0)
        loss_L2 = torch.sum((out1[:, :out2.shape[1]] - out2) ** 2)
        loss_models_diff = loss_L2


        column_losses = []
        for col1, col2 in zip(out1[:, :out2.shape[1]].t(), out2.t()):
            cos_similarity = F.cosine_similarity(col1, col2, dim=0)
            column_loss = torch.acos(torch.clamp(cos_similarity, -1.0, 1.0))
            column_losses.append(column_loss)
        loss_SAD = torch.sum(torch.stack(column_losses))

        loss_models_diff = (1-self.SAD_L2_factor) * loss_L2 + self.SAD_L2_factor * loss_SAD * L

        loss = (1 - self.factor) * loss_models_diff + self.factor * loss_model1 #+ model2_center_loss
        self.model1_losses.append(loss_model1.item())
        self.model2_center_losses.append(model2_center_loss.item())
        self.models_diff_losses.append(loss_models_diff.item())
        self.losses.append(loss.item())

        return loss

    def plot_loss(self):
        plt.plot(self.losses, label='Total loss')
        plt.plot(self.model1_losses, label='Model 1 loss')
        plt.plot(self.model2_center_losses, label='Model 2 center loss')
        plt.plot(self.models_diff_losses, label='Models diff loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Over Time')
        plt.show()


def run_2_models(model1, model2, input_matrix1, W, input_matrix2, num_epochs, lr, model_2_epoch_TH, param_search, center_factor, factor, SAD_L2_factor, clip_grad_max=0.5, plot_flag=False):

    optimizer1 = optim.Adam(model1.parameters(), lr=lr, betas=(0.5,0.999))
    optimizer2 = optim.Adam(model2.parameters(), lr=lr, betas=(0.5,0.999))

    loss_function = TwoModelsLoss(W, center_factor, SAD_L2_factor, factor=factor)

    scheduler1 = StepLR(optimizer1, step_size=10, gamma=0.9)
    scheduler2 = StepLR(optimizer2, step_size=10, gamma=0.9)

    for epoch in range(num_epochs):
        model1.train()
        model2.train()

        output1 = model1(input_matrix1)
        output2 = model2(input_matrix2)

        loss = loss_function(output1, output2, model1.parameters(), model2.parameters(), model2, input_matrix2, epoch)
        optimizer1.zero_grad()
        # optimizer2.zero_grad()
        if epoch < model_2_epoch_TH:
            optimizer2.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=clip_grad_max)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=clip_grad_max)


        optimizer1.step()
        scheduler1.step()
        # optimizer2.step()
        # scheduler2.step()
        if epoch < model_2_epoch_TH:
            optimizer2.step()
            scheduler2.step()


        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Model 1 Loss: {loss_function.model1_losses[-1]:.4f}, Model 2 Center Loss: {loss_function.model2_center_losses[-1]:.4f}, Models Difference Loss: {loss_function.models_diff_losses[-1]:.4f}")

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        if epoch >120:
            lr = lr / 1.2

    if plot_flag:
        loss_function.plot_loss()


    return {'loss': loss_function.name, 'loss_value': round(loss.item(), 4),
            'output_mat': output1.detach().numpy().squeeze(0),
            'total_losses':loss_function.losses,'models_diff_losses': loss_function.models_diff_losses,
            'model1_losses':loss_function.model1_losses, 'model2_center_losses': loss_function.model2_center_losses}


def deep_unmixing(W, U, SPA_Q, epochs=200, run_two_models_flag=True):
    W_torch = torch.from_numpy(W).float()
    U_torch = torch.from_numpy(U).float()
    SPA_Q_torch = torch.from_numpy(np.linalg.inv(SPA_Q)).float()
    L = W.shape[0]


    rand_input = torch.rand([L, L])/L
    n_heads = 2
    n_layers = 4
    J = SPA_Q.shape[0]


    num_epochs = epochs

    lr = 0.05

    two_models_factor = 0.2

    center_factor = L

    SAD_L2_factor = 0.2

    model_2_epoch_TH = 20

    patch = 5
    dim = 200


    model1 = MiSiCNet2(L, J, dropout=0)
    model1 = AutoEncoder(P=J, L=L, size=L,
                      patch=patch, dim=dim)
    model1.apply(model1.weights_init)
    # model1.apply(model1.weights_init)

    if run_two_models_flag:
        model2 = QinvU_estimator(SPA_Q_torch, dropout=0)

        d = run_2_models(model1, model2, W_torch, W_torch, U_torch, num_epochs, lr, model_2_epoch_TH=model_2_epoch_TH, factor=two_models_factor, param_search=False,
                         center_factor=center_factor, SAD_L2_factor=SAD_L2_factor, plot_flag=True)
    else:
        lr = 1e-4
        center_factor = 0.01
        d = run(model1, W_torch, W_torch, num_epochs, lr, param_search=False, center_factor=center_factor, plot_flag=True)
    # model1 = Transformer(L, J, n_heads, n_layers=n_layers, dropout=0)
    # model = MiSiCNet1(L, J)


    return d

def deep_unmixing_param_search(W, U, SPA_Q, id0, real_P, seconds, SNR, run_two_models_flag=True):
    W_torch = torch.from_numpy(W).float()
    U_torch = torch.from_numpy(U).float()
    SPA_Q_torch = torch.from_numpy(np.linalg.inv(SPA_Q)).float()

    L = W.shape[0]


    rand_input = torch.rand([L, L])/L
    n_heads = 2
    n_layers = 4
    J = SPA_Q.shape[0]
    num_epochs = 200
    model1 = MiSiCNet2(L, J)
    model2 = QinvU_estimator(SPA_Q_torch)
    best_lr = 0.01
    best_two_models_factor = 0.2
    best_center_factor = L/J*2
    best_SAD_L2_factor = 0.5

    lrs = [0.05, 0.01, 0.005, 0.001]
    center_factors = [L / J, L / J * 2, L / J * 3, L / J * 4, L / J * 5]
    two_models_factors = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
    SAD_L2_factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    best_L2 = 200

    for i in range(20):
        print(f'Trial num {i+1}')
        lr = random.choice(lrs)
        center_factor = random.choice(center_factors)
        two_models_factor = random.choice(two_models_factors)
        SAD_L2_factor = random.choice(SAD_L2_factors)
        d = run_2_models(model1, model2, W_torch, W_torch, U_torch, num_epochs, lr, factor=two_models_factor,
                         param_search=False,
                         center_factor=center_factor, SAD_L2_factor=SAD_L2_factor, plot_flag=False)

        P = np.array(d['output_mat'])
        P[:, :J] = P[:, id0]

        deep_L2 = np.sum((real_P[:, :J] - P[:, :J]) ** 2)

        if deep_L2 < best_L2:
            print(f'Best temp L2 = {deep_L2}')
            best_L2 = deep_L2
            best_lr = lr
            best_two_models_factor = two_models_factor
            best_center_factor = center_factor
            best_SAD_L2_factor = SAD_L2_factor
            best_P = P

    from sklearn.metrics import mean_squared_error
    best_MSE = mean_squared_error(real_P[:, :J], best_P[:, :J])

    print(f"For J = {J}, seconds = {seconds}, SNR = {SNR}, best L2 is: {best_L2}, best MSE is: {best_MSE} with the parameters:")
    print(f"Best LR: {best_lr} \nBest Two Models Factor: {best_two_models_factor} \nBest Center Factor: {best_center_factor} \nBest SAD L2 Factor: {best_SAD_L2_factor}")
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(
        ["J", "seconds", "SNR", "L2", "MSE", "Best LR", "Best Two Models Factor", "Best Center Factor", "Best SAD L2 Factor"])
    ws.append([J, seconds, SNR, best_L2, best_MSE, best_lr, best_two_models_factor, best_center_factor, best_SAD_L2_factor])
    wb.save("param_search_results.xlsx")



def plot_heat_mat(mat, plot_name, figs_directory, title=None, save_flag=True, show_flag=True, d=None):
    fig = plt.figure()
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.colorbar()
    if title:
        plt.title(title)
    if d:
        params_text = f'Model: {d["model_name"]}\n loss: {d["loss"]}\n value: {d["loss_value"]}\n lr: {d["lr"]}\n epochs: {d["epochs"]}'
        bbox_props = dict(boxstyle='round, pad=0.3', edgecolor='black', facecolor='white', alpha=0.8)
        plt.text(1, 1.05, params_text, transform=plt.gca().transAxes, fontsize=8, va='center', ha='left', bbox=bbox_props)
    if save_flag:
        plt.savefig(os.path.join(figs_directory, plot_name))
    if show_flag:
        plt.show()
    return mat

def plot_P_speakers(speakers, plot_name, figs_directory, noise=None, title=None, save_flag=True, show_flag=True, need_fig=True):
    if need_fig:
        plt.figure()
        plt.figure(figsize=(8, 6))
    for i in range(speakers.shape[1]):
        plt.plot(speakers[:,i], label=f'Speaker {i}')
    if noise is not None and np.any(noise):
        plt.plot(noise,'gray', label='Noise', linewidth=0.5)
    plt.legend(loc='upper right')
    if title:
        plt.title(title)
    if save_flag:
        plt.savefig(os.path.join(figs_directory, plot_name))
    if show_flag:
        plt.show()

def plot_simplex(vectors, vertices_indexes, Q, seconds=20, SNR=20, figs_dir='figures' ):

    fig = plt.figure(figsize=(8, 8))

    if Q==2:
        sc = plt.scatter(vectors[0], vectors[1])  # Scatter plot the points
        colors = ['red', 'green']
        for idx, ext in enumerate(vertices_indexes):
            plt.scatter(vectors[0][ext], vectors[1][ext], color=colors[idx], marker='o', s=100)
        plt.scatter(0,0, color='black')
        plt.xlabel('U0')
        plt.ylabel('U1')
        plt.title(f'SPA Simplex Figure from 2 Eigenvectors, Length = {seconds}, SNR = {SNR}')
        plt.grid(True)
        cbar = plt.colorbar(sc)
        plt.savefig(os.path.join(figs_dir, f'2D_SPA_simplex_length{seconds}_SNR{SNR}'))

    else:
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot the points
        ax.scatter(vectors[0], vectors[1], vectors[2], c='blue', label='Vectors')

        colors = ['red', 'green', 'yellow', 'pink', 'orange']

        # Plot the vertices
        for idx, ext in enumerate(vertices_indexes[:3]):
            ax.scatter(vectors[0][ext], vectors[1][ext], vectors[2][ext], color=colors[idx], marker='o', s=200,
                       label=f'Vertex {idx}')
        ax.scatter(0, 0, 0, color='black', marker='o', s=200,
                       label='(0,0,0)')
        # Set labels and title
        ax.set_xlabel('U0', fontsize=15)
        ax.set_ylabel('U1', fontsize=15)
        ax.set_zlabel('U2', fontsize=15)
        ax.set_title(f'SPA Simplex Figure from {Q} Eigenvectors, Length = {seconds}, SNR = {SNR}')

        # Add a legend
        ax.legend()

        ax.view_init(elev=20, azim=80)

        # Show the plot
        plt.savefig(os.path.join(figs_dir, f'3D_SPA_simplex_length{seconds}_SNR{SNR}'))

    # else:
    #     return
    plt.show()


#Test
# W_np = np.load('W_correlation_matrix.npy')

# SPA_Q_np = np.load('SPA_Q.npy')
# U_np = np.load('U_matrix.npy')
#
# deep_unmixing(W_np, U_np, SPA_Q_np, epochs=100, run_two_models_flag=False)