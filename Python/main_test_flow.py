import torch
import numpy as np

from data_helper import (extract_data, create_dataloaders_cov, create_dataloaders_weights, real_imag_to_hermitian,
                         abs_phase_rejoin, abs_phase_rejoin_hermitian)
from neural_nets import Stage1Network, Stage2Network
from scipy.io import savemat


if __name__ == "__main__":

    path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\dataForPython_train.mat"
    save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\netV3_results"
    Xw, Yw, XR, YR, params = extract_data(path)

    batch_size = 1024
    pre_method = 1

    _, test_loader_cov, _, idx_test = create_dataloaders_cov(XR, YR, pre_method, batch_size=batch_size, test_size=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage1_model = Stage1Network().to(device)
    stage1_model.load_state_dict(torch.load('checkpoint_stage1_finetuned.pth'))

    _, test_loader_weights, _, _ = create_dataloaders_weights(Xw, Yw, batch_size=batch_size)
    stage2_model = Stage2Network().to(device)
    stage2_model.load_state_dict(torch.load('checkpoint_stage2_finetuned.pth'))

    inputs_list = []
    outputs_list = []
    labels_list = []
    # Validation loop
    stage1_model.eval()
    stage2_model.eval()

    with torch.no_grad():
        for (inputs_weights, labels_weights), (inputs_cov, labels_cov) in zip(test_loader_weights, test_loader_cov):

            inputs_weights, labels_weights = inputs_weights.to(device), labels_weights.to(device)
            inputs_cov, labels_cov = inputs_cov.to(device), labels_cov.to(device)

            outputs_stage1 = stage1_model(inputs_cov)
            inputs_stacked = torch.cat((inputs_cov, outputs_stage1), dim=1)
            outputs = stage2_model(inputs_stacked)

            outputs_temp = outputs.numpy()
            labels_temp = labels_weights.numpy()
            inputs_temp = inputs_weights.numpy()

            outputs_list.extend(outputs_temp)
            labels_list.extend(labels_temp)
            inputs_list.extend(inputs_temp)

    outputs_np_temp = np.array(outputs_list)
    labels_np_temp = np.array(labels_list)
    inputs_np_temp = np.array(inputs_list)

    outputs_np = outputs_np_temp[:, 0:4] + 1j * outputs_np_temp[:, 4:8]
    inputs_np = inputs_np_temp[:, 0:4] + 1j * inputs_np_temp[:, 4:8]
    labels_np = labels_np_temp[:, 0:4] + 1j * labels_np_temp[:, 4:8]

    data = {
        'Yw': outputs_np,
        'Xw': inputs_np,
        'Yw_true': labels_np,
        'Indexes': idx_test,
        'pythonParams': params[0, idx_test]
    }
    savemat(save_path + r"\data.mat", data)
