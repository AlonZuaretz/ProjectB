import torch
import numpy as np
import os

from data_helper import extract_data, create_dataloaders, real_imag_to_hermitian
from neural_nets import Stage1Network
from scipy.io import savemat


if __name__ == "__main__":

    # path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
    # save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\NN_results"
    path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
    save_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\NN_results"
    stage1_load_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\NN_results\stage1_run3_20241129_074726"

    Xw, Yw, XR, XRd, YR, Ydoa, params = extract_data(path)

    batch_size = 1024
    pre_method = 1

    _, test_loader_cov, _, idx_test = create_dataloaders_cov(XR, YR, XRd, pre_method, batch_size=batch_size, test_size=0.2)
    _, test_loader_weights, _, _ = create_dataloaders_weights(Xw, Yw, batch_size=batch_size, test_size=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    checkpoint = torch.load(os.path.join(stage1_load_path, "checkpoint_stage1.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    inputs_list = []
    outputs_list = []
    labels_list = []
    # Validation loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in test_loader_cov:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            outputs_temp = torch.squeeze(outputs).numpy()
            labels_temp = torch.squeeze(labels).numpy()
            inputs_temp = torch.squeeze(inputs).numpy()

            outputs_list.extend(outputs_temp)
            labels_list.extend(labels_temp)
            inputs_list.extend(inputs_temp)

    outputs_np_temp = np.array(outputs_list)
    labels_np_temp = np.array(labels_list)
    inputs_np_temp = np.array(inputs_list)


    input_XRd = real_imag_to_hermitian(inputs_np_temp)
    label_YR = real_imag_to_hermitian(labels_np_temp[:, 0, :, :])
    label_XR = real_imag_to_hermitian(labels_np_temp[:,1,:,:])
    output_YR = real_imag_to_hermitian(outputs_np_temp[:, 0, :, :])
    output_XR = real_imag_to_hermitian(outputs_np_temp[:, 1, :, :])


    data = {
        'input_XRd': input_XRd,
        'label_YR': label_YR,
        'label_XR': label_XR,
        'output_YR': output_YR,
        'output_XR': output_XR,
        'Indexes': idx_test,
        'pythonParams': params[0, idx_test]
    }
    savemat(save_path + r"\data.mat", data)
