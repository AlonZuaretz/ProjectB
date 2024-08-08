import torch
import numpy as np

from data_helper import extract_data, create_dataloaders_cov, real_imag_to_hermitian
from neural_nets import Stage1Network
from scipy.io import savemat


if __name__ == "__main__":

    path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1_new\allSIR\dataForPython.mat"
    save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1_new\allSIR\netV1_results"
    Xw, Yw, XR, YR, params = extract_data(path)

    batch_size = 256

    _, test_loader_cov, _, idx_test = create_dataloaders_cov(XR, YR, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    model.load_state_dict(torch.load('checkpoint.pth'))

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

    outputs_np = real_imag_to_hermitian(outputs_np_temp)
    labels_np = real_imag_to_hermitian(labels_np_temp)
    inputs_np = real_imag_to_hermitian(inputs_np_temp)

    data = {
        'YR': outputs_np,
        'XR': inputs_np,
        'YR_true': labels_np,
        'Indexes': idx_test,
        'pythonParams': params[0, idx_test]
    }
    savemat(save_path + r"\data.mat", data)
