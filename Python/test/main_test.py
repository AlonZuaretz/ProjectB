import torch
import numpy as np
import os


from data.data_helper import extract_data, create_dataloaders, real_imag_to_hermitian
from NNs.neural_nets import Stage1Network
from scipy.io import savemat


if __name__ == "__main__":

    # base_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4"
    base_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4"
    path = base_path + r"\dataForPython_train.mat"
    save_path = base_path + r"\NN_results"
    load_path = base_path + r"\NN_results\stage1_run3_20241129_074726"

    save_flag = True

    Xw, Yw, XR, XRd, YR, Ydoa, params = extract_data(path)

    batch_size = 1024
    pre_method = 1

    _, _, cov_test_loader, _, _, _, _, _, _, _, _, idx_test = create_dataloaders(XR, XRd, YR, Yw, Ydoa)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    checkpoint_path = os.path.join(load_path, "checkpoint_stage1.pth")
    checkpoint = torch.load('checkpoint_stage1.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    inputs_list = []
    outputs_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for (inputs, labels) in cov_test_loader:

            labels = labels.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)

            inputs_list.extend(inputs.cpu().numpy())
            outputs_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    inputs_np_temp = np.array(inputs_list)
    outputs_np_temp = np.array(outputs_list)
    labels_np_temp = np.array(labels_list)

    input_XRd = real_imag_to_hermitian(inputs_np_temp[:, 0, :, :])
    output_YR = real_imag_to_hermitian(outputs_np_temp[:, 0, :, :])
    output_XR = real_imag_to_hermitian(outputs_np_temp[:, 1, :, :])
    label_YR = real_imag_to_hermitian(labels_np_temp[:, 0, :, :])
    label_XR = real_imag_to_hermitian(labels_np_temp[:, 1, :, :])

    data = {
        'input_XRd': input_XRd,
        'label_YR': label_YR,
        'label_XR': label_XR,
        'output_YR': output_YR,
        'output_XR': output_XR,
        'Indexes': idx_test,
        'pythonParams': params[0, idx_test]
    }
    savemat(save_path + r"\test_results.mat", data)
