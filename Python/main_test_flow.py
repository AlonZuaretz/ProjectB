import torch
import numpy as np

from data_helper import extract_data, create_dataloaders_cov, create_dataloaders_weights
from neural_nets import Stage1Network, Stage2Network
from scipy.io import savemat


if __name__ == "__main__":

    # path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
    path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
    # save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\NN_results"
    save_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\NN_results"

    Xw, Yw, XR, XRd, YR, Ydoa, params = extract_data(path)

    batch_size = 1024
    pre_method = 1

    _, test_loader_cov, _, idx_test = create_dataloaders_cov(XR, YR, XRd, pre_method, batch_size=batch_size, test_size=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage1_model = Stage1Network().to(device)
    checkpoint = torch.load('checkpoint_stage1.pth', map_location=device)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])

    _, test_loader_weights, _, _ = create_dataloaders_weights(Xw, Yw, batch_size=batch_size, test_size=0.2)
    stage2_model = Stage2Network().to(device)
    checkpoint = torch.load('checkpoint_stage2.pth', map_location=device)
    stage2_model.load_state_dict(checkpoint['model_state_dict'])

    outputs_list = []
    labels_list = []
    # Validation loop
    stage1_model.eval()
    stage2_model.eval()

    with torch.no_grad():
        for (_, labels_weights), (inputs_cov, _) in zip(test_loader_weights, test_loader_cov):

            labels_weights = labels_weights.to(device)
            inputs_cov = inputs_cov.to(device)

            outputs_stage1 = stage1_model(inputs_cov)
            outputs = stage2_model(outputs_stage1)

            outputs_temp = outputs.numpy()
            labels_temp = labels_weights.numpy()

            outputs_list.extend(outputs_temp)
            labels_list.extend(labels_temp)

    outputs_np_temp = np.array(outputs_list)
    labels_np_temp = np.array(labels_list)

    outputs_np = outputs_np_temp[:, 0:4] + 1j * outputs_np_temp[:, 4:8]
    labels_np = labels_np_temp[:, 0:4] + 1j * labels_np_temp[:, 4:8]

    data = {
        'output_Yw': outputs_np,
        'label_Yw': labels_np,
        'Indexes': idx_test,
        'pythonParams': params[0, idx_test]
    }
    savemat(save_path + r"\data.mat", data)
