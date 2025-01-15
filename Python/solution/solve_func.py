from data.data_helper import hermitian_to_real_imag, normalize_data, real_imag_to_hermitian, extract_data
from NNs.neural_nets import Stage1Network
from scipy.signal import find_peaks

import torch
import numpy as np
import matplotlib.pyplot as plt


def music_spect(X, sv_mat, M, num_srcs):
    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(X)
    idx = np.argsort(eigenvalues)  # Ascend order
    eigenvectors = eigenvectors[:, idx]

    # Define the noise subspace
    noise_subspace = eigenvectors[:, :M - num_srcs]

    # Pre-allocate MUSIC spectrum
    P = np.zeros(sv_mat.shape[0], dtype=float)

    for i in range(sv_mat.shape[0]):
        a = sv_mat[i, :].reshape(-1, 1)  # Steering vector for angle i
        P[i] = 1 / np.real(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)

    # Normalize the spectrum
    P = np.abs(P) / np.max(np.abs(P))
    return P


def estimate_angle(P, scan_angles):
    peaks, _ = find_peaks(P, prominence=1)
    if len(peaks) < 2:
        print(f"Cannot estimate signal's direction")
        return 0, 0
    else:
        peak_values = P[peaks]
        # Sort peaks by descending value
        sorted_indices = np.argsort(peak_values)[::-1]
        sorted_peaks = peaks[sorted_indices]
        locs = scan_angles[sorted_peaks]
        # Assume second highest peak to be the desired signal:
        est_angle = locs[1]
        idx = sorted_peaks[1]
    return est_angle, idx


def pc_beamformer(X, npc, sv):
    eigenvalues, eigenvectors = np.linalg.eig(X)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Construct reduced covariance matrix
    SSI = np.diag(eigenvalues[:npc])
    U_SI = eigenvectors[:, :npc]
    R_r = U_SI @ SSI @ U_SI.T

    # Calculate beamforming weights
    SSI_inv = np.linalg.inv(SSI)
    sv = np.conj(sv)
    w = (sv @ U_SI @ SSI_inv @ U_SI.T) / (sv @ U_SI @ SSI_inv @ U_SI.T @ sv.T)

    return w


def preprocess(X):
    """
        Inputs:

        Outputs:

        Description:

    """
    if len(X.shape) == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])

    X = normalize_data(X)
    X = hermitian_to_real_imag(X)

    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    X = torch.tensor(X, dtype=torch.float64)
    return X


def stage1(X):
    """
    Inputs:

    Outputs:

    Description:

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().double().to(device)
    checkpoint_path = 'checkpoint_dataV5_141656.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        output = model(X)
    output = output.cpu().numpy()
    Y1 = np.squeeze(output[:, 0, :, :]) # The MVDR matrix
    Y2 = np.squeeze(output[:, 1, :, :]) # The MPDR matrix

    if len(Y1.shape) == 2:
        Y1 = Y1.reshape(1, Y1.shape[0], Y1.shape[1])
        Y2 = Y2.reshape(1, Y2.shape[0], Y2.shape[1])

    Y1 = real_imag_to_hermitian(Y1)
    Y2 = real_imag_to_hermitian(Y2)

    return Y1, Y2


def stage2(X):
    """
        Inputs:

        Outputs:

        Description:

    """
    M = 4
    npc = 2
    scan_angles = np.arange(-60, 60, 0.2)
    sv_mat = np.zeros((len(scan_angles), M), dtype=complex)
    w = np.zeros((X.shape[0], M), dtype=complex)
    for i, angle in enumerate(scan_angles):
        sv_mat[i, :] = np.exp(1j * np.pi * np.arange(M) * np.sin(np.radians(angle)))

    for i in range(X.shape[0]):
        x = X[i, :, :]
        P = music_spect(x, sv_mat, M, npc)
        P_dB = 10*np.log10(P)
        est_angle, idx = estimate_angle(P_dB, scan_angles)
        sv = sv_mat[idx, :]

        w[i, :] = pc_beamformer(x, npc, sv)

    return w

def solve(X):
    """
        Inputs:

        Outputs:

        Description:

    """
    X = preprocess(X)
    Y1, Y2 = stage1(X)
    w = stage2(Y2)

    return Y1, Y2, w

if __name__ == "__main__":

    # base_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV5"
    base_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5"
    path = base_path + r"\dataForPython.mat"

    Xw, Yw, XR, XRd, YR, Ydoa, params = extract_data(path)

    indices = np.random.permutation(XRd.shape[0])
    XRd = XRd[indices]

    idx = range(100)
    X = XRd[idx, :, :]
    Y1, Y2, w = solve(X)

    # Analysis:
    a_idx = 1
    r_idx = indices[a_idx]
    M = 4
    npc = 2
    input_angle = params[0, r_idx]['inputAngle'][0, 0]
    int_angle = params[0, r_idx]['interferenceAngle'][0, 0]

    scan_angles = np.arange(-60, 60, 0.2)
    AF = np.zeros_like(scan_angles, dtype=complex)
    for i, angle in enumerate(scan_angles):
        sv = np.exp(1j * np.pi * np.arange(M) * np.sin(np.radians(angle)))
        AF[i] = np.abs(np.dot(np.conj(w[a_idx, :]), sv))

    # Normalize the response
    Pw = M * np.abs(AF)**2 / np.max(np.abs(AF)**2)
    Pw_dB = 10 * np.log10(Pw)

    # Plot the pattern
    plt.figure(figsize=(8, 6))
    plt.plot(scan_angles, Pw_dB, label="Beam Pattern")
    plt.axvline(x=input_angle, label='Input Angle', color='r')
    plt.axvline(x=int_angle, label='Interference Angle', color='b')
    plt.title("Beam Pattern")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.show()










