import numpy as np


def fit_line(spots):
    A = np.ndarray((len(spots), 2))
    y = np.ndarray((len(spots),))
    for i, spot in enumerate(spots):
        omega = spot[2]*spot[2] if len(spot) > 2 else 1  # spot[2]*spot[2] because spot[2] is actually the square root of omega
        A[i, 0] = omega
        A[i, 1] = omega * spot[0]
        y[i] = omega * spot[1]
    x, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return (x[0], -1, -1, x[0] + x[1]), np.sum(residuals)


def fit_model(spots, step=2):
    # Only works if the x coordinates of the spots are linspace(...)
    best_model = fit_line(spots)
    for i in range(2, len(spots) - 2, step):
        left_line, left_error = fit_line(spots[:i])
        right_line, right_error = fit_line(spots[i:])
        total_error = left_error + right_error
        if total_error < best_model[1]:
            x_inter = (left_line[0] - right_line[0]) / (right_line[3] - right_line[0] - left_line[3] + left_line[0])
            y_inter = left_line[0] + (left_line[3] - left_line[0]) * x_inter
            if x_inter < 0 or x_inter > 1:
                continue
            if y_inter > max(left_line[0], right_line[3]):
                continue
            best_model = ((left_line[0], x_inter, y_inter, right_line[3]), total_error)
    return best_model[0]
