import numpy as np

yCoeffR, yCoeffG, yCoeffB = 0.299, 0.587, 0.114
uCoeff, vCoeff = 0.564, 0.713
scaleExponent = 16

scaledYCoeffR = int(yCoeffR * float(1 << scaleExponent))
scaledYCoeffG = int(yCoeffG * float(1 << scaleExponent))
scaledYCoeffB = int(yCoeffB * float(1 << scaleExponent))

scaledUCoeff = int(uCoeff * float(1 << scaleExponent))
scaledVCoeff = int(vCoeff * float(1 << scaleExponent))
scaledInvUCoeff = int(float(1 << scaleExponent) / uCoeff)
scaledInvVCoeff = int(float(1 << scaleExponent) / vCoeff)

scaledGCoeffU = int(yCoeffB / (yCoeffG * uCoeff) * float(1 << scaleExponent))
scaledGCoeffV = int(yCoeffR / (yCoeffG * vCoeff) * float(1 << scaleExponent))


def bgr2rgb(bgr_img):
    return bgr_img[:, :, ::-1].astype(np.uint8)


def rgb2bgr(rgb_img):
    return rgb_img[:, :, ::-1].astype(np.uint8)


def bgr2yuv(bgr_img):
    bgr_img = bgr_img.astype(np.int64)
    y_img = (scaledYCoeffR * bgr_img[:, :, 2] + scaledYCoeffG * bgr_img[:, :, 1] + scaledYCoeffB * bgr_img[:, :, 0]) >> scaleExponent
    u_img = 128 + (((bgr_img[:, :, 0] - y_img) * scaledUCoeff) >> scaleExponent)
    v_img = 128 + (((bgr_img[:, :, 2] - y_img) * scaledVCoeff) >> scaleExponent)
    return np.dstack([np.clip(y_img, 0, 255), np.clip(u_img, 0, 255), np.clip(v_img, 0, 255)]).astype(np.uint8)


def yuv2rgb(yuv_img):
    yuv_img = yuv_img.astype(np.int64)
    u, v = yuv_img[:, :, 1]-128, yuv_img[:, :, 2]-128
    b = yuv_img[:, :, 0] + ((u * scaledInvUCoeff) >> scaleExponent)
    g = yuv_img[:, :, 0] - ((u * scaledGCoeffU + v * scaledGCoeffV) >> scaleExponent)
    r = yuv_img[:, :, 0] + ((v * scaledInvVCoeff) >> scaleExponent)
    return np.dstack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]).astype(np.uint8)
