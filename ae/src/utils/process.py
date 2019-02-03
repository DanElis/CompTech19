import numpy as np


def forward_fourier_1d(x, axis=1):
    x = x.copy()
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.fft(x, axis=axis)
    x = np.fft.fftshift(x, axes=axis)
    return x


def backward_fourie_1dr(x, axis=1):
    x = x.copy()
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.ifft(x, axis=axis)
    x = np.fft.ifftshift(x, axes=axis)
    return np.real(x)


def get_spectrum_1d(x, axis=1):
    ns = x.shape[axis]
    x = forward_fourier(x, axis=axis)
    nw = np.int32(np.floor(ns/2) + 1)
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(0, nw)
    return np.flip(x[slc], axis=axis)


def apply_filter_1d(x, f, axis=1):
    ns = x.shape[axis]
    nw = np.int32(np.floor(ns/2) + 1)
    j = np.abs(np.arange(-nw + 1, nw)[:ns])
    f = np.squeeze(f[j]).reshape(1, -1)
    if axis==0:
        f = f.T
    return backward_fourier(forward_fourier(x, axis=axis)*f, axis=axis)


def band_pass_filter_1d(ns, dt, b):

    nw = np.int32(np.floor(ns/2)) + 1
    b = np.int32(np.round(b*(ns-1)*dt))
#     print(b)
    b[b<0] = 0
    b[b>(nw-1)] = nw
    f = np.zeros(nw)
    f[b[0]: b[1]] = np.linspace(0,1,b[1]-b[0])
    f[b[1]: b[2]] = 1
    f[b[2]: b[3]] = np.linspace(1,0,b[3]-b[2])
    return f


def normalize_data(x, axis=2, eps=1e-16):
    x = x.copy()
    shape = x.shape
    if len(shape) == 2:
        mu = x.mean(axis=1, keepdims=True)
#         std = x.std(axis=1, keepdims=True)
        std = np.abs(x).max(axis=1, keepdims=True)

    elif len(shape) == 3:
        if shape[-1] == 1:
            mu = x.mean(axis=axis, keepdims=True)
            std = x.std(axis=axis, keepdims=True)
            

        elif shape[-1] > 1:
            mu = x.mean(axis=axis, keepdims=True)
            std = x.std(axis=axis, keepdims=True)
            
#     std = np.abs(x).max(axis=axis, keepdims=True)
    x = (x - mu) / (std + eps)
    return x


def normalize_image(img, eps=1e-16):
    img = img.copy()
    img /= np.abs(img).max() + eps
    
    return img


def get_image_sample(
    image_sgy,
    faults_map,
    location,
    sample_size=64,
    axis=0,
):
    if not np.mod(sample_size, 2) == 0:
        raise Exception('sample_size should be even. The value of sample_size was odd: {}'.format(sample_size))
        
    sample_size_half = np.int32(sample_size / 2)
    
    image_sgy = image_sgy.copy()
    faults_map = faults_map.copy()
    
    input_shape = np.array(image_sgy.shape, dtype=int)
    
    sample_img = image_sgy.take(location[axis], axis=axis)
    sample_map = faults_map.take(location[axis], axis=axis)
    
    mask = np.ones(len(location), dtype=bool)
    mask[axis] = False
    location = location[mask, ...]
    input_shape = input_shape[mask, ...]
    
    delta = (input_shape - location)
    mask = delta < sample_size_half
    location[mask] -= (sample_size_half - delta[mask])

    delta = (location - np.zeros(len(location), dtype=int))
    mask = delta < sample_size_half
    location[mask] += (sample_size_half - delta[mask])

    slc = [slice(None)] * len(input_shape)
    slc[0] = slice(
        location[0] - sample_size_half, 
        location[0] + sample_size_half
    )

    slc[1] = slice(
        location[1] - sample_size_half, 
        location[1] + sample_size_half
    )
    sample_img = sample_img[tuple(slc)]
    sample_map = sample_map[tuple(slc)]
    
    return sample_img, sample_map


def rms_agc(traces, window=301, des_std=1, eps=1e-16, axis=1, squeeze_window=False):
    g = np.zeros(traces.shape)
    ns = traces.shape[axis]
    slc_g = [slice(None)] * len(traces.shape)
    slc_t = [slice(None)] * len(traces.shape)
    window =  np.int32(np.floor(window/2))
    
    if squeeze_window:
        range_end = ns - window
        range_start = window
    else:
        range_end = ns
        range_start = 1
        
    for i in range(range_start, range_end):
        jfr = max([(i - window), 0])
        jto = min([ns, (i + window)])
        
        slc_t[axis] = slice(jfr, jto)
        slc_g[axis] = slice(i,i+1)
        
        temp = traces[tuple(slc_t)]
        measure = temp.std(axis=axis, keepdims=True)
        g[tuple(slc_g)] = measure
    g = des_std / (g + eps)
    slc_g[axis] = slice(2)
    g[tuple(slc_g)] = 0
    return g