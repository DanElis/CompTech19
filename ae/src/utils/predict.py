import numpy as np
from sklearn import metrics
import keras
def cut_traces(data, nr, ns):
    if data.shape[0] < nr:
        temp = np.zeros((nr - data.shape[0], data.shape[1]))
        data = np.vstack((data, temp))

    if data.shape[1] < ns:
        temp = np.zeros((data.shape[0], ns - data.shape[1]))
        data = np.hstack((data, temp))
    if data.shape[0] > nr:
        data = data[:nr]
    if data.shape[1] > ns:
        data = data[:ns]
    return data

def get_weight(offset, ns, vel0=2000, dt=.002):
    time0 = np.int32(np.round(offset / dt / vel0))
    time0[time0>(ns - 1)] = ns - 1
    weight = keras.utils.to_categorical(time0, ns)
    return np.float32(np.cumsum(weight, axis=1))

def get_weight_time(time0, ns):
    time0[time0>(ns - 1)] = ns - 1
    weight = keras.utils.to_categorical(time0, ns)
    return np.float32(np.cumsum(weight, axis=1))


def predict_idxmax(x, model, weight=1, pred_val=False):
    pred = model.predict(x, verbose=False).argmax(axis=2)
    pred = np.float32(pred)
    pred *= weight
    
    if pred_val:
        _pred = np.squeeze(pred.max(axis=1)[None,...])
        _pred = np.array(_pred, ndmin=1)
        
    pred = np.gradient(pred, axis=1)
    picks = pred.argmax(axis=1)
    if pred_val:
        return _pred, picks
    return pred, picks

def predict_argmax(x, model, _fr=4, _end=9, axis=1, weight=1, pred_val=False):
    pred = model.predict(x, verbose=False)[..., axis]
    pred[:,:_fr] = 0
    pred[:, -_end:] = 0
    pred *= weight
    picks = np.squeeze(pred.argmax(axis=1)[None,...])
    picks = np.array(picks, ndmin=1)
    if pred_val:
        pred = np.squeeze(pred.max(axis=1)[None,...])
        pred = np.array(pred, ndmin=1)
    return pred, picks

def predict_threshold(x, model, _fr=4, _end=9, axis=1, threshold=.7, gradient=False, is_abs=False, weight=1, pred_val=False):
    pred = model.predict(x, verbose=False)[..., axis]
    pred[:,:_fr] = 0
    pred[:, -_end:] = 0
    if gradient:
        pred = np.gradient(pred, axis=1)
        pred[:,:(_fr + 1)] = 0
        pred[:, -(_end + 1):] = 0
        
    if is_abs:
        pred = np.abs(pred)
    
    if pred_val:
        _pred = np.squeeze((pred*weight).max(axis=1)[None,...])
        _pred = np.array(_pred, ndmin=1)
        
    pred /= pred.max(axis=1, keepdims=True)
    picks = np.sign(pred - threshold)
    pred *= weight
    picks = np.squeeze(pred.argmax(axis=1))
    picks = np.array(picks, ndmin=1)
    
    if gradient:
        picks += 1
    if pred_val:
        return _pred, picks
    
    return pred, picks
    
def predict_conv1d_det_model(traces, model_conv1d):
    x = ut.normalize_data(traces)
    pred1d = model_conv1d.predict(x[..., None], verbose=False)[..., 1]
    pred1d[:,:4] = 0
    pred1d[:, -9:] = 0
    picks_pred1d = np.squeeze(pred1d.argmax(axis=1)[None,...])
    return np.squeeze(pred1d), picks_pred1d

def predict_conv1d_segm_model(traces, model_conv1d_mask):
    x = ut.normalize_data(traces)
    pred1d_mask = model_conv1d_mask.predict(x[..., None], verbose=False)[..., 1]
    pred1d_mask[:,:4] = 0
    pred1d_mask[:, -9:] = 1
    pred1d_mask = np.gradient(pred1d_mask, axis=1)
    picks_pred1d_mask = np.squeeze(pred1d_mask.argmax(axis=1)[None,...])
    return pred1d_mask, picks_pred1d_mask

def predict_conv1d_segm_model_new(traces, model_conv1d_mask):
    x = ut.normalize_data(traces)
    pred1d_mask = model_conv1d_mask.predict(x[..., None], verbose=False)[..., 2]
    pred1d_mask[:, :4] = 0
    pred1d_mask[:, -9:] = 1
    pred1d_mask /= pred1d_mask.max(axis=1, keepdims=True)
#     pred1d_mask = np.gradient(pred1d_mask, axis=1)
    picks_pred1d = np.sign(pred1d_mask - .7).argmax(axis=1)
#     picks_pred1d = np.squeeze(pred1d.argmax(axis=1)[None,...])
    return np.squeeze(pred1d_mask), picks_pred1d
    
def print_metrics(true, pred_dict):
    for x in pred_dict:
        pred = pred_dict[x]
        _metrics_format = '{:15} MAE={:.2f};STD={:.2f};MDAE={:.2f}'.format(
            x,
#             metrics.accuracy_score(true, pred),
            metrics.mean_absolute_error(true, pred),
            metrics.mean_squared_error(true, pred),
            metrics.median_absolute_error(true, pred)
        )
        print(_metrics_format)