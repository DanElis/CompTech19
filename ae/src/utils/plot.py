import pylab as plt
import numpy as np

## Choose colormap from
### https://matplotlib.org/users/colormaps.html
    
def _check_shape(
    image, 
    map_true,
    map_pred,
):
    if not image.shape == map_true.shape:
        raise Exception(
            'image.shape must be equal map_true.shape.'
            'The shape of image was: {}.'
            'The shape of map_true was: {}'.format(image.shape, map_true.shape)
        )
        
    if not image.shape == map_pred.shape:
        raise Exception(
            'image.shape must be equal map_pred.shape. '
            'The shape of image was: {}.'
            ' The shape of map_pred was: {}'.format(image.shape, map_pred.shape)
        )

        
def _check_isnan(
    image, 
    map_true,
    map_pred,
    titles,
):
    if np.all(np.isnan(image)):
        image = np.zeros(image.shape)
        
    if np.all(np.isnan(map_true)):
        map_true = np.zeros(image.shape)
        
    if np.all(np.isnan(map_pred)):
        map_pred = np.zeros(image.shape)
        
    titles = np.squeeze(np.array(titles))
    
    if not ((len(titles.shape)==1) & (len(titles) >= image.shape[0])):
        
        titles = np.arange(image.shape[0])
    
    return image, map_true, map_pred, titles


def _check_input(
    image, 
    map_true,
    map_pred,
    titles,
):
    if len(image.shape) < 3:
        image = image[None, ...].copy()
    
    image, map_true, map_pred, titles = _check_isnan(image, map_true, map_pred, titles)
    
    if len(map_true.shape) < 3:
        map_true = map_true[None, ...].copy()
    
    if len(map_pred.shape) < 3:
        map_pred = map_pred[None, ...].copy()
        
    _check_shape(image, map_true, map_pred)
    
    return image, map_true, map_pred, titles
    

def show_image(
    image, 
    map_true=np.nan, 
    map_pred=np.nan, 
    threshold=.5, 
    alpha=None, 
    titles=np.nan, 
    scale_map=True,
    fig_hight=3,
    fig_width=None,
    image_cmap='seismic',
    map_true_cmap='Greys_r',
    map_pred_cmap='Greens',
):
    title_format = '{}'
    
    image, map_true, map_pred, titles = _check_input(image, map_true, map_pred, titles)
      
    for i in range(image.shape[0]):
        if not alpha:
            if not fig_width:
                fig_width = fig_hight * 3
            plt.figure(figsize=(fig_width, fig_hight))
        else:
            if not fig_width:
                fig_width = fig_hight * int(image.shape[2] / image.shape[1])
            plt.figure(figsize=(fig_width, fig_hight))

        cur_image = image[i].copy()
        cur_map_true = map_true[i].copy()
        cur_map_pred = map_pred[i].copy()
        cur_map_true = np.float32(cur_map_true)
        cur_map_pred = np.float32(cur_map_pred)
        
        if scale_map:
            cur_map_pred /=  np.abs(cur_map_pred).max() + 1e-16
            
        cur_map_pred[cur_map_pred < threshold] = np.nan
        cur_map_true[cur_map_true < threshold] = np.nan
        
        if alpha:
            plt.imshow(cur_image, cmap=image_cmap)
            
            if not np.all(np.isnan(cur_map_pred)):
                plt.imshow(cur_map_pred, alpha=alpha, cmap=map_pred_cmap)
                
            if not np.all(np.isnan(cur_map_true)):
                plt.imshow(cur_map_true, alpha=alpha, cmap=map_true_cmap)
                
            plt.title(title_format.format(titles[i]))
            plt.show()
            continue
        
        plt.subplot(1, 3, 1)
        plt.imshow(cur_image, cmap=image_cmap)
        plt.title(title_format.format(titles[i]))
        
        plt.subplot(1, 3, 2)
        plt.imshow(cur_map_true, vmin=0, vmax=1)
        plt.title('True')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cur_map_pred, vmin=0, vmax=1)
        plt.title('Pred')
        
        plt.show()
        
def show_image_dashboard(
    image, 
    map_true=np.nan, 
    map_pred=np.nan, 
    threshold=.5, 
    alpha=.5, 
    titles=np.nan, 
    scale_map=True,
    fig_hight=3,
    n_cols=3,
    image_cmap='seismic',
    map_true_cmap='Greys_r',
    map_pred_cmap='Greens',
    axis_off=True,
):
    
    title_format = '{}'
    image, map_true, map_pred, titles = _check_input(image, map_true, map_pred, titles)
    
    n_rows = image.shape[0] / n_cols
    n_rows = np.int32(np.floor(n_rows)) + 1
    
    
    fig_width = fig_hight * int(image.shape[2] / image.shape[1]) * n_cols
    
    j_zx = 0
    for jr in range(n_rows):
        _image = image[jr*n_cols: (jr+1)*n_cols].copy()
        _map_true = map_true[jr*n_cols: (jr+1)*n_cols].copy()
        _map_pred = map_pred[jr*n_cols: (jr+1)*n_cols].copy()
        
        _, axs = plt.subplots(figsize=(fig_width, fig_hight), ncols=n_cols, nrows=1)
        for i in range(n_cols):
            j_img = jr*n_cols + i
            if j_img >= image.shape[0]:
                continue
            cur_image = image[j_img].copy()
            cur_map_true = map_true[j_img].copy()
            cur_map_pred = map_pred[j_img].copy()
            
            cur_map_true = np.float32(cur_map_true)
            cur_map_pred = np.float32(cur_map_pred)

            if scale_map:
                cur_map_pred /=  np.abs(cur_map_pred).max() + 1e-16

            cur_map_pred[cur_map_pred < threshold] = np.nan
            cur_map_true[cur_map_true < threshold] = np.nan

            axs[i].imshow(cur_image, cmap=image_cmap, aspect='equal')

            if not np.all(np.isnan(cur_map_pred)):
                axs[i].imshow(cur_map_pred, alpha=alpha, cmap=map_pred_cmap, aspect='equal')

            if not np.all(np.isnan(cur_map_true)):
                axs[i].imshow(cur_map_true, alpha=alpha, cmap=map_true_cmap, aspect='equal')

            axs[i].set_title(title_format.format(titles[j_img]))
            if axis_off:
                axs[i].set_axis_off()
            j_zx += 1
        [axs[i].set_axis_off() for i in range(n_cols) if i >=_image.shape[0]]
        plt.show()
        
        
import pylab as plt
import numpy as np

## Choose colormap from
### https://matplotlib.org/users/colormaps.html
    
def _check_shape(
    image, 
    map_true,
    map_pred,
):
    if not image.shape == map_true.shape:
        raise Exception(
            'image.shape must be equal map_true.shape.'
            'The shape of image was: {}.'
            'The shape of map_true was: {}'.format(image.shape, map_true.shape)
        )
        
    if not image.shape == map_pred.shape:
        raise Exception(
            'image.shape must be equal map_pred.shape. '
            'The shape of image was: {}.'
            ' The shape of map_pred was: {}'.format(image.shape, map_pred.shape)
        )

        
def _check_isnan(
    image, 
    map_true,
    map_pred,
    titles,
):
    if np.all(np.isnan(image)):
        image = np.zeros(image.shape)
        
    if np.all(np.isnan(map_true)):
        map_true = np.zeros(image.shape)
        
    if np.all(np.isnan(map_pred)):
        map_pred = np.zeros(image.shape)
        
    titles = np.squeeze(np.array(titles))
    
    if not ((len(titles.shape)==1) & (len(titles) >= image.shape[0])):
        
        titles = np.arange(image.shape[0])
    
    return image, map_true, map_pred, titles


def _check_input(
    image, 
    map_true,
    map_pred,
    titles,
):
    if len(image.shape) < 3:
        image = image[None, ...].copy()
    
    image, map_true, map_pred, titles = _check_isnan(image, map_true, map_pred, titles)
    
    if len(map_true.shape) < 3:
        map_true = map_true[None, ...].copy()
    
    if len(map_pred.shape) < 3:
        map_pred = map_pred[None, ...].copy()
        
    _check_shape(image, map_true, map_pred)
    
    return image, map_true, map_pred, titles
    

def show_image(
    image, 
    map_true=np.nan, 
    map_pred=np.nan, 
    threshold=.5, 
    alpha=None, 
    titles=np.nan, 
    scale_map=True,
    fig_hight=3,
    fig_width=None,
    image_cmap='seismic',
    map_true_cmap='Greys_r',
    map_pred_cmap='Greens',
):
    title_format = '{}'
    
    image, map_true, map_pred, titles = _check_input(image, map_true, map_pred, titles)
      
    for i in range(image.shape[0]):
        if not alpha:
            if not fig_width:
                fig_width = fig_hight * 3
            plt.figure(figsize=(fig_width, fig_hight))
        else:
            if not fig_width:
                fig_width = fig_hight * int(image.shape[2] / image.shape[1])
            plt.figure(figsize=(fig_width, fig_hight))

        cur_image = image[i].copy()
        cur_map_true = map_true[i].copy()
        cur_map_pred = map_pred[i].copy()
        cur_map_true = np.float32(cur_map_true)
        cur_map_pred = np.float32(cur_map_pred)
        
        if scale_map:
            cur_map_pred /=  np.abs(cur_map_pred).max() + 1e-16
            
        cur_map_pred[cur_map_pred < threshold] = np.nan
        cur_map_true[cur_map_true < threshold] = np.nan
        
        if alpha:
            plt.imshow(cur_image, cmap=image_cmap)
            
            if not np.all(np.isnan(cur_map_pred)):
                plt.imshow(cur_map_pred, alpha=alpha, cmap=map_pred_cmap)
                
            if not np.all(np.isnan(cur_map_true)):
                plt.imshow(cur_map_true, alpha=alpha, cmap=map_true_cmap)
                
            plt.title(title_format.format(titles[i]))
            plt.show()
            continue
        
        plt.subplot(1, 3, 1)
        plt.imshow(cur_image, cmap=image_cmap)
        plt.title(title_format.format(titles[i]))
        
        plt.subplot(1, 3, 2)
        plt.imshow(cur_map_true, vmin=0, vmax=1)
        plt.title('True')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cur_map_pred, vmin=0, vmax=1)
        plt.title('Pred')
        
        plt.show()
        
def show_image_dashboard(
    image, 
    map_true=np.nan, 
    map_pred=np.nan, 
    threshold=.5, 
    alpha=.5, 
    titles=np.nan, 
    scale_map=True,
    fig_hight=3,
    n_cols=3,
    image_cmap='seismic',
    map_true_cmap='Greys_r',
    map_pred_cmap='Greens',
    axis_off=True,
):
    
    title_format = '{}'
    image, map_true, map_pred, titles = _check_input(image, map_true, map_pred, titles)
    
    n_rows = image.shape[0] / n_cols
    n_rows = np.int32(np.floor(n_rows)) + 1
    
    
    fig_width = fig_hight * int(image.shape[2] / image.shape[1]) * n_cols
    
    j_zx = 0
    for jr in range(n_rows):
        _image = image[jr*n_cols: (jr+1)*n_cols].copy()
        _map_true = map_true[jr*n_cols: (jr+1)*n_cols].copy()
        _map_pred = map_pred[jr*n_cols: (jr+1)*n_cols].copy()
        
        _, axs = plt.subplots(figsize=(fig_width, fig_hight), ncols=n_cols, nrows=1)
        for i in range(n_cols):
            j_img = jr*n_cols + i
            if j_img >= image.shape[0]:
                continue
            cur_image = image[j_img].copy()
            cur_map_true = map_true[j_img].copy()
            cur_map_pred = map_pred[j_img].copy()
            
            cur_map_true = np.float32(cur_map_true)
            cur_map_pred = np.float32(cur_map_pred)

            if scale_map:
                cur_map_pred /=  np.abs(cur_map_pred).max() + 1e-16

            cur_map_pred[cur_map_pred < threshold] = np.nan
            cur_map_true[cur_map_true < threshold] = np.nan

            axs[i].imshow(cur_image, cmap=image_cmap, aspect='equal')

            if not np.all(np.isnan(cur_map_pred)):
                axs[i].imshow(cur_map_pred, alpha=alpha, cmap=map_pred_cmap, aspect='equal')

            if not np.all(np.isnan(cur_map_true)):
                axs[i].imshow(cur_map_true, alpha=alpha, cmap=map_true_cmap, aspect='equal')

            axs[i].set_title(title_format.format(titles[j_img]))
            if axis_off:
                axs[i].set_axis_off()
            j_zx += 1
        [axs[i].set_axis_off() for i in range(n_cols) if i >=_image.shape[0]]
        plt.show()
        
        