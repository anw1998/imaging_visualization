import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import nibabel as nib 
import numpy as np
import os
import nrrd


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
  
    window_image = image.copy()

    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image



def show_2D(images_list: list, labels_list: list, dataloader=False):
    
    """
    This function plots the images and mask
    images_list: list of image paths where each image is a slice
    masks_list: list of mask paths where each mask is a slice
    """
    if dataloader:
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(images_list, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("mask")
        plt.imshow(labels_list, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"Overlay")
        plt.imshow(images_list, cmap="gray", interpolation=None)
        plt.imshow(labels_list, "jet", interpolation=None, alpha=0.2)
        plt.show()

    else: 
        @ interact(case_number=widgets.IntSlider(min=0, max=len(images_list)-1, step=1, value=0, description='Case #:'))
        def plot(case_number):
            # get image and segmentation file
            im_path = images_list[case_number]
            ma_path = labels_list[case_number]

            im = nib.load(im_path)
            im_np = im.get_fdata()
            im_np = window_image(im_np, 50, 100)
            im_slice = im_np

            ma = nib.load(ma_path)
            ma_np = ma.get_fdata().astype('int8')
            ma_slice = ma_np

            masked = np.ma.masked_where(ma_slice == 0, ma_slice)

            print(f'Image path {im_path}')
            print(f'Label path {ma_path}')
            print(f'Shape of image: {im_np.shape}')
            print(f'Shape of label: {ma_np.shape}')
            print(f'Labels: {np.unique(ma_np)}')

            plt.figure(figsize=(18, 6))
            plt.subplot(1,3,1)
            plt.imshow(im_slice.T,'gray', interpolation='none')
            plt.title('Image')

            plt.subplot(1,3,2)
            plt.imshow(ma_slice.T, 'gray', interpolation='none')
            plt.title('Mask')

            plt.subplot(1,3,3)
            plt.imshow(im_slice.T,'gray', interpolation='none')
            plt.imshow(masked.T, 'jet', interpolation='none', alpha=0.5)
            plt.title('Overlay')
            plt.show()

        
        
def show_3D(images_list: list, labels_list: list, slc = 0, dataloader=False):
    
    """
    This function plots the images and mask
    images_list: list of image paths where each image is a volume
    masks_list: list of mask paths where each mask is a volume
    """
    if dataloader:
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(images_list[:,:,slc], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("mask")
        plt.imshow(labels_list[:,:,slc], cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"Overlay")
        plt.imshow(images_list[:,:,slc], cmap="gray", interpolation=None)
        plt.imshow(labels_list[:,:,slc], "jet", interpolation=None, alpha=0.2)
        plt.show()
    
    else: 
        @ interact(case_number=widgets.IntSlider(min=0, max=len(images_list)-1, step=1, value=0, description='Case #:'))
        def step_one(case_number):
            im_path = images_list[case_number]
            ma_path = labels_list[case_number]

            im = nib.load(im_path)
            im_np = im.get_fdata()
            im_np = window_image(im_np, 50, 100)

            ma = nib.load(ma_path)
            ma_np = ma.get_fdata().astype('int8')

            @ interact(slice_number=widgets.IntSlider(min=0, max=im_np.shape[-1]-1, step=1, value=15, description='Slice #:'))
            def plot(slice_number):

                im_slice = im_np[:,:,slice_number]

                ma_slice = ma_np[:,:,slice_number]

                masked = np.ma.masked_where(ma_slice == 0, ma_slice)

                print(f'Image path {im_path}')
                print(f'Label path {ma_path}')
                print(f'Shape of image: {im_np.shape}')
                print(f'Shape of label: {ma_np.shape}')
                print(f'Labels: {np.unique(ma_np)}')

                plt.figure(figsize=(18, 6))
                plt.subplot(1,3,1)
                plt.imshow(im_slice.T,'gray', interpolation='none')
                plt.title('Image')

                plt.subplot(1,3,2)
                plt.imshow(ma_slice.T, 'gray', interpolation='none')
                plt.title('Mask')

                plt.subplot(1,3,3)
                plt.imshow(im_slice.T,'gray', interpolation='none')
                plt.imshow(masked.T, 'jet', interpolation='none', alpha=0.5)
                plt.title('Overlay')
                plt.show()
                
def show_3D_multi(images_list: list, labels_list: list, slc = 0, dataloader=False):
    
    """
    This function plots the images and mask
    images_list: list of image paths where each image is a volume
    masks_list: list of mask paths where each mask is a volume
    """
    if dataloader:
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(images_list[:,:,slc], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("mask")
        plt.imshow(labels_list[:,:,slc], cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"Overlay")
        plt.imshow(images_list[:,:,slc], cmap="gray", interpolation=None)
        plt.imshow(labels_list[:,:,slc], "jet", interpolation=None, alpha=0.2)
        plt.show()
    
    else: 
        @ interact(case_number=widgets.IntSlider(min=0, max=len(images_list)-1, step=1, value=0, description='Case #:'))
        def step_one(case_number):
            im_path = images_list[case_number]
            ma_path = labels_list[case_number]

            im = nib.load(im_path)
            im_np = im.get_fdata()
            im_np = window_image(im_np, 50, 100)

            ma = nib.load(ma_path)
            ma_np = ma.get_fdata().astype('int8')

            @ interact(slice_number=widgets.IntSlider(min=0, max=im_np.shape[-1]-1, step=1, value=15, description='Slice #:'))
            def plot(slice_number):

                im_slice = im_np[:,:,slice_number]

                ma_slice = ma_np[:,:,slice_number]

                masked = np.ma.masked_where(ma_slice == 0, ma_slice)
                mask_ICH = masked == 1
                mask_IVH = masked == 2

                print(f'Image path {im_path}')
                print(f'Label path {ma_path}')
                print(f'Shape of image: {im_np.shape}')
                print(f'Shape of label: {ma_np.shape}')
                print(f'Labels: {np.unique(ma_np)}')

                plt.figure(figsize=(18, 6))
                plt.subplot(1,3,1)
                plt.imshow(im_slice.T,'gray', interpolation='none')
                plt.title('Image')

                plt.subplot(1,3,2)
                plt.imshow(ma_slice.T, 'gray', interpolation='none')
                plt.title('Mask')

                plt.subplot(1,3,3)
                plt.imshow(im_slice.T,'gray', interpolation='none')
                # plt.imshow(masked.T, 'jet', interpolation='none', alpha=0.5)
                plt.imshow(mask_ICH.T, 'hsv', interpolation='none', alpha=0.5)
                plt.imshow(mask_IVH.T, 'viridis', interpolation='none', alpha=0.5)
                plt.title('Overlay')
                plt.show()

            
def show_3D_nrrd(images_list: list, labels_list: list, slc = 0, dataloader=False):
    
    """
    This function plots the images and mask
    images_list: list of image paths where each image is a volume
    masks_list: list of mask paths where each mask is a volume
    """
    if dataloader:
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(images_list[:,:,slc], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("mask")
        plt.imshow(labels_list[:,:,slc], cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"Overlay")
        plt.imshow(images_list[:,:,slc], cmap="gray", interpolation=None)
        plt.imshow(labels_list[:,:,slc], "jet", interpolation=None, alpha=0.2)
        plt.show()
    
    else: 
        @ interact(case_number=widgets.IntSlider(min=0, max=len(images_list)-1, step=1, value=0, description='Case #:'))
        def step_one(case_number):
            im_path = images_list[case_number]
            ma_path = labels_list[case_number]

            im = nib.load(im_path)
            im_np = im.get_fdata()
            im_np = window_image(im_np, 50, 100)

            ma, header = nrrd.read(ma_path)
            ma_np = ma.astype('int8')

            @ interact(slice_number=widgets.IntSlider(min=0, max=im_np.shape[-1]-1, step=1, value=15, description='Slice #:'))
            def plot(slice_number):

                im_slice = im_np[:,:,slice_number]

                ma_slice = ma_np[:,:,slice_number]

                masked = np.ma.masked_where(ma_slice == 0, ma_slice)

                print(f'Image path {im_path}')
                print(f'Label path {ma_path}')
                print(f'Shape of image: {im_np.shape}')
                print(f'Shape of label: {ma_np.shape}')
                print(f'Labels: {np.unique(ma_np)}')

                plt.figure(figsize=(18, 6))
                plt.subplot(1,3,1)
                plt.imshow(im_slice.T,'gray', interpolation='none')
                plt.title('Image')

                plt.subplot(1,3,2)
                plt.imshow(ma_slice.T, 'gray', interpolation='none')
                plt.title('Mask')

                plt.subplot(1,3,3)
                plt.imshow(im_slice.T,'gray', interpolation='none')
                plt.imshow(masked.T, 'jet', interpolation='none', alpha=0.5)
                plt.title('Overlay')
                plt.show()

            
def plot_curves(base_dir, num_runs):
    
    """
    This function plots the training and validation curves
    base_dir: base directory of all the runs
    num_runs: number of runs 
    """
    
    train_loss = np.asarray([])
    train_metric = np.asarray([])
    test_loss = np.asarray([])
    test_metric = np.asarray([])
    
    for i in range(1, num_runs+1):
        run_dir = os.path.join(base_dir, f"run {i}")

        train_loss_run = np.load(os.path.join(run_dir, 'loss_train.npy'))
        train_metric_run = np.load(os.path.join(run_dir, 'metric_train.npy'))
        # test_loss_run = np.load(os.path.join(run_dir, 'loss_val.npy'))
        test_metric_run = np.load(os.path.join(run_dir, 'metric_val.npy'))

        train_loss = np.append(train_loss, train_loss_run)
        train_metric = np.append(train_metric, train_metric_run)
        # test_loss = np.append(test_loss, test_loss_run)
        test_metric = np.append(test_metric, test_metric_run)
        
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Train dice loss")
    x = [i + 1 for i in range(len(train_loss))]
    y = train_loss
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 2)
    plt.title("Train metric DICE")
    x = [i + 1 for i in range(len(train_metric))]
    y = train_metric
    plt.xlabel("epoch")
    plt.plot(x, y)

    # plt.subplot(2, 2, 3)
    # plt.title("Val dice loss")
    # x = [i + 1 for i in range(len(test_loss))]
    # y = test_loss
    # plt.xlabel("epoch")
    # plt.plot(x, y)

    plt.subplot(2, 2, 4)
    plt.title("Val metric DICE")
    x = [i + 1 for i in range(len(test_metric))]
    y = test_metric
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.tight_layout()
    plt.show()