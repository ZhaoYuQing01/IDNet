import torch
import numpy as np
import torch.utils.data as Data

def obtain_index(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def crop_patches(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len)]
    return selected_patch


def crop_hsi_patches(data_size, data_indices, whole_data, hsi_patch_padding, padded_data, pca_components):
    small_cubic_data = np.zeros((data_size, 2 * hsi_patch_padding, 2 * hsi_patch_padding, pca_components))
    data_assign = obtain_index(data_indices, whole_data.shape[0], whole_data.shape[1], hsi_patch_padding)
    for i in range(len(data_assign)):
        small_cubic_data[i] = crop_patches(padded_data, data_assign[i][0], data_assign[i][1], hsi_patch_padding)
    return small_cubic_data

def crop_lidar_patches(data_size, data_indices, whole_data, hsi_patch_padding, padded_data):
    small_cubic_data = np.zeros((data_size, 2 * hsi_patch_padding, 2 * hsi_patch_padding))
    data_assign = obtain_index(data_indices, whole_data.shape[0], whole_data.shape[1], hsi_patch_padding)
    for i in range(len(data_assign)):
        small_cubic_data[i] = crop_patches(padded_data, data_assign[i][0], data_assign[i][1], hsi_patch_padding)
    return small_cubic_data

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, whole_hsi, whole_lidar, patch_padding, padded_hsi, padded_lidar, gt, args):
    y_train = gt[train_indices]-1
    y_test = gt[test_indices]-1
    hsi_train_data = crop_hsi_patches(TRAIN_SIZE, train_indices, whole_hsi, patch_padding, padded_hsi, args.pca_components)
    lidar_train_data = crop_lidar_patches(TRAIN_SIZE, train_indices, whole_lidar, patch_padding, padded_lidar)

    hsi_test_data = crop_hsi_patches(TEST_SIZE, test_indices, whole_hsi, patch_padding, padded_hsi, args.pca_components)
    lidar_test_data = crop_lidar_patches(TEST_SIZE, test_indices, whole_lidar, patch_padding, padded_lidar)
    hsi_train = hsi_train_data.transpose(0, 3, 1, 2)
    hsi_test = hsi_test_data.transpose(0, 3, 1, 2)
    hsi_tensor_train = torch.from_numpy(hsi_train).type(torch.FloatTensor)
    lidar_tensor_train = torch.from_numpy(lidar_train_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_dataset_train = Data.TensorDataset(hsi_tensor_train, lidar_tensor_train, y1_tensor_train)

    hsi_tensor_test = torch.from_numpy(hsi_test).type(torch.FloatTensor)
    lidar_tensor_test = torch.from_numpy(lidar_test_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.LongTensor)
    torch_dataset_test = Data.TensorDataset(hsi_tensor_test, lidar_tensor_test, y1_tensor_test)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train, 
        batch_size=args.batch_size,  
        shuffle=True,
        num_workers=0,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
    )
    
    return train_iter, test_iter
