'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader根据给定的数据集（dataset）、数据集配置选项（dataset_opt）和阶段（phase）来创建 PyTorch 的 DataLoader 对象 '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset2 as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))

    # 调试打印数据集长度
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset [{dataset.__class__.__name__}] is empty. Please check dataroot or preprocessing logic.")

    # 打印前 5 个样本
    for i in range(min(5, len(dataset))):
        print(f"Sample {i}: {dataset[i]}")


    return dataset
