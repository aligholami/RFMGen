import argparse
import torch
from torchvision import datasets, transforms
from detectron2.config import get_cfg
from objectifier.rfg_baseline import RFGenerator
from tqdm import tqdm


def extract_region_features(model, loaders, device):
    """
    Passes all the data inside loaders module to the model for region feature extractions.
    :param model: A model inherited from Module.nn
    :param loaders: A dictionary of PyTorch loaders to extract the region features from.
    :param args: A set of arguments, specifying details of the extraction.
    :param device: A variable specifying the computing device.
    :return: A dictionary containing the region feature matrices for the corresponding keys in the loaders dictionary.
    """
    return_rfs = {}
    model.eval()

    with torch.no_grad():
        for loader_key, loader in loaders.items():
            rfs = []
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                data = data.to(device)
                batch_region_features = model([data])
                # rfs.append(batch_region_features)

            return_rfs[loader_key] = rfs

    return return_rfs

def main():
    parser = argparse.ArgumentParser(description='Region Feature Matrix Generation Script.')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--train-rf-path', type=str, default='/local-scratch/region-features/train')
    parser.add_argument('--val-rf-path', type=str, default='/local-scratch/region-features/val')
    parser.add_argument('--dataset-path', type=str, default='/local-scratch/cifar-dataset')
    parser.add_argument('--rpn-config-path', type=str, default='./detectron-configs/base_rcnn_rpn.yaml')
    parser.add_argument('--rpn-pretrained-path', type=str, default='./pretrained/model_final.pkl')
    parser.add_argument('--no-cuda', default=False)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(args.dataset_path, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))

        val_dataset = datasets.CIFAR10(args.dataset_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    else:
        print("Dataset not valid!")
        exit(0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    object_detection_config = get_cfg()
    object_detection_config.merge_from_file(args.rpn_config_path)
    object_detection_config.MODEL.WEIGHTS = args.rpn_pretrained_path
    rf_generator = RFGenerator(cfg=object_detection_config, device=device).to(device)
    # optimizer = optim.Adam(rf_generator.parameters(), lr=args.lr)

    loaders = {
        'train': train_loader,
        'validation': val_loader
    }

    # Region features will be a dictionary
    region_features = extract_region_features(rf_generator, loaders, device)


if __name__ == '__main__':
    main()
