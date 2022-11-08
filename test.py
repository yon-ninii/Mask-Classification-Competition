import argparse
import torch
from tqdm import tqdm
import os
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.model import EfficientNet
from parse_config import ConfigParser


def main(config, input_dir, save_dir):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=4
    )

    # build model architecture
    model = EfficientNet.from_name(config['arch']['type'], num_classes=config['arch']['args']['num_classes'])
    logger.info(model)

    info_path = os.path.join(input_dir, 'info.csv')
    info = pd.read_csv(info_path)

    # get function handles of loss and metrics
    #loss_fn = getattr(module_loss, config['loss'])
    #metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    #total_loss = 0.0
    #total_metrics = torch.zeros(len(metric_fns))
    preds = []

    with torch.no_grad():
        for i, (data) in enumerate(tqdm(data_loader)):
            data = data.to(device).float() #, target.to(device)
            output = model(data)
            pred = torch.argmax(output.detach().cpu(), dim=1).numpy()
            preds.extend(pred)

            # computing loss, metrics on test set
            #loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            #total_loss += loss.item() * batch_size
            #for i, metric in enumerate(metric_fns):
            #    total_metrics[i] += metric(output, target) * batch_size
    
    info['ans'] = preds
    save_path = os.path.join(save_dir, f'submission.csv')
    info.to_csv(save_path, index=False)
    
    #n_samples = len(data_loader.sampler)
    #log = {'loss': total_loss / n_samples}
    #log.update({
    #    met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    #})
    #logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='/opt/ml/code/Lv1/ENet_Implement/test.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    input_dir = '/opt/ml/input/data/eval/'
    save_dir = '/opt/ml/code/Lv1/ENet_Implement/outputs'
    config = ConfigParser.from_args(args)
    main(config, input_dir, save_dir)
