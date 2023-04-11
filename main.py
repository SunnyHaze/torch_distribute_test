import argparse
import datetime
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F 
import misc
import argparse, json, time, os
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

def get_args_parser():
    parser = argparse.ArgumentParser('Distributed training', add_help=False)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--batch_size', default=32, type=int)    
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

from typing import Iterable
class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),# 原题为三通道，此处转为单通道实现 # C1
            nn.ReLU(),
            nn.MaxPool2d(2,2), # S2
            nn.Conv2d(6,16,5), # C3  原始论文中C3与S2并不是全连接而是部分连接，这样能减少部分计算量。而现代CNN模型中，比如AlexNet，ResNet等，都采取全连接的方式了。我们的实现在这里做了一些简化。
            nn.ReLU(),
            nn.MaxPool2d(2,2) # S4
        )
        # 然后需要经过变形后，继续进行全连接
        self.layer2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), # C5
            nn.ReLU(),
            nn.Linear(120, 84),         # F6
            nn.ReLU(),
            nn.Linear(84,10), # Output 文章中使用高斯连接，现在方便起见仍然使用全连接
        )
    def forward(self,x):
        x = self.layer1(x) # 执行卷积神经网络部分
        x = x.view(-1,16 * 5 * 5) # 重新构建向量形状，准备全连接
        x = self.layer2(x) # 执行全连接部分
        return x
    
def train_one_epoch(
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        log_writer: SummaryWriter,
        args = argparse.ArgumentParser
    ):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    accum_iter = args.accum_iter
    
    optimizer.zero_grad()
    
    if log_writer is not None:
        print('log_dir : {}'.format(log_writer.log_dir))
    accum_correct = 0
    accum_items = 0
    for data_iter_step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        accum_correct += (outputs.argmax(1) == targets).sum().item()
        accum_items += images.shape[0]
        
        loss_value = loss.item()
        
        loss /= accum_iter
        loss.backward()
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        
        # metric_logger
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)
        
        # Tensorboard Summary
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_100x = int((data_iter_step / len(data_loader) + epoch) * 100)
            log_writer.add_scalar('train/loss', loss_value, epoch_100x)
            log_writer.add_scalar('train/lr', lr, epoch_100x)
    acc = accum_correct / accum_items
    if log_writer is not None:
        log_writer.add_scalar('acc/train', acc, epoch)
    metric_logger.update(acc=acc)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test_one_epoch(
    model : nn.Module,
    dataloader : Iterable,
    log_writer : SummaryWriter,
    device : torch.device,
    epoch : int,
    args = None
):     
    model.eval()
    torch.no_grad()
    acc_list = []
    for data_iter_step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(images)
        local_acc = (pred.argmax(1) == targets).sum().item() / images.shape[0]
        acc_list.append(local_acc)
        if data_iter_step == 0 and misc.is_main_process():
            log_writer.add_images('test', images, epoch)
            log_writer.add_text('test', str(pred.argmax(1)), epoch)
            log_writer.add_graph(model, images)   
    torch.cuda.synchronize()
    final_acc = sum(acc_list) / len(acc_list)
    if misc.is_main_process():
        log_writer.add_scalar('acc/test', final_acc, epoch)    

def main(args):
    misc.init_distributed_mode(args)
    
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    device = torch.device(args.device)
    
    seed = args.seed + misc.get_rank()
    
    cudnn.benchmark = True
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    


    # Prepare model parallel
    model = LeNet_5()
    model.to(device)
    print(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    # Prepare datasets
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    if args.distributed:
        world_size = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=world_size, 
            rank=global_rank, 
            shuffle=True
        )
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            test_data,
            num_replicas=world_size, 
            rank=global_rank, 
            shuffle=False
        )
        print('sampler_train: {}'.format(sampler_train))
        print('sampler_test: {}'.format(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_data)
        sampler_test = torch.utils.data.SequentialSampler(test_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, 
        sampler=sampler_train, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, 
        sampler=sampler_test, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    
    # Tensorboard summary writer
    if global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(args.log_dir)
        # log_writer.add_graph(model)
    else:
        log_writer = None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('optimizer: {}'.format(optimizer))
    start_time = time.time()
    # Train
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, log_writer, args)
        if args.output_dir and (epoch + 1) % 100 == 0 or epoch +1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        if epoch % 10 == 0:
            print(log_writer)
            test_one_epoch(model, test_loader,  log_writer, device, epoch, args)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

