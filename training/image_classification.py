# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import torch

from models import build_model
from datasets import get_dataset
from utils.evaluator import Evaluator
from utils.logger import create_logger
from utils.misc import bool_flag
from utils.trainer import Trainer
from utils.masks import generate_masks
import socket
import signal
import subprocess
import torch.nn as nn

def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm
    # logger.info("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    print('slurm job', params.is_slurm_job)
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            # logger.info(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        # logger.info(PREFIX + "Master address: %s" % params.master_addr)
        # logger.info(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1
    print('n_nodes', params.n_nodes)
    print('multi gpu', params.multi_gpu)
    print('world size', params.world_size)
    # summary
    PREFIX = "%i - " % params.global_rank
    # logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    # logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    # logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    # logger.info(PREFIX + "Global rank    : %i" % params.global_rank)
    # logger.info(PREFIX + "World size     : %i" % params.world_size)
    # logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    # logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    # logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    # logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    # logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        # logger.info("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )

def sig_handler(signum, frame):
    # logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    # logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        # logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        'nothing'
        # logger.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    'nothing'
    # logger.warning("Signal handler called with signal " + str(signum))
    # logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    # logger.warning("Signal handler installed.")

def check_parameters(params):
    assert params.dump_path is not None
    os.makedirs(params.dump_path, exist_ok=True)


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Train/evaluate image classification models')

    # config parameters
    parser.add_argument("--dump_path", type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument("--save_periodic", type=int, default=0)

    # Data parameters
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100","imagenet", "gaussian","credit", "hep", "adult", "mnist", "lfw"], default="cifar10")
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument('--n_data', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--data_num_dimensions', type=int, default=75)
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)

    # Model parameters
    parser.add_argument("--architecture", choices=["lenet", "smallnet", "alexnet", "kllenet", "linear", "mlp", "resnet18", "leaks"], default="lenet")

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--optimizer", default="sgd,lr=0.001,momentum=0.9")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--aug", type=bool_flag, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--private_train_split", type=float, default=0.25)
    parser.add_argument("--private_heldout_split", type=float, default=0.25)

    # privacy parameters
    parser.add_argument("--private", type=bool_flag, default=False)
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--privacy_epsilon", type=float, default=None)
    parser.add_argument("--privacy_delta", type=float, default=None)
    parser.add_argument("--log_gradients", type=bool_flag, default=False)
    parser.add_argument("--log_batch_models", type=bool_flag, default=False)
    parser.add_argument("--log_epoch_models", type=bool_flag, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    #multi gpu paramaeters
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)
    

    return parser


def train(params, mask):
    # Create logger and print params
    logger = create_logger(params)
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    if params.is_slurm_job:
        init_signal_handler()

    trainloader, n_data = get_dataset(params=params, is_train=True, mask=mask)
    validloader, _ = get_dataset(params=params, is_train=False)

    model = build_model(params)
    model.cuda()

    if params.multi_gpu:
        if params.private:
            raise NotImplementedError('Distributed training not implemented with privacy')
        else:
            print('Using multi gpu')
            model = nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)

    trainer = Trainer(model, params, n_data=n_data)
    trainer.reload_checkpoint()

    evaluator = Evaluator(model, params)

    # evaluation
    # if params.eval_only:
    #     scores = evaluator.run_all_evals(trainer, evals=['classif'], data_loader=validloader)

    #     for k, v in scores.items():
    #         logger.info('%s -> %.6f' % (k, v))
    #     logger.info("__log__:%s" % json.dumps(scores))
    #     exit()


    # training
    for epoch in range(trainer.epoch, params.epochs):

        # update epoch / sampler / learning rate
        trainer.epoch = epoch
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        # train
        for (idx, images, targets) in trainloader:
            trainer.classif_step(idx, images, targets)
            trainer.end_step()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate classification accuracy
        scores = evaluator.run_all_evals(evals=['classif'], data_loader=validloader)
        for name, val in trainer.get_scores().items():
            scores[name] = val

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.end_epoch(scores)

    return model



if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    check_parameters(params)

    if params.dataset=='imagenet':
        n_data=1281167
    elif params.dataset=='credit':
        n_data=800
    elif params.dataset=='hep':
        n_data=124
    elif params.dataset=='adult':
        n_data=32561
    elif params.dataset=='mnist':
        n_data=60000
    elif params.dataset=='lfw':
        #need to do the split here and fill this in
        n_data=912
    else:
        n_data=50000

    if params.mask_path=="none":
        split_config = {"public": {"train": 0.25,"heldout": 0.25}, "private": {"train": params.private_train_split,"heldout": params.private_heldout_split}}

        # Randomly split the data according to the configuration
        known_masks, hidden_masks = generate_masks(n_data, split_config)

        path = "data/"
        torch.save(known_masks['public'], path + "public.pth")
        torch.save(known_masks['private'], path + "private.pth")

        torch.save(hidden_masks['private']['train'], path + "hidden/train.pth")
        torch.save(hidden_masks['private']['heldout'], path + "hidden/heldout.pth")
        torch.save(hidden_masks['public']['train'], path + "hidden/public_train.pth")
        torch.save(hidden_masks['public']['heldout'], path + "hidden/public_heldout.pth")

        mask=hidden_masks['private']['train']
    else:
        mask = torch.load(params.mask_path)

    train(params, mask)
