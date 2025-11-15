import argparse

ALL_METHODS = [
    'Random', 'Cluster1', 'Cluster2', 'Pow-d', 'AFL', 'DivFL', 'GradNorm','MaxRate'
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST', help='dataset',
                        choices=['Reddit','MNISTDataset','CIFAR','FederatedEMNIST','FedCIFAR100','CelebA', 'PartitionedCIFAR10', 'FederatedEMNIST_IID', 'FederatedEMNIST_nonIID'
                        ,
        'PathMNISTDomain'    # ← add this one only
                        ])

    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--data_dir', type=str, default='/home/xingyu/FL_CSL/Active-Client-Selection-for-Communication-efficient-Federated-Learning/data/', help='dataset directory')
    parser.add_argument('--model', type=str, default='CNN', help='model', choices=['BLSTM','CNN','ResNet'])
    parser.add_argument('--method', type=str, default='Random', help='client selection',
                        choices=ALL_METHODS)

  

    parser.add_argument('--domains', type=int, default=4, help='number of domains for PathMNISTDomain')
    parser.add_argument('--num_total_clients', type=int, default=10, help='total number of clients for PathMNISTDomain')
    parser.add_argument('--val_ratio', type=float, default=0.1)

    parser.add_argument('--fed_algo', type=str, default='FedAvg', help='Federated algorithm for aggregation',
                        choices=['FedAvg', 'FedAdam'])
    
    # optimizer
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.1, help='learning rate for client optim')
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for server optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha1', type=float, default=0.75, help='alpha1 for AFL')
    parser.add_argument('--alpha2', type=float, default=1, help='alpha2 for AFL')
    parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3 for AFL')
    # ---- Straggler Simulation Arguments ----
    parser.add_argument(
    '--simulate_stragglers', 
    type=str, 
    default='0,1',
    help='Comma-separated client IDs to simulate as stragglers (e.g., "0,1,2").'
)
    parser.add_argument(
    '--delay_base_sec', 
    type=float, 
    default=10.0,
    help='Base delay time (in seconds) for straggler clients.'
)

    parser.add_argument(
    '--delay_jitter_sec', 
    type=float, 
    default=3.0,
    help='Random jitter (in seconds) added to base delay for stragglers.'
)

    parser.add_argument(
    '--delay_prob', 
    type=float, 
    default=1.0,
    help='Probability [0,1] that a straggler actually delays in a round.'
)
    # parser.add_argument('--power_limit', type=int, default=None,
    #                 help='Power limit for each client')
    # parser.add_argument('--bandwidth_limit', type=int, default=None,
    #                 help='Bandwidth limit for each client')
    # parser.add_argument('--num_clients', type=int, default=None,
    #                 help='num_clients for selection')

    parser.add_argument('--power_limit', type=int, default=100,
                    help='power limit for clients')
    parser.add_argument('--bandwidth_limit', type=int, default=100,
                    help='bandwidth limit for clients')

    
    # training setting
    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    parser.add_argument('-R', '--num_round', type=int, default=2000, help='total number of rounds')
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    parser.add_argument('-K', '--total_num_clients', type=int, default=None, help='total number of clients')

    parser.add_argument('-u', '--num_updates', type=int, default=None, help='number of updates')
    parser.add_argument('-n', '--num_available', type=int, default=None, help='number of available clients at each round')
    parser.add_argument('-d', '--num_candidates', type=int, default=None, help='buffer size; d of power-of-choice')

    parser.add_argument('--loss_div_sqrt', action='store_true', default=False, help='loss_div_sqrt')
    parser.add_argument('--loss_sum', action='store_true', default=False, help='sum of losses')
    parser.add_argument('--num_gn', type=int, default=0, help='number of group normalization')

    parser.add_argument('--distance_type', type=str, default='L1', help='distance type for clustered sampling 2')
    parser.add_argument('--subset_ratio', type=float, default=None, help='subset size for DivFL')

    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help='ratio of data partition from dirichlet distribution')
    
    parser.add_argument('--min_num_samples', type=int, default=None, help='mininum number of samples')
    parser.add_argument('--schedule', type=int, nargs='+', default=[0, 5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300],
                        help='splitting points (epoch number) for multiple episodes of training')
    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    # experiment setting
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')
    parser.add_argument('--save_probs', action='store_true', default=False, help='save probs')
    parser.add_argument('--no_save_results', action='store_true', default=False, help='save results')
    parser.add_argument('--test_freq', type=int, default=1, help='test all frequency')

    parser.add_argument('--comment', type=str, default='', help='comment')
    
    # ----- Checkpointing for long Kaggle runs -----
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='/kaggle/working/checkpoints/testhhh',
        help='Directory to save checkpoints (default: /kaggle/working/checkpoints)'
    )
    parser.add_argument(
        '--ckpt_freq',
        type=int,
        default=20,
        help='Save a checkpoint every ckpt_freq global rounds (default: 20)'
    )
    parser.add_argument(
        '--resume_ckpt',
        type=str,
        default=None,
        help='Path to checkpoint .pt file to resume from (default: None)'
    )
    parser.add_argument(
        '--resume_round',
        type=int,
        default=0,
        help='Round index to resume from (0 = auto from checkpoint)'
    )
    args = parser.parse_args()
    return args
