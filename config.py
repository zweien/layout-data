from argparse import ArgumentParser

def get_parser():
        # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    # root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)
    
    

    # gpu args
    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='how many gpus'
    )

    
    parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parser.add_argument('--test', action='store_true', default=False, help='print args')

    hparams = parser.parse_args()

    return parser