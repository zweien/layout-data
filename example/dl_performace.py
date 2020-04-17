import tqdm
from torch.utils.data import DataLoader
import configargparse
from layout_data.data.layout import LayoutDataset


def get_parser():
    parser = configargparse.ArgParser(
        default_config_files=["config.yml"], description="Hyper-parameters."
    )
    parser.add_argument(
        "--config", is_config_file=True, default=False, help="config file path"
    )
    parser.add_argument("--data_root", type=str, default="d:/work/dataset")

    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="num_workers in DatasetLoader",
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--test_args", action="store_true", help="print args")

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    # print(parser.format_values())
    print(args)

    dataset = LayoutDataset(args.data_root)
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        # pin_memory=True
    )

    for epoch in tqdm.trange(args.max_epochs, ncols=50):
        for idx, batch in enumerate(dataloader):
            pass


if __name__ == "__main__":

    main()
