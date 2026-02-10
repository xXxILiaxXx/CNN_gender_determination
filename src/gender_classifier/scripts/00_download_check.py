from ..config import Config
from ..data.dataset import prepare_dataloaders


def main():
    cfg = Config()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    dls = prepare_dataloaders(cfg)

    print("DATA ROOT:", dls.data_root)
    print("class_to_idx:", dls.class_to_idx)
    print("train/val/test sizes:",
          len(dls.train.dataset), len(dls.val.dataset), len(dls.test.dataset))

    x, y = next(iter(dls.train))
    print("batch x:", x.shape, x.dtype)
    print("batch y:", y.shape, y.dtype)
    print("y sample:", y[:10].tolist())


if __name__ == "__main__":
    main()