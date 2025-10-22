"""CLI training script (minimal)."""
from models.custom_cnn import build_simple_cnn
from training.trainer import train


def main():
    model = build_simple_cnn()
    dataset = []
    train(model, dataset, epochs=1)


if __name__ == '__main__':
    main()
