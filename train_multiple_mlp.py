import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import time
from pathlib import Path
import wandb


from dataset import DataPoints2D
from model import MultiMLP
import torch.nn.functional as F


def get_args_parser():
    parser = argparse.ArgumentParser("Multiple MLP arguments", add_help=False)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--drop_out", default=0.2, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--loss", default="CE", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint_index", default=None, type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--data_range", default=[-5, 5], type=list)

    return parser


def print_args(args):
    print("RUNNING AT SETTINGS \n")
    print(
        "--lr {} --epochs {} --drop_out {} --optimizer {} --loss {} --batch_size {} --num_workers {} --checkpoint_index {} --mode {} --data_range {}".format(
            args.lr, args.epochs, args.drop_out, args.optimizer, args.loss, args.batch_size, args.num_workers, args.checkpoint_index, args.mode, args.data_range
        )
    )
    print("\n")


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_accuracy(predictions, labels):
    top_pred = predictions.argmax(1, keepdim=True)
    correct = top_pred.eq(labels.view_as(top_pred)).sum()
    acc = correct.float() / labels.shape[0]
    return acc


def train(args, model, training_loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (data_points, labels) in tqdm(training_loader, desc="Training", leave=False):
        if args.loss == "CE":
            labels = labels.type(torch.LongTensor)
        elif args.loss == "L2":
            labels = labels.type(torch.FloatTensor)

        optimizer.zero_grad()
        predictions = model(data_points)  # forward
        loss = criterion(predictions, labels)  # loss
        acc = calculate_accuracy(predictions, labels)
        loss.backward()  # backward
        optimizer.step()  # optimize

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(training_loader), epoch_acc / len(training_loader)


def evaluate(args, model, val_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (data_points, labels) in tqdm(val_loader, desc="Evaluating", leave=False):
            predictions = model(data_points)
            if args.loss == "CE":
                labels = labels.type(torch.LongTensor)
            elif args.loss == "L2":
                labels = labels.type(torch.FloatTensor)
            loss = criterion(predictions, labels)

            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(val_loader), epoch_acc / len(val_loader)


def train_and_evaluate(args, train_loader, val_loader, model, criterion, optimizer):
    best_valid_loss = float("inf")

    for epoch in range(args.epochs):
        start_time = time.monotonic()

        train_loss, train_acc = train(args, model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(args, model, val_loader, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "./checkpoints/model-ckpt-best.pt")

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        wandb.log(
            {
                "Train loss": train_loss,
                "Train Acc": train_acc * 100,
                "Val. Loss": valid_loss,
                "Val. Acc": valid_acc * 100,
            }
        )


def get_predictions(model, data_loader):

    model.eval()

    images = []
    labels = []
    output = []

    with torch.no_grad():

        for (data_points, labels) in data_loader:
            predictions = model(data_points)

            y_prob = F.Sigmoid(predictions, dim=-1)

            images.append(data_points)
            labels.append(labels)
            output.append(y_prob)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    output = torch.cat(output, dim=0)

    return images, labels, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLP training script", parents=[get_args_parser()])
    args = parser.parse_args()

    wandb.init(project="BB_Homework_DL", entity="mediaeval-sport")

    # prepare data points
    training_data = DataPoints2D(100000, args.data_range)
    validation_data = DataPoints2D(20000, args.data_range)
    testing_data = DataPoints2D(20000, args.data_range)

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # define loss functions
    if args.loss == "L2":
        criterion = nn.MSELoss()
    elif args.loss == "CE":
        criterion = nn.CrossEntropyLoss()

    # model and optimizer
    model = MultiMLP(2, args.drop_out)
    wandb.watch(model)
    print_args(args)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training phase
    if args.mode == "train":
        train_and_evaluate(args, training_loader, validation_loader, model, criterion, optimizer)
        model.load_state_dict(torch.load("./checkpoints/model-ckpt-best.pt"))
        test_loss, test_acc = evaluate(args, model, testing_loader, criterion)
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
        # wandb.log({"Test Loss": test_loss, "Test Acc": test_acc*100})
    # elif args.mode == "test":
    #     model.load_state_dict(torch.load("model-ckpt-{}.pt".format(args.checkpoint_index)))
    #     test_loss, test_acc = evaluate(args, model, testing_loader, criterion)
    #     print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
    # elif args.mode == "predict":
    #     images, labels, probs = get_predictions(model, testing_loader)
    #     pred_labels = torch.argmax(probs, 1)
