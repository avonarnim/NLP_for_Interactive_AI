import tqdm
import torch
import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score
from actionModel import ActionIdet
from targetModel import TargetIdet

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)


# To encode the data, I tokenize and label the words based on v2i,
# which stores the most common words and their indices
def encode_data(data, v2i, seq_len, a2i, t2i):
    n_lines = len(data)
    n_actions = len(a2i)
    n_targets = len(t2i)

    x = np.zeros((n_lines, seq_len), dtype=np.int32)
    yAction = np.zeros((n_lines), dtype=np.int32)
    yTarget = np.zeros((n_lines), dtype=np.int32)

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for command, actionTarget in data:
        command = preprocess_string(command)
        x[idx][0] = v2i["<start>"]
        jdx = 1
        for word in command.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        yAction[idx] = a2i[actionTarget[0]]
        yTarget[idx] = t2i[actionTarget[1]]
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    return x, yAction, yTarget

# To create the DataLoaders, I called the build_tokenizer_table function,
# flattened the input data, encoded the data, and transformed them into Tensors
def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    minibatch_size = 256

    args = vars(args)
    input_file = open(args['in_data_fn'])
    input_data = json.load(input_file)
    train_data = input_data['train']
    val_data = input_data['valid_seen']

    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)

    maps = {"v2i": vocab_to_index, "a2i": actions_to_index, "t2i": targets_to_index}
    
    train_lines = []
    for episode in train_data:
        for insts, outseq in episode:
            train_lines.append([insts, outseq])
    val_lines = []
    for episode in val_data:
        for insts, outseq in episode:
            val_lines.append([insts, outseq])

    # Encode the training and validation set inputs/outputs.
    train_np_x, train_np_yAction, train_np_yTarget = encode_data(train_lines, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    train_yAction_weight = np.array([1. / (sum([train_np_yAction[jdx] == idx for jdx in range(len(train_np_yAction))]) / len(train_np_yAction)) for idx in range(len(actions_to_index))], dtype=np.float32)
    train_yTarget_weight = np.array([1. / (sum([train_np_yTarget[jdx] == idx for jdx in range(len(train_np_yTarget))]) / len(train_np_yTarget)) for idx in range(len(targets_to_index))], dtype=np.float32)
    trainAction_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_yAction))
    trainTarget_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_yTarget))

    val_np_x, val_np_yAction, val_np_yTarget = encode_data(val_lines, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    valAction_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_yAction))
    valTarget_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_yTarget))

    # Create data loaders
    trainAction_loader = torch.utils.data.DataLoader(trainAction_dataset, shuffle=True, batch_size=minibatch_size)
    trainTarget_loader = torch.utils.data.DataLoader(trainTarget_dataset, shuffle=True, batch_size=minibatch_size)
    valAction_loader = torch.utils.data.DataLoader(valAction_dataset, shuffle=True, batch_size=minibatch_size)
    valTarget_loader = torch.utils.data.DataLoader(valTarget_dataset, shuffle=True, batch_size=minibatch_size)

    return trainAction_loader, trainTarget_loader, valAction_loader, valTarget_loader, maps, len_cutoff, train_yAction_weight, train_yTarget_weight

# To set up the model, I just initialize the individual model classes
def setup_model(args, maps, device, len_cutoff):
    """
    return:
        - model: YourOwnModelClass
    """

    embedding_dim = 128
    actionModel = ActionIdet(device, len(maps["v2i"]), len_cutoff, len(maps["a2i"]), embedding_dim)
    targetModel = TargetIdet(device, len(maps["v2i"]), len_cutoff, len(maps["t2i"]), embedding_dim)
    
    return actionModel, targetModel

# To set up the optimizer, I used a cross entropy loss as a base and stuck with it, seeing 
# minimal benefit when trying out other loss functions. I switched from SGD to Adam because 
# when running SGD, I had a prohibitively long runtime. I initially started with a learning 
# rate of 10^-4 but that required many epochs to train, so I switched to a rate of 10^-3, 
# which gave me reasonably positive results within as few as 3 epochs
def setup_optimizer(args, model, weight, aOrT):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    learning_rate = 0.001

    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer


# To train over an epoch for the two models, I iterate through the loaders and sum 
# their losses, calling both optimizers, and calculating individual accuracies.
def train_epoch(
    args,
    actionModel,
    targetModel,
    actionLoader,
    targetLoader,
    action_optimizer,
    target_optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    targetLoader_iterator = iter(targetLoader)
    for (inputs, actionLabels) in actionLoader:
        # put model inputs to device
        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        inputs, actionLabels = inputs.to(device), actionLabels.to(device)
        actions_out = actionModel(inputs)

        try:
            (inputs, targetLabels) = next(targetLoader_iterator)
        except StopIteration:
            targetLoader_iterator = iter(targetLoader)
            (inputs, targetLabels) = next(targetLoader_iterator)
        inputs, targetLabels = inputs.to(device), targetLabels.to(device)
        targets_out = targetModel(inputs)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), actionLabels.long())
        target_loss = target_criterion(targets_out.squeeze(), targetLabels.long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            action_optimizer.zero_grad()
            target_optimizer.zero_grad()
            loss.backward()
            action_optimizer.step()
            target_optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(actionLabels.cpu().numpy())
        target_labels.extend(targetLabels.cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, actionModel, targetModel, loaders, action_optimizer, target_optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    actionModel.eval()
    targetModel.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            actionModel,
            targetModel,
            loaders["valAction"],
            loaders["valTarget"],
            action_optimizer,
            target_optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

        print(
            f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
        )
        print(
            f"val action acc : {action_acc} | val target acc: {target_acc}"
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, actionModel, targetModel, loaders, action_optimizer, target_optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    actionModel.train()
    targetModel.train()

    for epoch in tqdm.tqdm(range(int(args.num_epochs))):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            actionModel,
            targetModel,
            loaders["trainAction"],
            loaders["trainTarget"],
            action_optimizer,
            target_optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % int(args.val_every) == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                actionModel,
                targetModel,
                loaders,
                action_optimizer,
                target_optimizer,
                action_criterion,
                target_criterion,
                device,
            )

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    trainAction_loader, trainTarget_loader, valAction_loader, valTarget_loader, maps, len_cutoff, train_yAction_weight, train_yTarget_weight = setup_dataloader(args)
    loaders = {"trainAction": trainAction_loader, "trainTarget": trainTarget_loader, "valAction": valAction_loader, "valTarget": valTarget_loader}

    # build model
    actionModel, targetModel = setup_model(args, maps, device, len_cutoff)

    # get optimizer and loss functions
    action_criterion, action_optimizer = setup_optimizer(args, actionModel, train_yAction_weight, True)
    target_criterion, target_optimizer = setup_optimizer(args, targetModel, train_yTarget_weight, False)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            actionModel,
            targetModel,
            loaders,
            action_optimizer,
            target_optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, actionModel, targetModel, loaders, action_optimizer, target_optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
