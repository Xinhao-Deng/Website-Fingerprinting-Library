import numpy as np
import torch
import os
import json
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from .evaluator import measurement
from .netclr_augmentor import Augmentor

def knn_monitor(net, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().cuda()
        feature_labels = torch.cat(feature_labels,dim=0).t().contiguous().cuda()
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()    
    return y_true, y_pred

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs
    return adjusted_lengths

def build_augmentor(x_random):
    x_random = x_random.numpy().squeeze(1)
    outgoing_burst_sizes = []
    for packets in x_random:
        packets = np.trim_zeros(packets, "fb")
        bursts = fast_count_burst(packets)
        outgoing_burst_sizes += list(bursts[bursts>0])
    max_outgoing_burst_size = int(max(outgoing_burst_sizes))

    count, bins = np.histogram(outgoing_burst_sizes, bins=max_outgoing_burst_size - 1)
    PDF = count/np.sum(count)
    OUTGOING_BURST_SIZE_CDF = np.zeros_like(bins)
    OUTGOING_BURST_SIZE_CDF[1:] = np.cumsum(PDF)
    augmentor = Augmentor(max_outgoing_burst_size, outgoing_burst_sizes, OUTGOING_BURST_SIZE_CDF)
    return augmentor

def model_train(
    model,
    optimizer,
    train_iter,
    valid_iter,
    loss_name,
    save_metric,
    eval_metrics,
    train_epochs,
    out_file,
    num_classes
):
    if loss_name == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name == "TripletMarginLoss": # TF
        criterion = losses.TripletMarginLoss(margin=0.1)
        miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
    else:
        raise ValueError(f"Loss function {loss_name} is not matched.")
    
    assert save_metric in eval_metrics, f"save_metric {save_metric} should be included in {eval_metrics}"
    metric_best_value = 0

    for epoch in range(train_epochs):
        model.train()
        sum_loss = 0
        sum_count = 0
        for index, cur_data in enumerate(train_iter):
            cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
            optimizer.zero_grad()
            outs = model(cur_X)

            if loss_name ==  "TripletMarginLoss":
                hard_pairs = miner(outs, cur_y)
                loss = criterion(outs, cur_y, hard_pairs)
            else:
                loss = criterion(outs, cur_y)
            loss.backward()

            optimizer.step()
            sum_loss += loss.data.cpu().numpy() * outs.shape[0]
            sum_count += outs.shape[0]

        # Calculate and print training loss
        train_loss = round(sum_loss/sum_count, 3)
        print(f"epoch {epoch}: train_loss = {train_loss}")

        # Validation step
        if loss_name in ["TripletMarginLoss"]:
            valid_true, valid_pred = knn_monitor(model, train_iter, valid_iter, num_classes, 10)
        else:
            with torch.no_grad():
                model.eval()
                sum_loss = .0
                sum_count = 0
                valid_pred = []
                valid_true = []

                # Validate over batches
                for index, cur_data in enumerate(valid_iter):
                    cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
                    outs = model(cur_X)
                    outs = torch.argsort(outs, dim=1, descending=True)[:,0]
                    valid_pred.append(outs.cpu().numpy())
                    valid_true.append(cur_y.cpu().numpy())
                
                # Concatenate predictions and measure validation performance
                valid_pred = np.concatenate(valid_pred)
                valid_true = np.concatenate(valid_true)
        
        valid_result = measurement(valid_true, valid_pred, eval_metrics)
        print(f"{epoch}: {valid_result}")
        if valid_result[save_metric] > metric_best_value:
            metric_best_value = valid_result[save_metric]
            torch.save(model.state_dict(), out_file)

def model_eval(model, test_iter, valid_iter, eval_method, eval_metrics, out_file, num_classes):

    if eval_method == "common":
        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []

            # Validate over batches
            for index, cur_data in enumerate(test_iter):
                cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
                outs = model(cur_X)
                outs = torch.argsort(outs, dim=1, descending=True)[:,0]
                y_pred.append(outs.cpu().numpy())
                y_true.append(cur_y.cpu().numpy())

            y_pred = np.concatenate(y_pred).flatten()
            y_true = np.concatenate(y_true).flatten()
    elif eval_method == "kNN":
        y_true, y_pred = knn_monitor(model, valid_iter, test_iter, num_classes, 10)
    elif eval_method == "holmes":
        pass
    else:
        raise ValueError(f"Evaluation method {eval_method} is not matched.")
    
    result = measurement(y_true, y_pred, eval_metrics)
    print(result)

    with open(out_file, "w") as fp:
        json.dump(result, fp, indent=4)

def info_nce_loss(features, batch_size):
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim = 0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits / 0.5
    return logits, labels

def pretrian_accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)
        return res.cpu().numpy()[0]

def model_pretrian(model, optimizer, train_iter, train_epochs, out_file, batch_size):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(train_epochs):
        model.train()
        mean_acc = .0
        iter_count = 0
        for index, cur_data in enumerate(train_iter):
            cur_X, cur_y = cur_data[0], cur_data[1]
            cur_X = torch.cat(cur_X, dim=0)
            cur_X = cur_X.view(cur_X.size(0), 1, cur_X.size(1))
            cur_X = cur_X.float().cuda()

            optimizer.zero_grad()
            features = model(cur_X)
            logits, labels = info_nce_loss(features, batch_size)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            iter_count += 1
            mean_acc += pretrian_accuracy(logits, labels)
        mean_acc /= iter_count
        print(f"epoch {epoch}: {mean_acc}")
    torch.save(model.state_dict(), out_file)