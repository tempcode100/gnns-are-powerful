## Part of the following code has been modified from https://github.com/weihua916/powerful-gnns (Copyright (c) 2021 Weihua Hu)#####

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from util import load_data, separate_data, add_attributes_norm
from models.graphcnn import GraphCNN
criterion = nn.CrossEntropyLoss()
def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    # pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        # pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    # print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)

    f1_micro_train = f1_score(labels.cpu(), pred.cpu(), average = 'micro')
    f1_macro_train = f1_score(labels.cpu(), pred.cpu(), average = 'macro')

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)

    f1_micro_test = f1_score(labels.cpu(), pred.cpu(), average = 'micro')
    f1_macro_test = f1_score(labels.cpu(), pred.cpu(), average = 'macro')

    return round(f1_micro_train,4), round(f1_macro_train,4), round(f1_micro_test,4), round(f1_macro_test,4)

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="IMDBMULTI",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()



    for data in ["PROTEINS","PTC","IMDBBINARY","IMDBMULTI","MUTAG","NCI1","REDDITBINARY","REDDITMULTI5K"]:

        args.dataset = data

        print("Dataset:",args.dataset)

    #set up seeds and gpu device
        torch.manual_seed(0)
        np.random.seed(0)    
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)


        graphs, num_classes = load_data(args.dataset)

        graphs = add_attributes_norm(graphs, 10)

        for dim in [8,16,32,64,128,256]:
            args.hidden_dim = dim
            print(dim)


            ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
            micro_test = np.zeros((10,args.epochs))
            macro_test = np.zeros((10,args.epochs))
            micro_train = np.zeros((10,args.epochs))
            macro_train = np.zeros((10,args.epochs))
            for fold in range(10):
                torch.manual_seed(0)
                np.random.seed(0)    
                device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)

                # print('fold number:',fold)
                args.fold_idx = fold


                train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

                model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim
                , num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                score_list = [ "f1_micro_train", "f1_macro_train", "f1_micro_test", "f1_macro_test"]
                score_dict = defaultdict(float)
                for score in score_list:
                    score_dict[score] = 0

                for epoch in range(1, args.epochs + 1):
                    

                    avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
                    scheduler.step()
                    f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test = test(args, model, device, train_graphs, test_graphs, epoch)
                    scores = [ f1_micro_train, f1_macro_train, f1_micro_test, f1_macro_test]
                    for i, score in enumerate(score_list):
                        if score_dict[score] <  scores[i]:
                            score_dict[score] = scores[i]


                    micro_test[fold,epoch-1] = f1_micro_test
                    macro_test[fold,epoch-1] = f1_macro_test
                    micro_train[fold,epoch-1] = f1_micro_train
                    macro_train[fold,epoch-1] = f1_macro_train

            micro_test_epoch = np.mean(micro_test,0)
            macro_test_epoch = np.mean(macro_test,0)
            micro_train_epoch = np.mean(micro_train,0)
            macro_train_epoch = np.mean(macro_train,0)

            mic_test_ind =  micro_test_epoch.argmax() 
            mac_test_ind = macro_test_epoch.argmax()
            mic_train_ind = micro_train_epoch.argmax()
            mac_train_ind = macro_train_epoch.argmax() 

            ind = [mic_test_ind,mac_test_ind,mic_train_ind,mac_train_ind]


            print('test micro mean:',np.round(np.mean(micro_test,0)[ind],3),'std',np.round(np.std(micro_test,0)[ind],3))
            print('test macro mean:',np.round(np.mean(macro_test,0)[ind],3),'std',np.round(np.std(macro_test,0)[ind],3))
            print('train micro mean:',np.round(np.mean(micro_train,0)[ind],3),'std',np.round(np.std(micro_train,0)[ind],3))
            print('train macro mean:',np.round(np.mean(macro_train,0)[ind],3),'std',np.round(np.std(macro_train,0)[ind],3))
            
if __name__ == '__main__':
    main()
