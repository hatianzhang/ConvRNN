import time
import os
import numpy as np
import random

import torch
import torch.nn as nn
from torchtext import data

from args import get_args
from model import SmConvRNN
from trec_dataset import TrecDataset
import operator
import heapq
from torch.nn import functional as F

from evaluate import evaluate

args = get_args()
config = args
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

def set_vectors(field, vector_path):
    if os.path.isfile(vector_path):
        stoi, vectors, dim = torch.load(vector_path)
        field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

        for i, token in enumerate(field.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                field.vocab.vectors[i] = vectors[wv_index]
            else:
                # initialize <unk> with U(-0.25, 0.25) vectors
                field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
    else:
        print("Error: Need word embedding pt file")
        print("Error: Need word embedding pt file")
        exit(1)
    return field


# Set random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("You have Cuda but you're using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

QID = data.Field(sequential=False)
AID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
EXTERNAL = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
            postprocessing=data.Pipeline(lambda arr, _, train: [float(y) for y in arr]))

train, dev, test = TrecDataset.splits(QID, QUESTION, AID, ANSWER, EXTERNAL, LABEL)

QID.build_vocab(train, dev, test)
AID.build_vocab(train, dev, test)
QUESTION.build_vocab(train, dev, test)
ANSWER.build_vocab(train, dev, test)
LABEL.build_vocab(train, dev, test)

QUESTION = set_vectors(QUESTION, args.vector_cache)
ANSWER = set_vectors(ANSWER, args.vector_cache)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                           sort=False, shuffle=False)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                         sort=False,    shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                          sort=False, shuffle=False)

config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)

print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num", len(QUESTION.vocab))
print("LABEL.target_class:", len(LABEL.vocab))
print("LABELS:", LABEL.vocab.itos)
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))

if args.resume_snapshot:
    if args.cuda:
        pw_model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        pw_model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = SmConvRNN(config)
    model.static_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.nonstatic_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.static_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)
    model.nonstatic_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)

    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


parameter = filter(lambda p: p.requires_grad, model.parameters())

# the SM model originally follows SGD but Adadelta is used here
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay, eps=1e-6)
# A good lr is required to use Adam
# optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

# criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([0,0.2,0.8]))
criterion = nn.CrossEntropyLoss()
# marginRankingLoss = nn.MarginRankingLoss(margin = 1, size_average = True)



early_stop = False
iterations = 0
iters_not_improved = 0
epoch = 0
q2neg = {} # a dict from qid to a list of aid
question2answer = {} # a dict from qid to the information of both pos and neg answers
best_dev_map = 0
best_dev_mrr = 0
false_samples = {}

start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)

index2label = np.array(LABEL.vocab.itos) # ['<unk>', '0', '1']
index2qid = np.array(QID.vocab.itos) # torchtext index to qid in the TrecQA dataset
index2aid = np.array(AID.vocab.itos) # torchtext index to aid in the TrecQA dataset
index2question = np.array(QUESTION.vocab.itos)  # torchtext index to words appearing in questions in the TrecQA dataset
index2answer = np.array(ANSWER.vocab.itos) # torchtext index to words appearing in answers in the TrecQA dataset


# Questions, answers and ext_features for each qid
qid_correct_ans_dict = {}
qid_questions_dict = {}
qid_anwers_dict = {}
qid_ext_feat_dict = {}
qid_label_dict = {}


# pack the lists of question/answer/ext_feat into a torchtext batch
def get_qid_batch(qid, qid_questions_dict, qid_anwers_dict, qid_ext_feat_dict, qid_label_dict):
    new_batch = data.Batch()
    question = qid_questions_dict[qid]
    answers = qid_anwers_dict[qid]
    labels = qid_label_dict[qid]
    ext_feats = qid_ext_feat_dict[qid]



    size = len(qid_anwers_dict[qid])
    new_batch.batch_size = size
    new_batch.dataset = "trecqa"

    
    max_len_a = max([ans.size()[0] for ans in answers])

    padding_answers = []

    for ans in answers:
        padding_answers.append(F.pad(ans,(0, max_len_a - ans.size()[0]), value=1))

    setattr(new_batch, "answer", torch.stack(padding_answers))
    setattr(new_batch, "question", torch.stack(question.repeat(size,1)))
    setattr(new_batch, "ext_feat", torch.stack(ext_feats))
    setattr(new_batch, "label", torch.stack(labels))
    return new_batch



while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_map))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        model.train(); optimizer.zero_grad()
        scores = model(batch)
        n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, batch.label)
        loss.backward()
        optimizer.step()

        # Evaluate performance on validation set
        if iterations % args.dev_every == 1:
            # switch model into evaluation mode
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0
            dev_losses = []
            instance = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                qid_array = index2qid[np.transpose(dev_batch.qid.cpu().data.numpy())]
                true_label_array = index2label[np.transpose(dev_batch.label.cpu().data.numpy())]

                scores = model(dev_batch)
                n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(scores, dev_batch.label)
                dev_losses.append(dev_loss.data[0])
                index_label = np.transpose(torch.max(scores, 1)[1].view(dev_batch.label.size()).cpu().data.numpy())
                label_array = index2label[index_label]
                # get the relevance scores
                score_array = scores[:, 2].cpu().data.numpy()
                for i in range(dev_batch.batch_size):
                    this_qid, predicted_label, score, gold_label = qid_array[i], label_array[i], score_array[i], true_label_array[i]
                    instance.append((this_qid, predicted_label, score, gold_label))


            dev_map, dev_mrr = evaluate(instance, config.dataset, 'valid', config.mode)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          sum(dev_losses) / len(dev_losses), train_acc, dev_map))

            # Update validation results
            if dev_map > best_dev_map:
                iters_not_improved = 0
                best_dev_map = dev_map
                snapshot_path = os.path.join(args.save_path, args.dataset, args.mode+'_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        # if iterations % args.log_every == 1:
        #     # print progress message
        #     print(log_template.format(time.time() - start,
        #                               epoch, iterations, 1 + batch_idx, len(train_iter),
        #                               100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
        #                               n_correct / n_total * 100, ' ' * 12))

      
