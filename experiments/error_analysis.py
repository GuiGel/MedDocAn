""" CF: Context when predicting with FLERT models #2650
https://github.com/flairNLP/flair/issues/2650

TODO Verify that the context is not considered after and before each
-DOCSTART- ...
"""
from copy import copy
from pathlib import Path
from typing import List, Set

import flair
import pandas as pd
import torch
import torch.nn.functional as F
from flair.data import Dataset
from flair.datasets.base import DataLoader, FlairDatapointDataset
from flair.file_utils import Tqdm
from flair.models import SequenceTagger

from meddocan.data.corpus import MEDDOCAN

meddocan = MEDDOCAN(
    sentences=True, in_memory=True, document_separator_token="-DOCSTART-"
)

model_loc = "/home/wave/Project/MedDocAn/experiments_2/corpus_sentence_grid_search_flert_beto_final_models/an_wh_rs_False_dpt_0_emb_beto-cased-context_FT_True_Ly_-1_seed_1_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/best-model.pt"
model = SequenceTagger.load(model_loc)

data_points = copy(meddocan.dev)


mini_batch_size, stop = 4, -1
# mini_batch_size, stop = 4*16, 80/16

return_loss = True

probabilities_for_all_classes = True
label_name = "predicted"

if not isinstance(data_points, Dataset):
    data_points = FlairDatapointDataset(data_points)


with torch.no_grad():

    # loss calculation
    eval_loss = torch.zeros(1, device=flair.device)
    average_over = 0

    # variables for printing
    lines: List[str] = []

    # variables for computing scores
    all_spans: Set[str] = set()
    all_true_values = {}
    all_predicted_values = {}

    loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=4)
    sentence_id = 0

    overall_loss = torch.zeros(1, device=flair.device)
    batch_no = 0
    label_count = 0

    all_tokens: List[str] = []
    all_gold_labels: List[str] = []
    all_predicted_labels: List[str] = []
    all_scores: List[torch.Tensor] = []
    all_loss: List[torch.Tensor] = []

    for batch in Tqdm.tqdm(loader):
        # remove any previously predicted labels
        for datapoint in batch:
            datapoint.remove_labels(label_name)
        # loss_and_count = model.predict(
        #                     batch,
        #                     embedding_storage_mode="cpu",
        #                     mini_batch_size=mini_batch_size,
        #                     label_name="predicted",
        #                     return_loss=True,
        #                 )
        #
        # print(loss_and_count)

        # This should be interesting to have the different result without reorder...
        # label_name = model.tag_type

        with torch.no_grad():
            # filter empty sentences
            sentences = [sentence for sentence in batch if len(sentence) > 0]
            # print(f"\n------------- {batch_no=} ---------------------")
            # print(f"{sentences}")
            batch_no += 1

            # stop if all sentences are empty
            if not sentences:
                continue

            # get features from forward propagation the same as the function model.forward but without reordering
            # a, b = model.forward(sentences)
            # print(f"{a.shape=}, {b=}")

            if not isinstance(sentences, list):
                sentences = [sentences]

            model.embeddings.embed(sentences)

            # make a zero-padded tensor for the whole sentence
            lengths, sentence_tensor = model._make_padded_tensor_for_batch(
                sentences
            )  # concatenate sentences together
            # print(f"{lengths=}, {sentence_tensor.shape=}")

            # sort tensor in decreasing order based on lengths of sentences in batch
            # sorted_lengths, length_indices = lengths.sort(dim=0, descending=True)
            # sentences = [sentences[i] for i in length_indices]
            # sentence_tensor = sentence_tensor[length_indices]

            # ----- Forward Propagation -----
            if model.use_dropout:
                sentence_tensor = model.dropout(sentence_tensor)
            if model.use_word_dropout:
                sentence_tensor = model.word_dropout(sentence_tensor)
            if model.use_locked_dropout:
                sentence_tensor = model.locked_dropout(sentence_tensor)

            if model.reproject_embeddings:
                sentence_tensor = model.embedding2nn(sentence_tensor)

            if model.use_rnn:
                from torch.nn.utils.rnn import (
                    pack_padded_sequence,
                    pad_packed_sequence,
                )

                packed = pack_padded_sequence(
                    sentence_tensor,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                rnn_output, hidden = model.rnn(packed)
                sentence_tensor, output_lengths = pad_packed_sequence(
                    rnn_output, batch_first=True
                )

            if model.use_dropout:
                sentence_tensor = model.dropout(sentence_tensor)
            if model.use_locked_dropout:
                sentence_tensor = model.locked_dropout(sentence_tensor)

            # linear map to tag space
            features = model.linear(sentence_tensor)
            # print(
            #     f"model.linear({sentence_tensor.shape=}) --> {features.shape=}"
            # )

            # Depending on whether we are using CRF or a linear layer, scores is either:
            # -- A tensor of shape (batch size, sequence length, tagset size, tagset size) for CRF
            # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
            if model.use_crf:
                features = model.crf(features)
                scores = (features, lengths, model.crf.transitions)
            else:
                scores = model._get_scores_from_features(features, lengths)
                # print(f"model._get_scores_from_features({features.shape=}) --> {scores.shape=}")

            # get the gold labels
            gold_labels = model._get_gold_labels(sentences)
            # print(
            #     f"model._get_gold_labels({len(sentences)=}) --> {gold_labels=}"
            # )
            # -------------

            # remove previously predicted labels of this type
            for sentence in batch:
                sentence.remove_labels(label_name)

            # if return_loss, get loss value
            if return_loss:
                loss = model._calculate_loss(scores, gold_labels)
                # print(
                #   f"model._calculate_loss({scores.shape=}, {[len(l) for l in gold_labels]}) --> {loss}"
                # )
                overall_loss += loss[0]
                label_count += loss[1]

                if not any(gold_labels):
                    loss = (
                        torch.tensor(
                            0.0, requires_grad=True, device=flair.device
                        ),
                        1,
                    )

                # create labels tensor
                labels = torch.tensor(
                    [
                        model.label_dictionary.get_idx_for_item(label[0])
                        if len(label) > 0
                        else model.label_dictionary.get_idx_for_item("O")
                        for label in gold_labels
                    ],
                    dtype=torch.long,
                    device=flair.device,
                )
                # print(f"{labels=}")
                token_loss_func = torch.nn.CrossEntropyLoss(
                    weight=model.loss_weights, reduction="none"
                )
                loss_tokens = token_loss_func(scores, labels)
                # print(
                #     f"token_loss_func({scores.shape}, {labels.shape=})) --> {loss_tokens.shape=}"
                # )

                # We have a bug in our code since we are note able to compute the summed loss from the token's loss
                # print(torch.eq(loss[0], torch.sum(loss_token)), f"{loss[0]} != {torch.sum(loss_token)}")

                # Compute predictions
                softmax_batch = F.softmax(scores, 1).cpu()
                # print(f"{softmax_batch.shape=}")
                scores_batch, prediction_batch = torch.max(
                    softmax_batch, dim=1
                )
                predictions = []
                all_tags = []

                # print(f"{scores_batch.shape=}, {prediction_batch.shape=}")

                for sentence in sentences:
                    # print(sentence, len(sentence))
                    scores = scores_batch[: len(sentence)]
                    predictions_for_sentence = prediction_batch[
                        : len(sentence)
                    ]
                    token_loss = loss_tokens[: len(sentence)]

                    predictions.append(
                        [
                            (
                                model.label_dictionary.get_item_for_index(
                                    prediction.tolist()
                                ),
                                score.item(),
                                tok_loss.tolist(),
                            )
                            for token, score, prediction, tok_loss in zip(
                                sentence,
                                scores,
                                predictions_for_sentence,
                                token_loss,
                            )
                        ]
                    )
                    scores_batch = scores_batch[len(sentence) :]
                    prediction_batch = prediction_batch[len(sentence) :]
                    loss_tokens = loss_tokens[len(sentence) :]

            if probabilities_for_all_classes:
                lengths = [len(sentence) for sentence in batch]
                # Returns all scores for each tag in tag dictionary.
                all_tags = model._all_scores_for_token(
                    batch, softmax_batch, lengths
                )
                # print(f"{[len(l) for l in all_tags]=}")

            # add prediction to Sentence
            tokens_with_loss = []
            for sentence, sentence_predictions in zip(sentences, predictions):
                for token, label in zip(sentence.tokens, sentence_predictions):
                    if label[0] in ["O", "_"]:
                        # print(token.text, label)
                        continue
                    token.add_label(
                        typename=label_name, value=label[0], score=label[1]
                    )
                    # print(token.text, f"{label=}")

            # prepare pandas dataframe row
            for sentence, sentence_predictions in zip(sentences, predictions):
                sentence_gold_labels = gold_labels[: len(sentence)]
                input_ids = []
                tokens_list = []
                score_list = []
                gold_labels_list = []
                predicted_labels_list = []
                loss_list = []
                for token, (predicted_label, _score, _loss), gold_label in zip(
                    sentence, sentence_predictions, sentence_gold_labels
                ):
                    tokens_list.append(token.text)
                    gold_labels_list.append(gold_label[0])
                    predicted_labels_list.append(predicted_label)
                    loss_list.append(_loss)
                    score_list.append(_score)

                # print(input_ids)
                # print(f"{tokens_list=}")
                # print(f"{gold_labels_list=}")
                # print(f"{predicted_labels_list=}")
                # print(f"{loss_list=}")
                gold_labels = gold_labels[len(sentence) :]

                # Fill list to make dataframe
                all_tokens.append(tokens_list)
                all_scores.append(score_list)
                all_predicted_labels.append(predicted_labels_list)
                all_gold_labels.append(gold_labels_list)
                all_loss.append(loss_list)

            # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
            for (sentence, sent_all_tags) in zip(sentences, all_tags):
                for (token, token_all_tags) in zip(
                    sentence.tokens, sent_all_tags
                ):
                    # print(f"{label_name=}, {token_all_tags=}")
                    token.add_tags_proba_dist(label_name, token_all_tags)

            if batch_no == stop:
                # print(f"{overall_loss=}")
                break


df = pd.DataFrame(
    [all_tokens, all_predicted_labels, all_gold_labels, all_loss, all_scores],
    index=["token", "predicted_label", "label", "loss", "scores"],
).T
df.to_csv(Path(__file__).parent / "dev.csv")
