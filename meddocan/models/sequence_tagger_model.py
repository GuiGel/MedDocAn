"""This module patch the ``flair.models.sequence_tagger_model.SequenceTagger\
.predict`` method in order to not reverse sort all sequences by their length \
before creating the batch that will be feed to the model. By doing this we \
preserve the possibility for the ``flair.data.Sentence`` to have the good \
left and right context (see Flair issues #2350 and #2650).
"""
import logging
from typing import List, Optional, Union

import flair.nn
import torch
import torch.nn
from fastcore.basics import patch_to
from flair.data import Sentence, Span
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.models.sequence_tagger_model import SequenceTagger
from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio
from flair.training_utils import store_embeddings
from tqdm import tqdm

log = logging.getLogger("flair")


@patch_to(SequenceTagger)
def predict(
    self: SequenceTagger,
    sentences: Union[List[Sentence], Sentence],
    mini_batch_size: int = 32,
    return_probabilities_for_all_classes: bool = False,
    verbose: bool = False,
    label_name: Optional[str] = None,
    return_loss=False,
    embedding_storage_mode="none",
    force_token_predictions: bool = False,
):  # type: ignore
    """
    Predicts labels for current batch with CRF or Softmax.
    :param sentences: List of sentences in batch
    :param mini_batch_size: batch size for test data
    :param return_probabilities_for_all_classes: Whether to return probabilities for all classes
    :param verbose: whether to use progress bar
    :param label_name: which label to predict
    :param return_loss: whether to return loss value
    :param embedding_storage_mode: determines where to store embeddings - can be "gpu", "cpu" or None.
    """
    if label_name is None:
        label_name = self.tag_type

    with torch.no_grad():
        if not sentences:
            return sentences

        # make sure its a list
        if not isinstance(sentences, list) and not isinstance(
            sentences, flair.data.Dataset
        ):
            sentences = [sentences]

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]

        # reverse sort all sequences by their length
        reordered_sentences = (
            sentences  # sorted(sentences, key=lambda s: len(s), reverse=True)
        )

        if len(reordered_sentences) == 0:
            return sentences

        dataloader = DataLoader(
            dataset=FlairDatapointDataset(reordered_sentences),
            batch_size=mini_batch_size,
        )
        # progress bar for verbosity
        if verbose:
            dataloader = tqdm(dataloader, desc="Batch inference")

        overall_loss = torch.zeros(1, device=flair.device)
        batch_no = 0
        label_count = 0
        for batch in dataloader:

            batch_no += 1

            # stop if all sentences are empty
            if not batch:
                continue

            # get features from forward propagation
            features, gold_labels = self.forward(batch)

            # remove previously predicted labels of this type
            for sentence in batch:
                sentence.remove_labels(label_name)

            # if return_loss, get loss value
            if return_loss:
                loss = self._calculate_loss(features, gold_labels)
                overall_loss += loss[0]
                label_count += loss[1]

            # Sort batch in same way as forward propagation
            lengths = torch.LongTensor([len(sentence) for sentence in batch])
            _, sort_indices = lengths.sort(dim=0, descending=True)
            batch = [batch[i] for i in sort_indices]

            # make predictions
            if self.use_crf:
                predictions, all_tags = self.viterbi_decoder.decode(
                    features, return_probabilities_for_all_classes, batch
                )
            else:
                predictions, all_tags = self._standard_inference(
                    features, batch, return_probabilities_for_all_classes
                )

            # add predictions to Sentence
            for sentence, sentence_predictions in zip(batch, predictions):

                # BIOES-labels need to be converted to spans
                if self.predict_spans and not force_token_predictions:
                    sentence_tags = [
                        label[0] for label in sentence_predictions
                    ]
                    sentence_scores = [
                        label[1] for label in sentence_predictions
                    ]
                    predicted_spans = get_spans_from_bio(
                        sentence_tags, sentence_scores
                    )
                    for predicted_span in predicted_spans:
                        span: Span = sentence[
                            predicted_span[0][0] : predicted_span[0][-1] + 1
                        ]
                        span.add_label(
                            label_name,
                            value=predicted_span[2],
                            score=predicted_span[1],
                        )

                # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
                else:
                    for token, label in zip(
                        sentence.tokens, sentence_predictions
                    ):
                        if label[0] in ["O", "_"]:
                            continue
                        token.add_label(
                            typename=label_name, value=label[0], score=label[1]
                        )

            # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
            for (sentence, sent_all_tags) in zip(batch, all_tags):
                for (token, token_all_tags) in zip(
                    sentence.tokens, sent_all_tags
                ):
                    token.add_tags_proba_dist(label_name, token_all_tags)

            store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return overall_loss, label_count
