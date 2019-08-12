from typing import List, Union

import flair
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from flair.data import Sentence
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings, StackedEmbeddings
from flair.nn import LockedDropout, WordDropout


class CNN_Text(nn.Module):

    def __init__(self,
                 input_dim = 100,
                 output_dim = 64,
                 kernel_sizes = [1]
):
        super(CNN_Text, self).__init__()

        self.convs1 = nn.ModuleList([nn.Conv2d(1, output_dim, (kernel_size, input_dim)) for kernel_size in kernel_sizes])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):

        x = x.permute(1,0,2).unsqueeze(1)  # (batch_size, 1, seq_len, dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        return x


class DocumentCNNEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: List[TokenEmbeddings],
        hidden_size=128,
        reproject_words: bool = True,
        reproject_words_dimension: int = None,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
    ):
        """
        code mainly copy-pasted from DocumentRNNEmbeddings-class
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)


        self.reproject_words = reproject_words

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False

        self.__embedding_length: int = hidden_size

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )
        self.cnn = CNN_Text(input_dim=self.embeddings_dimension,
                            output_dim=hidden_size,
                            kernel_sizes=[1]
                            )

        self.name = "document_" + self.cnn._get_name()

        # dropouts
        if locked_dropout > 0.0:
            self.dropout: torch.nn.Module = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.cnn.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        # first, sort sentences by number of tokens
        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            # PADDING: pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(
                        self.length_of_all_token_embeddings, dtype=torch.float
                    ).unsqueeze(0)
                )

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(flair.device)

            sentence_states = word_embeddings_tensor

            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # --------------------------------------------------------------------
        # GET REPRESENTATION FOR ENTIRE BATCH
        # --------------------------------------------------------------------
        sentence_tensor = torch.cat(all_sentence_tensors, 1)

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        # use word dropout if set
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        sentence_tensor = self.dropout(sentence_tensor)

        outputs = self.cnn(sentence_tensor)

        outputs = self.dropout(outputs)

        # --------------------------------------------------------------------
        # EXTRACT EMBEDDINGS FROM RNN
        # --------------------------------------------------------------------
        for sentence_no, length in enumerate(lengths):
            sentences[sentence_no].set_embedding(self.name, outputs[sentence_no])

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass
