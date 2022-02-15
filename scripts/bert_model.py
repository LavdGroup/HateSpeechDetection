"""
bert_model.py : customized BERT models
"""

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from utils import pad_sents
import sys


def sents_to_tensor(tokenizer, sents, device):
    """
    Convert the sentences (sequences) to tensor token representation
    :param tokenizer: BertTokenizer
    :param sents: list[str], list of untokenized sentences 
    :param device: torch.device
    :return: sents_tensor: torch.Tensor, shape(batch_size, max_sent_length)
    :return: masks_tensor: torch.Tensor, shape(batch_size, max_sent_length)
    :return: sents_lengths: torch.Tensor, shape(batch_size)
    """
    tokens_list = [tokenizer.tokenize(sent) for sent in sents]
    sents_lengths = [len(tokens) for tokens in tokens_list]
    tokens_list_padded = pad_sents(tokens_list, '[PAD]')
    sents_lengths = torch.tensor(sents_lengths, device=device)

    masks = []
    for tokens in tokens_list_padded:
        mask = [0 if token=='[PAD]' else 1 for token in tokens]
        masks.append(mask)
    masks_tensor = torch.tensor(masks, dtype=torch.long, device=device)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long, device=device)

    return sents_tensor, masks_tensor, sents_lengths


class DefaultModel(nn.Module):

    def __init__(self, device, n_class):
        """

        :param device: torch.device
        :param n_class: int
        """

        super(DefaultModel, self).__init__()

        self.n_class = n_class
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.n_class)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = device

    def forward(self, sents):
        """

        :param sents: list[str], list of sentences
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        pre_softmax = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = DefaultModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class NonlinearModel(nn.Module):

    def __init__(self, device, n_class, dropout_rate):
        """

        :param device: torch.device
        :param n_class: int
        """

        super(NonlinearModel, self).__init__()
        self.n_class = n_class
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, self.n_class)
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.LeakyReLU()

    def forward(self, sents):
        """

        :param sents: list[str], list of sentences
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=False)
        hidden1 = self.dropout(self.activation(self.linear1(pooled_output)))
        hidden2 = self.activation(self.linear2(hidden1))
        hidden3 = self.activation(self.linear3(hidden2))

        return hidden3

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NonlinearModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(n_class=self.n_class, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CustomBertLSTMModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class):
        """

        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        """

        super(CustomBertLSTMModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.lstm_hidden_size = 128

        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(768, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, sents):
        """

        :param sents: list[str], list of sentences
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=False)
        encoded_layers = encoded_layers.permute(1, 0, 2)  # (n_tokens, 768, n_sentences]
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(encoded_layers, sents_lengths.to("cpu"), enforce_sorted=False))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertLSTMModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CustomBertConvModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, out_channel=16):
        """

        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param out_channel: int, NOTE: out_channel per layer of BERT
        """

        super(CustomBertConvModel, self).__init__()

        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.out_channels = self.bert.config.num_hidden_layers * self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.conv = nn.Conv2d(in_channels=12,
                              out_channels=self.out_channels,
                              kernel_size=(3, 768),
                              groups=12)
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, sents):
        """

        :param sents:
        :return:
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=True)
        encoded_stack_layer = torch.stack(encoded_layers, 1)  # (batch_size, channel, max_sent_length, hidden_size)

        conv_out = self.conv(encoded_stack_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertConvModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(out_channel=self.out_channel,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)