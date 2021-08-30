import torch
import sys
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, attender_dim, attendee_dim, output_dim, kq_dim):
        super(Attention,self).__init__()

        self.attender_dim = attender_dim
        self.attendee_dim = attendee_dim
        self.output_dim = output_dim
        self.kq_dim = kq_dim

        self.weightsQ =  torch.nn.Parameter(torch.zeros((attender_dim, kq_dim)))
        torch.nn.init.xavier_uniform_(self.weightsQ)

        self.weightsK = torch.nn.Parameter(torch.zeros((attendee_dim, kq_dim)))
        torch.nn.init.xavier_uniform_(self.weightsK)

        self.weightsV = torch.nn.Parameter(torch.zeros((attender_dim, output_dim)))
        torch.nn.init.xavier_uniform_((self.weightsV))

        self.bias = torch.nn.Parameter(torch.zeros((output_dim)))


    def forward(self, attenders, attendees):
        a = ((attenders @ self.weightsQ) @ (attendees @ self.weightsK).T) / torch.sqrt(len(attenders))
        b = attendees @ self.weightsV + self.bias
        z = torch.softmax(a) * b

        return z


def main(argv):
    print("hello")


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, kq_dim):
        super(SelfAttention, self).__init__()

        self.attention = Attention(input_dim, input_dim, output_dim, kq_dim)

    def forward(self, input):

        return self.attention.forward(input, input)

if __name__ == "__main__":
    main(sys.argv)