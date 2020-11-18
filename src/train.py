"""
Adapted from 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py
"""

import torch
from dataset import LocalizedCOCO
from models import Encoder, DecoderWithAttention
from utils import AverageMeter, clip_gradient

class PARAMS:

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    trace_weight = 1.0
    desc = 'trace_weight_1.0'

    # Training parameters
    start_epoch = 0
    epochs = 15  # number of epochs to train for (if early stopping is not triggered)
    batch_size = 32
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    grad_clip = 5.  # clip gradients at an absolute value of
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
    print_freq = 100  # print training/validation stats every __ batches
    fine_tune_encoder = False  # fine-tune encoder?
    checkpoint = None  # path to checkpoint, None if none

def main():
    """
    Training and validation.
    """

    trainset = LocalizedCOCO('coco_localized/train', 'coco_localized/vocab.pickle')

    decoder = DecoderWithAttention(
        attention_dim=PARAMS.attention_dim,
        embed_dim=PARAMS.emb_dim,
        decoder_dim=PARAMS.decoder_dim,
        vocab_size=len(trainset.vocab),
        dropout=PARAMS.dropout,
        trace_weight=PARAMS.trace_weight,
        device=PARAMS.device).to(PARAMS.device)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=PARAMS.decoder_lr)
    encoder = Encoder().to(PARAMS.device)
    encoder.fine_tune(False)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trainset.vocab['<pad>']).to(PARAMS.device)

    train_loader = torch.utils.data.DataLoader(trainset,
        batch_size=PARAMS.batch_size, 
        shuffle=True, 
        num_workers=6)

    # Epochs
    for epoch in range(0, PARAMS.epochs):

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=None,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

    # Save checkpoint
    state = {'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': None,
             'decoder_optimizer': decoder_optimizer}
    filename = f'checkpoint-{PARAMS.desc}.pth.tar'
    torch.save(state, filename)

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    losses = AverageMeter()  # loss (per word decoded)

    # Batches
    for i, (imgs, caps, caplens, traces) in enumerate(train_loader):
        
        # print('got here first')

        # Move to GPU, if available
        imgs = imgs.to(PARAMS.device)
        caps = caps.to(PARAMS.device)
        caplens = caplens.to(PARAMS.device)
        traces = traces.to(PARAMS.device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, traces)
        
        # print('got here')

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Calculate loss
        loss = criterion(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))

        # Add doubly stochastic attention regularization
        loss += PARAMS.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if PARAMS.grad_clip is not None:
            clip_gradient(decoder_optimizer, PARAMS.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, PARAMS.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))

        # Print status
        if i % PARAMS.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})')

if __name__ == '__main__':
    main()
