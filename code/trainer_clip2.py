from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses_clip import words_loss
from miscc.losses_clip import discriminator_loss, generator_loss, KL_loss
import os, shutil
import time
import numpy as np
import sys, pickle

import clip

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self, model):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()  #load encoder
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches

        if cfg.TRAIN.CLIP_SENTENCODER:
            print("CLIP Sentence Encoder: True")

        if cfg.TRAIN.CLIP_LOSS:
            print("CLIP Loss: True")

        if cfg.TRAIN.EXTRA_LOSS:
            print("Extra DAMSM Loss in G: True")
            print("DAMSM Weight: ", cfg.TRAIN.WEIGHT_DAMSM_LOSS)

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()

                # imgs, captions, cap_lens, class_ids, keys = prepare_data(data) #new sents:, sents
                # new: return raw texts
                imgs, captions, cap_lens, class_ids, keys, texts = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                # new: rename
                words_embs_damsm, sent_emb_damsm = text_encoder(captions, cap_lens, hidden)
                #print('captions shape from trainer: ', captions.shape) torch.Size([12, 18])
                #print('sentence emb size: ', sent_emb.shape) torch.Size([12, 256])
                words_embs_damsm, sent_emb_damsm = words_embs_damsm.detach(), sent_emb_damsm.detach()
                #print('sentence emb size after detach: ', sent_emb[0]) torch.Size([12, 256])
                mask = (captions == 0)
                num_words = words_embs_damsm.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                # new: use clip sentence encoder
                if cfg.TRAIN.CLIP_SENTENCODER or cfg.TRAIN.CLIP_LOSS:
                    sents = []
                    # randomly select one paragraph for each training example
                    for idx in range(len(texts)):
                        sents_per_image = texts[idx].split('\n')    #new: '\n' rather than '.'
                        if len(sents_per_image)>1:
                            sent_ix = np.random.randint(0, len(sents_per_image)-1)
                        else:
                            sent_ix = 0
                        sents.append(sents_per_image[sent_ix])
                    #print('sents: ', sents)

                    sent = clip.tokenize(sents)#.to(device)

                    # load clip
                    #model = torch.jit.load("model.pt").cuda().eval()    # ViT-B/32
                    sent_input = sent.cuda()

                    with torch.no_grad():
                        sent_emb_clip = model.encode_text(sent_input).float()
                        if cfg.TRAIN.CLIP_SENTENCODER:
                            sent_emb = sent_emb_clip
                        else:
                            sent_emb = sent_emb_damsm
                else:
                    sent_emb_clip = 0
                    sent_emb = sent_emb_damsm

                words_embs = words_embs_damsm

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                
                # new: pass clip model and sent_emb_damsm for CLIP_LOSS = True
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                        words_embs, sent_emb, match_labels, cap_lens, class_ids, model, sent_emb_damsm, sent_emb_clip)

                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or epoch % 10 ==0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir, model):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            if cfg.GPU_ID != -1:
                netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            if cfg.GPU_ID != -1:
                text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM

            with torch.no_grad():
                noise = Variable(torch.FloatTensor(batch_size, nz))
                if cfg.GPU_ID != -1:
                    noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            #new
            if cfg.TRAIN.CLIP_SENTENCODER:
                print("Use CLIP SentEncoder for sampling")

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break

                    #imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                    #new
                    imgs, captions, cap_lens, class_ids, keys, texts = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb= text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    # new
                    if cfg.TRAIN.CLIP_SENTENCODER:

                        # random select one paragraph for each training example
                        sents = []
                        for idx in range(len(texts)):
                            sents_per_image = texts[idx].split('\n') # new 3/11
                            if len(sents_per_image) > 1:
                                sent_ix = np.random.randint(0, len(sents_per_image) - 1)
                            else:
                                sent_ix = 0
                            sents.append(sents_per_image[sent_ix])
                            with open('%s/%s' % (save_dir,'eval_sents.txt'),'a+') as f:
                                f.write(sents_per_image[sent_ix]+'\n')
                        # print('sents: ', sents)

                        sent = clip.tokenize(sents)  # .to(device)

                        # load clip
                        #model = torch.jit.load("model.pt").cuda().eval()
                        sent_input = sent
                        if cfg.GPU_ID != -1:
                            sent_input = sent.cuda()
                        # print("text input", sent_input)
                        with torch.no_grad():
                            sent_emb = model.encode_text(sent_input).float()

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/fake/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                            print('Make a new folder: ', f'{save_dir}/real')
                            mkdir_p(f'{save_dir}/real')
                            print('Make a new folder: ', f'{save_dir}/text')
                            mkdir_p(f'{save_dir}/text')
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)
                        temp = keys[j].replace('b','').replace("'",'')
                        shutil.copy(f"../data/Face/images/{temp}.jpg", f"{save_dir}/real/")
                        shutil.copy(f"../data/Face/text/{temp}.txt", f"{save_dir}/text/")

    def embedding(self, split_dir, model):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            if cfg.GPU_ID != -1:
                netG.cuda()
            netG.eval()
            #
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            print(img_encoder_path)
            print('Load image encoder from:', img_encoder_path)
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            if cfg.GPU_ID != -1:
                image_encoder = image_encoder.cuda()
            image_encoder.eval()
            
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            if cfg.GPU_ID != -1:
                text_encoder = text_encoder.cuda()
            text_encoder.eval()
        
            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM

            with torch.no_grad():
                noise = Variable(torch.FloatTensor(batch_size, nz))
                if cfg.GPU_ID != -1:
                    noise = noise.cuda()

            # the path to save generated images
            save_dir = model_dir[:model_dir.rfind('.pth')]

            cnt = 0

            # new
            if cfg.TRAIN.CLIP_SENTENCODER:
                print("Use CLIP SentEncoder for sampling")
            img_features = dict()
            txt_features = dict()

            with torch.no_grad():
                for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                    for step, data in enumerate(self.data_loader, 0):
                        cnt += batch_size
                        if step % 100 == 0:
                            print('step: ', step)

                        imgs, captions, cap_lens, class_ids, keys, texts = prepare_data(data)

                        hidden = text_encoder.init_hidden(batch_size)
                        # words_embs: batch_size x nef x seq_len
                        # sent_emb: batch_size x nef
                        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                        mask = (captions == 0)
                        num_words = words_embs.size(2)
                        if mask.size(1) > num_words:
                            mask = mask[:, :num_words]

                        if cfg.TRAIN.CLIP_SENTENCODER:

                            # random select one paragraph for each training example
                            sents = []
                            for idx in range(len(texts)):
                                sents_per_image = texts[idx].split('\n') # new 3/11
                                if len(sents_per_image) > 1:
                                    sent_ix = np.random.randint(0, len(sents_per_image) - 1)
                                else:
                                    sent_ix = 0
                                sents.append(sents_per_image[0])
                            # print('sents: ', sents)

                            sent = clip.tokenize(sents)  # .to(device)

                            # load clip
                            #model = torch.jit.load("model.pt").cuda().eval()
                            sent_input = sent
                            if cfg.GPU_ID != -1:
                                sent_input = sent.cuda()
                            # print("text input", sent_input)
                            sent_emb_clip = model.encode_text(sent_input).float()
                            if CLIP:
                                sent_emb = sent_emb_clip
                        #######################################################
                        # (2) Generate fake images
                        ######################################################
                        noise.data.normal_(0, 1)
                        fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                        if CLIP:
                            images=[]
                            for j in range(fake_imgs[-1].shape[0]):
                                image = fake_imgs[-1][j].cpu().clone()
                                image = image.squeeze(0)
                                unloader = transforms.ToPILImage()
                                image = unloader(image)

                                image = preprocess(image.convert("RGB"))    # 256*256 -> 224*224
                                images.append(image)

                            image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
                            image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

                            image_input = torch.tensor(np.stack(images)).cuda()
                            image_input -= image_mean[:, None, None]
                            image_input /= image_std[:, None, None]
                            cnn_codes = model.encode_image(image_input).float()
                        else:
                            region_features, cnn_codes = image_encoder(fake_imgs[-1])
                        for j in range(batch_size):
                            cnn_code = cnn_codes[j]

                            temp = keys[j].replace('b','').replace("'",'')
                            img_features[temp] = cnn_code.cpu().numpy()
                            txt_features[temp] = sent_emb[j].cpu().numpy()
            with open(save_dir+".pkl", 'wb') as f:
                pickle.dump(img_features, f)
            with open(save_dir+"_text.pkl", 'wb') as f:
                pickle.dump(txt_features, f)
    ''' # new: disable due to unmodified
    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM

                with torch.no_grad():
                    captions = Variable(torch.from_numpy(captions))
                    cap_lens = Variable(torch.from_numpy(cap_lens))

                    captions = captions.cuda()
                    cap_lens = cap_lens.cuda()

                for i in range(1):  # 16
                    with torch.no_grad():
                        noise = Variable(torch.FloatTensor(batch_size, nz))
                        noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
    '''