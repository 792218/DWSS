import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

import lib.utils as utils
from models.basic_model import BasicModel

from models.backbone.swin_transformer_backbone import SwinTransformer as STBackbone

from models.encoder_decoder.PureT_encoder import Encoder
from models.encoder_decoder.PureT_decoder import Decoder

from models.encoder_decoder.my_encoder import PlainEncoder
from models.encoder_decoder.my_decoder import PlainDecoder

from models.Swin import swin_transformer

# For masked MSA
"""
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
"""
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return subsequent_mask == 0

class Captioner(BasicModel):
    def __init__(self):
        super(Captioner, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE
        
        weight_path = None
        if cfg.MODEL.BACKBONE == 'small':
            self.backbone = swin_transformer.SwinTransformer(
                            img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=21841,
                            embed_dim=96,
                            depths=[2, 2, 18, 2],
                            num_heads=[ 3, 6, 12, 24 ],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            drop_path_rate=0.1,
                            ) # small

            weight_path = '../swin_small_patch4_window7_224_22kto1k_finetune.pth'
        elif cfg.MODEL.BACKBONE == 'base':
            self.backbone = swin_transformer.SwinTransformer(
                            img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=21841,
                            embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            drop_path_rate=0.1, 
                            ) # base
            weight_path = '../swin_base_patch4_window7_224_22kto1k.pth'
        elif cfg.MODEL.BACKBONE == 'base384':
            self.backbone = swin_transformer.SwinTransformer(
                            img_size=384,
                            patch_size=4,
                            in_chans=3,
                            num_classes=21841,
                            embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=12,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            drop_path_rate=0.1,
                            ) # base 384
            weight_path = '../swin_base_patch4_window12_384_22kto1k.pth'
        elif cfg.MODEL.BACKBONE == 'large':
            self.backbone = swin_transformer.SwinTransformer(
                            img_size=384,
                            patch_size=4,
                            in_chans=3,
                            num_classes=21841,
                            embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=12,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            drop_path_rate=0.1,
                            ) # large
            weight_path = '../swin_large_patch4_window12_384_22kto1k.pth'
        else:
            self.backbone = nn.Identity()

        if weight_path is not None:
            self.backbone.load_state_dict(
                torch.load(weight_path, map_location='cpu')['model'], 
                strict=False)

        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
                )
            
        self.encoder = PlainEncoder(
            num_layers = cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
            dim = cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            num_heads = cfg.MODEL.ATT_FEATS_EMBED_DIM // 64,
            attn_drop = 0.1,
            proj_drop = 0.1,
            drop_path = 0.,
            qkv_bias = True,             
            act_layer = utils.activation(cfg.MODEL.ATT_ACT),
            )
       
        self.decoder = PlainDecoder(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            depth = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads = cfg.MODEL.BILINEAR.HEAD, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            qkv_bias = True, 
            act_layer = utils.activation(cfg.MODEL.BILINEAR.ACT),
            ) 
        
 
    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        
        # words mask [B, L, L]
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:, 0] = 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        att_mask = None
        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats) # [b, n, d]
        att_feats = self.encoder(att_feats, att_mask)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        logits = self.decoder(seq, att_feats, seq_mask, att_mask) # [b*5, seq_len, vocab_size] 
        
        return logits

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT] 
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        
        # state[0]: [b, t-1]
        # wt: [b]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0], wt.unsqueeze(dim=1)], dim=1) # [b, t] 
            
        seq_mask = None
        logits = self.decoder(wt.unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(dim=1) # [b, vocab_size]
        return logits, [ys]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        
        att_mask = None
        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        att_feats = self.encoder(att_feats, att_mask)

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        
        seq_logprob = torch.zeros((batch_size, 1, 1), device=torch.device('cuda'))
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1), device=torch.device('cuda'))
        state = None
        wt = torch.zeros(batch_size, dtype=torch.int32, device=torch.device('cuda'))
        outputs = []

        self.decoder.init_buffer(batch_size)
        
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state

            logits, state = self.get_logprobs_state(**kwargs)
            word_logprob = F.log_softmax(logits, dim=-1) # @@@@@

            # [b*cur_beam_size, vocab_size] --> [b, cur_beam_size, vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [b, cur_beam_size, vocab_size]
            candidate_logprob = seq_logprob + word_logprob # t=0: [b, 1, vocab_size] t>0: [b, beam_size, vocab_size]

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1) # 0 is the token <end>
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [b, beam_size], [b, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # update buffer
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
       
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                att_feats = utils.expand_tensor(att_feats, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                
                # state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                # state[0] = state[0].unsqueeze(0)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            if seq_mask.sum() == 0:
                break
   
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
       
        outputs = torch.cat(outputs, -1) # [b, beam_size, seq_len]
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, outputs.shape[-1])) # [b, beam_size, seq_len]
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, log_probs.shape[-1]))

        if not self.training:
            outputs = outputs.contiguous()[:, 0] # [b, seq_len]
            log_probs = log_probs.contiguous()[:, 0] # [b, seq_len]
        else:
            outputs = outputs.contiguous().view(batch_size*beam_size, -1)
            log_probs = log_probs.contiguous().view(batch_size*beam_size, -1)

        self.decoder.clear_buffer()

        return outputs, log_probs

    def decode(self, **kwargs):
        # beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]

        att_mask = None
        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        att_feats = self.encoder(att_feats, att_mask)

        if self.training:
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)

        batch_size = att_feats.size(0)
        self.decoder.init_buffer(batch_size)
        
        state = None
        sents = torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.int32, device=torch.device('cuda'))
        logprobs = torch.zeros(batch_size, cfg.MODEL.SEQ_LEN, device=torch.device('cuda'))
        wt = torch.zeros(batch_size, dtype=torch.int32, device=torch.device('cuda'))
        unfinished = wt.eq(wt)
        
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        # inference word by word
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            
            logits, state = self.get_logprobs_state(**kwargs) # [b*5, vocab_size]

            if greedy_decode:
                logprobs_t = F.log_softmax(logits, dim=-1)
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                if cfg.MODEL.MY_P > 0:
                    wt, logP_t = utils.dynamic_sample(logits, p=cfg.MODEL.MY_P, low=cfg.MODEL.MY_LOW, high=cfg.MODEL.MY_HIGH)
                else:
                    wt, logP_t = utils.normal_sample(logits) 
                
            wt = wt.view(-1).int() # [b*5]
            unfinished = unfinished * (wt > 0) # [b*5] 
            wt = wt * unfinished.type_as(wt) # 
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break

        self.decoder.clear_buffer()
        
        if self.training:
            return sents, logprobs
        else:
            return sents, logprobs
