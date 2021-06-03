# coding: utf-8
import torch
from torch import nn

from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.tts.layers.tacotron.tacotron2 import Decoder, Encoder, Postnet
from TTS.tts.models.tacotron_abstract import TacotronAbstract
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram


class Tacotron2(TacotronAbstract):
    """Tacotron2 as in https://arxiv.org/abs/1712.05884

    It's an autoregressive encoder-attention-decoder-postnet architecture.

    Args:
        num_chars (int): number of input characters to define the size of embedding layer.
        num_speakers (int): number of speakers in the dataset. >1 enables multi-speaker training and model learns speaker embeddings.
        r (int): initial model reduction rate.
        postnet_output_dim (int, optional): postnet output channels. Defaults to 80.
        decoder_output_dim (int, optional): decoder output channels. Defaults to 80.
        attn_type (str, optional): attention type. Check ```TTS.tts.layers.tacotron.common_layers.init_attn```. Defaults to 'original'.
        attn_win (bool, optional): enable/disable attention windowing.
            It especially useful at inference to keep attention alignment diagonal. Defaults to False.
        attn_norm (str, optional): Attention normalization method. "sigmoid" or "softmax". Defaults to "softmax".
        prenet_type (str, optional): prenet type for the decoder. Defaults to "original".
        prenet_dropout (bool, optional): prenet dropout rate. Defaults to True.
        prenet_dropout_at_inference (bool, optional): use dropout at inference time. This leads to a better quality for
            some models. Defaults to False.
        forward_attn (bool, optional): enable/disable forward attention.
            It is only valid if ```attn_type``` is ```original```.  Defaults to False.
        trans_agent (bool, optional): enable/disable transition agent in forward attention. Defaults to False.
        forward_attn_mask (bool, optional): enable/disable extra masking over forward attention. Defaults to False.
        location_attn (bool, optional): enable/disable location sensitive attention.
            It is only valid if ```attn_type``` is ```original```. Defaults to True.
        attn_K (int, optional): Number of attention heads for GMM attention. Defaults to 5.
        separate_stopnet (bool, optional): enable/disable separate stopnet training without only gradient
            flow from stopnet to the rest of the model.  Defaults to True.
        bidirectional_decoder (bool, optional): enable/disable bidirectional decoding. Defaults to False.
        double_decoder_consistency (bool, optional): enable/disable double decoder consistency. Defaults to False.
        ddc_r (int, optional): reduction rate for the coarse decoder of double decoder consistency. Defaults to None.
        encoder_in_features (int, optional): input channels for the encoder. Defaults to 512.
        decoder_in_features (int, optional): input channels for the decoder. Defaults to 512.
        d_vector_dim (int, optional): external speaker conditioning vector channels. Defaults to None.
        use_gst (bool, optional): enable/disable Global style token module.
        gst (Coqpit, optional): Coqpit to initialize the GST module. If `None`, GST is disabled. Defaults to None.
        gradual_training (List): Gradual training schedule. If None or `[]`, no gradual training is used.
            Defaults to `[]`.
    """

    def __init__(
        self,
        num_chars,
        num_speakers,
        r,
        postnet_output_dim=80,
        decoder_output_dim=80,
        attn_type="original",
        attn_win=False,
        attn_norm="softmax",
        prenet_type="original",
        prenet_dropout=True,
        prenet_dropout_at_inference=False,
        forward_attn=False,
        trans_agent=False,
        forward_attn_mask=False,
        location_attn=True,
        attn_K=5,
        separate_stopnet=True,
        bidirectional_decoder=False,
        double_decoder_consistency=False,
        ddc_r=None,
        encoder_in_features=512,
        decoder_in_features=512,
        d_vector_dim=None,
        use_gst=False,
        gst=None,
        gradual_training=None,
    ):
        super().__init__(
            num_chars,
            num_speakers,
            r,
            postnet_output_dim,
            decoder_output_dim,
            attn_type,
            attn_win,
            attn_norm,
            prenet_type,
            prenet_dropout,
            prenet_dropout_at_inference,
            forward_attn,
            trans_agent,
            forward_attn_mask,
            location_attn,
            attn_K,
            separate_stopnet,
            bidirectional_decoder,
            double_decoder_consistency,
            ddc_r,
            encoder_in_features,
            decoder_in_features,
            d_vector_dim,
            use_gst,
            gst,
            gradual_training,
        )

        # speaker embedding layer
        if self.num_speakers > 1:
            if not self.use_d_vectors:
                d_vector_dim = 512
                self.speaker_embedding = nn.Embedding(self.num_speakers, d_vector_dim)
                self.speaker_embedding.weight.data.normal_(0, 0.3)

        # speaker and gst embeddings is concat in decoder input
        if self.num_speakers > 1:
            self.decoder_in_features += d_vector_dim  # add speaker embedding dim

        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)
        self.decoder = Decoder(
            self.decoder_in_features,
            self.decoder_output_dim,
            r,
            attn_type,
            attn_win,
            attn_norm,
            prenet_type,
            prenet_dropout,
            forward_attn,
            trans_agent,
            forward_attn_mask,
            location_attn,
            attn_K,
            separate_stopnet,
        )
        self.postnet = Postnet(self.postnet_output_dim)

        # setup prenet dropout
        self.decoder.prenet.dropout_at_g = prenet_dropout_at_inference

        # global style token layers
        if self.gst and use_gst:
            self.gst_layer = GST(
                num_mel=decoder_output_dim,
                d_vector_dim=d_vector_dim,
                num_heads=gst.gst_num_heads,
                num_style_tokens=gst.gst_num_style_tokens,
                gst_embedding_dim=gst.gst_embedding_dim,
            )

        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                self.decoder_in_features,
                self.decoder_output_dim,
                ddc_r,
                attn_type,
                attn_win,
                attn_norm,
                prenet_type,
                prenet_dropout,
                forward_attn,
                trans_agent,
                forward_attn_mask,
                location_attn,
                attn_K,
                separate_stopnet,
            )

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, mel_lengths=None, cond_input=None):
        """
        Shapes:
            text: [B, T_in]
            text_lengths: [B]
            mel_specs: [B, T_out, C]
            mel_lengths: [B]
            cond_input: 'speaker_ids': [B, 1] and  'd_vectors':[B, C]
        """
        cond_input = self._format_cond_input(cond_input)
        outputs = {"alignments_backward": None, "decoder_outputs_backward": None}
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs, cond_input["d_vectors"])
        if self.num_speakers > 1:
            if not self.use_d_vectors:
                # B x 1 x speaker_embed_dim
                embedded_speakers = self.speaker_embedding(cond_input["speaker_ids"])[:, None]
            else:
                # B x 1 x speaker_embed_dim
                embedded_speakers = torch.unsqueeze(cond_input["d_vectors"], 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)

        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)

        # B x mel_dim x T_out -- B x T_out//r x T_in -- B x T_out//r
        decoder_outputs, alignments, stop_tokens = self.decoder(encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs)
        # B x mel_dim x T_out
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        # sequence masking
        if output_mask is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(1).expand_as(postnet_outputs)
        # B x T_out x mel_dim -- B x T_out x mel_dim -- B x T_out//r x T_in
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_pass(mel_specs, encoder_outputs, input_mask)
            outputs["alignments_backward"] = alignments_backward
            outputs["decoder_outputs_backward"] = decoder_outputs_backward
        if self.double_decoder_consistency:
            decoder_outputs_backward, alignments_backward = self._coarse_decoder_pass(
                mel_specs, encoder_outputs, alignments, input_mask
            )
            outputs["alignments_backward"] = alignments_backward
            outputs["decoder_outputs_backward"] = decoder_outputs_backward
        outputs.update(
            {
                "model_outputs": postnet_outputs,
                "decoder_outputs": decoder_outputs,
                "alignments": alignments,
                "stop_tokens": stop_tokens,
            }
        )
        return outputs

    @torch.no_grad()
    def inference(self, text, cond_input=None):
        cond_input = self._format_cond_input(cond_input)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, cond_input["style_mel"], cond_input["d_vectors"])
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                x_vector = self.speaker_embedding(cond_input['speaker_ids'])[:, None]
                x_vector = torch.unsqueeze(x_vector, 0).transpose(1, 2)
            else:
                embedded_speakers = cond_input["d_vectors"]

            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(decoder_outputs, postnet_outputs, alignments)
        outputs = {
            "model_outputs": postnet_outputs,
            "decoder_outputs": decoder_outputs,
            "alignments": alignments,
            "stop_tokens": stop_tokens,
        }
        return outputs

    def train_step(self, batch, criterion):
        """Perform a single training step by fetching the right set if samples from the batch.

        Args:
            batch ([type]): [description]
            criterion ([type]): [description]
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        linear_input = batch["linear_input"]
        stop_targets = batch["stop_targets"]
        speaker_ids = batch["speaker_ids"]
        d_vectors = batch["d_vectors"]

        # forward pass model
        outputs = self.forward(
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            cond_input={"speaker_ids": speaker_ids, "d_vectors": d_vectors},
        )

        # set the [alignment] lengths wrt reduction factor for guided attention
        if mel_lengths.max() % self.decoder.r != 0:
            alignment_lengths = (
                mel_lengths + (self.decoder.r - (mel_lengths.max() % self.decoder.r))
            ) // self.decoder.r
        else:
            alignment_lengths = mel_lengths // self.decoder.r

        cond_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, cond_input)

        # compute loss
        loss_dict = criterion(
            outputs["model_outputs"],
            outputs["decoder_outputs"],
            mel_input,
            linear_input,
            outputs["stop_tokens"],
            stop_targets,
            mel_lengths,
            outputs["decoder_outputs_backward"],
            outputs["alignments"],
            alignment_lengths,
            outputs["alignments_backward"],
            text_lengths,
        )

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(outputs["alignments"])
        loss_dict["align_error"] = align_error
        return outputs, loss_dict

    def train_log(self, ap, batch, outputs):
        postnet_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        alignments_backward = outputs["alignments_backward"]
        mel_input = batch["mel_input"]

        pred_spec = postnet_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        if self.bidirectional_decoder or self.double_decoder_consistency:
            figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy(), output_fig=False)

        # Sample audio
        train_audio = ap.inv_melspectrogram(pred_spec.T)
        return figures, train_audio

    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)

    def eval_log(self, ap, batch, outputs):
        return self.train_log(ap, batch, outputs)
