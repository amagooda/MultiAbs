from transformers.models.bart import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import *
import transformers.models.bart.modeling_bart as m_bart
import torch

# class BartForMultiSum(PretrainedBartModel):
#     base_model_prefix = "model"
#
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         base_model = BartModel(config)
#         self.model = base_model
#         self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
#         self.ex_classification_head = BartClassificationHead(config.d_model, config.d_model, 2,
#                                                           config.classif_dropout,)
#         self.model._init_weights(self.ex_classification_head.dense)
#         self.model._init_weights(self.ex_classification_head.out_proj)
#
#         self.para_classification_head = BartClassificationHead(config.d_model, config.d_model, 2,
#                                                           config.classif_dropout,)
#         self.model._init_weights(self.para_classification_head.dense)
#         self.model._init_weights(self.para_classification_head.out_proj)
#
#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         old_num_tokens = self.model.shared.num_embeddings
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self.model.shared = new_embeddings
#         self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
#         return new_embeddings
#
#     def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)
#
#     # @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
#     # @add_end_docstrings(BART_GENERATION_EXAMPLE)
#     def forward(
#         self,
#         input_ids,
#         attention_mask=None,
#         encoder_outputs=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         decoder_cached_states=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         **unused,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the masked language modeling loss.
#             Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
#             Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
#             with labels
#             in ``[0, ..., config.vocab_size]``.
#
#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
#         masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Masked language modeling loss.
#         prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#
#     Conditional generation example::
#
#             # Mask filling only works for bart-large
#             from transformers import BartTokenizer, BartForConditionalGeneration
#             tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
#             TXT = "My friends are <mask> but they eat too many carbs."
#
#             model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
#             input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
#             logits = model(input_ids)[0]
#
#             masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
#             probs = logits[0, masked_index].softmax(dim=0)
#             values, predictions = probs.topk(5)
#
#             tokenizer.decode(predictions).split()
#             # ['good', 'great', 'all', 'really', 'very']
#         """
#         if "lm_labels" in unused:
#             warnings.warn(
#                 "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
#                 DeprecationWarning,
#             )
#             labels = unused.pop("lm_labels")
#
#         if labels is not None:
#             use_cache = False
#
#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             decoder_cached_states=decoder_cached_states,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
#         outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             # TODO(SS): do we need to ignore pad tokens in labels?
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#             outputs = (masked_lm_loss,) + outputs
#
#         return outputs
#
#     def forward_ex(self, input_ids, attention_mask=None, encoder_outputs=None,
#                 decoder_input_ids=None, decoder_attention_mask=None, labels=None,
#                 output_attentions=None, output_hidden_states=None, use_cache=None,):
#         if labels is not None:
#             use_cache = False
#
#         outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
#                              decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
#                              output_attentions=output_attentions, output_hidden_states=output_hidden_states,
#                              use_cache=use_cache,)
#         x = outputs[0]  # last hidden state
#         eos_mask = input_ids.eq(self.config.eos_token_id)
#         if len(torch.unique(eos_mask.sum(1))) > 1:
#             raise ValueError("All examples must have the same number of <eos> tokens.")
#         sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
#         logits = self.ex_classification_head(sentence_representation)
#         # Prepend logits
#         outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
#         if labels is not None:  # prepend loss to output,
#             loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
#             outputs = (loss,) + outputs
#
#         return outputs
#
#     def forward_para(self, input_ids, attention_mask=None, encoder_outputs=None,
#                 decoder_input_ids=None, decoder_attention_mask=None, labels=None,
#                 output_attentions=None, output_hidden_states=None, use_cache=None,):
#         if labels is not None:
#             use_cache = False
#
#         outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
#                              decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
#                              output_attentions=output_attentions, output_hidden_states=output_hidden_states,
#                              use_cache=use_cache,)
#         x = outputs[0]  # last hidden state
#         eos_mask = input_ids.eq(self.config.eos_token_id)
#         if len(torch.unique(eos_mask.sum(1))) > 1:
#             raise ValueError("All examples must have the same number of <eos> tokens.")
#         sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
#         logits = self.para_classification_head(sentence_representation)
#         # Prepend logits
#         outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
#         if labels is not None:  # prepend loss to output,
#             loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
#             outputs = (loss,) + outputs
#
#         return outputs
#
#     def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
#         assert past is not None, "past has to be defined for encoder_outputs"
#
#         encoder_outputs, decoder_cached_states = past
#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "decoder_cached_states": decoder_cached_states,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }
#
#     def adjust_logits_during_generation(self, logits, cur_len, max_length):
#         if cur_len == 1:
#             self._force_token_ids_generation(logits, self.config.bos_token_id)
#         if cur_len == max_length - 1 and self.config.eos_token_id is not None:
#             self._force_token_ids_generation(logits, self.config.eos_token_id)
#         return logits
#
#     def _force_token_ids_generation(self, scores, token_ids) -> None:
#         """force one of token_ids to be generated by setting prob of all other tokens to 0"""
#         if isinstance(token_ids, int):
#             token_ids = [token_ids]
#         all_but_token_ids_mask = torch.tensor(
#             [x for x in range(self.config.vocab_size) if x not in token_ids],
#             dtype=torch.long,
#             device=next(self.parameters()).device,
#         )
#         assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
#         scores[:, all_but_token_ids_mask] = -float("inf")
#
#     # @staticmethod
#     # def _reorder_cache(past, beam_idx):
#     #     ((enc_out, enc_mask), decoder_cached_states) = past
#     #     reordered_past = []
#     #     for layer_past in decoder_cached_states:
#     #         # get the correct batch idx from decoder layer's batch dim for cross and self-attn
#     #         layer_past_new = {
#     #             attn_key: m_bart._reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
#     #         }
#     #         reordered_past.append(layer_past_new)
#     #
#     #     new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
#     #     new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)
#     #
#     #     past = ((new_enc_out, new_enc_mask), reordered_past)
#     #     return past
#
#     def get_encoder(self):
#         return self.model.encoder
#
#     # def get_output_embeddings(self):
#     #     return m_bart._make_linear_from_emb(self.model.shared)  # make it on the fly




# class BertForTokenClassification(BertPreTrainedModel):
#
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#         self.init_weights()
#
#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
#             1]``.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)
#                 active_labels = torch.where(
#                     active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#                 )
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

class BartForMultiSum(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.ex_classification_head = BartClassificationHead(config.d_model, config.d_model, 2, config.classif_dropout,)
        self.model._init_weights(self.ex_classification_head.dense)
        self.model._init_weights(self.ex_classification_head.out_proj)

        self.para_classification_head = BartClassificationHead(config.d_model, config.d_model, 2, config.classif_dropout,)

        self.model._init_weights(self.para_classification_head.dense)
        self.model._init_weights(self.para_classification_head.out_proj)

        ########################## Concept Classification #####################
        self.concept_dropout = nn.Dropout(config.classif_dropout)
        self.concept_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )



    def forward_lm(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # labels = None
        labels = input_ids
        if labels is not None:
            # if decoder_input_ids is None:
            # print(labels)
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        input_ids = input_ids[:,:2]
        attention_mask = attention_mask[:,:2]
        outputs = self.model(input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
            decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def forward_ex(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, use_cache=None,):
        if labels is not None:
            use_cache = False

        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             use_cache=use_cache,)
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.ex_classification_head(sentence_representation)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            loss = F.cross_entropy(logits.view(-1, 2), labels)#view(-1))
            outputs = (loss,) + outputs

        return outputs

    def forward_para(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, use_cache=None,):
        if labels is not None:
            use_cache = False

        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             use_cache=use_cache,)
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.para_classification_head(sentence_representation)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            loss = F.cross_entropy(logits.view(-1, 2), labels)#labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


    def forward_concept(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, use_cache=None,):
        if labels is not None:
            use_cache = False

        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             use_cache=use_cache,)
        # x = outputs[2]  # token states
        x = outputs['encoder_last_hidden_state']
        sequence_output = self.concept_dropout(x)
        logits = self.concept_classifier(sequence_output)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1,1).squeeze(1),ignore_index=-100)#labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past