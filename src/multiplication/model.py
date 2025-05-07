from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, GenerationConfig
from transformers.modeling_outputs import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)


@dataclass
class LatentOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    latent_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        

@dataclass
class LatentGenerateOutPut(ModelOutput):
    sequences: torch.LongTensor = None
    latent_logits: torch.FloatTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class LatentTokenizer(Qwen2Tokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        # use existing special tokens
        self.latent_token = '<|fim_middle|>' # 151660
        self.latent_token_id = 151660

        self.model_input_names = ["input_ids", "attention_mask", "latent_embeds", "token_types"]


class LatentQwen2(Qwen2ForCausalLM):
    def __init__(self, config, latent_dim=20, **kwargs):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.generation_config.do_sample = False

        self.latent_dim = latent_dim

        # Latent embeddings
        self.latent_projection = nn.Linear(self.latent_dim, config.hidden_size)
        self.latent_head = nn.Linear(config.hidden_size, self.latent_dim)
        self.latent_type = 1

        self.latent_token_id = 151660 # '<|fim_middle|>'

        self.post_init()

    def bind_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        assert len(input_ids.shape) == 2, 'Input ids should be in the shape of batch_size * len'

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            **kwargs,
        )
        model_inputs.update({
            'latent_embeds': kwargs['latent_embeds'],
            'token_types': kwargs['token_types'],
        })

        return model_inputs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        latent_embeds: Optional[torch.FloatTensor] = None, # Latent binary encodings
        token_types: Optional[torch.Tensor] = None, # Token types (0: normal, 1: latent)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True, # True by default
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        inputs_embeds = self.model.embed_tokens(input_ids)

        # print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
        # print(token_types[0])
        # print(latent_embeds[0])
        # input()

        # add binary encodings
        if (latent_embeds is not None):
            projected_latent_encodings = self.latent_projection(latent_embeds).to(self.device)
            latent_encoding_mask = (token_types == self.latent_type).unsqueeze(-1).repeat(1,1,self.hidden_size)
            inputs_embeds = torch.where(
                latent_encoding_mask,
                projected_latent_encodings,
                inputs_embeds,
            )

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        latent_logits = self.latent_head(hidden_states)

        loss = None
        if labels is not None:
            # === Loss 1: LM head loss ===
            # Shift so that tokens < n predict n
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

            # === Loss 2: Latent loss ===
            latent_logits = latent_logits.float()

            latent_labels = latent_embeds[:, :, :]
            # Shift so that tokens < n predict n
            shift_logits = latent_logits[..., :-1, :].contiguous()
            shift_labels = latent_labels[..., 1:, :].contiguous()
            # Flatten the tokens
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            latent_loss = loss_fct(shift_logits, shift_labels)
            
            latent_loss = torch.mean(latent_loss, dim=-1)
            latent_loss_mask = (token_types == self.latent_type) * ((labels != -100)) # both latent and non-prompt
            latent_loss_mask = latent_loss_mask[..., 1:].contiguous().to(latent_loss.device)
            latent_loss = latent_loss * latent_loss_mask
            latent_loss = (latent_loss.sum(dim=-1) / latent_loss_mask.sum(dim=-1)).mean()

            loss = lm_loss + latent_loss

        if not return_dict:
            output = (logits,) + (latent_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LatentOutput(
            loss=loss,
            logits=logits,
            latent_logits=latent_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"

        # Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["latent_embeds", "token_types"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        return model_inputs
    

    def _update_latent_kwargs(
        self,
        model_kwargs: Dict[str, Any],
        next_token_types,
        next_latent_logits,
    ):
        # update token types
        token_types = model_kwargs['token_types']
        model_kwargs['token_types'] = torch.cat(
            [token_types, next_token_types],
            dim=-1,
        )

        # update latent embeds
        next_latent_embeds = (next_latent_logits >= 0.5).float()
        latent_embeds = model_kwargs['latent_embeds']
        model_kwargs['latent_embeds'] = torch.cat(
            [latent_embeds, next_latent_embeds],
            dim=1,
        )

        return model_kwargs

    def _sample( # override for greedy search
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer,
        **model_kwargs,
    ) -> Union[ModelOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # Latent: init latent features
        if ('latent_embeds' not in model_kwargs):
            latent_embeds = torch.zeros(batch_size, cur_len, self.latent_dim).float().to(self.device)
            model_kwargs['latent_embeds'] = latent_embeds
        latent_embeds = model_kwargs['latent_embeds']
        
        if ('token_types' not in model_kwargs):
            token_types = torch.zeros(batch_size, cur_len).long().to(self.device)
            model_kwargs['token_types'] = token_types

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # === Latent features ===
            # Token types
            next_token_types = torch.zeros(next_tokens.shape).view(-1)
            for i, element in enumerate(next_tokens.view(-1)):
                if (element == self.latent_token_id):
                    next_token_types[i] = self.latent_type
            next_token_types = next_token_types.view(next_tokens.shape).to(next_tokens.device)
            next_token_types = next_token_types[:, None]

            # Latent logits
            next_latent_logits = torch.sigmoid(outputs.latent_logits[:, -1, :].unsqueeze(1))
            latent_embeds = torch.cat([latent_embeds, next_latent_logits], dim=1)

            model_kwargs = self._update_latent_kwargs(
                model_kwargs,
                next_token_types=next_token_types,
                next_latent_logits=next_latent_logits,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return LatentGenerateOutPut(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                latent_logits=latent_embeds,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        