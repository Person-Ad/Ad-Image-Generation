from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    AttentionProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin

class MLP(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):  # b, 257,1280
        return self.net(x)


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    Args:
        predicted_image_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: torch.FloatTensor


class Stage1_PriorTransformer(ModelMixin, ConfigMixin):
    """
    The prior transformer from unCLIP is used to predict CLIP image embeddings from CLIP text embeddings. Note that the
    transformer predicts the image embeddings through a denoising diffusion process.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    For more details, see the original paper: https://arxiv.org/abs/2204.06125

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the CLIP embeddings. Note that CLIP
            image embeddings and text embeddings are both the same dimension.
        num_embeddings (`int`, *optional*, defaults to 77): The max number of clip embeddings allowed. I.e. the
            length of the prompt after it has been tokenized.
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected hidden_states. The actual length of the used hidden_states is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.

    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        embedding_dim: int = 768,
        num_embeddings=77,
        additional_embeddings=4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        self.pose_encoder = MLP(in_dim=36, hidden_dim=512, out_dim=1024)
        self.pose_encoder1 = MLP(in_dim=36, hidden_dim=512, out_dim=1024)

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, inner_dim)

        self.proj_in = nn.Linear(embedding_dim, inner_dim)

        self.embedding_proj = nn.Linear(embedding_dim, inner_dim)
        self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
        self.encoder_hidden_states_proj1 = nn.Linear(embedding_dim, inner_dim)

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings + additional_embeddings, inner_dim))

        self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))


        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for d in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim)
        self.proj_to_clip_embeddings = nn.Linear(inner_dim, embedding_dim)

        causal_attention_mask = torch.full(
            [num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], -10000.0
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistent=False)

        self.clip_mean = torch.tensor(-0.016)
        self.clip_std = torch.tensor(0.415)
    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    def forward(
        self,
        hidden_states,
        timestep: Union[torch.Tensor, float, int],
        proj_embedding: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_hidden_states1: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
        do_classifier_free_guidance: bool = False,
        test_flag: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                x_t, the currently predicted image embeddings.
            timestep (`torch.long`):
                Current denoising step.
            proj_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
            [`~models.prior_transformer.PriorTransformerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        batch_size = hidden_states.shape[0]

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=hidden_states.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(batch_size, dtype=timesteps.dtype, device=timesteps.device)

        timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timesteps_projected = timesteps_projected.to(dtype=self.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        proj_embeddings = self.embedding_proj(proj_embedding)

        encoder_hidden_states = self.pose_encoder(encoder_hidden_states)
        encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)

        encoder_hidden_states1 = self.pose_encoder1(encoder_hidden_states1)
        encoder_hidden_states1 = self.encoder_hidden_states_proj1(encoder_hidden_states1)
        hidden_states = self.proj_in(hidden_states) # bs,1024  ->  bs, 2048
        prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        if test_flag:
            negative_prompt_embeds = torch.zeros(encoder_hidden_states.shape).to(encoder_hidden_states.device, encoder_hidden_states.dtype)
            encoder_hidden_states = torch.cat([negative_prompt_embeds, encoder_hidden_states])
            encoder_hidden_states1 = torch.cat([negative_prompt_embeds, encoder_hidden_states1])

        hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states1,  proj_embeddings, time_embeddings[:, None, :], hidden_states.unsqueeze(0), prd_embedding,], dim=1,)


        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = F.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config.num_attention_heads, dim=0)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -1]

        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
