from diffusers import TextToVideoSDPipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.models import UNet3DConditionModel
from diffusers import DDIMScheduler, EulerDiscreteScheduler, DDPMScheduler
import gc
import os
from PIL import Image
from torchvision.transforms import PILToTensor
import copy
import torch.nn.functional as F

#from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from einops import rearrange
import math
import random
import PIL
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor




    


class MyUNet3DConditionModelUPDOWN(UNet3DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        up_ft_indices: int = 2,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        r"""
        The [`UNet3DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_channels, num_frames, height, width`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].

        Returns:
            [`~models.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_3d_condition.UNet3DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        with torch.no_grad() :
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                # logger.info("Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            num_frames = sample.shape[2]
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=self.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)
            emb = emb.repeat_interleave(repeats=num_frames, dim=0)
            
            # 2. pre-process

            #print("-------sample shape", sample.shape)
            sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
            #print("-------sample shape", sample.shape)
            #print(llll)

        #encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        
        sample = self.conv_in(sample)
        #print("pre transin-------sample shape", sample.shape)

        sample = self.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        #print("post transin-------sample shape", sample.shape)
        #print(llll)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)
            # print(sample.shape)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            # print(sample.shape)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )
            # print(sample.shape)
            #if i in up_ft_indices:
            #print("------>>>>>>>>",i)
            #up_ft[i] = sample#.detach()

        if self.conv_norm_out:
            print("-----NORM----")
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        sample = self.conv_out(sample)
        print("-----chk----")

        
        return sample
















class OneStepPipeline(TextToVideoSDPipeline):
    @torch.no_grad()
    def __call__(
        self,
        video_input,
        up_ft_indices,
        t: int = 0,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ):

        device = self._execution_device
        num_images_per_prompt = 1
        do_classifier_free_guidance = False

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )



        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])


        # ?
        b, f, c, h, w = video_input.shape
        video_input = rearrange(video_input, 'b f c h w -> (b f) c h w')

        t_add = self.scheduler.timesteps[-1-t]

        
        # import pdb 
        # pdb.set_trace()
        latents = self.vae.encode(video_input).latent_dist.mode() * self.vae.config.scaling_factor
        # ? encode_first stage
        noise = torch.randn_like(latents).to(device)
        noisy_latents = self.scheduler.add_noise(latents, noise, t_add)

        latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_add)

        latent_model_input = rearrange(latent_model_input, '(b f) c h w -> b c f h w', b = b, f = f)

        unet_output = self.unet(
                sample = latent_model_input,
                timestep = t_add,
                up_ft_indices = up_ft_indices,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )
        
        return unet_output

    @property
    def do_classifier_free_guidance(self):
        return False
    





        








    







class VDiffFeatExtractor_lean_updown(nn.Module):

    def __init__(self, unet, tokenizer, vae, text_encoder, scheduler):# unet, vae, scheduler):
        super().__init__()
        

        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.vae.enable_xformers_memory_efficient_attention()
        self.unet.enable_xformers_memory_efficient_attention()
        #self.set_untrained_all()
        #self.set_untrained_vae_textenc()
        

        
    
    def set_untrained_all(self):
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def set_untrained_vae_textenc(self):
        
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False


    def set_untrained_vaeenc_textenc(self):
        
        for name, param in self.vae.named_parameters():

            if "encoder" in name :
                param.requires_grad = False
            if "quant" in name and "post" not in name :
                param.requires_grad = False
                
        for param in self.text_encoder.parameters():
            param.requires_grad = False



    def forward(self,
        video_input,
        prompt_embeds,
        up_ft_index,
        t = 0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        device = video_input.device
        num_images_per_prompt = 1
        do_classifier_free_guidance = False

        # ?
        b, f, c, h, w = video_input.shape
        video_input = rearrange(video_input, 'b f c h w -> (b f) c h w')
        print("timestep used ::::::::::::::::: ", t)

        t_add = self.scheduler.timesteps[-1-t]

        
        # import pdb 
        # pdb.set_trace()
        with torch.no_grad() :
            latents = self.vae.encode(video_input).latent_dist.mode() * self.vae.config.scaling_factor
        # ? encode_first stage
        noise = torch.randn_like(latents).to(device)
        noisy_latents = self.scheduler.add_noise(latents, noise, t_add)

        latent_model_input = noisy_latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_add)

        latent_model_input = rearrange(latent_model_input, '(b f) c h w -> b c f h w', b = b, f = f)

        #prompt_embeds = self.prompt_embeds.to(device).repeat(b,1,1)
        # import pdb
        # pdb.set_trace()
        sample = self.unet(
                sample = latent_model_input,
                timestep = t_add,
                up_ft_indices = up_ft_index,
                encoder_hidden_states= prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )
        
        
        return sample
    



def build_vdiff_lean_updown(sd_id='ali-vilab/text-to-video-ms-1.7b'):

    unet = MyUNet3DConditionModelUPDOWN.from_pretrained(sd_id, subfolder="unet")
    onestep_pipe = OneStepPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
    onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
    gc.collect()
    onestep_pipe.enable_attention_slicing()
    onestep_pipe.enable_xformers_memory_efficient_attention()
    

    real_pipe = VDiffFeatExtractor_lean_updown(unet, onestep_pipe.tokenizer, onestep_pipe.vae, onestep_pipe.text_encoder, onestep_pipe.scheduler)
    return real_pipe






class Vdiff_updown(nn.Module) :

    def __init__(self, vdiff) :
        super().__init__()


        self.backbone = vdiff
        self.backbone.set_untrained_vae_textenc()
        for n,p in self.backbone.unet.named_parameters() :
            print("===>",n)
            if "time_embedding" in n :
                print("------>",n)
                p.requires_grad = False


    def forward(self, frames, prompt_embeds) :


        pred_latents = self.backbone.forward(
            frames,
            prompt_embeds,
            t=0,
            up_ft_index = 3)#doesn't matter we are using all of the unet


        return pred_latents


































