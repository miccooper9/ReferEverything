from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import export_to_video, USE_PEFT_BACKEND
from diffusers.models.modeling_outputs import Transformer2DModelOutput

import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils.import_utils import is_ftfy_available, is_torch_xla_available, is_torch_npu_available, logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.models.attention_processor import Attention
from torch import nn
import torch.nn.functional as F

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False





logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class WanDiT(WanTransformer3DModel) :
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3", "norm_q", "norm_k", "to_q", "to_k", "to_v", "to_out"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        

        

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        print("timestep inside DiT : ", timestep, timestep.dtype, timestep.shape, timestep.device)

        print("inside DiT 1 : ", hidden_states.shape, hidden_states.dtype)

        rotary_emb = self.rope(hidden_states)

        print("inside DiT 2 rotary : ", rotary_emb.shape, rotary_emb.dtype)

        hidden_states = self.patch_embedding(hidden_states)

        print("inside DiT 3 hidden states post patch: ", hidden_states.shape, hidden_states.dtype)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        print("inside DiT 4 hidden states post flatten: ", hidden_states.shape, hidden_states.dtype)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        print("inside DiT 5 timestep proj: ", timestep_proj.shape, timestep_proj.dtype)

        print("inside DiT 6 encoder_hidden_states: ", encoder_hidden_states.shape, encoder_hidden_states.dtype)


        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        print("inside DiT 7 shift scale : ", shift.dtype, scale.dtype)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        

        

        return output#Transformer2DModelOutput(sample=output)
    




class DebugWanPipeline(WanPipeline) :



    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        print("heught , width, num_frames :", height, width, num_frames)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        print("prompt_embeds : ", prompt_embeds.shape, prompt_embeds.dtype)
        print("negative_prompt_embeds : ", negative_prompt_embeds.shape, negative_prompt_embeds.dtype)


        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        print("latents : ", latents.shape, latents.dtype)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
    



class WanVdiff_wrapper(nn.Module) :

    def __init__(self, vdiff) :
        super().__init__()


        self.backbone = vdiff

        

        self.backbone.set_untrained_vae_textenc()

        for n,p in self.backbone.dit.named_parameters() :



            if "patch_embedding" in n :
                p.requires_grad = False
                

            if "condition_embedder" in n :
                p.requires_grad = False

            

    def forward(self, latents, prompt_embeds, timesteps) :


        pred_latents = self.backbone(
            latents,
            timesteps,
            prompt_embeds)


        return pred_latents
    






class WanVdiff_txtfree(nn.Module):

    def __init__(self, dit, vae, scheduler):# unet, vae, scheduler):
        super().__init__()
        

        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler


    def set_untrained_all(self):
        for param in self.dit.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        

    def set_untrained_vae_textenc(self):
        
        for param in self.vae.parameters():
            param.requires_grad = False
        

    def forward(
        self,
        latents,
        t,
        prompt_embeds,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        ):

        device = latents.device


        with torch.no_grad() :

            t_add = self.scheduler.timesteps[-1-t]
            print("t_add : ", t_add)
            t_dit = t_add.expand(latents.shape[0]).to(device=latents.device)
            print("t_dit : ", t_dit)
            
            noise = torch.randn_like(latents).to(device)
            print("noise shape : ", noise.shape)

            noisy_latents = self.scheduler.scale_noise(latents, t_dit, noise)
            print("noise_scaled")


            t_dit_in = torch.tensor([0.0]).to(device=latents.device, dtype=latents.dtype)#self.scheduler.config.num_train_timesteps*self.scheduler.sigmas[-1-t].expand(latents.shape[0]).to(device=latents.device)
            print("t_dit_in : " , t_dit_in, t_dit_in.shape, t_dit_in.dtype)

        sample = self.dit(
                    hidden_states=latents.to(device=latents.device, dtype=latents.dtype),
                    timestep=t_dit_in,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                )
        
        return sample



class WanVdifflarge_txt(nn.Module):

    def __init__(self, dit, vae, scheduler, text_encoder, tokenizer):# unet, vae, scheduler):
        super().__init__()
        

        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder 


    def set_untrained_all(self):
        for param in self.dit.parameters():
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
        

    def forward(
        self,
        latents,
        t,
        prompt_embeds,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        ):

        device = latents.device


        with torch.no_grad() :

            t_add = self.scheduler.timesteps[-1-t]
            print("t_add : ", t_add)
            t_dit = t_add.expand(latents.shape[0]).to(device=latents.device)
            print("t_dit : ", t_dit)
            
            noise = torch.randn_like(latents).to(device)
            print("noise shape : ", noise.shape)

            noisy_latents = self.scheduler.scale_noise(latents, t_dit, noise)
            print("noise_scaled")


            t_dit_in = torch.tensor([0.0]).to(device=latents.device, dtype=latents.dtype)#self.scheduler.config.num_train_timesteps*self.scheduler.sigmas[-1-t].expand(latents.shape[0]).to(device=latents.device)
            print("t_dit_in : " , t_dit_in, t_dit_in.shape, t_dit_in.dtype)

        sample = self.dit(
                    hidden_states=latents.to(device=latents.device, dtype=latents.dtype),
                    timestep=t_dit_in,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                )
        
        return sample





def scale_fn(self) :

    def debug_scale_noise(
            sample: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            noise: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:
            """
            Forward process in flow-matching

            Args:
                sample (`torch.FloatTensor`):
                    The input sample.
                timestep (`int`, *optional*):
                    The current timestep in the diffusion chain.

            Returns:
                `torch.FloatTensor`:
                    A scaled input sample.
            """
            # Make sure sigmas and timesteps have the same device and dtype as original_samples
            sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

            print("--------==---------", timestep)

            if sample.device.type == "mps" and torch.is_floating_point(timestep):
                # mps does not support float64
                schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
                timestep = timestep.to(sample.device, dtype=torch.float32)
            else:
                schedule_timesteps = self.timesteps.to(sample.device)
                timestep = timestep.to(sample.device)

            print(schedule_timesteps)

            # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
            if self.begin_index is None:
                print("1")
                step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
            elif self.step_index is not None:
                print("2")
                # add_noise is called after first denoising step (for inpainting)
                step_indices = [self.step_index] * timestep.shape[0]
            else:
                print("3")
                # add noise is called before first denoising step to create initial latent(img2img)
                step_indices = [self.begin_index] * timestep.shape[0]

            print("=========idx : ", step_indices)

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < len(sample.shape):
                sigma = sigma.unsqueeze(-1)

            #print(sigmas)
            print("----------")
            print("sigma used to scale : ", sigma)

            sample = sigma * noise + (1.0 - sigma) * sample

            return sample
    
    return debug_scale_noise




def time_forward_fn(self) :


    def debug_forward(sample, condition=None):

        print("1debugtimemb===<><><====", sample.dtype)
        if condition is not None:
            print("condition")
            sample = sample + self.cond_proj(condition)
        print("2debugtimemb===<><><====", sample.dtype)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
    
    return debug_forward





def condition_forward_fn(self) :


    def debug_cond_forward(
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        
        print("0inside cond <><><><>: ", timestep.dtype)

        timestep = self.timesteps_proj(timestep)

        print("1inside cond <><><><>: ", timestep.dtype)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype

        print("2inside cond <><><><>:time_embedder_dtype ", time_embedder_dtype)


        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            print("<><><><> to")
            timestep = timestep.to(time_embedder_dtype)


        print("3inside cond <><><><>: ", timestep.dtype)

        print("4inside cond <><><><>: ", encoder_hidden_states.dtype)



        temb = self.time_embedder(timestep.to(encoder_hidden_states.dtype)).type_as(encoder_hidden_states)

        print("5inside cond <><><><>: ", temb.dtype)
        
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image
    
    return debug_cond_forward
        


def RMS_norm_forward(self) :

    def debug_rmsnorm_forward(hidden_states):
        if is_torch_npu_available():
            import torch_npu

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            print("r1 inside rms norm : ", hidden_states.dtype)
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

            print("r2 inside rms norm : ", hidden_states.dtype, self.weight.dtype)

            if self.weight is not None:
                # convert into half-precision if necessary
                '''if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    print("r2.1 inside rms norm")
                    hidden_states = hidden_states.to(self.weight.dtype)'''
                hidden_states = hidden_states * self.weight.float()
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias.float()
            else:
                hidden_states = hidden_states.to(input_dtype)

            print("r3 inside rms norm : ", hidden_states.dtype)

        return hidden_states
    
    return debug_rmsnorm_forward





def fp32_ln_forward(self) :



    def debug_ln_forward(inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype

        print("ln1 inside fp32ln : ", inputs.dtype, self.weight.dtype, self.bias.dtype)
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        print("ln2 inside fp32ln : ", out.dtype)
        return out.to(origin_dtype)
    

    return debug_ln_forward






class WanAttnProcessor2_0_1:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states


        origin_dtype = encoder_hidden_states.dtype

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        print("================attnproc", origin_dtype, hidden_states.dtype, query.dtype, key.dtype)

        if attn.norm_q is not None:
            print("==-=-=-=-=normq")
            
        if attn.norm_k is not None:
            print("==-=-=-=-=normk")

        #query = attn.norm_q(query.float()).to(dtype=hidden_states.dtype)
        #key = attn.norm_k(key.float()).to(dtype=hidden_states.dtype)

        query = attn.norm_q(query)
        key = attn.norm_k(key)



        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states#.to(dtype=origin_dtype)







def build_want2v_txtfree_model(sd_id=""):

    
    wandit = WanDiT.from_pretrained(f"{sd_id}/transformer", torch_dtype=torch.bfloat16)#torch.float32

    vae = AutoencoderKLWan.from_pretrained(f"{sd_id}/vae", torch_dtype=torch.bfloat16)#torch.float32

    
    pipe = DebugWanPipeline.from_pretrained(sd_id, transformer=wandit, vae=vae, torch_dtype=torch.bfloat16)
    print(pipe.device, pipe.transformer.device)


    pipe.scheduler.scale_noise = scale_fn(pipe.scheduler)
    wandit.condition_embedder.forward = condition_forward_fn(wandit.condition_embedder)
    wandit.condition_embedder.time_embedder.forward = time_forward_fn(wandit.condition_embedder.time_embedder)

    
    vdiff = WanVdiff_txtfree(wandit, pipe.vae, pipe.scheduler)
    return vdiff




def build_want2vlarge_txt_model(sd_id=""):

    print(sd_id)
    wandit = WanDiT.from_pretrained(f"{sd_id}/transformer", torch_dtype=torch.bfloat16)

    vae = AutoencoderKLWan.from_pretrained(f"{sd_id}/vae", torch_dtype=torch.bfloat16)

    
    pipe = DebugWanPipeline.from_pretrained(sd_id, transformer=wandit, vae=vae, torch_dtype=torch.bfloat16)
    


    pipe.scheduler.scale_noise = scale_fn(pipe.scheduler)
    wandit.condition_embedder.forward = condition_forward_fn(wandit.condition_embedder)
    wandit.condition_embedder.time_embedder.forward = time_forward_fn(wandit.condition_embedder.time_embedder)

    vdiff = WanVdifflarge_txt(wandit, pipe.vae, pipe.scheduler, pipe.text_encoder, pipe.tokenizer)
    return vdiff






def build_want2v_txt(sd_id=""):


    print(sd_id)
    wandit = WanDiT.from_pretrained(f"{sd_id}/transformer", torch_dtype=torch.bfloat16)

    vae = AutoencoderKLWan.from_pretrained(f"{sd_id}/vae", torch_dtype=torch.bfloat16)

    print(".....")
    pipe = DebugWanPipeline.from_pretrained(sd_id, transformer=wandit, vae=vae, torch_dtype=torch.bfloat16)
    print(pipe.device, pipe.transformer.device)


    return pipe.tokenizer, pipe.text_encoder






