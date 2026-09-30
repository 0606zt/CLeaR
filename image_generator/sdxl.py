import warnings
from diffusers.utils import logging
warnings.filterwarnings('ignore')
logging.disable_progress_bar()

import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import deprecate, is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class SDXL(nn.Module):
    def __init__(self, model, refiner):
        super().__init__()
        self.model = model
        self.refiner = refiner

    @classmethod
    def load_model(cls, device, dtype=torch.float16, is_train=False, **kwargs):
        model_dir = kwargs.get('model_dir', 'stabilityai/stable-diffusion-xl-base-1.0')
        variant = None if dtype == torch.float32 else 'fp16'
        model = StableDiffusionXLPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            variant=variant
        ).to(device)
        # model.vae.enable_tiling()
        refiner = StableDiffusionXLImg2ImgPipeline.from_pipe(model)
        sdxl = cls(model=model, refiner=refiner)
        return sdxl

    def load_ip_adapter(self, **kwargs):
        ip_adapter_dir = kwargs.get('ip_adapter_dir', 'h94/IP-Adapter')
        ip_adapter_name = kwargs.get('ip_adapter_name', 'ip-adapter_sdxl')
        self.model.load_ip_adapter(
            ip_adapter_dir,
            subfolder='sdxl_models',
            weight_name=f'{ip_adapter_name}.bin'
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def calib_call(
            self,
            ####################################### for style calibration
            style_image,
            style_content_image,
            vision_encoders: dict,
            vision_encoder_loss_funcs: dict,
            vision_weights: list[float] = None,
            calib_guidance_scale=0.1,
            lr=1e-2,
            epochs=5,
            eps=8/255,
            #############################################################
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = 'pil',
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ['latents'],
            **kwargs
    ):
        callback = kwargs.pop('callback', None)
        callback_steps = kwargs.pop('callback_steps', None)

        if callback is not None:
            deprecate(
                'callback',
                '1.0.0',
                'Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`',
            )
        if callback_steps is not None:
            deprecate(
                'callback_steps',
                '1.0.0',
                'Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`',
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.model.default_sample_size * self.model.vae_scale_factor
        width = width or self.model.default_sample_size * self.model.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.model.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self.model._guidance_scale = guidance_scale
        self.model._guidance_rescale = guidance_rescale
        self.model._clip_skip = clip_skip
        self.model._cross_attention_kwargs = cross_attention_kwargs
        self.model._denoising_end = denoising_end
        self.model._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device1 = self.model._execution_device

        ####################################### for style calibration
        device2 = next(iter(vision_encoders.values())).device
        dtype2 = next(iter(vision_encoders.values())).dtype

        self.model.vae.to(device2)

        if vision_weights is None:
            v_w = torch.ones(len(vision_encoders), device=device2) / len(vision_encoders)  # average weights
        else:
            assert len(vision_weights) == len(vision_encoders)
            v_w = torch.tensor(vision_weights, device=device2)

        style_image_embeds = []
        style_content_image_embeds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype2):
            for t_i, model_name in enumerate(vision_encoders):
                model = vision_encoders[model_name]
                image_embeds = model.get_image_embeds(style_image, is_train=False)
                style_image_embeds.append(image_embeds)
                image_embeds = model.get_image_embeds(style_content_image, is_train=False)
                style_content_image_embeds.append(image_embeds)
        #############################################################

        # 3. Encode input prompt
        lora_scale = (
            self.model.cross_attention_kwargs.get('scale', None) if self.model.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device1,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.model.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.model.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler, num_inference_steps, device1, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device1,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.model.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.model.text_encoder_2.config.projection_dim

        add_time_ids = self.model._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.model._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.model.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device1)
        add_text_embeds = add_text_embeds.to(device1)
        add_time_ids = add_time_ids.to(device1).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.model.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device1,
                batch_size * num_images_per_prompt,
                self.model.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.model.denoising_end is not None
            and isinstance(self.model.denoising_end, float)
            and self.model.denoising_end > 0
            and self.model.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.model.scheduler.config.num_train_timesteps
                    - (self.model.denoising_end * self.model.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.model.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.model.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.model.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.unet.config.time_cond_proj_dim
            ).to(device=device1, dtype=latents.dtype)

        self.model._num_timesteps = len(timesteps)
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for t_i, t in enumerate(timesteps):
                if self.model.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents

                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs['image_embeds'] = image_embeds
                noise_pred = self.model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.model.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.model.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.model.do_classifier_free_guidance and self.model.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.model.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                ####################################### for style calibration
                if t_i >= int(num_inference_steps * (1 - calib_guidance_scale)):
                    with torch.enable_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
                        latents_copy = latents.detach().clone().to(device2)
                        z = torch.randn_like(latents_copy, device=latents_copy.device, dtype=latents_copy.dtype, requires_grad=True)
                        optimizer = torch.optim.AdamW([z], lr=lr)

                        for epoch in range(epochs):
                            total_loss = torch.tensor(0.0, device=device2)
                            z.data = torch.clamp(z.data, -eps, eps)
                            calibrated_latents = latents_copy + z

                            calibrated_latents = calibrated_latents / self.model.vae.config.scaling_factor
                            calibrated_image = self.model.vae.decode(calibrated_latents, return_dict=False)[0]

                            for m_i, (model_name, model) in enumerate(vision_encoders.items()):
                                calibrated_image_embeds = model.get_image_embeds(calibrated_image, is_train=True)  # ensure augment_mode is set to 'none'

                                loss_func = vision_encoder_loss_funcs[model_name]
                                loss_scale = getattr(model, 'style_weight', 1.0)

                                if model_name in ['csd_clip', 'vgg']:
                                    loss = loss_func(calibrated_image_embeds, style_image_embeds[m_i], loss_scale=loss_scale, is_pairwise=False, is_neg=False)
                                else:
                                    loss = loss_func(calibrated_image_embeds, style_content_image_embeds[m_i], loss_scale=loss_scale, is_pairwise=False, is_neg=True)

                                total_loss += loss * v_w[m_i]

                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()

                            # print(f'Step {t_i}, Epoch {epoch}, Loss: {total_loss.item():.4f}')

                        torch.cuda.synchronize(device2)
                        torch.cuda.empty_cache()

                    latents = latents + z.clamp(-eps, eps).detach().to(device1)
                #############################################################

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, t_i, t, callback_kwargs)

                    latents = callback_outputs.pop('latents', latents)
                    prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds)
                    add_text_embeds = callback_outputs.pop('add_text_embeds', add_text_embeds)
                    add_time_ids = callback_outputs.pop('add_time_ids', add_time_ids)

                # call the callback, if provided
                if t_i == len(timesteps) - 1 or ((t_i + 1) > num_warmup_steps and (t_i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and t_i % callback_steps == 0:
                        step_idx = t_i // getattr(self.model.scheduler, 'order', 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        ####################################### for style calibration
        self.model.vae.to(device1)
        #############################################################

        if not output_type == 'latent':
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.model.vae.dtype == torch.float16 and self.model.vae.config.force_upcast

            if needs_upcasting:
                self.model.upcast_vae()
                latents = latents.to(next(iter(self.model.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.model.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.model.vae = self.model.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.model.vae.config, 'latents_mean') and self.model.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.model.vae.config, 'latents_std') and self.model.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.model.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.model.vae.config.scaling_factor

            image = self.model.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.model.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == 'latent':
            # apply watermark if available
            if self.model.watermark is not None:
                image = self.model.watermark.apply_watermark(image)

            image = self.model.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.model.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    @torch.no_grad()
    def calib_call_2(
            self,
            ####################################### for style calibration
            style_image,
            style_content_image,
            vision_encoders: dict,
            vision_encoder_loss_funcs: dict,
            vision_weights: list[float] = None,
            calib_guidance_scale=0.1,
            lr=1e-2,
            epochs=5,
            eps=8/255,
            #############################################################
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = 'pil',
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ['latents'],
            **kwargs
    ):
        callback = kwargs.pop('callback', None)
        callback_steps = kwargs.pop('callback_steps', None)

        if callback is not None:
            deprecate(
                'callback',
                '1.0.0',
                'Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`',
            )
        if callback_steps is not None:
            deprecate(
                'callback_steps',
                '1.0.0',
                'Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`',
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.model.default_sample_size * self.model.vae_scale_factor
        width = width or self.model.default_sample_size * self.model.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.model.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self.model._guidance_scale = guidance_scale
        self.model._guidance_rescale = guidance_rescale
        self.model._clip_skip = clip_skip
        self.model._cross_attention_kwargs = cross_attention_kwargs
        self.model._denoising_end = denoising_end
        self.model._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device1 = self.model._execution_device

        ####################################### for style calibration
        device2 = next(iter(vision_encoders.values())).device
        dtype2 = next(iter(vision_encoders.values())).dtype

        self.model.upcast_vae()  # make sure vae is fp32
        self.model.vae.to(device2)

        if vision_weights is None:
            v_w = torch.ones(len(vision_encoders), device=device2) / len(vision_encoders)  # average weights
        else:
            assert len(vision_weights) == len(vision_encoders)
            v_w = torch.tensor(vision_weights, device=device2)

        style_image_embeds = []
        style_content_image_embeds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype2):
            for t_i, model_name in enumerate(vision_encoders):
                model = vision_encoders[model_name]
                image_embeds = model.get_image_embeds(style_image, is_train=False)
                style_image_embeds.append(image_embeds)
                image_embeds = model.get_image_embeds(style_content_image, is_train=False)
                style_content_image_embeds.append(image_embeds)
        #############################################################

        # 3. Encode input prompt
        lora_scale = (
            self.model.cross_attention_kwargs.get('scale', None) if self.model.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device1,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.model.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.model.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler, num_inference_steps, device1, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device1,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.model.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.model.text_encoder_2.config.projection_dim

        add_time_ids = self.model._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.model._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.model.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device1)
        add_text_embeds = add_text_embeds.to(device1)
        add_time_ids = add_time_ids.to(device1).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.model.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device1,
                batch_size * num_images_per_prompt,
                self.model.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.model.denoising_end is not None
            and isinstance(self.model.denoising_end, float)
            and self.model.denoising_end > 0
            and self.model.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.model.scheduler.config.num_train_timesteps
                    - (self.model.denoising_end * self.model.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.model.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.model.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.model.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.unet.config.time_cond_proj_dim
            ).to(device=device1, dtype=latents.dtype)

        self.model._num_timesteps = len(timesteps)
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for t_i, t in enumerate(timesteps):
                if self.model.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents

                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs['image_embeds'] = image_embeds
                noise_pred = self.model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.model.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.model.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.model.do_classifier_free_guidance and self.model.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.model.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                ####################################### for style calibration
                if t_i >= int(num_inference_steps * (1 - calib_guidance_scale)):
                    latents_dtype = latents.dtype
                    image_t = self.model.vae.decode(latents.to(device=device2, dtype=next(iter(self.model.vae.post_quant_conv.parameters())).dtype) / self.model.vae.config.scaling_factor).sample
                    image_t = (image_t / 2 + 0.5).clamp(0, 1)

                    with torch.enable_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
                        z = torch.randn_like(image_t, device=image_t.device, dtype=image_t.dtype, requires_grad=True)
                        optimizer = torch.optim.AdamW([z], lr=lr)

                        for epoch in range(epochs):
                            z.data = torch.clamp(z.data, -eps, eps)
                            calibrated_image_t = image_t + z
                            total_loss = torch.tensor(0.0, device=device2)

                            for m_i, (model_name, model) in enumerate(vision_encoders.items()):
                                calibrated_image_embeds = model.get_image_embeds(calibrated_image_t, is_train=True)  # make sure augment_mode is set to 'none'

                                loss_func = vision_encoder_loss_funcs[model_name]
                                loss_scale = getattr(model, 'style_weight', 1.0)

                                if model_name in ['csd_clip', 'vgg']:
                                    loss = loss_func(calibrated_image_embeds, style_image_embeds[m_i], loss_scale=loss_scale, is_pairwise=False, is_neg=False)
                                else:
                                    loss = loss_func(calibrated_image_embeds, style_content_image_embeds[m_i], loss_scale=loss_scale, is_pairwise=False, is_neg=True)

                                total_loss += loss * v_w[m_i]

                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()

                            # print(f'Step {t_i}, Epoch {epoch}, Loss: {total_loss.item():.4f}')

                    calibrated_image_t = image_t + z.clamp(-eps, eps).detach()
                    calibrated_image_t = 2.0 * calibrated_image_t - 1.0
                    latents = self.model.vae.encode(calibrated_image_t).latent_dist.sample() * self.model.vae.config.scaling_factor
                    latents = latents.to(device=device1, dtype=latents_dtype)
                #############################################################

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, t_i, t, callback_kwargs)

                    latents = callback_outputs.pop('latents', latents)
                    prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds)
                    add_text_embeds = callback_outputs.pop('add_text_embeds', add_text_embeds)
                    add_time_ids = callback_outputs.pop('add_time_ids', add_time_ids)

                # call the callback, if provided
                if t_i == len(timesteps) - 1 or ((t_i + 1) > num_warmup_steps and (t_i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and t_i % callback_steps == 0:
                        step_idx = t_i // getattr(self.model.scheduler, 'order', 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        ####################################### for style calibration
        self.model.vae.to(device=device1)
        latents = latents.to(next(iter(self.model.vae.post_quant_conv.parameters())).dtype)
        #############################################################

        if not output_type == 'latent':
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.model.vae.dtype == torch.float16 and self.model.vae.config.force_upcast

            if needs_upcasting:
                self.model.upcast_vae()
                latents = latents.to(next(iter(self.model.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.model.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.model.vae = self.model.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.model.vae.config, 'latents_mean') and self.model.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.model.vae.config, 'latents_std') and self.model.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.model.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.model.vae.config.scaling_factor

            image = self.model.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.model.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == 'latent':
            # apply watermark if available
            if self.model.watermark is not None:
                image = self.model.watermark.apply_watermark(image)

            image = self.model.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.model.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    @torch.no_grad()
    def text2image(
            self,
            prompt: str = '',
            negative_prompt: str = '',
            height=1024,
            width=1024,
            num_samples=1,
            num_inference_steps=50,
            guidance_scale=5.0
    ) -> list[Image.Image]:
        results = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images
        return results

    @torch.no_grad()
    def noising_image2image(
            self,
            image: Image.Image,
            image_guidance_scale=0.7,
            prompt: str = '',
            negative_prompt: str = '',
            prompt_guidance_scale=7.5,
            height=1024,
            width=1024,
            num_samples=1,
            num_inference_steps=50
    ) -> list[Image.Image]:
        results = self.refiner(
            image=image,
            strength=1 - image_guidance_scale,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=prompt_guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps
        ).images
        return results

    @torch.no_grad()
    def ip_adapter_image2image(
            self,
            image: Image.Image,
            image_guidance_scale: Union[float, dict] = 0.5,
            prompt: str = '',
            negative_prompt: str = '',
            prompt_guidance_scale=5.0,
            height=1024,
            width=1024,
            num_samples=1,
            num_inference_steps=50,
            edit_mode='none',
            **kwargs
    ) -> list[Image.Image]:
        self.model.set_ip_adapter_scale(image_guidance_scale)

        if edit_mode == 'calib_style':
            style_image = kwargs.get('style_image', None)
            style_content_image = kwargs.get('style_content_image', None)
            vision_encoders = kwargs.get('vision_encoders', None)
            vision_encoder_loss_funcs = kwargs.get('vision_encoder_loss_funcs', None)
            vision_weights = kwargs.get('vision_weights', None)
            calib_guidance_scale = kwargs.get('calib_guidance_scale', 0.1)
            lr = kwargs.get('lr', 1e-2)
            epochs = kwargs.get('epochs', 5)
            eps = kwargs.get('eps', 8/255)

            results = self.calib_call_2(
                style_image=style_image,
                style_content_image=style_content_image,
                vision_encoders=vision_encoders,
                vision_encoder_loss_funcs=vision_encoder_loss_funcs,
                vision_weights=vision_weights,
                calib_guidance_scale=calib_guidance_scale,
                lr=lr,
                epochs=epochs,
                eps=eps,
                ip_adapter_image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=prompt_guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps
            ).images
        else:
            results = self.model(
                ip_adapter_image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=prompt_guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps
            ).images
        return results

