<!-- accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --output_dir="/mnt/nas/athiwat/ControlNet_models/18-07-2025-lod3to1-canny_with_freestylegreyscale" \
  --resume_from_checkpoint="checkpoint-3000" \
  --train_data_dir="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think" \
  --image_column="target_images" \
  --conditioning_image_column="conditioning_images" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --num_train_epochs=64 \
  --validation_steps=200 \
  --validation_image "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/50/lod3/lod3_az120_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod3/lod3_az345_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod3/lod3_az330_el00.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/48/lod3/lod3_az060_el15.png" \
  --validation_prompt "a simplified building wireframe drawing" "a simplified building wireframe drawing" "a simplified building wireframe drawing" "a simplified building wireframe drawing" \
  --num_validation_images=4 \
  --tracker_run_name="18-07-2025-lod3to1-canny_with_freestylegreyscale_b4_ep16to32_controlnet" \
  --report_to=wandb \


## try grad accujmm
  accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --output_dir="/mnt/nas/athiwat/ControlNet_models/19-07-2025-lod3to1-canny_with_freestylegreyscale" \
  --train_data_dir="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think" \
  --image_column="target_images" \
  --conditioning_image_column="conditioning_images" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=16 \
  --dataloader_num_workers=4 \
  --num_train_epochs=128 \
  --validation_steps=100 \
  --validation_image "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/50/lod3/lod3_az120_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod3/lod3_az345_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod3/lod3_az330_el00.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/48/lod3/lod3_az060_el15.png" \
  --validation_prompt "a simplified building wireframe drawing" "a simplified building wireframe drawing" "a simplified building wireframe drawing" "a simplified building wireframe drawing" \
  --num_validation_images=4 \
  --tracker_run_name="19-07-2025-lod3to1-canny_with_freestylegreyscale_b16_gradaccum16_ep128_controlnet" \
  --report_to=wandb \


### lod2 to lod1
  accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --output_dir="/mnt/nas/athiwat/ControlNet_models/22-07-2025-lod2to1-canny_with_freestylegreyscale" \
  --train_data_dir="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think" \
  --image_column="target_images" \
  --conditioning_image_column="conditioning_images" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --num_train_epochs=16 \
  --validation_steps=200 \
  --validation_image "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/50/lod2/lod2_az120_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod2/lod2_az345_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod2/lod2_az330_el00.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/48/lod2/lod2_az060_el15.png" \
  --validation_prompt "A simplified architectural sketch of a building, with only black boundary lines and rough outlines on a white background. No colors, no shading." "A simplified architectural sketch of a building, with only black boundary lines and rough outlines on a white background. No colors, no shading." "A simplified architectural sketch of a building, with only black boundary lines and rough outlines on a white background. No colors, no shading." "A simplified architectural sketch of a building, with only black boundary lines and rough outlines on a white background. No colors, no shading." \
  --num_validation_images=4 \
  --tracker_run_name="22-07-2025-lod2to1-canny_with_freestylegreyscale_b4_gradaccum4_ep16_controlnet" \
  --report_to=wandb \

### stabilityai/stable-diffusion-2-1 lod2 to lod 1
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --output_dir="/mnt/nas/athiwat/ControlNet_models/22-07-2025-sd21-lod2to1-canny_with_freestylegreyscale" \
  --train_data_dir="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think" \
  --image_column="target_images" \
  --conditioning_image_column="conditioning_images" \
  --resolution=768 \
  --learning_rate=1e-5 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --num_train_epochs=16 \
  --validation_steps=200 \
  --validation_image "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/50/lod2/lod2_az120_el30.png" "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/49/lod2/lod2_az330_el00.png" \
  --validation_prompt "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients" "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients" \
  --num_validation_images=2 \
  --tracker_run_name="22-07-2025-sd21-lod2to1-canny_with_freestylegreyscale_b4_accum4_ep16" \
  --report_to=wandb