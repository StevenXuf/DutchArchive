import fire
import torch

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig, set_seed
from qwen_vl_utils import process_vision_info

from prompts import get_prompts
from dataset import get_archive_loader
from configuration import get_default_config
from utils import get_image_transform

def generate_captions(cfg, **kwargs):
    image_folder = kwargs.get('image_folder',cfg['IMAGE_FOLDER'])
    anno_folder = kwargs.get('anno_folder',cfg['ANNOTATION_FOLDER'])
    batch_size = kwargs.get('batch_size',cfg['BATCH_SIZE'])
    image_size = kwargs.get('image_size',cfg['IMAGE_SIZE'])
    mean = kwargs.get('mean',cfg['MEAN'])
    std = kwargs.get('std',cfg['STD'])
    device = torch.device(f"cuda:{kwargs['device']}") if kwargs.get('device') is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temperature = kwargs.get('temperature', cfg['TEXT-GENERATION']['GLOBAL']['TEMPERATURE'])
    top_p = kwargs.get('top_p', cfg['TEXT-GENERATION']['GLOBAL']['TOP_P'])
    llm_top_k = kwargs.get('llm_top_k', cfg['TEXT-GENERATION']['GLOBAL']['TOP_K'])
    max_new_tokens = kwargs.get('max_new_tokens', cfg['TEXT-GENERATION']['GLOBAL']['MAX_NEW_TOKENS'])
    gen_config = GenerationConfig(do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=llm_top_k,
                                    max_new_tokens=max_new_tokens
                                )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', 
                                                                torch_dtype=torch.bfloat16, 
                                                                device_map={"": device}, 
                                                                attn_implementation='flash_attention_2'
                                                                ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                              use_fast=True,
                                              padding_side='left')

    img_transform = get_image_transform(image_size, mean, std)
    dataloader = get_archive_loader(image_folder=image_folder, anno_folder=anno_folder, batch_size=batch_size, transform=img_transform)

    for i,batch in enumerate(tqdm(dataloader)):
        messages = [get_prompts(image) for image in batch['pil_image']]

        text = [processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side='left'
        ).to(model.device)

        generated_ids = model.generate(**inputs, generation_config=gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

def main(**kwargs):
    cfg = get_default_config("config.yaml")
    seed = kwargs.get('seed', cfg['SEED'])
    set_seed(seed)
    generate_captions(cfg, **kwargs)

if __name__ == '__main__':
    fire.Fire(main)