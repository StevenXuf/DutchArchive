import io
import base64

def convert_pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64

def get_prompts(image):
    base64_image = convert_pil_to_base64(image)
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{base64_image}",
                },
                {"type": "text", 
                "text": """
                    Describe the image in accurate and detailed manner and distinguish as much historical information as possible, including events, persons, sourrounding environment, and possibly related historical background. 
                    Use no more than 100 words in English.
                    Only output your description.
                """
                },
            ],
        }
    ]
    return message