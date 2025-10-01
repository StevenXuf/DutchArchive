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
            "role": "system", 
            "content": 
            """
                You are an historian good at describing historical images. 
                Given a historical image, observe the image and connect the image with the historical background indicated by the image. 
                Describe the image in accurate and detailed manner and distinguish as much historical information as possible, including events, persons, sourrounding environment, and possibly related historical background. 
                For example, if the image has a well-known celebrity, you should distinguish the person.
                If the image has a well-known place, you should distinguish the place.
                If the image has a well-known event, you should distinguish the event.
                Write 1 to 3 coherent and complete sentences in English.
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{base64_image}",
                },
                {"type": "text", 
                "text": """
                    Format your description in a complete, structured way.
                    Only output your description in 1 to 3 complete sentences.
                    Now describe the given image.
                """
                },
            ],
        }
    ]
    return message