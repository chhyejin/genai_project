from PIL import Image

def load_image(image_path):
    if isinstance(image_path, Image.Image):
        return image_path
    return Image.open(image_path).convert("RGB")

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []

    for msg in messages:
        contents = msg["content"]
        for content in contents:
            if content["type"] == "image":
                image_path = content["image"]
                image = load_image(image_path)
                image_inputs.append(image)
            elif content["type"] == "video":
                # 동영상 지원 필요 시 여기에 처리 추가
                raise NotImplementedError("Video input not supported in this example.")

    return image_inputs, video_inputs
