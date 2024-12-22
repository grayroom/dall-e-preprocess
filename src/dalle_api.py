from typing import List

from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field

from .models.request_response import Ratio

# DALL·E 3의 최종 API 엔드포인트와 파라미터는 OpenAI 문서를 통해 확인 필요

ratio_lookup = {
    Ratio.SQAURE: "1024x1024",
    Ratio.LANDSCAPE: "1792x1024",
    Ratio.PORTRAIT: "1024x1792",
}


def generate_image(prompt: str, n: int = 1, ratio: Ratio = Ratio.SQAURE):
    """
    prompt를 바탕으로 DALL·E3에서 이미지를 생성하는 함수
    """
    if ratio not in ratio_lookup:
        raise ValueError(f"Invalid ratio: {ratio}\nAvailable ratios: {', '.join(ratio_lookup.keys())}")

    pp = preprocess_prompt_via_structured_output(prompt)
    final_prompt = compose_final_prompt(pp)

    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=final_prompt,
        n=n,
        size=ratio_lookup.get(ratio),
        quality="hd", # standard
        style="vivid" # vivid standard
    )

    pass

    return [(img.url, img.revised_prompt) for img in response.data]


class DallESchema(BaseModel):
    subject: list[str] = Field(description="Subject of the image")
    objects: list[str] = Field(description="Objects in the image")
    mood: list[str] = Field(description="Mood of the image")
    style: list[str] = Field(description="Style of the image")
    negative: list[str] = Field(description="Keywords that should not be included in the image")

def preprocess_prompt_via_structured_output(user_input: str):
    client = OpenAI()
    # (4) ChatCompletion.create에서 function_call 명시
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extracting structured data from your input..."},
            {"role": "user", "content": user_input},
        ],
        response_format=DallESchema
    )

    return response.choices[0].message.parsed


def compose_final_prompt(pp: DallESchema):
    return f"Subject: {pp.subject}\nObjects: {pp.objects}\nMood: {pp.mood}\nStyle: {pp.style}\nNegative: {pp.negative}\n"

if __name__ == "__main__":

    class EntitiesModel(BaseModel):
        attributes: List[str]
        colors: List[str]
        animals: List[str]

    client = OpenAI()

    with client.beta.chat.completions.stream(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract entities from the input text"},
                {
                    "role": "user",
                    "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes",
                },
            ],
            response_format=EntitiesModel,
    ) as stream:
        """
            아래와 같은 완결된 형태의 JSON을 얻을 수 있음
            content.delta parsed: {}
            content.delta parsed: {'attributes': []}
            content.delta parsed: {'attributes': []}
            content.delta parsed: {'attributes': ['quick']}
            content.delta parsed: {'attributes': ['quick']}
            content.delta parsed: {'attributes': ['quick', 'lazy']}
            content.delta parsed: {'attributes': ['quick', 'lazy']}
            content.delta parsed: {'attributes': ['quick', 'lazy']}
            content.delta parsed: {'attributes': ['quick', 'lazy']}
            content.delta parsed: {'attributes': ['quick', 'lazy']}
            content.delta parsed: {'attributes': ['quick', 'lazy', 'piercing blue eyes']}
            content.delta parsed: {'attributes': ['quick', 'lazy', 'piercing blue eyes']}
            content.delta parsed: {'attributes': ['quick', 'lazy', 'piercing blue eyes']}
        """
        for event in stream:
            if event.type == "content.delta":
                if event.parsed is not None:
                    # Print the parsed data as JSON
                    print("content.delta parsed:", event.parsed)
            elif event.type == "content.done":
                print("content.done")
            elif event.type == "error":
                print("Error in stream:", event.error)

    final_completion = stream.get_final_completion()
    print("Final completion:", final_completion)